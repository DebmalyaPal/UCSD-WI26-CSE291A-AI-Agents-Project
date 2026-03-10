"""
utils.py  –  HMAS_3
====================
Key differences from HMAS_2:
  - propose_actions() is now REGIONAL: the central planner prompt is scoped to
    agents in a single region. Called once per active region per timestep.
  - provide_feedback() is unchanged in logic but now labels agents with their region.
  - global_leader_plan() handles cross-region reallocation decisions.
  - translate_action() is identical to HMAS_2 (no changes needed).
"""

import os
import re
from openai import OpenAI
from pydantic import BaseModel
from typing import List, Tuple, Dict, Optional

from crew_algorithms.wildfire_alg.algorithms.HMAS_3.agent import Agent, ROLE_REGIONAL_LEADER, ROLE_GLOBAL_LEADER
from crew_algorithms.wildfire_alg.algorithms.HMAS_3.region_manager import RegionManager, ReallocRequest

# ---------------------------------------------------------------------------
# Action model  (identical to HMAS_2)
# ---------------------------------------------------------------------------

class Action(BaseModel):
    type: int
    param_1: int
    param_2: int
    description: str

    def print_action(self) -> None:
        print([self.type, self.param_1, self.param_2, self.description])


# ---------------------------------------------------------------------------
# translate_action  (identical to HMAS_2 — untouched)
# ---------------------------------------------------------------------------

def translate_action(action_str: str, type: int, global_data: dict) -> Action:
    type_string = {0: "firefighter", 1: "bulldozer", 2: "drone", 3: "helicopter"}.get(type, "firefighter")

    prompt_path = os.path.join("algorithms", "HMAS_3", "prompts", "translator", f"{type_string}_translator.txt")
    with open(prompt_path, "r", encoding="utf-8") as file:
        translator_prompt = file.read().replace("ACTION", action_str)

    system_message = (
        "You are the controller of a highly trained embodied agent within a grid forest world. "
        "Your job is to convert a string action into a structured format for robotic control."
    )

    client = OpenAI(api_key=global_data.get("api_key"))
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": system_message},
            {"role": "user", "content": translator_prompt},
        ],
        temperature=0,
    )
    global_data["api_calls"] += 1
    global_data["input_tokens"] += response.usage.prompt_tokens
    global_data["output_tokens"] += response.usage.completion_tokens
    result = response.choices[0].message.content

    def extract(tag: str) -> str:
        m = re.search(fr"<{tag}>\s*(.*?)\s*</{tag}>", result, re.DOTALL)
        if not m:
            raise ValueError(f"Missing tag {tag}")
        return m.group(1).strip()

    try:
        return Action(
            type=int(extract("type")),
            param_1=int(extract("param_1")),
            param_2=int(extract("param_2")),
            description=extract("description"),
        )
    except Exception:
        return Action(type=0, param_1=0, param_2=0, description="ERROR EXECUTING: " + result)


# ---------------------------------------------------------------------------
# Regional propose_actions
# ---------------------------------------------------------------------------

def propose_regional_actions(
    region_id: int,
    regional_leader: Agent,
    region_workers: List[Agent],
    global_data: dict,
    past_conversation: list,
) -> Tuple[Dict[str, str], list]:
    """
    The regional leader acts as the central planner for its own 20x20 region.
    Runs the HMAS_2 planning loop scoped to only the agents in this region.

    Returns:
        proposed_actions: {AGENT_<id>: action_string}
        messages: updated conversation history
    """
    region = global_data["region_manager"].regions[region_id]
    all_region_agents = [a for a in region_workers]  # workers only; leader observes but also gets a slot

    # Regional leader also gets an action slot (it is physically present in the grid)
    all_region_agents_with_leader = [regional_leader] + all_region_agents

    if len(past_conversation) == 0:
        current_state = _build_state_dict(all_region_agents_with_leader)

        generate_action_string = f"""
You are the Regional Planner (AGENT_{regional_leader.id}) directing agents within Region {region_id}
(grid area x:[{region.x_min},{region.x_max}), y:[{region.y_min},{region.y_max})).

Your team's task is:
{global_data['agents'][0].current_task}
---

Your team's previous state-action pairs:
{global_data["step_history"]}
---

Your team's current state and available actions (region {region_id} only):
{current_state}
---

Provide the next best action for each agent in your region. 
One action per agent. Use explicit coordinate targets.

Format:
<reasoning>(reasoning)</reasoning>
<AGENT_X>'action'</AGENT_X> for each agent.
"""
        system_message = (
            f"You are Regional Planner AGENT_{regional_leader.id} directing agents in Region {region_id} "
            f"of a cooperative wildfire response task."
        )
        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": generate_action_string},
        ] + past_conversation
    else:
        while len(past_conversation) > 8:
            past_conversation.pop(2)
            past_conversation.pop(2)
        messages = past_conversation

    response = global_data["client"].chat.completions.create(
        model="gpt-4o",
        messages=messages,
        temperature=0,
    )
    global_data["api_calls"] += 1
    global_data["input_tokens"] += response.usage.prompt_tokens
    global_data["output_tokens"] += response.usage.completion_tokens
    content = response.choices[0].message.content
    messages.append({"role": "assistant", "content": content})

    proposed_actions: Dict[str, str] = {}
    for agent in all_region_agents_with_leader:
        tag = f"AGENT_{agent.id}"
        m = re.search(fr"<{tag}>\s*(.*?)\s*</{tag}>", content, re.DOTALL)
        if m:
            proposed_actions[tag] = m.group(1).strip()

    if len(proposed_actions) != len(all_region_agents_with_leader):
        print(f"[Region {region_id}] INVALID action count — replanning")
        return propose_regional_actions(region_id, regional_leader, region_workers, global_data, past_conversation)

    _log_plan(global_data, region_id, messages)
    return proposed_actions, messages


# ---------------------------------------------------------------------------
# provide_feedback  (same logic as HMAS_2, region-scoped context)
# ---------------------------------------------------------------------------

def provide_feedback(
    agent: Agent,
    region_id: int,
    region_agents: List[Agent],
    proposed_actions: Dict[str, str],
    global_data: dict,
) -> str:
    client = global_data["client"]
    type_string = {0: "Firefighter", 1: "Bulldozer", 2: "Drone", 3: "Helicopter"}.get(agent.type, "Agent")

    current_state = _build_state_dict(region_agents)
    region = global_data["region_manager"].regions[region_id]

    generate_feedback_string = f"""
You are AGENT_{agent.id}, a {type_string} in Region {region_id}
(grid x:[{region.x_min},{region.x_max}), y:[{region.y_min},{region.y_max})).

Your team's task:
{global_data['agents'][0].current_task}
---

Previous state-action pairs:
{global_data["step_history"]}
---

Current state and available actions (region {region_id}):
{current_state}
---

Action plan from the regional planner:
{proposed_actions}
---

Provide feedback specifically about your assigned action.
If satisfactory, respond only with: ACCEPT

<reasoning>(reasoning)</reasoning>
<feedback>'feedback'</feedback>
"""
    system_message = (
        f"You are AGENT_{agent.id}, a {type_string} in Region {region_id}. "
        f"Evaluate the plan from your regional planner."
    )
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": system_message},
            {"role": "user", "content": generate_feedback_string},
        ],
        temperature=0,
    )
    global_data["api_calls"] += 1
    global_data["input_tokens"] += response.usage.prompt_tokens
    global_data["output_tokens"] += response.usage.completion_tokens
    result = response.choices[0].message.content
    agent.log_chat("Providing Feedback (Region)", [("user", generate_feedback_string), ("assistant", result)])

    m = re.search(r"<feedback>\s*(.*?)\s*</feedback>", result, re.DOTALL)
    if not m:
        print(f"[Region {region_id}] No feedback tag found for AGENT_{agent.id}")
        return "ACCEPT"
    return m.group(1).strip()


# ---------------------------------------------------------------------------
# Global leader: reallocation planning
# ---------------------------------------------------------------------------

def global_leader_manage_reallocation(
    global_agent: Agent,
    global_data: dict,
) -> List[Tuple[Agent, int, int]]:
    """
    The global leader surveys all regions, identifies any with fires but no workers,
    and resolves pending reallocation requests from regional leaders.

    Returns list of (worker, from_region_id, to_region_id) enacted moves.
    """
    region_manager: RegionManager = global_data["region_manager"]

    # Step 1: detect empty regions with fire context and auto-raise requests
    for rid, region in region_manager.regions.items():
        if not region.active:
            continue
        if len(region.workers) == 0 and region.leader is not None:
            # Check if leader's perception mentions fire
            ldr = region.leader
            if ldr.last_perception and ("fire" in ldr.last_perception.lower() or "ignited" in ldr.last_perception.lower()):
                if not region.pending_request:
                    req = ReallocRequest(
                        from_region_id=rid,
                        urgency="HIGH",
                        fire_cells=[],  # leader will provide details in its own plan
                        workers_needed=1,
                    )
                    region_manager.submit_reallocation_request(req)
                    print(f"[GlobalLeader] Auto-raised HIGH realloc request for empty region {rid}")

    # Step 2: have global leader reason about pending requests (LLM call)
    if region_manager.pending_requests:
        _global_leader_llm_arbitration(global_agent, global_data)

    # Step 3: resolve requests via region_manager logic
    moves = region_manager.resolve_reallocation_requests()

    # Step 4: handle recall — workers in resolved regions whose fire is gone
    _check_and_recall_workers(global_data)

    return moves


def _global_leader_llm_arbitration(global_agent: Agent, global_data: dict):
    """
    Optional LLM call: global leader reasons about which pending requests to
    prioritise and whether to merge/split assignments.
    Result is advisory — actual movement is handled by region_manager.resolve_reallocation_requests().
    """
    region_manager: RegionManager = global_data["region_manager"]
    roster_summary = region_manager.summary()
    requests_summary = str(region_manager.pending_requests)

    prompt = f"""
You are the Global Leader (AGENT_{global_agent.id}) coordinating wildfire response across all regions.

Current region roster:
{roster_summary}

Pending worker reallocation requests:
{requests_summary}

Your job:
1. Confirm or adjust urgency priorities for pending requests.
2. Identify any donor regions that can spare workers without compromising their own coverage.
3. Output your decisions in the format:

<reasoning>(your analysis)</reasoning>
<decisions>
  SEND AGENT_<id> FROM REGION_<from> TO REGION_<to>
  ... (one line per move, or NONE if no moves needed)
</decisions>
"""
    response = global_data["client"].chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are the Global Leader coordinating multi-region wildfire response."},
            {"role": "user", "content": prompt},
        ],
        temperature=0,
    )
    global_data["api_calls"] += 1
    global_data["input_tokens"] += response.usage.prompt_tokens
    global_data["output_tokens"] += response.usage.completion_tokens
    content = response.choices[0].message.content
    global_agent.log_chat("Global Leader Arbitration", [("user", prompt), ("assistant", content)])

    # Log to file
    log_path = os.path.join(global_data["path"], "global_leader.txt")
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(f"--- Global Leader Arbitration ---\n{content}\n\n")


def _check_and_recall_workers(global_data: dict):
    """
    After reallocation, check if any temporarily reallocated workers should be
    recalled to their home region (fire suppressed in their temporary region).
    """
    region_manager: RegionManager = global_data["region_manager"]
    for rid, region in region_manager.regions.items():
        if not region.active:
            continue
        # Workers whose home is elsewhere and current region appears fire-free
        fire_present = any(
            w.last_perception and ("fire" in w.last_perception.lower() or "ignited" in w.last_perception.lower())
            for w in region.workers
        )
        if not fire_present:
            displaced = [w for w in region.workers if w.home_region_id != rid]
            for worker in displaced:
                region_manager.recall_worker(worker)
                print(f"[GlobalLeader] Recalled AGENT_{worker.id} from region {rid} to home {worker.home_region_id}")


# ---------------------------------------------------------------------------
# Regional leader empty-region check + request
# ---------------------------------------------------------------------------

def regional_leader_check_and_request(
    region_id: int,
    regional_leader: Agent,
    global_data: dict,
) -> Optional[ReallocRequest]:
    """
    Called at start of each timestep by each regional leader.
    If the region has no workers but the leader perceives fire, raise a request.
    """
    region_manager: RegionManager = global_data["region_manager"]
    region = region_manager.regions[region_id]

    if len(region.workers) > 0:
        return None  # region is staffed

    if regional_leader.last_perception is None:
        return None

    if "fire" in regional_leader.last_perception.lower() or "ignited" in regional_leader.last_perception.lower():
        if not region.pending_request:
            req = ReallocRequest(
                from_region_id=region_id,
                urgency="HIGH",
                fire_cells=[],
                workers_needed=1,
            )
            region_manager.submit_reallocation_request(req)
            print(f"[RegionalLeader AGENT_{regional_leader.id}] Submitted realloc request for region {region_id}")
            return req
    return None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_state_dict(agents: List[Agent]) -> dict:
    state = {}
    for agent in agents:
        if agent.type == 0 and len(agent.extra_variables) > 2 and agent.extra_variables[2] == 1:
            state[f"AGENT_{agent.id}"] = {
                "perception": agent.last_perception,
                "available_actions": "- Do nothing since you are in a helicopter.",
            }
            continue
        type_string = {0: "firefighter", 1: "bulldozer", 2: "drone", 3: "helicopter"}.get(agent.type, "firefighter")
        desc_path = os.path.join("algorithms", "HMAS_3", "prompts", "descriptions", f"{type_string}_description.txt")
        try:
            with open(desc_path, "r", encoding="utf-8") as f:
                abilities_string = f.read()
        except FileNotFoundError:
            abilities_string = "- Move to target location.\n- Do nothing."
        state[f"AGENT_{agent.id}"] = {
            "perception": agent.last_perception,
            "available_actions": abilities_string,
        }
    return state


def _log_plan(global_data: dict, region_id: int, messages: list):
    if len(messages) == 3:
        chat_string = f"Region {region_id}: Proposing Action Plan\n" + "-" * 20 + "\n\n"
        for m in messages:
            chat_string += f"{m['role']}\n\n{m['content']}\n\n"
    else:
        chat_string = f"Region {region_id}: Revising Action Plan\n" + "-" * 20 + "\n\n"
        for m in messages[-2:]:
            chat_string += f"{m['role']}\n-----\n{m['content']}\n-----\n\n"
    chat_string += "-" * 20 + "\nEND CHAT\n\n\n"
    filepath = os.path.join(global_data["path"], f"region_{region_id}_leader.txt")
    with open(filepath, "a", encoding="utf-8") as f:
        f.write(chat_string)