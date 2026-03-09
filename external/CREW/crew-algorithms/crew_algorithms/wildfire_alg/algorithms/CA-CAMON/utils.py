import re
from openai import OpenAI
from pydantic import BaseModel
from crew_algorithms.wildfire_alg.algorithms.CAMON.agent import Agent

# ============================================================
# [CA-CAMON]  Capability constants
# ============================================================

AGENT_TYPE_NAMES = {
    0: "firefighter",
    1: "bulldozer",
    2: "drone",
    3: "helicopter",
}

VISION_RANGE = {
    "helicopter": 20,
    "drone":      15,
    "firefighter": 5,
    "bulldozer":   5,
}

MOBILITY_BONUS = {
    "helicopter": 5,
    "drone":      3,
    "firefighter": 0,
    "bulldozer":   0,
}

# Only these types may NORMALLY hold leadership.
# Workers can act as reluctant regional leaders only when no eligible agent
# exists in their region.
LEADER_ELIGIBLE_TYPES = {"helicopter", "drone"}

# Side-length of every region tile (must be <= helicopter vision = 20)
REGION_SIZE = 20


# ============================================================
# [CA-CAMON]  Leadership helpers
# ============================================================

def get_agent_type_name(agent: Agent) -> str:
    return AGENT_TYPE_NAMES.get(agent.type, "firefighter")


def leadership_score(agent: Agent) -> int:
    """
    score = vision_range + mobility_bonus.
    Workers always return 0 so they never win a score comparison against
    eligible types, but can still act as reluctant regional leaders when
    they are the only agents in a region.
    """
    type_name = get_agent_type_name(agent)
    if type_name not in LEADER_ELIGIBLE_TYPES:
        return 0
    return VISION_RANGE[type_name] + MOBILITY_BONUS[type_name]


def can_be_leader(agent: Agent) -> bool:
    """True iff the agent type is eligible to lead (helicopter/drone)."""
    return get_agent_type_name(agent) in LEADER_ELIGIBLE_TYPES


def should_transfer_leadership(proposer: Agent, current_leader: Agent) -> bool:
    """
    Regional leadership transfer rule:
        eligible(proposer) AND score(proposer) > score(current_leader)
    Workers never satisfy the first condition, so they never trigger transfer.
    """
    if not can_be_leader(proposer):
        return False
    return leadership_score(proposer) > leadership_score(current_leader)


# ============================================================
# [CA-CAMON]  Regional partition
# ============================================================

def compute_regions(grid_size: int) -> list:
    """
    Tile the square grid into REGION_SIZE x REGION_SIZE blocks.

    Each region dict:
        id               – unique index
        x_min/y_min/x_max/y_max  – inclusive cell bounds
        leader           – Agent acting as regional leader, or None (dormant)
        reluctant_leader – True when the leader is a worker fallback
    """
    regions = []
    rid = 0
    x = 0
    while x < grid_size:
        y = 0
        while y < grid_size:
            regions.append({
                "id":               rid,
                "x_min":            x,
                "y_min":            y,
                "x_max":            min(x + REGION_SIZE - 1, grid_size - 1),
                "y_max":            min(y + REGION_SIZE - 1, grid_size - 1),
                "leader":           None,
                "reluctant_leader": False,
            })
            rid += 1
            y += REGION_SIZE
        x += REGION_SIZE
    return regions


def _agent_in_region(region: dict, agent: Agent) -> bool:
    if agent.last_position is None:
        return False
    px, py = agent.last_position[0], agent.last_position[1]
    return (region["x_min"] <= px <= region["x_max"] and
            region["y_min"] <= py <= region["y_max"])


def get_region_for_agent(regions: list, agent: Agent):
    """Return the region that contains the agent, or None."""
    for r in regions:
        if _agent_in_region(r, agent):
            return r
    return None


def get_region_leader(region: dict):
    """Return the regional leader Agent, or None if the region is dormant."""
    return region.get("leader")


def assign_regional_leaders(regions: list, agents: list) -> None:
    """
    [CA-CAMON] Rebuild region["leader"] for every region from scratch.
    Called each timestep AFTER the dead-agent removelist has been applied,
    so dead agents are already absent from `agents`.

    Priority inside each region:
        1. Helicopter with the highest score.
        2. Drone with the highest score.
        3. Reluctant leader: any worker — only when no eligible agent present.
        4. No agents → region["leader"] = None  (dormant).
    """
    for region in regions:
        candidates = [a for a in agents if _agent_in_region(region, a)]

        if not candidates:
            region["leader"]           = None
            region["reluctant_leader"] = False
            continue

        chosen = None
        for preferred in ("helicopter", "drone"):
            typed = [c for c in candidates if get_agent_type_name(c) == preferred]
            if typed:
                chosen = max(typed, key=leadership_score)
                break

        if chosen is not None:
            region["leader"]           = chosen
            region["reluctant_leader"] = False
        else:
            region["leader"]           = max(candidates, key=lambda a: a.id)
            region["reluctant_leader"] = True
            print(
                f"[CA-CAMON] Region {region['id']}: no eligible leader. "
                f"AGENT_{region['leader'].id} ({get_agent_type_name(region['leader'])}) "
                f"acting as reluctant leader."
            )


def check_and_reassign_regional_leader(region: dict, agents: list) -> None:
    """
    [CA-CAMON] Called each timestep after assign_regional_leaders.

    Case A: current leader has walked out of the region → reassign.
    Case B: a reluctant-leader region now has an eligible agent → upgrade.
    """
    current = region.get("leader")

    # Case A: leader left the region
    if current is not None and not _agent_in_region(region, current):
        print(
            f"[CA-CAMON] Region {region['id']}: "
            f"AGENT_{current.id} left the region. Reassigning."
        )
        assign_regional_leaders([region], agents)
        return

    # Case B: reluctant leader can be upgraded to an eligible agent
    if region.get("reluctant_leader") and current is not None:
        for preferred in ("helicopter", "drone"):
            eligible = [
                a for a in agents
                if _agent_in_region(region, a) and get_agent_type_name(a) == preferred
            ]
            if eligible:
                new_leader = max(eligible, key=leadership_score)
                print(
                    f"[CA-CAMON] Region {region['id']}: upgrading reluctant leader "
                    f"AGENT_{current.id} → AGENT_{new_leader.id} "
                    f"({get_agent_type_name(new_leader)})."
                )
                region["leader"]           = new_leader
                region["reluctant_leader"] = False
                return


# ============================================================
# [CA-CAMON]  Global leader helpers
# ============================================================

def elect_global_leader(agents: list, regions: list, exclude: Agent = None):
    """
    [CA-CAMON] Elect the highest-scoring eligible agent as global leader.

    The global leader is chosen from agents that are also regional leaders,
    giving priority to the highest-scoring one. If no regional leader is
    eligible, fall back to any eligible agent, then to any agent.

    `exclude` is used when the previous global leader has died — we skip
    that agent even if it is still referenced in stale region dicts.

    Priority:
        1. Helicopter regional leader with highest score.
        2. Drone regional leader with highest score.
        3. Any helicopter (not necessarily a regional leader).
        4. Any drone.
        5. First surviving agent (last-resort fallback).
    """
    if not agents:
        return None

    # Collect current regional leaders (alive, not excluded)
    regional_leaders = {
        region["leader"]
        for region in regions
        if region.get("leader") is not None and region["leader"] is not exclude
    }

    for preferred in ("helicopter", "drone"):
        candidates = [
            a for a in regional_leaders
            if get_agent_type_name(a) == preferred and a in agents
        ]
        if candidates:
            return max(candidates, key=leadership_score)

    # Fall back to any eligible agent
    for preferred in ("helicopter", "drone"):
        candidates = [
            a for a in agents
            if get_agent_type_name(a) == preferred and a is not exclude
        ]
        if candidates:
            return max(candidates, key=leadership_score)

    # Last resort
    surviving = [a for a in agents if a is not exclude]
    if surviving:
        print("[CA-CAMON] WARNING: No eligible global leader found; falling back.")
        return surviving[0]
    return None


def handle_global_leader_death(
    dead_global_leader: Agent,
    agents: list,
    regions: list,
) -> Agent:
    """
    [CA-CAMON] Called when the global leader has been destroyed.

    Steps:
        1. Elect a new global leader from surviving regional leaders
           (helicopter > drone > fallback), excluding the dead agent.
        2. The promoted agent was a regional leader; its region now needs
           a new regional leader → call assign_regional_leaders for that
           region alone so the next-best agent in the region takes over.

    Returns the newly elected global leader Agent.
    """
    print(
        f"[CA-CAMON] Global leader AGENT_{dead_global_leader.id} "
        f"({get_agent_type_name(dead_global_leader)}) destroyed. Electing replacement."
    )

    new_global = elect_global_leader(agents, regions, exclude=dead_global_leader)
    if new_global is None:
        print("[CA-CAMON] WARNING: No agents remain. Cannot elect global leader.")
        return None

    # Find the region the promoted agent was leading and fill the vacancy
    vacated_region = new_global.assigned_region
    if vacated_region is not None:
        # Temporarily clear so assign_regional_leaders picks the next agent
        vacated_region["leader"] = None
        assign_regional_leaders([vacated_region], agents)
        new_regional = vacated_region.get("leader")
        if new_regional and new_regional is not new_global:
            new_regional.assigned_region = vacated_region
            print(
                f"[CA-CAMON] Region {vacated_region['id']}: "
                f"AGENT_{new_regional.id} ({get_agent_type_name(new_regional)}) "
                f"fills vacancy left by new global leader AGENT_{new_global.id}."
            )
        elif new_regional is new_global:
            # Assign_regional_leaders keeps picking the same agent because
            # it's still physically in the region. Force it to look for others.
            candidates = [
                a for a in agents
                if _agent_in_region(vacated_region, a) and a is not new_global
            ]
            if candidates:
                # Best non-global candidate
                for preferred in ("helicopter", "drone"):
                    typed = [c for c in candidates if get_agent_type_name(c) == preferred]
                    if typed:
                        vacated_region["leader"] = max(typed, key=leadership_score)
                        vacated_region["reluctant_leader"] = False
                        break
                else:
                    vacated_region["leader"] = max(candidates, key=lambda a: a.id)
                    vacated_region["reluctant_leader"] = True
                vacated_region["leader"].assigned_region = vacated_region
                print(
                    f"[CA-CAMON] Region {vacated_region['id']}: "
                    f"AGENT_{vacated_region['leader'].id} takes over as regional leader."
                )
            else:
                # New global leader is the only agent in the region
                vacated_region["leader"] = None
                print(
                    f"[CA-CAMON] Region {vacated_region['id']}: dormant after "
                    f"global leader promotion (no remaining agents in region)."
                )

    # The new global leader is no longer a regional leader for its old region
    new_global.assigned_region = None

    print(
        f"[CA-CAMON] New global leader: AGENT_{new_global.id} "
        f"({get_agent_type_name(new_global)}, score={leadership_score(new_global)})."
    )
    return new_global


def orchestrate_global(global_leader: Agent, regions: list, agents: list,
                       global_data: dict) -> None:
    """
    [CA-CAMON] Global leader orchestration step – runs once per timestep
    BEFORE the per-region planning loop.

    Responsibilities:
        1. Health-check every region's leader; trigger reassignment for any
           region whose leader died or left.
        2. Detect under-staffed regions (no subordinates, only a reluctant
           leader, or dormant) and request an agent transfer from a region
           that has a surplus.  The global leader acts as the cross-region
           communicator: it messages the donating region's leader to redirect
           one subordinate toward the needy region.

    This function does NOT call the LLM. It works purely through the
    structured region/agent data and appends messages via agent.add_message()
    so the LLM prompt for each agent already contains the directive.
    """
    t = global_data.get("time", 0)

    # ── 1. Health-check all regional leaders ───────────────────────────────
    for region in regions:
        rl = region.get("leader")
        if rl is not None and rl not in agents:
            # Regional leader died between the removelist step and now
            print(
                f"[CA-CAMON] Global leader detects Region {region['id']} leader "
                f"AGENT_{rl.id} is gone. Triggering reassignment."
            )
            assign_regional_leaders([region], agents)
            new_rl = region.get("leader")
            if new_rl:
                new_rl.assigned_region = region
                print(
                    f"[CA-CAMON] Region {region['id']}: "
                    f"AGENT_{new_rl.id} ({get_agent_type_name(new_rl)}) is new regional leader."
                )

    # ── 2. Cross-region agent redistribution ───────────────────────────────
    # Identify needy regions: dormant OR only has a reluctant leader with no
    # subordinates, OR has a reluctant leader at all.
    needy_regions = []
    for region in regions:
        agents_in = [a for a in agents if _agent_in_region(region, a)]
        rl = region.get("leader")
        if rl is None:
            # Dormant: no agents at all
            needy_regions.append((region, "dormant", 0))
        elif region.get("reluctant_leader") and len(agents_in) == 1:
            # Single worker with no support
            needy_regions.append((region, "reluctant_solo", 1))
        elif region.get("reluctant_leader"):
            # Has workers but no eligible leader
            needy_regions.append((region, "reluctant_team", len(agents_in)))

    if not needy_regions:
        return  # nothing to do

    # Find donor regions: those with an eligible leader AND at least 2 agents
    # (so they can spare one subordinate)
    donor_regions = []
    for region in regions:
        rl = region.get("leader")
        if rl is None or region.get("reluctant_leader"):
            continue
        agents_in = [a for a in agents if _agent_in_region(region, a)]
        subordinates = [a for a in agents_in if a is not rl]
        if subordinates:
            donor_regions.append((region, subordinates))

    if not donor_regions:
        return  # no surplus anywhere

    # Match each needy region to the nearest donor and issue a redirect message
    for needy_region, need_type, n_agents in needy_regions:
        # Centre of needy region
        nx = (needy_region["x_min"] + needy_region["x_max"]) / 2
        ny = (needy_region["y_min"] + needy_region["y_max"]) / 2

        # Closest donor by distance between region centres
        best_donor_region, best_subs = min(
            donor_regions,
            key=lambda dr: abs((dr[0]["x_min"] + dr[0]["x_max"]) / 2 - nx)
                         + abs((dr[0]["y_min"] + dr[0]["y_max"]) / 2 - ny)
        )
        donor_rl = best_donor_region["leader"]
        # Pick the subordinate closest to the needy region's centre
        transfer_agent = min(
            best_subs,
            key=lambda a: (abs(a.last_position[0] - nx) + abs(a.last_position[1] - ny))
            if a.last_position else float('inf')
        )

        # Message the donor regional leader to redirect the transfer agent
        redirect_target = (int(nx), int(ny))
        gl_msg = (
            f"[GLOBAL LEADER DIRECTIVE] Region {needy_region['id']} needs support "
            f"({need_type}). Please redirect AGENT_{transfer_agent.id} to move toward "
            f"location {redirect_target} to reinforce that region."
        )
        donor_rl.add_message(
            source=f"GLOBAL_LEADER_AGENT_{global_leader.id}",
            content=gl_msg,
            time=t,
        )
        # Also message the transfer agent directly so it is aware
        agent_msg = (
            f"[GLOBAL LEADER DIRECTIVE] You are being redirected to Region "
            f"{needy_region['id']} (centre {redirect_target}) to provide support. "
            f"Proceed there when your current action completes."
        )
        transfer_agent.add_message(
            source=f"GLOBAL_LEADER_AGENT_{global_leader.id}",
            content=agent_msg,
            time=t,
        )
        print(
            f"[CA-CAMON] Global leader AGENT_{global_leader.id}: redirecting "
            f"AGENT_{transfer_agent.id} from Region {best_donor_region['id']} "
            f"to Region {needy_region['id']} ({need_type})."
        )

        # Remove used subordinate from donor list to avoid double-assignment
        best_subs.remove(transfer_agent)
        if not best_subs:
            donor_regions = [(r, s) for r, s in donor_regions if r is not best_donor_region]


# ============================================================
# Action model (unchanged from original)
# ============================================================

class Action(BaseModel):
    """
    Represents an action that can be taken by an agent in the CAMON algorithm.

    Attributes:
        type (int): The type of action to be performed
        param_1 (int): First parameter (typically x-coordinate)
        param_2 (int): Second parameter (typically y-coordinate)
        description (str): Human-readable description of the action
    """
    type: int
    param_1: int
    param_2: int
    description: str

    def print_option(self) -> None:
        print([self.type, self.param_1, self.param_2, self.description])


# ============================================================
# translate_action  (unchanged from original)
# ============================================================

def translate_action(option_str: str, type: int, global_data: dict) -> Action:
    """
    Translates a natural language action description into a structured Action.
    """
    system_message = (
        "You are the controller of a highly trained embodied agent within a grid forest world. "
        "Your job is to convert a string action into a structured format for robotic control."
    )
    if type == 0:
        type_string = 'firefighter'
    elif type == 1:
        type_string = 'bulldozer'
    elif type == 2:
        type_string = 'drone'
    else:
        type_string = 'helicopter'

    prompt_path = (
        f'crew_algorithms/wildfire_alg/algorithms/CAMON/prompts/translator/'
        f'{type_string}_translator.txt'
    )
    with open(prompt_path, 'r', encoding='utf-8') as f:
        prompt = f.read().replace("ACTION", option_str)

    client = OpenAI(
        base_url="https://tritonai-api.ucsd.edu",
        api_key=global_data['leader_agent'].api_key
    )
    response = client.chat.completions.create(
        model='api-gpt-oss-120b',
        messages=[
            {'role': 'system', 'content': system_message},
            {'role': 'user',   'content': prompt}
        ],
        temperature=0.7
    )
    global_data["api_calls"]     += 1
    global_data["input_tokens"]  += response.usage.prompt_tokens
    global_data["output_tokens"] += response.usage.completion_tokens
    result = response.choices[0].message.content

    def extract(tag: str) -> str:
        m = re.search(fr"<{tag}>\s*(.*?)\s*</{tag}>", result, re.DOTALL)
        if not m:
            raise ValueError(f"Missing tag {tag}")
        return m.group(1).strip()

    try:
        return Action(
            type=int(extract('type')),
            param_1=int(extract('param_1')),
            param_2=int(extract('param_2')),
            description=extract('description')
        )
    except Exception:
        return Action(
            type=0, param_1=0, param_2=0,
            description='ERROR EXECUTING: ' + option_str
        )


# ============================================================
# Private helpers shared by generate_plan / propose_plan
# ============================================================

def _build_team_comp_string(global_data: dict) -> str:
    s = ''
    if global_data.get('firefighters'):
        s += (
            f"{[f'AGENT_{a.id}' for a in global_data['firefighters']]} "
            f"{'is' if len(global_data['firefighters']) == 1 else 'are'} Firefighter Agents. "
            "Firefighter agents are general purpose agents with decent speed and observation "
            "capabilities. They can move, cut trees, spray water, and rescue civilians.\n\n"
        )
    if global_data.get('bulldozers'):
        s += (
            f"{[f'AGENT_{a.id}' for a in global_data['bulldozers']]} "
            f"{'is' if len(global_data['bulldozers']) == 1 else 'are'} Bulldozer Agents. "
            "Bulldozer agents are specialized agents with exceptional tree-cutting abilities "
            "but limited speed.\n\n"
        )
    if global_data.get('drones'):
        s += (
            f"{[f'AGENT_{a.id}' for a in global_data['drones']]} "
            f"{'is' if len(global_data['drones']) == 1 else 'are'} Drone Agents. "
            "Drone agents are specialized recon agents with exceptional speed and observations.\n\n"
        )
    if global_data.get('helicopters'):
        s += (
            f"{[f'AGENT_{a.id}' for a in global_data['helicopters']]} "
            f"{'is' if len(global_data['helicopters']) == 1 else 'are'} Helicopter Agents. "
            "Helicopter agents are general support agents with exceptional speed and observations. "
            "They can move, pick up and drop off Firefighter Agents, and spray water.\n\n"
        )
    return s


def _build_team_abilities(global_data: dict) -> str:
    abilities = ''
    for kind in ('firefighter', 'bulldozer', 'drone', 'helicopter'):
        if global_data.get(kind + 's'):
            path = (
                f'crew_algorithms/wildfire_alg/algorithms/CAMON/prompts/descriptions/'
                f'{kind}_description.txt'
            )
            with open(path, 'r', encoding='utf-8') as f:
                abilities += f.read()
    return abilities


# ============================================================
# generate_plan  (logic unchanged; called on regional leader)
# ============================================================

def generate_plan(agent: Agent, global_data: dict) -> None:
    """
    Generates a plan for the regional (or global) leader.
    global_data['leader_agent'] must point to `agent` before this call.
    """
    print(f"agent {agent.id}: generating actions")

    if agent.type == 0:
        type_string = 'Firefighter'
    elif agent.type == 1:
        type_string = 'Bulldozer'
    elif agent.type == 2:
        type_string = 'Drone'
    else:
        type_string = 'Helicopter'

    # Indicate global leader role in prompt if applicable
    is_global = global_data.get("global_leader") is agent
    role_label = "global leader and regional leader" if is_global else "regional leader"

    team_comp_string = _build_team_comp_string(global_data)
    team_abilities   = _build_team_abilities(global_data)
    past_string      = ''.join(str(o.description) + '\n' for o in agent.past_options)
    chat_string      = ''.join(f"{t}: \n{msg}\n\n" for t, msg in agent.chat_history.items())
    global_str       = [(k, v) for k, v in global_data.items() if "AGENT" in str(k)]

    generate_plan_string = f"""
            You are AGENT_{agent.id} a {type_string} Agent, currently acting as the {role_label} in a cooperative multi-agent robotic task.
            This is your team composition, including you:
            {team_comp_string}

            Your team's current task is:
            {agent.current_task}
            ---

            Your past actions were:
            {past_string}

            ---
            This is your chat history with agents in your team:
            {chat_string}

            ---
            This is your teams'(including you) collective observations, locations, current actions, and past actions of all agents.
            {str(global_str)}

            Now your job is to provide the next best action for yourself, and OPTIONALLY: the next best action for any other agents.
            Remember, you are AGENT_{agent.id} a {type_string} Agent, located at {agent.last_position}.

            These are all the possible actions for each type of agent. This is a comprehensive list, so the action MUST be one of these types. NO other responses are allowed.

            {team_abilities}

            Provide your output in the following format:

            <reasoning>(any reasoning or calculations)</reasoning>

            <action>'MY NEXT ACTION'</action>

            OPTIONAL-for other agents:

            <AGENT_ID-action>(AGENT_ID'S NEXT ACTION)<AGENT_ID-action>
            <AGENT_ID-message>(message to AGENTID)<AGENT_ID-message>

            For example:
            <AGENT_A-action>'action'</AGENT_A-action>
            <AGENT_A-message>'action'</AGENT_A-message>
            """

    client = OpenAI(base_url="https://tritonai-api.ucsd.edu", api_key=agent.api_key)
    system_message = (
        f"You are AGENT_{agent.id}, currently acting as the {role_label} in a cooperative "
        f"multi-agent robotic task. Your team is in a {agent.cfg.envs.map_size} by "
        f"{agent.cfg.envs.map_size} forest grid world that spans x:[0 to {agent.cfg.envs.map_size}]"
        f" and y:[0 to {agent.cfg.envs.map_size}]. You have access to the collective observations "
        "and the progress of all agents. Plan the next best action for yourself and OPTIONALLY "
        "for any other agents."
    )
    response = client.chat.completions.create(
        model='api-gpt-oss-120b',
        messages=[
            {'role': 'system', 'content': system_message},
            {'role': 'user',   'content': generate_plan_string}
        ],
        temperature=0.7
    )
    global_data["api_calls"]     += 1
    global_data["input_tokens"]  += response.usage.prompt_tokens
    global_data["output_tokens"] += response.usage.completion_tokens
    result = response.choices[0].message.content
    agent.log_chat("Generating Plan", [("user", generate_plan_string), ("assistant", result)])

    m = re.search(r"<action>\s*(.*?)\s*</action>", result, re.DOTALL)
    if m:
        option = translate_action(m.group(1), agent.type, global_data)
        agent.options = [option]
        print(f"agent {agent.id}: provided action: '{m.group(1)}' to itself")
    else:
        print("ERROR NO ACTION FOUND")
        return

    for a in global_data["agents"]:
        ma = re.search(fr"<AGENT_{a.id}-action>\s*(.*?)\s*</AGENT_{a.id}-action>", result, re.DOTALL)
        mm = re.search(fr"<AGENT_{a.id}-message>\s*(.*?)\s*</AGENT_{a.id}-message>", result, re.DOTALL)
        if ma and mm:
            if a.type == 0 and a.extra_variables[2] == 1:
                a.options = [Action(type=0, param_1=0, param_2=0, description="ride helicopter")]
                continue
            print(f"agent {agent.id}: provided action '{ma.group(1)}' to {a.id}")
            opt = translate_action(ma.group(1), a.type, global_data)
            a.options      = [opt]
            a.action_queue = []
            a.add_message(source=f"AGENT_{agent.id}", content=mm.group(1), time=global_data['time'])
            global_data.update({
                f'AGENT_{a.id}': {
                    'name':           f'AGENT_{a.id}',
                    'perception':     a.last_perception,
                    'position':       a.last_position,
                    'current_action': opt,
                    'past_actions':   a.past_options,
                }
            })


# ============================================================
# propose_plan  – [CA-CAMON] advisory mode, routed to regional leader
# ============================================================

def propose_plan(agent: Agent, global_data: dict) -> None:
    """
    [CA-CAMON] Advisory Proposal Mechanism.

    The proposal is reviewed by the agent's REGIONAL leader
    (global_data['leader_agent'] is set per-region by __main__.py).

    Leadership transfer rule:
        eligible(proposer) AND score(proposer) > score(regional_leader)
    Workers never satisfy eligible(), so they never trigger a transfer.
    """
    leader = global_data["leader_agent"]
    print(f"agent {agent.id}: proposing action to regional leader AGENT_{leader.id}")

    if agent.type == 0:
        type_string = 'Firefighter'
    elif agent.type == 1:
        type_string = 'Bulldozer'
    elif agent.type == 2:
        type_string = 'Drone'
    else:
        type_string = 'Helicopter'

    team_comp_string = _build_team_comp_string(global_data)
    team_abilities   = _build_team_abilities(global_data)
    past_string      = ''.join(str(o.description) + '\n' for o in agent.past_options)
    chat_string      = ''.join(f"{t}: \n{msg}\n\n" for t, msg in agent.chat_history.items())

    desc_path = (
        f'crew_algorithms/wildfire_alg/algorithms/CAMON/prompts/descriptions/'
        f'{type_string.lower()}_description.txt'
    )
    with open(desc_path, 'r', encoding='utf-8') as f:
        description_string = f.read()

    # ── Step 1: agent drafts its proposed action ────────────────────────────
    proposal_str = f"""
                    You are AGENT_{agent.id}, an embodied {type_string} agent within a {agent.cfg.envs.map_size} by {agent.cfg.envs.map_size} forest grid world and part of a collaborative team of {len(global_data["agents"])} Agents.

                    This is your team's composition (including yourself):
                    {team_comp_string}

                    These are your current observations:
                    '{agent.last_perception}'

                    ---
                    This is your team's overall task: '{agent.current_task}'

                    Your past actions were:
                    {past_string}

                    ---
                    This is your chat history with agents in your team:
                    {chat_string}

                    ---
                    Your job is to propose your next action. These are your possible actions:
                    {description_string}

                    This is a comprehensive list, so your action MUST be one of these types. NO other responses are allowed.

                    Provide your output in the following format:

                    <reasoning>(any reasoning or calculations)</reasoning>
                    <action>'MY NEXT ACTION'</action>
                    """

    client = OpenAI(base_url="https://tritonai-api.ucsd.edu", api_key=agent.api_key)
    system_msg = (
        f"You are AGENT_{agent.id}, an embodied {type_string} agent. "
        "You propose your next action based on your task, observations, past actions, and chat history."
    )
    response = client.chat.completions.create(
        model='api-gpt-oss-120b',
        messages=[{'role': 'system', 'content': system_msg},
                  {'role': 'user',   'content': proposal_str}],
        temperature=0.7
    )
    global_data["api_calls"]     += 1
    global_data["input_tokens"]  += response.usage.prompt_tokens
    global_data["output_tokens"] += response.usage.completion_tokens
    proposal_result = response.choices[0].message.content
    agent.log_chat('Proposing an Action', [('user', proposal_str), ('assistant', proposal_result)])

    m = re.search(r"<action>\s*(.*?)\s*</action>", proposal_result, re.DOTALL)
    if not m:
        print("ERROR NO ACTION FOUND")
        return
    proposed_action = m.group(1).strip()

    # ── Step 2: regional leader reviews the proposal ────────────────────────
    global_str     = [(k, v) for k, v in global_data.items() if "AGENT" in str(k)]
    proposer_score = leadership_score(agent)
    leader_score   = leadership_score(leader)

    capability_note = (
        f"Note: AGENT_{agent.id} is a {type_string} with a leadership score of {proposer_score}. "
        + (
            "This agent is NOT eligible to become regional leader (worker type)."
            if not can_be_leader(agent)
            else (
                f"Your own leadership score is {leader_score}. "
                + ("This agent's score exceeds yours and it may become the new regional leader."
                   if proposer_score > leader_score
                   else "Your score is equal or higher; you will remain regional leader.")
            )
        )
    )

    review_prompt = f"""
            You are AGENT_{leader.id}, currently acting as the REGIONAL LEADER in a cooperative multi-agent robotic task.
            This is your team composition, including you:
            {team_comp_string}
            ---

            Your team's current task is:
            {agent.current_task}
            ---

            This is your teams'(including you) collective observations, locations, current actions, and past actions of all agents.
            {str(global_str)}
            ---

            Your teammate AGENT_{agent.id}, a {type_string} Agent, is proposing a new action for itself:
            {proposed_action}
            ---

            {capability_note}
            ---

            Your job is to review this action and ACCEPT or REJECT it.
            Then provide the next best action for AGENT_{agent.id}.
            Also send a message to AGENT_{agent.id} describing your choice.
            You may also override actions for other agents in your region.

            These are all the possible actions for each type of agent:
            {team_abilities}

            Provide your output in the following format:

            <reasoning>(any reasoning or calculations)</reasoning>

            <decision> ACCEPT OR REJECT </decision>
            <action> AGENT_{agent.id}'s next action </action>
            <message> message to AGENT_{agent.id} </message>

            OPTIONAL-for other agents:

            <AGENT_ID-action>(AGENTID'S NEXT ACTION)<AGENT_ID-action>
            <AGENT_ID-message>(message to AGENTID)<AGENT_ID-message>

            YOU MUST HAVE AT LEAST THE <reasoning>, <decision>, <action>, <message> TAGS.
            """

    sys_review = (
        f"You are AGENT_{leader.id}, acting as the REGIONAL LEADER in a cooperative "
        f"multi-agent robotic task in a {agent.cfg.envs.map_size} by {agent.cfg.envs.map_size} "
        "grid world. Review proposals from agents in your region and assign them actions."
    )
    rev_resp = OpenAI(
        base_url="https://tritonai-api.ucsd.edu",
        api_key=leader.api_key
    ).chat.completions.create(
        model='api-gpt-oss-120b',
        messages=[{'role': 'system', 'content': sys_review},
                  {'role': 'user',   'content': review_prompt}],
        temperature=0.7
    )
    global_data["api_calls"]     += 1
    global_data["input_tokens"]  += rev_resp.usage.prompt_tokens
    global_data["output_tokens"] += rev_resp.usage.completion_tokens
    review = rev_resp.choices[0].message.content
    agent.log_chat("Review Proposal", [('user', review_prompt), ('assistant', review)])

    action_match  = re.search(r"<action>\s*(.*?)\s*</action>",   review, re.DOTALL)
    message_match = re.search(r"<message>\s*(.*?)\s*</message>", review, re.DOTALL)

    if action_match:
        action_str = action_match.group(1)
        option = translate_action(action_str, type=agent.type, global_data=global_data)
        agent.options = [option]
        if message_match:
            agent.add_message(
                source=f"AGENT_{leader.id}",
                content=message_match.group(1),
                time=global_data["time"]
            )
            print(f"agent {leader.id}: {message_match.group(1)}")
        print(f"agent {leader.id}: provided action: '{action_str}' to {agent.id}")

        for a in global_data["agents"]:
            aa = re.search(fr"<AGENT_{a.id}-action>\s*(.*?)\s*</AGENT_{a.id}-action>", review, re.DOTALL)
            am = re.search(fr"<AGENT_{a.id}-message>\s*(.*?)\s*</AGENT_{a.id}-message>", review, re.DOTALL)
            if aa and am:
                if a.type == 0 and a.extra_variables[2] == 1:
                    a.options = [Action(type=0, param_1=0, param_2=0, description="ride helicopter")]
                    continue
                opt = translate_action(aa.group(1), type=a.type, global_data=global_data)
                a.options      = [opt]
                a.action_queue = []
                a.add_message(source=f"AGENT_{leader.id}", content=am.group(1), time=global_data["time"])
                global_data.update({
                    f'AGENT_{a.id}': {
                        'name':           f'AGENT_{a.id}',
                        'perception':     a.last_perception,
                        'position':       a.last_position,
                        'current_action': a.options[0] if a.options else "IDLE",
                        'past_actions':   a.past_options,
                    }
                })
    else:
        print("ERROR NO ACTION FOUND")

    # ── [CA-CAMON] Conditional regional leadership transfer ─────────────────
    #
    # ORIGINAL vanilla CAMON: global_data.update({"leader_agent": agent})
    # CA-CAMON: transfer only when eligible AND score(proposer) > score(leader).
    if should_transfer_leadership(agent, leader):
        global_data.update({"leader_agent": agent})
        if agent.assigned_region is not None:
            agent.assigned_region["leader"]           = agent
            agent.assigned_region["reluctant_leader"] = False
        print(
            f"[CA-CAMON] Regional leadership TRANSFERRED: "
            f"AGENT_{agent.id} ({get_agent_type_name(agent)}, score={leadership_score(agent)}) "
            f"replaces AGENT_{leader.id} "
            f"({get_agent_type_name(leader)}, score={leadership_score(leader)})."
        )
    else:
        print(
            f"[CA-CAMON] Regional leadership RETAINED by AGENT_{leader.id} "
            f"({get_agent_type_name(leader)}, score={leadership_score(leader)}). "
            f"AGENT_{agent.id} proposal was advisory only."
        )
