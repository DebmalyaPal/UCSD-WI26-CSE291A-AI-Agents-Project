"""
__main__.py  –  HMAS_3  (Regional Hierarchical Multi-Agent System)
===================================================================

Three-tier hierarchy:
  Global Leader   → 1 agent, manages cross-region worker reallocation
  Regional Leader → 1 per 20×20 region, runs HMAS_2 planning loop internally
  Workers         → execute tasks, accept/reject plans

Under-resourced fallback (total_agents < target_size):
  RegionManager automatically reduces the number of active regions so the
  three-tier hierarchy degrades gracefully rather than collapsing to flat HMAS_2.
  See region_manager.py for details.
"""

import hydra
from attrs import define
from crew_algorithms.envs.configs import EnvironmentConfig, register_env_configs
from crew_algorithms.wildfire_alg.config.configs import LLMConfig
from crew_algorithms.utils.wandb_utils import WandbConfig
from crew_algorithms.wildfire_alg.config.build_config import update_config, create_level_presets
from hydra.core.config_store import ConfigStore
from omegaconf import MISSING
import numpy as np
from crew_algorithms.wildfire_alg.core.alg_utils import get_agent_observations, parse_game_data, check_game_done
import datetime
import csv
import os
import torch
import certifi
from openai import OpenAI

from crew_algorithms.wildfire_alg.libraries.firefighter_action_library import Run_Firefighter_Action
from crew_algorithms.wildfire_alg.libraries.bulldozer_action_library import Run_Bulldozer_Action
from crew_algorithms.wildfire_alg.libraries.drone_action_library import Run_Drone_Action
from crew_algorithms.wildfire_alg.libraries.helicopter_action_library import Run_Helicopter_Action
from crew_algorithms.wildfire_alg.data.render_logs import compile_split_screen_video


@define(auto_attribs=True)
class Config:
    envs: EnvironmentConfig = MISSING
    wandb: WandbConfig = WandbConfig(project="wildfire")
    collect_data: bool = False
    llms: LLMConfig = LLMConfig()


cs = ConfigStore.instance()
cs.store(name="base_config", node=Config)
register_env_configs()


@hydra.main(version_base=None, config_path="../../../conf", config_name="wildfire_alg")
def wildfire_alg(cfg: Config):
    """HMAS_3: Hierarchical Multi-Agent System with Regional Leaders."""
    import uuid
    from crew_algorithms.envs.channels import ToggleTimestepChannel
    from crew_algorithms.wildfire_alg.core.utils import make_env
    from crew_algorithms.wildfire_alg.algorithms.HMAS_3.agent import Agent, ROLE_WORKER, ROLE_REGIONAL_LEADER, ROLE_GLOBAL_LEADER
    from crew_algorithms.wildfire_alg.algorithms.HMAS_3.region_manager import RegionManager
    from crew_algorithms.wildfire_alg.algorithms.HMAS_3.utils import (
        propose_regional_actions,
        provide_feedback,
        translate_action,
        global_leader_manage_reallocation,
        regional_leader_check_and_request,
    )

    os.environ["SSL_CERT_FILE"] = certifi.where()
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    toggle_timestep_channel = ToggleTimestepChannel(uuid.uuid4())

    # -----------------------------------------------------------------------
    # Config + environment setup
    # -----------------------------------------------------------------------
    cfg.envs.algorithm = "HMAS_3"
    level = cfg.envs.level
    seed = cfg.envs.seed
    levels = create_level_presets()

    firefighters  = levels[level].get("starting_firefighter_agents", 0)
    bulldozers    = levels[level].get("starting_bulldozer_agents", 0)
    drones        = levels[level].get("starting_drone_agents", 0)
    helicopters   = levels[level].get("starting_helicopter_agents", 0)
    agent_count   = firefighters + bulldozers + drones + helicopters

    update_config(preset=levels[level], config=cfg.envs, log_trajectory=True, seed=seed)
    cfg.envs.timestamp = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    env = make_env(cfg.envs, toggle_timestep_channel, device)
    state = env.reset()

    os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
    api_key = os.environ["OPENAI_API_KEY"]

    path = os.path.join("results", "logs", "HMAS_3", level, str(seed), cfg.envs.timestamp)
    os.makedirs(path, exist_ok=True)

    # -----------------------------------------------------------------------
    # Agent instantiation  (types identical to HMAS_2)
    # -----------------------------------------------------------------------
    game_data = parse_game_data(state, cfg)
    print(f"Task: {game_data['task_description']}")

    agents: list[Agent] = []
    for i in range(firefighters):
        agents.append(Agent(i + 1, 0, cfg, path, game_data["task_description"], api_key, agent_count))
    for i in range(bulldozers):
        agents.append(Agent(firefighters + i + 1, 1, cfg, path, game_data["task_description"], api_key, agent_count))
    for i in range(drones):
        agents.append(Agent(firefighters + bulldozers + i + 1, 2, cfg, path, game_data["task_description"], api_key, agent_count))
    for i in range(helicopters):
        agents.append(Agent(firefighters + bulldozers + drones + i + 1, 3, cfg, path, game_data["task_description"], api_key, agent_count))

    # -----------------------------------------------------------------------
    # Region Manager  (assigns roles: global leader, regional leaders, workers)
    # -----------------------------------------------------------------------
    map_size = cfg.envs.map_size
    region_manager = RegionManager(map_size=map_size, agents=agents)
    print(region_manager.summary())

    client = OpenAI(api_key=api_key)

    global_data = {
        "api_calls": 0,
        "input_tokens": 0,
        "output_tokens": 0,
        "score": 0,
        "step_history": {},
        "api_key": api_key,
        "client": client,
        "path": path,
        "agents": agents,
        "region_manager": region_manager,
    }

    csv_filename = os.path.join(path, "data.csv")
    with open(csv_filename, "w", newline="") as f:
        csv.writer(f).writerow(["cumulative_score", "cumulative_api_calls", "cumulative_input_tokens", "cumulative_output_tokens"])

    # Identify global leader once (role assigned by RegionManager)
    global_agent = next((a for a in agents if a.role == ROLE_GLOBAL_LEADER), None)

    # -----------------------------------------------------------------------
    # Main loop
    # -----------------------------------------------------------------------
    for t in range(cfg.envs.max_steps):
        print(f"\n{'='*60}\nTIME: {t}\n{'='*60}")
        game_data = parse_game_data(state, cfg)
        past_score = global_data["score"]

        global_data.update({
            "firefighters": [],
            "bulldozers": [],
            "drones": [],
            "helicopters": [],
            "agents": agents,
            "time": t,
            "score": game_data["score"],
        })

        # Trim step history
        remove_times = [k for k in global_data["step_history"] if t - int(k.split(": ")[1]) > 5]
        for k in remove_times:
            global_data["step_history"].pop(k)

        with open(csv_filename, "a", newline="") as f:
            csv.writer(f).writerow([
                global_data["score"], global_data["api_calls"],
                global_data["input_tokens"], global_data["output_tokens"]
            ])

        # ---- Observe & update agent states --------------------------------
        removelist = []
        agent_states = {}
        for agent in agents:
            obs = get_agent_observations(state, agent.id)
            if obs["agent_type"] >= 4:
                print(f"Agent {agent.id} DESTROYED")
                removelist.append(agent)
                continue
            type_map = {0: "firefighters", 1: "bulldozers", 2: "drones", 3: "helicopters"}
            global_data[type_map[obs["agent_type"]]].append(agent)
            agent.last_observation = obs["perception_grid"]
            agent.last_position    = obs["position"]
            agent.last_current_cell = obs["current_cell"]
            agent.map_range        = obs["map_range"]
            agent.extra_variables  = obs["extra_variables"]
            agent_states[agent.id] = agent.last_position

        for r in removelist:
            agents.remove(r)
            # Also remove from region roster
            for region in region_manager.regions.values():
                if r in region.workers:
                    region.workers.remove(r)
                if region.leader == r:
                    region.leader = None

        if check_game_done(global_data=global_data, cfg=cfg.envs, past_score=past_score):
            break

        # ---- Generate perceptions -----------------------------------------
        for agent in agents:
            agent.generate_perception(cfg.envs, agent_states, global_data)

        # ---- Step 1: Regional leaders check for empty-region situations ---
        for rid, region in region_manager.regions.items():
            if not region.active or region.leader is None:
                continue
            regional_leader_check_and_request(rid, region.leader, global_data)

        # ---- Step 2: Global leader manages reallocation -------------------
        if global_agent is not None:
            realloc_moves = global_leader_manage_reallocation(global_agent, global_data)
            for worker, from_rid, to_rid in realloc_moves:
                print(f"  [Realloc] AGENT_{worker.id}: region {from_rid} → {to_rid}")

        # ---- Step 3: Per-region HMAS_2 planning loop ----------------------
        all_proposed_actions: dict[str, str] = {}
        global_data["step_history"][f"time: {t}"] = {}

        for rid, region in region_manager.regions.items():
            if not region.active:
                continue
            if region.leader is None:
                print(f"  [Region {rid}] No leader — skipping planning")
                continue

            regional_leader = region.leader
            region_workers  = region.workers

            # If no workers in region, leader does nothing this turn
            # (it already raised a realloc request above if fire detected)
            if not region_workers and regional_leader.last_perception and \
               "fire" not in regional_leader.last_perception.lower():
                print(f"  [Region {rid}] Empty + no fire — leader standing by")
                all_proposed_actions[f"AGENT_{regional_leader.id}"] = "do nothing and conserve energy"
                continue

            # Run regional planning loop (mirrors HMAS_2 loop)
            proposed_actions, messages = propose_regional_actions(
                region_id=rid,
                regional_leader=regional_leader,
                region_workers=region_workers,
                global_data=global_data,
                past_conversation=[],
            )

            # Worker feedback loop
            region_agents_all = [regional_leader] + region_workers
            max_replan_rounds = 3
            replan_round = 0
            while replan_round < max_replan_rounds:
                satisfactory = True
                feedback = {}
                for agent in region_agents_all:
                    fb = provide_feedback(
                        agent=agent,
                        region_id=rid,
                        region_agents=region_agents_all,
                        proposed_actions=proposed_actions,
                        global_data=global_data,
                    )
                    if "ACCEPT" not in fb:
                        satisfactory = False
                    feedback[f"AGENT_{agent.id}"] = fb

                if satisfactory:
                    break

                messages.append({
                    "role": "user",
                    "content": (
                        f"Team feedback for Region {rid}. Adjust the plan accordingly. "
                        f"Same format.\n\n{feedback}"
                    ),
                })
                proposed_actions, messages = propose_regional_actions(
                    region_id=rid,
                    regional_leader=regional_leader,
                    region_workers=region_workers,
                    global_data=global_data,
                    past_conversation=messages,
                )
                replan_round += 1

            all_proposed_actions.update(proposed_actions)

        # ---- Step 4: Global leader's own action (observation / standby) ---
        if global_agent is not None and f"AGENT_{global_agent.id}" not in all_proposed_actions:
            # Global leader patrols or stands by — minimal action
            all_proposed_actions[f"AGENT_{global_agent.id}"] = "do nothing and conserve energy"

        # ---- Step 5: Execute actions --------------------------------------
        env_action = [[0, 0, 0] for _ in range(cfg.envs.num_agents)]
        libraries = {
            0: Run_Firefighter_Action,
            1: Run_Bulldozer_Action,
            2: Run_Drone_Action,
            3: Run_Helicopter_Action,
        }

        for agent in agents:
            action_str = all_proposed_actions.get(f"AGENT_{agent.id}", "do nothing and conserve energy")
            action = translate_action(action_str=action_str, type=agent.type, global_data=global_data)
            agent.log_chat("Executing Action", [("system", action)])
            print(f"  AGENT_{agent.id} [{agent.role}, R{agent.region_id}]: {action.description}")

            try:
                if agent.type in libraries:
                    action_array = libraries[agent.type](agent, action)
                    env_action[agent.id] = action_array
                    global_data["step_history"][f"time: {t}"][f"AGENT_{agent.id}"] = {
                        "state": agent.last_position,
                        "action": action.description,
                        "role": agent.role,
                        "region": agent.region_id,
                    }
            except Exception as e:
                agent.log_chat("ERROR", [("system", f"ERROR EXECUTING ACTION: {action} — {e}")])
                global_data["step_history"][f"time: {t}"][f"AGENT_{agent.id}"] = {
                    "state": agent.last_position,
                    "action": f"ERROR: {action.description}",
                    "role": agent.role,
                    "region": agent.region_id,
                }

        print(f"  env_action: {env_action}")
        action_tensor = torch.from_numpy(np.array(env_action)).to(device)
        state["agents"]["action"] = action_tensor
        newstate = env.step(state)
        state["agents"]["observation"] = newstate["next"]["agents"]["observation"]

    # -----------------------------------------------------------------------
    env.close()
    print("HMAS_3 TEST COMPLETE")
    compile_split_screen_video(path, os.path.join(path, "render.mp4"))


if __name__ == "__main__":
    wildfire_alg()