import hydra
from attrs import define
from crew_algorithms.envs.configs import EnvironmentConfig, register_env_configs
from crew_algorithms.wildfire_alg.config.configs import LLMConfig
from crew_algorithms.utils.wandb_utils import WandbConfig
from crew_algorithms.wildfire_alg.config.build_config import update_config, create_level_presets
from hydra.core.config_store import ConfigStore
from omegaconf import MISSING
import numpy as np
from crew_algorithms.wildfire_alg.core.alg_utils import (
    get_agent_observations,
    generate_action_from_option,
    parse_game_data,
    check_if_option_done,
    check_game_done,
)
import datetime
import csv
import certifi
from crew_algorithms.wildfire_alg.data.render_logs import compile_split_screen_video


@define(auto_attribs=True)
class Config:
    envs: EnvironmentConfig = MISSING
    """Settings for the environment to use."""
    wandb: WandbConfig = WandbConfig(project="wildfire")
    """WandB logger configuration."""
    collect_data: bool = False
    """Whether or not to collect data and save a new dataset to WandB."""
    llms: LLMConfig = LLMConfig()


cs = ConfigStore.instance()
cs.store(name="base_config", node=Config)

register_env_configs()


@hydra.main(version_base=None, config_path="../../../conf", config_name="wildfire_alg")
def wildfire_alg(cfg: Config):
    """An implementation of the CA-CAMON wildfire algorithm."""
    import os
    import uuid
    import torch
    from crew_algorithms.envs.channels import ToggleTimestepChannel
    from crew_algorithms.wildfire_alg.core.utils import make_env
    from crew_algorithms.wildfire_alg.algorithms.CAMON.agent import Agent
    from crew_algorithms.wildfire_alg.algorithms.CAMON.utils import (
        generate_plan,
        propose_plan,
        Action,
        # [CA-CAMON] regional helpers
        compute_regions,
        assign_regional_leaders,
        check_and_reassign_regional_leader,
        get_region_for_agent,
        get_region_leader,
        _agent_in_region,
        # [CA-CAMON] global leader helpers
        elect_global_leader,
        handle_global_leader_death,
        orchestrate_global,
    )

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    toggle_timestep_channel = ToggleTimestepChannel(uuid.uuid4())

    cfg.envs.algorithm = 'CAMON'
    level  = cfg.envs.level
    seed   = cfg.envs.seed
    levels = create_level_presets()

    firefighters = levels[level].get("starting_firefighter_agents", 0)
    bulldozers   = levels[level].get("starting_bulldozer_agents",   0)
    drones       = levels[level].get("starting_drone_agents",       0)
    helicopters  = levels[level].get("starting_helicopter_agents",  0)

    update_config(preset=levels[level], config=cfg.envs, log_trajectory=True, seed=seed)
    cfg.envs.timestamp = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    env   = make_env(cfg.envs, toggle_timestep_channel, device)
    state = env.reset()

    os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
    os.environ["SSL_CERT_FILE"]  = certifi.where()
    api_key   = os.environ['OPENAI_API_KEY']
    logs_root = os.path.join(str(cfg.envs.render_folder_path), "wildfire_alg")
    path = os.path.join(
        logs_root, "results", "logs", "CAMON", level, str(seed), cfg.envs.timestamp
    )
    os.makedirs(path, exist_ok=True)

    # ── Build agent list ────────────────────────────────────────────────────
    agents    = []
    game_data = parse_game_data(state, cfg)
    print(f"Task: {game_data['task_description']}")

    for i in range(firefighters):
        agents.append(Agent(i + 1, 0, cfg, path,
                            current_task=game_data["task_description"], api_key=api_key))
    for i in range(bulldozers):
        agents.append(Agent(firefighters + i + 1, 1, cfg, path,
                            current_task=game_data["task_description"], api_key=api_key))
    for i in range(drones):
        agents.append(Agent(firefighters + bulldozers + i + 1, 2, cfg, path,
                            current_task=game_data["task_description"], api_key=api_key))
    for i in range(helicopters):
        agents.append(Agent(firefighters + bulldozers + drones + i + 1, 3, cfg, path,
                            current_task=game_data["task_description"], api_key=api_key))

    print(f"Agent Count: {len(agents)}")

    # ── [CA-CAMON]  Build regional partition ───────────────────────────────
    regions = compute_regions(cfg.envs.map_size)
    print(f"[CA-CAMON] Grid partitioned into {len(regions)} region(s).")

    # ── [CA-CAMON]  Initial global leader election ──────────────────────────
    # global_leader  – single Agent that orchestrates all regional leaders.
    #                  Also acts as the leader of its own region.
    # Elected now with no positions yet (positions come from first observation),
    # so we use type priority only; the full position-aware election runs at the
    # top of the first timestep after positions are known.
    global_leader = _bootstrap_global_leader(agents)
    global_leader.is_global_leader = True
    print(
        f"[CA-CAMON] Initial global leader: AGENT_{global_leader.id} "
        f"({global_leader.type_name}, score={global_leader.leadership_score})"
    )

    global_data = {
        "api_calls":     0,
        "input_tokens":  0,
        "output_tokens": 0,
        "score":         0,
        # Pointer to the current global leader Agent.
        # Updated here and by handle_global_leader_death() when it dies.
        "global_leader": global_leader,
        # leader_agent is set per-region inside the planning loop;
        # it is NOT the global leader — it is the regional leader for the
        # agent currently being processed.
        "leader_agent":  global_leader,
    }

    header = ["cumulative_score", "cumulative_api_calls",
              "cumulative_input_tokens", "cumulative_output_tokens"]
    csv_filename = os.path.join(path, "data.csv")
    with open(csv_filename, 'w', newline='') as f:
        csv.writer(f).writerow(header)

    print(f"Max Steps: {cfg.envs.max_steps}")

    for t in range(cfg.envs.max_steps):
        print(f"TIME: {t}")
        game_data  = parse_game_data(state, cfg)
        removelist = []
        past_score = global_data["score"]

        global_data.update({
            'firefighters': [],
            'bulldozers':   [],
            'drones':       [],
            'helicopters':  [],
            'time':         t,
            'score':        game_data['score'],
        })

        with open(csv_filename, 'a', newline='') as f:
            csv.writer(f).writerow([
                global_data['score'],
                global_data['api_calls'],
                global_data['input_tokens'],
                global_data['output_tokens'],
            ])

        agent_states = {}

        # ── Observation collection ──────────────────────────────────────────
        for agent in agents:
            observations = get_agent_observations(state, agent.id)

            if observations["agent_type"] >= 4:
                print(f"AGENT_{agent.id} DESTROYED")
                removelist.append(agent)
                continue

            type_map = {0: 'firefighters', 1: 'bulldozers',
                        2: 'drones',        3: 'helicopters'}
            bucket = type_map.get(observations["agent_type"])
            if bucket:
                global_data[bucket].append(agent)

            agent.last_observation  = observations["perception_grid"]
            agent.last_position     = observations["position"]
            agent.last_current_cell = observations["current_cell"]
            agent.map_range         = observations["map_range"]
            agent.extra_variables   = observations["extra_variables"]
            check_if_option_done(agent=agent)

            if agent.type == 0 and agent.extra_variables[2] == 1:
                agent.options = [Action(type=0, param_1=0, param_2=0,
                                        description="ride helicopter")]

            agent_states[agent.id] = agent.last_position

        # ── Remove destroyed agents ─────────────────────────────────────────
        for r in removelist:
            agents.remove(r)

        global_data["agents"] = agents

        if check_game_done(global_data=global_data, cfg=cfg.envs, past_score=past_score):
            break

        # ── [CA-CAMON]  Global leader death check ───────────────────────────
        #
        # Must happen BEFORE regional reassignment so handle_global_leader_death
        # can find alternate regional leaders that are still alive.
        #
        # global_leader is the Python object reference; if it was removed from
        # `agents` above, it is dead.
        global_leader = global_data["global_leader"]
        if global_leader not in agents:
            global_leader = handle_global_leader_death(
                dead_global_leader=global_leader,
                agents=agents,
                regions=regions,
            )
            if global_leader is None:
                # No agents remain at all
                print("[CA-CAMON] All agents destroyed. Ending simulation.")
                break
            global_leader.is_global_leader = True
            global_data["global_leader"]   = global_leader

        # ── [CA-CAMON]  Regional leader assignment ──────────────────────────
        #
        # assign_regional_leaders rebuilds all region["leader"] values from
        # scratch.  Because dead agents are already gone from `agents`, this
        # naturally handles regional leader death: the next-best candidate is
        # selected automatically.
        #
        # check_and_reassign_regional_leader then handles:
        #   • Leader walked out of region (boundary exit).
        #   • Reluctant-leader upgrade when an eligible agent has arrived.
        assign_regional_leaders(regions, agents)
        for region in regions:
            check_and_reassign_regional_leader(region, agents)

        # Sync assigned_region and is_global_leader flags onto agents
        for agent in agents:
            agent.assigned_region  = None
            agent.is_global_leader = (agent is global_leader)
        for region in regions:
            rl = region.get("leader")
            if rl is not None:
                rl.assigned_region = region

        # Ensure global leader's region pointer is updated after reassignment
        global_leader_region = get_region_for_agent(regions, global_leader)
        if global_leader_region is not None:
            # Global leader also acts as regional leader for its own region
            # (assign_regional_leaders already set this, but confirm)
            if global_leader_region.get("leader") is not global_leader:
                # Another agent was chosen as regional leader for this region;
                # the global leader is present but not the top regional pick —
                # that is fine.  The global leader still runs orchestrate_global.
                pass
            global_leader.assigned_region = global_leader_region

        # ── [CA-CAMON]  Global leader orchestration ─────────────────────────
        #
        # orchestrate_global() does two things without calling the LLM:
        #   1. Health-check every region's leader; reassign any stale pointer.
        #   2. Detect under-staffed / dormant regions and issue cross-region
        #      redistribution messages via agent.add_message(), so the LLM
        #      prompts in the planning loop already contain the directive.
        orchestrate_global(
            global_leader=global_leader,
            regions=regions,
            agents=agents,
            global_data=global_data,
        )

        # ── Perception generation ───────────────────────────────────────────
        for agent in agents:
            agent.generate_perception(cfg.envs, agent_states, global_data)
            global_data[f'AGENT_{agent.id}'] = {
                'name':           f'AGENT_{agent.id}',
                'perception':     agent.last_perception,
                'position':       agent.last_position,
                'current_action': agent.options[0].description if agent.options else "IDLE",
                'past_actions':   agent.past_options,
            }

        # ── [CA-CAMON]  Per-region planning loop ────────────────────────────
        #
        # For each ACTIVE region (leader is not None):
        #   • Set global_data['leader_agent'] = the region's leader so that
        #     generate_plan / propose_plan use the correct reviewer.
        #   • Regional leader calls generate_plan() (also runs for global leader
        #     when it processes its own region).
        #   • Subordinates call propose_plan().
        # DORMANT regions (leader is None) are skipped.
        for region in regions:
            regional_leader = get_region_leader(region)

            if regional_leader is None:
                print(
                    f"[CA-CAMON] Region {region['id']} is dormant "
                    f"(no agents present). Skipping."
                )
                continue

            # Collect all agents physically in this region
            region_agents = [a for a in agents if _agent_in_region(region, a)]

            # Point leader_agent at THIS region's leader for the duration
            # of this region's planning loop
            global_data['leader_agent'] = regional_leader

            for agent in region_agents:
                if len(agent.options) > 0:
                    print(f"agent {agent.id}: continuing current action")
                    continue

                if agent is regional_leader:
                    # ── Regional leader (may also be global leader) ──────
                    role = ("global+regional leader"
                            if agent is global_leader else "regional leader")
                    print(
                        f"[CA-CAMON] Region {region['id']}: "
                        f"AGENT_{agent.id} ({agent.type_name}, {role}) generating plan."
                    )
                    generate_plan(agent, global_data)

                else:
                    # ── Subordinate: advisory proposal ──────────────────
                    propose_plan(agent, global_data)

                    # Re-read in case leadership transferred inside propose_plan
                    new_leader = global_data['leader_agent']
                    if new_leader is not regional_leader:
                        regional_leader        = new_leader
                        region["leader"]       = new_leader
                        new_leader.assigned_region = region
                        print(
                            f"[CA-CAMON] Region {region['id']}: "
                            f"regional leader updated to AGENT_{new_leader.id}."
                        )

        # ── Action execution ────────────────────────────────────────────────
        env_action = [[0, 0, 0] for _ in range(cfg.envs.num_agents)]

        for agent in agents:
            if agent.type == 0 and agent.extra_variables[2] == 1:
                agent.options = [Action(type=0, param_1=0, param_2=0,
                                        description="ride helicopter")]

            agent.log_chat("Executing Actions", [("system", agent.options[0])])
            print(f"AGENT_{agent.id}: {agent.options[0].description}")

            action_array = generate_action_from_option(agent=agent)
            env_action[agent.id] = action_array
            agent.log_chat("", [("system", f"{t}: {action_array}")])

        print(env_action)

        action_tensor = torch.from_numpy(np.array(env_action)).to(device)
        state["agents"]["action"] = action_tensor
        newstate = env.step(state)
        state["agents"]["observation"] = newstate["next"]["agents"]["observation"]

    env.close()
    print("TEST COMPLETE")
    compile_split_screen_video(path, os.path.join(path, "render.mp4"))


# ── [CA-CAMON]  Bootstrap helper (used before first observation) ────────────

def _bootstrap_global_leader(agents: list):
    """
    Elect the initial global leader by type priority only (positions not yet known).
    helicopter (highest id) → drone (highest id) → first agent.
    """
    for preferred in ("helicopter", "drone"):
        # type int: helicopter=3, drone=2
        type_int = 3 if preferred == "helicopter" else 2
        candidates = [a for a in agents if a.type == type_int]
        if candidates:
            return max(candidates, key=lambda a: a.leadership_score)
    print("[CA-CAMON] WARNING: No eligible bootstrap leader; falling back to agents[0].")
    return agents[0]


if __name__ == "__main__":
    wildfire_alg()
