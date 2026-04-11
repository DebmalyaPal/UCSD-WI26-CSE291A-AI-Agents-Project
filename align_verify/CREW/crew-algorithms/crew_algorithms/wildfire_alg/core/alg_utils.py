import torch
import numpy as np
from pydantic import BaseModel


class Action(BaseModel):
    """
    Represents an action that can be taken by an agent in the environment.
    
    Attributes:
        type (int): The type of action to be performed
        param_1 (int): First parameter for the action (typically x-coordinate)
        param_2 (int): Second parameter for the action (typically y-coordinate)
        description (str): Human-readable description of the action
    """
    type: int
    param_1: int
    param_2: int
    description: str

    def print_option(self) -> None:
        """
        Prints the action details in a list format [type, param_1, param_2, description].
        """
        print([self.type, self.param_1, self.param_2, self.description])

from crew_algorithms.wildfire_alg.libraries.firefighter_option_library import Run_Firefighter_Option
from crew_algorithms.wildfire_alg.libraries.bulldozer_option_library import Run_Bulldozer_Option
from crew_algorithms.wildfire_alg.libraries.drone_option_library import Run_Drone_Option
from crew_algorithms.wildfire_alg.libraries.helicopter_option_library import Run_Helicopter_Option

def get_agent_observations(state, agent_index, device="cpu"):
    """
    Extract observations for a specific agent from environment state data.
    
    Args:
        state (dict or Tensor): Environment state data containing agent observations
        agent_index (int): Index of the agent to get observations for
        device (str): Device to put the tensor on (default: "cpu")
        
    Returns:
        dict: Dictionary containing parsed observations including:
            - raw_observation: The raw observation tensor
            - position: (x, y) tuple of agent position
            - agent_type: Type of agent (0-5)
            - current_cell: Value of current cell
            - perception_grid: Formatted string representation of perception grid
            - map_range: Size of the perception grid
            - extra_variables: List of additional state variables
    """
    state = state.float().to(device) if not isinstance(state, dict) else state
    
    # Handle different state formats
    if "agents" in state.keys() and "observation" in state["agents"].keys():
        if state["agents"]["observation"]["obs_0"].shape == torch.Size([state["agents"]["observation"]["obs_0"].shape[0], 3728]):
            observations = state["agents"]["observation"]["obs_0"].squeeze()
        else:
            observations = state["agents"]["observation"]["obs_0"][-1]
    else:
        observations = state

    # Get the specific agent's observation
    agent_observation = observations[agent_index]
    
    # Parse agent type and position
    agent_type = int(agent_observation[0].detach().item())
    
    # Determine map range based on agent type
    map_range = 0
    l = len(agent_observation)
    position = (agent_observation[l-5].detach().item(), agent_observation[l-4].detach().item())

    extra_variables = [agent_observation[l-3].detach().item(), agent_observation[l-2].detach().item(), agent_observation[l-1].detach().item()]

    if agent_type == 0:
        map_range = 21
    elif agent_type == 1:
        map_range = 21
    elif agent_type == 2:  # Drone
        map_range = 51
    elif agent_type == 3:  # Helicopter
        map_range = 61

    elif agent_type == 5:  # Default or destroyed
        return {"raw_observation": agent_observation, 
                "position": None,
                "agent_type": agent_type,
                "current_cell": None,
                "perception_grid": None,
                "map_range": None,
                "extra_variables": extra_variables
                }
    
    # Get position (assuming it's at the end of the observation)

    
    # Parse the grid
    perception_grid = ""
    current_cell = None
    
    for i in range(len(agent_observation)):
        if i == 0:  # Skip agent type
            continue
        elif agent_observation[i].detach().item() == -1:
            break
        else:
            cell_value = int(agent_observation[i].detach().item())
            
            # Get current cell (middle of the grid)
            if i == ((map_range*map_range)//2)+1:
                current_cell = translate_cell(cell_value)
                perception_grid += ("*" + current_cell + "*,")
            else:
                perception_grid += translate_cell(cell_value) + ","
                
            # Add newline after each row
            if (i % map_range == 0):
                perception_grid += "\n"
    
    return {
        "raw_observation": agent_observation,
        "position": position,
        "agent_type": agent_type,
        "current_cell": current_cell,
        "perception_grid": perception_grid,
        "map_range": map_range,
        "extra_variables": extra_variables
    }


def translate_cell(x):
    """
    Translate numerical cell values to human-readable characters in the environment grid.
    
    Args:
        x (int): Numerical cell value representing a cell state
        
    Returns:
        str: Character representation of the cell:
            - "-": Empty cell (0)
            - "'0'" to "'3'": Burning cells with different intensities (1-4)
            - "0" to "3": Trees with different densities (5-8)
            - "i": Initial fire (9)
            - "f": Firefighter (10)
            - "e": Extinguished cell (11)
            - "x": Destroyed cell (12)
            - "w": Water source (13)
            - "B": Bulldozer (14)
            - "C": Cut trees (15)
            - " ": Unknown or invalid cell
    """
    match(x):
        case 0:
            return "-"
        case 1:
            return "'0'"
        case 2:
            return "'1'"
        case 3:
            return "'2'"
        case 4:
            return "'3'"
        case 5:
            return "0"
        case 6:
            return "1"
        case 7:
            return "2"
        case 8:
            return "3"
        case 9:
            return "i"
        case 10:
            return "f"
        case 11:
            return "e"
        case 12:
            return "x"
        case 13:
            return "w"
        case 14:
            return "B"
        case 15:
            return "C"
        
        case _:
            return " "

def submit_actions_to_environment(state, actions, device="cpu"):
    """
    Format and submit actions to the environment by converting them to tensors.
    
    Args:
        state (dict): Current environment state containing agent information
        actions (list): List of action arrays [action_type, x, y] for each agent
        device (str): Device to put tensors on (default: "cpu")
        
    Returns:
        dict: Updated state dictionary with the new actions added to state["agents"]["action"]
    """
    # Convert actions to tensor
    action_tensor = torch.from_numpy(np.array(actions)).to(device)
    
    # Update state with actions
    if isinstance(state, dict) and "agents" in state:
        state["agents"]["action"] = action_tensor
    
    return state


def parse_game_data(state, cfg):
    """
    Parse game environment information from the first observation to determine scenario settings.
    
    Args:
        state (dict or Tensor): Environment state containing game metadata
        cfg: Configuration object containing game settings
        
    Returns:
        dict: Parsed game information including:
            - game_type: Type of game/scenario (int)
            - map_size: Size of the map grid (int)
            - score: Current score (int)
            - task_description: Human-readable task description (str)
            - task_parameters: Specific parameters for the task (dict)
    """

    
    # Handle different state formats
    if "agents" in state.keys() and "observation" in state["agents"].keys():
        if state["agents"]["observation"]["obs_0"].shape == torch.Size([state["agents"]["observation"]["obs_0"].shape[0], 3728]):
            observations = state["agents"]["observation"]["obs_0"].squeeze()
        else:
            observations = state["agents"]["observation"]["obs_0"][-1]
    else:
        observations = state
    
    game_data = observations[0]

    game_type = int(game_data[1].item())
    map_size = int(game_data[2].item())
    score = int(game_data[3].item())
    
    task_description = ""
    task_parameters = {}
    
    # Parse specific task parameters based on game type
    match game_type:
        # Cut Trees
        case 0:
            lines = game_data[4].item()
            task_parameters["lines"] = lines
            
            if lines == 0:
                # Sparse tree cutting
                count = game_data[5].item()
                task_parameters["count"] = count
                
                target_trees = []
                for i in range(int(count)):
                    target_trees.append((int(game_data[6+(i*2)]), int(game_data[7+(i*2)])))
                
                task_parameters["target_trees"] = target_trees
                coords = (", ".join(f"({x}, {y})" for x, y in target_trees))
                task_description = f"Cut all trees at {coords}"
            else:
                # Line tree cutting
                count = game_data[5].item()
                task_parameters["count"] = count
                
                target_lines = []
                for i in range(int(count)):
                    target_lines.append((
                        int(game_data[6+(i*4)]), 
                        int(game_data[7+(i*4)]), 
                        int(game_data[8+(i*4)]), 
                        int(game_data[9+(i*4)])
                    ))
                
                task_parameters["target_lines"] = target_lines
                coords = (", ".join(f"from ({x_1}, {y_1}) to ({x_2}, {y_2})" for x_1, y_1, x_2, y_2 in target_lines))
                task_description = f"Cut all trees in the following lines: {coords}"
        
        # Scout Fire
        case 1:
            task_description = f"Scout and confirm a fire within the map x: [0 to {map_size}] and y: [0 to {map_size}]. You need two agents directly over the fire to confirm it."
        
        # Pick and Place
        case 2:
            target_x = game_data[4].item()
            target_y = game_data[5].item()
            task_parameters["target_x"] = target_x
            task_parameters["target_y"] = target_y
            task_description = f"Transport all Firefighter Agents to the target location: [{target_x}, {target_y}]"
        
        # Contain Fire
        case 3:
            fire_known = cfg.envs.known  # Assuming this is how you determine if fire location is known
            task_parameters["fire_known"] = fire_known
            
            if fire_known:
                fire_x = game_data[4].item()
                fire_y = game_data[5].item()
                task_parameters["fire_x"] = fire_x
                task_parameters["fire_y"] = fire_y
                
                water = cfg.envs.water  # Assuming this is how you determine if water is available
                task_parameters["water"] = water
                
                if water:
                    water_x = game_data[6].item()
                    water_y = game_data[7].item()
                    task_parameters["water_x"] = water_x
                    task_parameters["water_y"] = water_y
                    
                    task_description = f"Fully contain/suppress the fire spreading near [{fire_x}, {fire_y}]. Be cautious to not touch it. Use the water source at [{water_x}, {water_y}] to refill water supplies."
                else:
                    task_description = f"Fully contain the fire near [{fire_x}, {fire_y}] Be cautious to not touch it. Cut trees to make firebreaks."
            else:
                task_description = f"Find and fully contain the fire within the map x: [0 to {map_size}] and y: [0 to {map_size}]. Be cautious to not touch it. Cut trees to make firebreaks."
        
        # Move Civilians
        case 4:
            target_x = game_data[4].item()
            target_y = game_data[5].item()
            task_parameters["target_x"] = target_x
            task_parameters["target_y"] = target_y
            
            civilian_known = cfg.envs.known  # Assuming this is how you determine if civilian locations are known
            task_parameters["civilian_known"] = civilian_known
            
            civilian_clusters = cfg.envs.civilian_clusters
            civilian_count = cfg.envs.civilian_count
            task_parameters["civilian_clusters"] = civilian_clusters
            task_parameters["civilian_count"] = civilian_count
            
            if civilian_known:
                clusters = []
                for i in range(civilian_clusters):
                    clusters.append((int(game_data[6+(i*2)]), int(game_data[7+(i*2)])))
                
                task_parameters["clusters"] = clusters
                coords = (", ".join(f"({x}, {y})" for x, y in clusters))
                task_description = f"There are {civilian_clusters} groups of civilians scattered near {coords}. Each group has {civilian_count} civilians. Transport all civilians to the target safe location of [{target_x}, {target_y}]."
            else:
                task_description = f"There are {civilian_clusters} groups of civilians scattered within the map x: [0 to {map_size}] and y: [0 to {map_size}]. Each group has {civilian_count} civilians. Find and transport all civilians to the target safe location of [{target_x}, {target_y}]."
    
            # Both
        case 5:

            known = cfg.envs.known

            if known:
                fire_x = game_data[4].item()
                fire_y = game_data[5].item()
                task_parameters["fire_x"] = fire_x
                task_parameters["fire_y"] = fire_y


                civilian_clusters = cfg.envs.civilian_clusters
                civilian_count = cfg.envs.civilian_count
                
                clusters = []
                for i in range(civilian_clusters):
                    clusters.append((int(game_data[8+(i*2)]), int(game_data[9+(i*2)])))
                
                task_parameters["clusters"] = clusters
                coords = (", ".join(f"({x}, {y})" for x, y in clusters))

                water = cfg.envs.water  # Assuming this is how you determine if water is available
                task_parameters["water"] = water
                if water:
                    water_x = game_data[6].item()
                    water_y = game_data[7].item()
                    task_parameters["water_x"] = water_x
                    task_parameters["water_y"] = water_y

                    task_description = f"There is a fire near [{fire_x}, {fire_y}]. Be cautious to not touch it. Use the water source at [{water_x}, {water_y}] to refill water supplies. BeThere are also {civilian_clusters} groups of civilians scattered near {coords}. Each group has {civilian_count} civilians. Contain/Suppress the fire while transporting the civilians away from danger."

                else:
                    task_description = f"There is a fire near [{fire_x}, {fire_y}]. Be cautious to not touch it. There are also {civilian_clusters} groups of civilians scattered near {coords}. Each group has {civilian_count} civilians. Contain/Suppress the fire while transporting the civilians away from danger."
            else:
                task_description = f"There is a fire within the map x: [0 to {map_size}] and y: [0 to {map_size}. Be cautious to not touch it. There are also {civilian_clusters} groups of civilians scattered within the map. Each group has {civilian_count} civilians. Search for and Contain/Suppress the fire while transporting civilians away from danger."
                
                


    return {
        "game_type": game_type,
        "map_size": map_size,
        "score": score,
        "task_description": task_description,
        "task_parameters": task_parameters
    }


def generate_action_from_option(agent):
    """
    Generate an action based on the current option for an agent. This function manages the action queue
    and converts high-level options into specific actions that can be executed in the environment.
    
    Args:
        agent: The agent object containing:
            - id: Agent identifier
            - type: Type of agent (0: Firefighter, 1: Bulldozer, 2: Drone, 3: Helicopter)
            - options: List of pending options to execute
            - action_queue: Queue of specific actions for the current option
            - past_options: List of completed options
    
    Returns:
        list: Action representation as [action_type, x, y] or [0, 0, 0] if there's an error
    """
    if len(agent.options)==0:
        return
    current_option = agent.options[0]

    print(f"agent {agent.id}: current option: {current_option}, action queue: {agent.action_queue}")

    # Initialize action queue if needed
    if not hasattr(agent, 'action_queue') or len(agent.action_queue) == 0:
        # Default action libraries

        try:

            action_libraries = {
                    0: Run_Firefighter_Option,  # Firefighter
                    1: Run_Bulldozer_Option,    # Bulldozer
                    2: Run_Drone_Option,        # Drone
                    3: Run_Helicopter_Option    # Helicopter
                }
            # Generate queue of actions for the option
            if agent.type in action_libraries:
                action_libraries[agent.type](agent, current_option)
                

            else:
                raise ValueError(f"No action library defined for agent type {agent.type}")
        except:
            error_option = Action(type=0, param_1=0, param_2=0, description=f"ERROR EXECUTING ACTION: {current_option}")
            agent.past_options.append(error_option)
            agent.options.pop(0)
            return [0,0,0]


    
    # Get next action from queue
    action = agent.action_queue.pop(0)
    
    # Handle completion of option
    if action.done:

        agent.past_options.append(current_option)
        agent.options.pop(0)

        # Generate next action if there are more options
        if len(agent.options) > 0:
            return generate_action_from_option(agent)

    
    # Format the action as [action_type, x, y]
    action_array = [action.action, action.x, action.y]
    #print("action" + str(action_array))
    
    
    return action_array


def check_if_option_done(agent):
    """
    Check if the current option for an agent is completed.
    
    Args:
        agent: The agent object containing:
            - options: List of pending options
            - action_queue: Queue of actions for current option
            - type: Type of agent
            - past_options: List of completed options
    
    Returns:
        bool: True if the current option is completed, False otherwise
    """
    if len(agent.options)==0 or len(agent.action_queue)!=0:
        return False
    current_option = agent.options[0]
    #should just be conditional options

    # Initialize action queue if needed
    if not hasattr(agent, 'action_queue') or len(agent.action_queue) == 0:
        # Default action libraries
        action_libraries = {
                0: Run_Firefighter_Option,  # Firefighter
                1: Run_Bulldozer_Option,    # Bulldozer
                2: Run_Drone_Option,        # Drone
                3: Run_Helicopter_Option    # Helicopter
            }
        
        # Generate queue of actions for the option
        if agent.type in action_libraries:
            action_libraries[agent.type](agent, current_option)
        else:
            raise ValueError(f"No action library defined for agent type {agent.type}")
    
    # Get next action from queue
    action = agent.action_queue.pop(0)
    
    # Handle completion of option
    if action.done:

        agent.past_options.append(current_option)
        agent.options.pop(0)
        return True

    
    return False

def check_game_done(global_data, cfg, past_score):
    """
    Check if the current game/scenario is completed based on the game type and score.
    
    Args:
        global_data (dict): Dictionary containing current game state including score
        cfg: Configuration object containing game settings and win conditions
        past_score (int): Previous score for comparison in certain game types
        
    Returns:
        bool: True if the game is completed according to its win conditions, False otherwise
        
    Game Types:
        0: Cut trees - Complete when score >= tree_count * trees_per_line * 3
        1: Scout fire - Complete when score >= 2
        2: Pick and place - Complete when score >= starting_firefighter_agents
        3: Contain fire - Never complete (returns False)
        4: Rescue civilians - Complete when score >= civilian_count * civilian_clusters
        5: Custom scenario - Never complete (returns False)
    """
    gametype = cfg.game_type
    score = global_data["score"]

    # Cut trees
    if gametype == 0:
        if score >= cfg.tree_count *cfg.trees_per_line * 3:
            return True
        else:
            return False
    # scout fire
    elif gametype == 1:
        if score>= 2:
            return True
        else:
            return False
    # pick and place
    elif gametype == 2:
        if score >= cfg.starting_firefighter_agents:
            return True
        else:
            return False
    # contain fire
    elif gametype == 3:
        # if len(global_data["firefighters"])==0 or (score == past_score and score!=0):
        #     return True
        # else:
            return False
    # rescue civilians
    elif gametype == 4:
        if score >= cfg.civilian_count*cfg.civilian_clusters:
            return True
        else:
            return False
    elif gametype == 5:
        # if len(global_data["firefighters"])==0 or (score == past_score and score!=0):
        #     return True
        # else:
            return False
    else:
        print("invalid game type")
        return True
    