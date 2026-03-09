import os
from openai import OpenAI
from crew_algorithms.wildfire_alg.algorithms.CAMON.__main__ import Config
from typing import List, Tuple, Dict


class Agent:
    def __init__(self, id: int, type: int, cfg: Config, path, current_task: str, api_key) -> None:
        self.id   = id
        self.type = type
        self.cfg  = cfg

        self.options      = []
        self.action_queue = []
        self.past_options = []

        self.chat_history = {}

        self.current_task      = current_task
        self.last_observation  = None
        self.last_position     = None
        self.last_current_cell = None
        self.last_perception   = None
        self.map_range         = 0
        self.path              = path
        self.model_client      = None
        self.api_key           = api_key
        self.extra_variables   = []

        # [CA-CAMON] Regional assignment.
        # Set by __main__.py each timestep after assign_regional_leaders().
        # None when this agent is not currently acting as any region's leader.
        self.assigned_region: dict | None = None

        # [CA-CAMON] Global leader flag.
        # True for exactly one agent at any point in time.
        # Set/cleared by __main__.py when the global leader is elected or dies.
        self.is_global_leader: bool = False

    # ------------------------------------------------------------------ #
    # [CA-CAMON]  Capability class-level tables                           #
    # ------------------------------------------------------------------ #

    _TYPE_NAMES = {
        0: "firefighter",
        1: "bulldozer",
        2: "drone",
        3: "helicopter",
    }
    _VISION_RANGE = {
        "helicopter": 20,
        "drone":      15,
        "firefighter": 5,
        "bulldozer":   5,
    }
    _MOBILITY_BONUS = {
        "helicopter": 5,
        "drone":      3,
        "firefighter": 0,
        "bulldozer":   0,
    }
    _LEADER_ELIGIBLE = {"helicopter", "drone"}

    # ------------------------------------------------------------------ #
    # [CA-CAMON]  Capability properties                                   #
    # ------------------------------------------------------------------ #

    @property
    def type_name(self) -> str:
        """Human-readable type: 'helicopter', 'drone', 'firefighter', 'bulldozer'."""
        return self._TYPE_NAMES.get(self.type, "firefighter")

    @property
    def leadership_score(self) -> int:
        """
        leadership_score = vision_range + mobility_bonus.
        Workers always return 0.
        """
        if self.type_name not in self._LEADER_ELIGIBLE:
            return 0
        return (
            self._VISION_RANGE.get(self.type_name, 0)
            + self._MOBILITY_BONUS.get(self.type_name, 0)
        )

    @property
    def can_be_leader(self) -> bool:
        """True only for helicopters and drones."""
        return self.type_name in self._LEADER_ELIGIBLE

    # ------------------------------------------------------------------ #
    # Original methods (unchanged)                                        #
    # ------------------------------------------------------------------ #

    def log_chat(self, chat_name: str, messages: List[Tuple[str, str]]):
        chat_string = f"{chat_name}\n" + "-" * 20 + "\n\n"
        for source, content in messages:
            chat_string += f"{source}\n-----\n{content}\n-----\n\n"
        chat_string += "-" * 20 + "\nEND CHAT\n\n"
        filepath = os.path.join(self.path, f"Agent_{self.id}\\chats.txt")
        with open(filepath, "a", encoding="utf-8") as file:
            file.write(chat_string)
        file.close

    def add_message(self, source: str, content: str, time: int):
        self.chat_history.update({f'TIME {time}: {source}': content})

    def generate_perception(
        self,
        cfg: Config,
        agent_states: Dict[int, Tuple[int, int]],
        global_data: dict,
    ):
        print(f"agent {self.id}: generating perception")

        if self.type == 0 and self.extra_variables[2] == 1:
            obs_string = f"""
            You are AGENT_{self.id} and you are within a helicopter. You are unable to perform actions. Your current location is POSITION.

            """
        else:
            obs_string = f"""
                You are AGENT_{self.id}, and your current location is POSITION and thus your minimap view will be the range MAPRANGE with the top corner of the map being (0,0).
                
                This is your minimap view: 
                \n OBS \n 
                Each cell is represented by a character corresponding to the type of terrain:
                    0: brush (no trees)
                    1: light forest (1 tree)
                    2: medium forest (2 trees)
                    3: dense forest (3 trees)
                    i: Ignited
                    f: On Fire
                    e: Extinguishing
                    x: Fully Extinguished
                    w: Water Source Cell (no trees)
                    B: building (no trees)
                    
                IGNORE ALL "-". Those are unrevealed cells. They will reveal themselves when you get closer to them.

                The cells in single quotations are wet cells. 'C' cells are civilians.
                
                The bolded cell is the current cell you are in. It is a CURRCELL cell at POSITION. There are other nearby agents at: 

                OTHERS

                EXTRA

                """

        extra_string = ""
        if self.type == 0:
            if self.extra_variables[0] == 0:
                extra_string += "You are not carrying any civilians.\n"
            else:
                extra_string += "You are carrying a civilian.\n"
            extra_string += f"You current have {int(self.extra_variables[1])} water to spray"
        if self.type == 3:
            if self.extra_variables[0] == 0:
                extra_string += "You are not carrying any firefighters.\n"
            else:
                extra_string += f"You are carrying {int(self.extra_variables[0])} firefighters.\n"
            extra_string += f"You current have {int(self.extra_variables[1])}/5 water to deploy"

        maprange_string = (
            f"x: [{self.last_position[0] - self.map_range // 2}, "
            f"{self.last_position[0] + self.map_range // 2}]\n"
            f"y: [{self.last_position[1] - self.map_range // 2}, "
            f"{self.last_position[1] + self.map_range // 2}]"
        )
        obs_string = obs_string.replace("POSITION",  str(self.last_position))
        obs_string = obs_string.replace("MAPRANGE",  maprange_string)
        obs_string = obs_string.replace("OBS",       self.last_observation)
        obs_string = obs_string.replace("CURRCELL",  self.last_current_cell)
        obs_string = obs_string.replace("EXTRA",     extra_string)
        others = ""
        for a, pos in agent_states.items():
            if (a != self.id
                    and abs(pos[0] - self.last_position[0]) < self.map_range
                    and abs(pos[1] - self.last_position[1]) < self.map_range):
                others += f"AGENT_{a}: {pos} "
        obs_string = obs_string.replace("OTHERS", others)

        if self.type == 0:
            type_string = 'Firefighter'
        elif self.type == 1:
            type_string = 'Bulldozer'
        elif self.type == 2:
            type_string = 'Drone'
        else:
            type_string = 'Helicopter'

        system_message = f"""
                You are the Perception Module of an embodied {type_string} agent, AGENT_{self.id}, within a large grid world spanning from x:[0 to {cfg.map_size - 1}] and y:[0 to {cfg.map_size - 1}]. 
                Your job is to process and understand your surroundings. 
                Do not directly report explicit information from the minimap, but rather spatially understand your surroundings.
                Do not refer to character representations of the minimap, only what they actually represent.
                Report general observations in general directions.
                Also report if there are specific cells of interest, such as fires, civilians, water, etc. 
                If there are any, calculate their exact locations by explicitly counting cells. 

                You should return a detailed but concise text summary paragraph of all relevant information, including location, surroundings, and presence of important cells.
                """
        user_message = (
            f"Here are your observations: \n\n{obs_string}\n\n"
            f"Create a detailed text summary of all relevant information, such as location, "
            f"surroundings, presence of fire and civilians, etc. "
            f"Speak only in first person as AGENT_{self.id}. "
        )

        client = OpenAI(
            base_url="https://tritonai-api.ucsd.edu",
            api_key=self.api_key
        )
        response = client.chat.completions.create(
            model="api-gpt-oss-120b",
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user",   "content": user_message}
            ],
            temperature=0
        )
        global_data["api_calls"]     += 1
        global_data["input_tokens"]  += response.usage.prompt_tokens
        global_data["output_tokens"] += response.usage.completion_tokens

        perception = response.choices[0].message.content
        self.log_chat(
            "Summarizing Observations",
            [("user", user_message), ("assistant", perception)]
        )
        self.last_perception = perception
        client.close()
        return perception
