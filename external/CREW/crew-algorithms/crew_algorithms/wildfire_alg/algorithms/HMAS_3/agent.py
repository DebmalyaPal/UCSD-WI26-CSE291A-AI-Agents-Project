import os
from openai import OpenAI
from typing import List, Tuple, Dict, Optional

# Role constants
ROLE_WORKER = "worker"
ROLE_REGIONAL_LEADER = "regional_leader"
ROLE_GLOBAL_LEADER = "global_leader"


class Agent:
    def __init__(
        self,
        id: int,
        type: int,
        cfg,
        path: str,
        current_task: str,
        api_key: str,
        agent_count: int,
        role: str = ROLE_WORKER,
        region_id: Optional[int] = None,
    ) -> None:
        self.id = id
        self.type = type
        self.cfg = cfg
        self.path = path
        self.current_task = current_task
        self.api_key = api_key
        self.role = role

        # Region assignment
        # region_id is the index into the flat list of 20x20 regions
        self.region_id: Optional[int] = region_id
        # home_region_id is the region this agent was originally assigned to.
        # Used for recall after temporary reallocation.
        self.home_region_id: Optional[int] = region_id

        # Chat history per peer + GLOBAL channel
        self.chat_history: Dict[str, List[Tuple[str, str, int]]] = {
            f"AGENT_{i + 1}": [] for i in range(agent_count) if (i + 1) != id
        }
        self.chat_history["GLOBAL"] = []

        self.past_actions: List[str] = []
        self.last_observation: str = None
        self.last_position: Tuple[int, int] = None
        self.last_current_cell: str = None
        self.last_perception: str = None
        self.map_range: int = 0
        self.extra_variables = []

    # ------------------------------------------------------------------
    # Logging
    # ------------------------------------------------------------------

    def log_chat(self, chat_name: str, messages: List[Tuple[str, str]]):
        chat_string = f"{chat_name}\n" + "-" * 20 + "\n\n"
        for source, content in messages:
            chat_string += f"{source}\n-----\n{content}\n-----\n\n"
        chat_string += "-" * 20 + "\nEND CHAT\n\n"
        agent_dir = os.path.join(self.path, f"Agent_{self.id}")
        os.makedirs(agent_dir, exist_ok=True)
        filepath = os.path.join(agent_dir, "chats.txt")
        with open(filepath, "a", encoding="utf-8") as file:
            file.write(chat_string)

    # ------------------------------------------------------------------
    # Perception generation  (identical to HMAS_2 but region-aware label)
    # ------------------------------------------------------------------

    def generate_perception(self, cfg, agent_states: Dict[int, Tuple[int, int]], global_data: dict):
        print(f"agent {self.id} [role={self.role}, region={self.region_id}]: generating perception")

        if self.type == 0 and len(self.extra_variables) > 2 and self.extra_variables[2] == 1:
            obs_string = (
                f"You are AGENT_{self.id} and you are within a helicopter. "
                f"You are unable to perform actions. Your current location is {self.last_position}.\n"
            )
        else:
            obs_string = f"""
You are AGENT_{self.id}, and your current location is {self.last_position} and thus your minimap view will be the range
x: [{self.last_position[0] - self.map_range // 2}, {self.last_position[0] + self.map_range // 2}]
y: [{self.last_position[1] - self.map_range // 2}, {self.last_position[1] + self.map_range // 2}]
with the top corner of the map being (0,0).

This is your minimap view:
{self.last_observation}

Each cell is represented by a character:
    0: brush, 1: light forest, 2: medium forest, 3: dense forest
    i: Ignited, f: On Fire, e: Extinguishing, x: Fully Extinguished
    w: Water Source, B: Building
IGNORE ALL "-" (unrevealed). Single-quoted cells are wet. 'C' = civilian.
The bolded cell is your current cell. It is a {self.last_current_cell} cell at {self.last_position}.
"""
            others = " ".join(
                f"AGENT_{a}: {pos}"
                for a, pos in agent_states.items()
                if a != self.id
                and abs(pos[0] - self.last_position[0]) < self.map_range
                and abs(pos[1] - self.last_position[1]) < self.map_range
            )
            obs_string += f"\nNearby agents: {others}\n"

            extra_string = ""
            if self.type == 0:
                extra_string += "Not carrying civilians.\n" if self.extra_variables[0] == 0 else "Carrying a civilian.\n"
                extra_string += f"Water remaining: {int(self.extra_variables[1])}\n"
            if self.type == 3:
                extra_string += (
                    "Not carrying firefighters.\n"
                    if self.extra_variables[0] == 0
                    else f"Carrying {int(self.extra_variables[0])} firefighters.\n"
                )
                extra_string += f"Water remaining: {int(self.extra_variables[1])}/5\n"
            obs_string += extra_string

        type_string = {0: "Firefighter", 1: "Bulldozer", 2: "Drone", 3: "Helicopter"}.get(self.type, "Agent")
        role_note = f" You are acting as {self.role.replace('_', ' ').title()} for region {self.region_id}." if self.region_id is not None else ""

        system_message = (
            f"You are the Perception Module of an embodied {type_string} agent, AGENT_{self.id}, "
            f"within a large grid world spanning x:[0 to {cfg.map_size - 1}] and y:[0 to {cfg.map_size - 1}].{role_note} "
            "Your job is to process and understand your surroundings. Report spatially, not by character codes. "
            "Report general observations in general directions. Calculate exact locations of fire, civilians, water."
        )
        user_message = (
            f"Here are your observations:\n\n{obs_string}\n\n"
            f"Create a detailed text summary of all relevant information. Speak in first person as AGENT_{self.id}."
        )

        client = OpenAI(api_key=self.api_key)
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": user_message},
            ],
            temperature=0,
        )
        global_data["api_calls"] += 1
        global_data["input_tokens"] += response.usage.prompt_tokens
        global_data["output_tokens"] += response.usage.completion_tokens

        perception = response.choices[0].message.content
        self.log_chat("Summarizing Observations", [("user", user_message), ("assistant", perception)])
        self.last_perception = perception
        client.close()
        return perception