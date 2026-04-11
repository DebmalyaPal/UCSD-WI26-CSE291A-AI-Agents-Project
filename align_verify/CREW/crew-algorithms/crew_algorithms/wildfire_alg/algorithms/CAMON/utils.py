import re
from openai import OpenAI
from pydantic import BaseModel
from crew_algorithms.wildfire_alg.algorithms.CAMON.agent import Agent


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

    def print_option(self)->None:
        """Prints the action details in a list format [type, param_1, param_2, description]."""
        print([self.type, self.param_1, self.param_2, self.description])


def _normalize_decision(decision: str) -> str:
    """
    Normalize a decision string from the LLM.
    Returns one of: 'STILL_POSSIBLE', 'NOT_POSSIBLE', or the raw string uppercased.
    """
    d = decision.strip().upper()
    if "NOT" in d and "POSSIBLE" in d:
        return "NOT_POSSIBLE"
    if "STILL" in d or "POSSIBLE" in d or d == "YES":
        return "STILL_POSSIBLE"
    return d


def translate_action(option_str: str, type: int, global_data: dict) -> Action:
    """
    Translates a natural language action description into a structured Action object using GPT-4.
    
    Args:
        option_str (str): Natural language description of the action
        type (int): Agent type (0=firefighter, 1=bulldozer, 2=drone, 3=helicopter)
        global_data (dict): Global state containing API keys and other shared data
        
    Returns:
        Action: Structured action object with type, parameters and description
    """
    system_message = f"""
                        You are the controller of a highly trained embodied agent within a grid forest world. 
                        Your job is to convert a string action into a structured format for robotic control."""
    # Determine type
    if type == 0:
        type_string = 'firefighter'
    elif type == 1:
        type_string = 'bulldozer'
    elif type == 2:
        type_string = 'drone'
    else:
        type_string = 'helicopter'
    # Load prompt
    prompt_path = f'crew_algorithms/wildfire_alg/algorithms/CAMON/prompts/translator/{type_string}_translator.txt'
    with open(prompt_path, 'r', encoding='utf-8') as f:
        prompt = f.read().replace("ACTION", option_str)
    # Call OpenAI
    client = OpenAI(api_key=global_data['leader_agent'].api_key, base_url="https://tritonai-api.ucsd.edu")
    response = client.chat.completions.create(
        model='api-gpt-oss-120b',
        messages=[
            {'role':'system', 'content':system_message},
            {'role':'user', 'content':prompt}
        ],
        temperature=0.7
    )
    global_data["api_calls"]+=1
    global_data["input_tokens"]+=response.usage.prompt_tokens
    global_data["output_tokens"]+=response.usage.completion_tokens
    result = response.choices[0].message.content
    
    # Extract fields
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
        # Fall back to a safe no-op style action but preserve description where possible
        try:
            desc = extract('description')
        except Exception:
            desc = "UNKNOWN ACTION"
        return Action(
            type=0,
            param_1=0,
            param_2=0,
            description='ERROR EXECUTING: ' + desc
        )


def review_option_feasibility(agent: Agent, global_data: dict, horizon: int = 2) -> None:
    """
    Periodically ask an LLM whether the agent's current high-level option is still feasible
    given its latest perception and location. If not feasible, the option is cancelled.
    Optionally, the model may suggest an alternative action that replaces the current option.

    This is intended to improve plan adaptation: rather than blindly executing long
    option queues, agents can abort or revise when the environment has changed.
    """
    # Basic guards: need an active option and tracking metadata
    if len(agent.options) == 0:
        return

    t = global_data.get("time")
    if t is None:
        return

    # Only review every `horizon` timesteps per option to bound cost
    if agent.last_option_review_time is not None:
        if (t - agent.last_option_review_time) < horizon:
            return

    # Only review if the option has been active for at least `horizon` steps
    if (t - agent.option_start_time) < horizon:
        return

    # Need a perception summary and position to reason about feasibility
    if agent.last_perception is None or agent.last_position is None:
        return

    print(f"[CAMON] Agent {agent.id}: reviewing option feasibility")

    current_option = agent.options[0]
    type_string = (
        "Firefighter" if agent.type == 0 else
        "Bulldozer" if agent.type == 1 else
        "Drone" if agent.type == 2 else
        "Helicopter"
    )

    system_msg = f"""
                You are AGENT_{agent.id}, an embodied {type_string} agent executing a plan
                in a dynamic wildfire environment.
                Your job is to decide whether your current high-level action is still feasible
                and safe given your most recent observations.
                Be conservative: if the environment has changed in a way that makes the
                current goal unsafe, unreachable, or clearly suboptimal, you should say it
                is NOT_POSSIBLE. Otherwise, say it is STILL_POSSIBLE.
                """

    user_msg = f"""
                Current timestep: {t}
                Your current location: {agent.last_position}

                Your current high-level action:
                '{current_option.description}'

                Your latest perception summary:
                '{agent.last_perception}'

                Question:
                - Is it still feasible and reasonable to continue executing this current high-level action,
                  given the above perception?

                Respond in this exact format:
                <decision>STILL_POSSIBLE or NOT_POSSIBLE</decision>
                """

    print(f"[CAMON] Agent {agent.id}: reviewing feasibility of option '{current_option.description}'")

    client = OpenAI(api_key=agent.api_key, base_url="https://tritonai-api.ucsd.edu")
    response = client.chat.completions.create(
        model="api-gpt-oss-120b",
        messages=[
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg},
        ],
        temperature=0
    )
    global_data["api_calls"] += 1
    global_data["input_tokens"] += response.usage.prompt_tokens
    global_data["output_tokens"] += response.usage.completion_tokens

    content = response.choices[0].message.content
    client.close()

    agent.log_chat(
        "Reviewing Option Feasibility",
        [("user", user_msg), ("assistant", content)],
    )

    dec_match = re.search(r"<decision>\s*(.*?)\s*</decision>", content, re.DOTALL | re.IGNORECASE)
    decision_raw = dec_match.group(1).strip() if dec_match else ""
    decision = _normalize_decision(decision_raw) if decision_raw else ""

    agent.last_option_review_time = t

    # If no clear decision, do nothing
    if not decision:
        print(f"[CAMON] Agent {agent.id}: feasibility review returned no clear decision; keeping option.")
        return

    # If the option is no longer considered possible, cancel it
    if decision == "NOT_POSSIBLE":
        print(f"[CAMON] Agent {agent.id}: feasibility review -> NOT_POSSIBLE, cancelling current option.")
        agent.past_options.append(current_option)
        agent.option_cancelled_recently = True
        agent.last_cancelled_option = current_option.description
        agent.force_stop_next_step = True
        agent.options.pop(0)
        agent.action_queue = []
        agent.option_start_time = None
        # Allow leader or future planning stages to assign a new option
        return

    # If STILL_POSSIBLE, keep executing current option as-is
    print(f"[CAMON] Agent {agent.id}: feasibility review -> STILL_POSSIBLE, continuing current option.")



def generate_plan(agent: Agent, global_data: dict) -> None:
    """
    Generates a plan for an agent based on current state and team composition.
    
    Args:
        agent (Agent): The agent to generate a plan for
        global_data (dict): Global state containing team composition and environment info
        
    Effects:
        - Updates agent's options list with next planned action
        - Logs planning process in agent's chat history
    """
    print(f"[CAMON] Leader {agent.id}: generating actions")
    # Identify type
    if agent.type == 0:
        type_string = 'Firefighter'
    elif agent.type == 1:
        type_string = 'Bulldozer'
    elif agent.type == 2:
        type_string = 'Drone'
    else:
        type_string = 'Helicopter'
    # Team composition
    team_comp_string = ''
    if global_data['firefighters']:
        team_comp_string += f"{[f'AGENT_{a.id}' for a in global_data['firefighters']]} {'is' if len(global_data['firefighters'])==1 else 'are'} Firefighter Agents. Firefighter agents are general purpose agents with decent speed and observation capabilities. They can move, cut trees, spray water, and rescue civilians.\n\n"
    if global_data['bulldozers']:
        team_comp_string += f"{[f'AGENT_{a.id}' for a in global_data['bulldozers']]} {'is' if len(global_data['bulldozers'])==1 else 'are'} Bulldozer Agents. Bulldozer agents are specialized agents with exceptional tree-cutting abilities but limited speed.\n\n"
    if global_data['drones']:
        team_comp_string += f"{[f'AGENT_{a.id}' for a in global_data['drones']]} {'is' if len(global_data['drones'])==1 else 'are'} Drone Agents. Drone agents are specialized recon agents with exceptional speed and observations.\n\n"
    if global_data['helicopters']:
        team_comp_string += f"{[f'AGENT_{a.id}' for a in global_data['helicopters']]} {'is' if len(global_data['helicopters'])==1 else 'are'} Helicopter Agents. Helicopter agents are general support agents with exceptional speed and observations. They can move, pick up and drop off Firefighter Agents, and spray water\n\n"
    # Abilities descriptions

    team_abilities = ''

    desc_files = ['firefighter', 'bulldozer', 'drone', 'helicopter']

    for kind in desc_files:

        if global_data.get(kind+'s'):

            path = f'crew_algorithms/wildfire_alg/algorithms/CAMON/prompts/descriptions/{kind}_description.txt'
            with open(path, 'r', encoding='utf-8') as f:
                team_abilities += f.read()

    # Past and chat

    past_string = ''.join(str(o.description)+'\n' for o in agent.past_options)
    chat_string = ''.join(f"{time}: \n{msg}\n\n" for time, msg in agent.chat_history.items())
    global_str = []
    for data in global_data.items():
        if data[0].__contains__("AGENT"):
            global_str.append(data)

    # Prompt
    generate_plan_string = f"""
            You are AGENT_{agent.id} a {type_string} Agent, currently acting as the leader in a cooperative multi-agent robotic task. 
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
    # Call OpenAI
    client = OpenAI(api_key=agent.api_key, base_url="https://tritonai-api.ucsd.edu")
    system_message = f"""
                    You are AGENT_{agent.id}, currently acting as the leader in a cooperative multi-agent robotic task. Your team is in a  {agent.cfg.envs.map_size} by {agent.cfg.envs.map_size} forest grid world that spans x:[0 to {agent.cfg.envs.map_size}] and y:[0 to {agent.cfg.envs.map_size}].
                    You have access to the collective observations and the progress of all agents. Your job is to plan the next best action for yourself, and OPTIONALLY: the next best action for any other agents."""
    user_message = generate_plan_string
    response = client.chat.completions.create(
        model='api-gpt-oss-120b',
        messages=[{'role':'system','content':system_message},{'role':'user','content':user_message}],
        temperature=0.7
    )
    global_data["api_calls"]+=1
    global_data["input_tokens"]+=response.usage.prompt_tokens
    global_data["output_tokens"]+=response.usage.completion_tokens

    result = response.choices[0].message.content

    agent.log_chat("Generating Plan", [("user", user_message), ("assistant", result)])

    # Own action

    m = re.search(r"<action>\s*(.*?)\s*</action>", result, re.DOTALL)
    if m:
        option = translate_action(m.group(1), agent.type, global_data)
        agent.options = [option]
        agent.action_queue = []
        agent.option_start_time = global_data.get("time", agent.option_start_time)
        agent.last_option_review_time = None
        print(f"agent {agent.id}: provided action: '{m.group(1)}' to itself")


    else:
        print("ERROR NO ACTION FOUND")
        return
    
    # Optional for others
    for a in global_data["agents"]:
        ma = re.search(fr"<AGENT_{a.id}-action>\s*(.*?)\s*</AGENT_{a.id}-action>", result, re.DOTALL)
        mm = re.search(fr"<AGENT_{a.id}-message>\s*(.*?)\s*</AGENT_{a.id}-message>", result, re.DOTALL)



        if ma and mm:
             # if in helicopter
            if a.type==0 and a.extra_variables[2]==1:
                a.options = [Action(type=0, param_1=0, param_2=0,description="ride helicopter")]
                a.action_queue = []
                a.option_start_time = global_data.get("time")
                a.last_option_review_time = None
                continue
            print(f"[CAMON] Leader {agent.id}: assigned action to agent {a.id} -> '{ma.group(1)}'")
            opt = translate_action(ma.group(1), a.type, global_data)
            a.options = [opt]
            a.action_queue = []
            a.option_start_time = global_data.get("time")
            a.last_option_review_time = None
            a.add_message(source=f"AGENT_{agent.id}", content=mm.group(1), time=global_data['time'])
            global_data.update({f'AGENT_{a.id}':{'name':f'AGENT_{a.id}','perception':a.last_perception,'position':a.last_position,'current_action':opt,'past_actions':a.past_options}})
    global_data['leader_agent'] = agent

    
def propose_plan(agent: Agent, global_data: dict) -> None:
    print(f"[CAMON] Agent {agent.id}: proposing action to leader {global_data['leader_agent'].id}")

    # Identify type
    if agent.type == 0:
        type_string = 'Firefighter'
    elif agent.type == 1:
        type_string = 'Bulldozer'
    elif agent.type == 2:
        type_string = 'Drone'
    else:
        type_string = 'Helicopter'
    # Team composition
    
    team_comp_string = ''
    if global_data['firefighters']:
        team_comp_string += f"{[f'AGENT_{a.id}' for a in global_data['firefighters']]} {'is' if len(global_data['firefighters'])==1 else 'are'} Firefighter Agents. Firefighter agents are general purpose agents with decent speed and observation capabilities. They can move, cut trees, spray water, and rescue civilians.\n\n"
    if global_data['bulldozers']:
        team_comp_string += f"{[f'AGENT_{a.id}' for a in global_data['bulldozers']]} {'is' if len(global_data['bulldozers'])==1 else 'are'} Bulldozer Agents. Bulldozer agents are specialized agents with exceptional tree-cutting abilities but limited speed.\n\n"
    if global_data['drones']:
        team_comp_string += f"{[f'AGENT_{a.id}' for a in global_data['drones']]} {'is' if len(global_data['drones'])==1 else 'are'} Drone Agents. Drone agents are specialized recon agents with exceptional speed and observations.\n\n"
    if global_data['helicopters']:
        team_comp_string += f"{[f'AGENT_{a.id}' for a in global_data['helicopters']]} {'is' if len(global_data['helicopters'])==1 else 'are'} Helicopter Agents. Helicopter agents are general support agents with exceptional speed and observations. They can move, pick up and drop off Firefighter Agents, and spray water\n\n"
    
    # Past and chat
    past_string = ''.join(str(o.description)+'\n' for o in agent.past_options)
    chat_string = ''.join(f"{time}: \n{msg}\n\n" for time, msg in agent.chat_history.items())

    # Description prompt
    desc_path = f'crew_algorithms/wildfire_alg/algorithms/CAMON/prompts/descriptions/{type_string.lower()}_description.txt'
    with open(desc_path, 'r', encoding='utf-8') as f:
        description_string = f.read()

    # Construct prompt
    proposal_str = f"""
                    You are AGENT_{agent.id}, an embodied {type_string} agent within a {agent.cfg.envs.map_size} by {agent.cfg.envs.map_size} forest grid world and part of a collaboratve team of {len(global_data["agents"])} Agents.

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
    client = OpenAI(api_key=agent.api_key, base_url="https://tritonai-api.ucsd.edu")
    system_msg = f"""
                    You are AGENT_{agent.id}, an embodied {type_string} agent.
                    You propose your next action based on your task, observations, past actions, and chat history.
                    """
    response = client.chat.completions.create(
        model='api-gpt-oss-120b',
        messages=[{'role':'system','content':system_msg},{'role':'user','content':proposal_str}],
        temperature=0.7
    )
    global_data["api_calls"]+=1
    global_data["input_tokens"]+=response.usage.prompt_tokens
    global_data["output_tokens"]+=response.usage.completion_tokens
    result = response.choices[0].message.content

    agent.log_chat('Proposing an Action', [('user', proposal_str), ('assistant', result)])

    m = re.search(r"<action>\s*(.*?)\s*</action>", result, re.DOTALL)

    if not m:
        print("[CAMON] Agent {agent.id}: ERROR no <action> tag found in proposal response")
        return
    proposed_action = m.group(1).strip()


    leader = global_data["leader_agent"]

    team_abilities = ''

    desc_files = ['firefighter', 'bulldozer', 'drone', 'helicopter']

    for kind in desc_files:

        if global_data.get(kind+'s'):

            path = f'crew_algorithms/wildfire_alg/algorithms/CAMON/prompts/descriptions/{kind}_description.txt'
            with open(path, 'r', encoding='utf-8') as f:
                team_abilities += f.read()

    global_str = []
    for data in global_data.items():
        if data[0].__contains__("AGENT"):
            global_str.append(data)

    review_prompt = f"""

            You are AGENT_{leader.id}, currently acting as the leader in a cooperative multi-agent robotic task. 
            This is your team composition, including you:
            {team_comp_string}
            ---

            Your team's current task is:
            {agent.current_task}
            ---
            
            This is your teams'(including you) collective observations, locations, current actions, and past actions of all agents. Only you have all of this data.
            {str(global_str)}
            ---
            
            Your teammate AGENT_{agent.id}, a {type_string} Agent, is proposing a new action for itself:
            {proposed_action}
            ---


            Your job is to review this action and ACCEPT or REJECT it.

            Then provide the next best action for AGENT_{agent.id}, choosing a better one if REJECT or repeating/rewriting the proposed one if ACCEPT.
            Also send a message to AGENT_{agent.id} describing your choice.

            Additionally, you may announce information to other agents in your team with information.
            You may also choose to override actions for other agents as well. You must send a message to that agent if you do so. This interrupts their action, so only do this if you want to change their current action.
            

            These are all the possible actions for each type of agent. This is a comprehensive list, so the action MUST be one of these types. NO other responses are allowed.

            {team_abilities}

            
            Provide your output in the following format:

            <reasoning>(any reasoning or calculations)</reasoning>

            <decision> ACCEPT OR REJECT </decision>
            <action> AGENT_{agent.id}'s next action </action>
            <message> message to AGENT_{agent.id} </message>

            OPTIONAL-for other agents:

            <AGENT_ID-action>(AGENTID'S NEXT ACTION)<AGENT_ID-action>
            <AGENT_ID-message>(message to AGENTID)<AGENT_ID-message>

            For example: <AGENT_A-action>'action'</AGENT_A-action>

            Make sure 'action's are specific and include all information needed to execute, such as coordinates.
            
            YOU MUST HAVE AT LEAST THE <reasoning>, <decision>, <action>, <message> TAGS. SENDING MESSAGES OR PROPOSING ACTIONS TO OTHER AGENTS IS OPTIONAL.

            """
    
    system_msg = f"""
                    You are AGENT_{agent.id}, currently acting as the leader in a cooperative multi-agent robotic task. Your team is in a  {agent.cfg.envs.map_size} by {agent.cfg.envs.map_size} forest grid world that spans x:[0 to {agent.cfg.envs.map_size}] and y:[0 to {agent.cfg.envs.map_size}].
                    You have access to the collective observations and the progress of all agents. Your job is to review the proposed actions of your teammates and assign them actions.
                    """
    rev_resp = OpenAI(api_key=leader.api_key, base_url="https://tritonai-api.ucsd.edu").chat.completions.create(
        model='api-gpt-oss-120b',
        messages=[{'role':'system','content':system_msg},{'role':'user','content':review_prompt}],
        temperature=0.7
    )
    global_data["api_calls"]+=1
    global_data["input_tokens"]+=rev_resp.usage.prompt_tokens
    global_data["output_tokens"]+=rev_resp.usage.completion_tokens
    review = rev_resp.choices[0].message.content
    agent.log_chat("Review Proposal", [('user', review_prompt), ('assistant', review)])

    action_match = re.search(r"<action>\s*(.*?)\s*</action>", result, re.DOTALL)
    message_match = re.search(r"<message>\s*(.*?)\s*</message>", result, re.DOTALL)

    if action_match:
            action_str= action_match.group(1)

            option = translate_action(action_str, type = agent.type, global_data=global_data)
            agent.options = [option]
            agent.action_queue = []
            agent.option_start_time = global_data.get("time")
            agent.last_option_review_time = None
            if message_match:
                message_str= message_match.group(1)
                agent.add_message(source=f"AGENT_{leader.id}", content=message_str, time=global_data["time"])

                print(f"agent {leader.id}: {message_str}")
            print(f"agent {leader.id}: provided action: '{action_str}' to {agent.id}")


            for a in global_data["agents"]:

                agent_action_match = re.search(fr"<AGENT_{a.id}-action>\s*(.*?)\s*</AGENT_{a.id}-action>", result, re.DOTALL)
                agent_message_match = re.search(fr"<AGENT_{a.id}-message>\s*(.*?)\s*</AGENT_{a.id}-message>", result, re.DOTALL)

                
                
                if agent_action_match and agent_message_match:

                    if a.type==0 and a.extra_variables[2]==1:
                        a.options = [Action(type=0, param_1=0, param_2=0,description="ride helicopter")]
                        a.action_queue = []
                        a.option_start_time = global_data.get("time")
                        a.last_option_review_time = None
                        continue

                    agent_action_str= agent_action_match.group(1)
                    agent_message_str= agent_message_match.group(1)
                    print(f"agent {leader.id}: provided action: '{agent_action_str}' to {a.id}")

                    option = translate_action(agent_action_str, type = a.type, global_data=global_data)
                    a.options = [option]
                    a.action_queue = []
                    a.option_start_time = global_data.get("time")
                    a.last_option_review_time = None
                    a.add_message(source = f"AGENT_{leader.id}", content=agent_message_str, time = global_data["time"])

                    agent_data = {'name': f'AGENT_{a.id}', 
                        'perception': a.last_perception, 
                        'position': a.last_position, 
                        'current_action': a.options[0] if len(a.options)>0 else "IDLE", 
                        'past_actions': a.past_options}
                    global_data.update({f'AGENT_{a.id}': agent_data})
                
    else:
        print(f"[CAMON] Leader review for agent {agent.id}: ERROR no <action> tag found in review")

    global_data.update({"leader_agent": agent})
    print(f"agent {agent.id}: is now leader")