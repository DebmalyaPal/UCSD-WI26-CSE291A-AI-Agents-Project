from attrs import define


@define(auto_attribs=True)
class LLMConfig:
    actor_model: str = "gpt-4o-mini"
    critic_model: str = "gpt-4o-mini"
    planner_model: str = "gpt-4o"
    translator_model: str = "gpt-4o"

    large_model: str = "gpt-4o"
    small_model: str = "gpt-4o-mini"
    reasoning_model: str = "o1-mini"


    

    verbose: bool = False
    memory_buffer_size: int = 10
    use_adaptive_options: bool = False
    observations_summary_size: int = 200
    plan_summary_size: int = 200



