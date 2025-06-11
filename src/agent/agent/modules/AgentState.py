from typing import TypedDict, Any, Dict, Optional

class AgentState(TypedDict):
    input: str
    llm: Any
    weather_info: str
    news_info: str
    db_info: bool
    fall_alert: bool
    check_routine: str
    routine_data: str
    final_answer: str
    routine_alarm: Dict[str, Any]
    agent_components: Dict[str, Any]
    voice_response: Optional[str]
    is_routine_flow: Optional[bool]
    routine_result_message: Optional[str]
