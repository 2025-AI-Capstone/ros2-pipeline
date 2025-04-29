from agent.modules.AgentState import AgentState
from typing import Dict

def task_selector(state: AgentState) -> Dict:
    user_input = state["input"].strip().lower()

    if user_input in ["weather", "날씨", "기상", "온도", "기온", "예보"]:
        state["task_type"] = "call_weather"
    elif user_input in ["news", "뉴스", "기사", "소식", "정보"]:
        state["task_type"] = "call_news"
    elif user_input in ['저장', "기억해", "일정 추가", "알람 설정", "알람", "일정", "기억"]:
        state["task_type"] = "call_db"
    else:
        state["task_type"] = "normal"

    return state

def check_routine_edge(state: AgentState) -> Dict:
    check_routine_chain = state["agent_components"].get("check_routine_chain")
    if not check_routine_chain:
        raise ValueError("check_routine_chain not found in agent_components")

    response = check_routine_chain.invoke({"user_input": state["input"]})
    state["check_routine"] = response.content
    return state

def await_voice_response(state: AgentState) -> Dict:
    fall_response = state.get("fall_response", "").strip()
    check_chain = state["agent_components"].get("check_emergency_chain")

    if not fall_response:
        state["voice_response"] = "no_response"
        return state

    response = check_chain.invoke({"fall_response": fall_response}).content.strip().lower()

    if response in ["report", "ok", "no_response"]:
        state["voice_response"] = response
    else:
        state["voice_response"] = "no_response"

    return state
