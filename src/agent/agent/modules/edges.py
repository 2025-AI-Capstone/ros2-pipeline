from agent.modules.AgentState import AgentState
from typing import Dict
import re

def task_selector(state: AgentState) -> Dict:
    if state.get("fall_alert"):
        state["task_type"] = "emergency_voice_check"
        return state

    chain = state["agent_components"]["task_selector_chain"]
    response = chain.invoke({"user_input": state["input"]})
    print("[Raw LLM output]:", response.content)

    raw_output = response.content.strip().lower()
    allowed = ["call_weather", "call_news", "call_routine", "normal"]
    found = [task for task in allowed if task in raw_output]

    task_type = found[0] if found else "normal"
    state["task_type"] = task_type
    return state

def check_routine_edge(state: AgentState) -> Dict:
    check_routine_chain = state["agent_components"].get("check_routine_chain")
    if not check_routine_chain:
        raise ValueError("check_routine_chain not found in agent_components")

    response = check_routine_chain.invoke({"user_input": state["input"]})
    state["check_routine"] = response.content.strip()
    state["is_routine_flow"] = True
    print(response.content.strip())
    return state

def await_voice_response(state: AgentState) -> Dict:
    fall_response = state["input"]
    check_chain = state["agent_components"].get("check_emergency_chain")

    if not fall_response:
        state["voice_response"] = "no_response"
        return state

    response = check_chain.invoke({"fall_response": fall_response}).content.strip().lower()

    if response in ["report", "ok", "no_response"]:
        state["voice_response"] = response
    else:
        state["voice_response"] = "no_response"
    if response == "ok":
        state['final_answer'] = "알겠습니다. 도움이 필요하면 말씀해주세요"
    return state
