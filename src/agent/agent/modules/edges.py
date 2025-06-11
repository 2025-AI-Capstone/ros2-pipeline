from agent.modules.AgentState import AgentState
from typing import Dict
import re

def task_selector(state: AgentState) -> Dict:
    chain = state["agent_components"]["task_selector_chain"]
    response = chain.invoke({"user_input": state["input"]})
    print("[Raw LLM output]:", response.content)
    
    raw_output = response.content.strip().lower()

    # 허용된 작업 유형
    allowed = ["call_weather", "call_news", "call_routine", "normal"]

    # 정확한 단어만 출력한 경우
    if raw_output in allowed:
        task_type = raw_output
    else:
        # 출력 중에 허용된 단어가 포함되어 있으면 추출
        found = [task for task in allowed if task in raw_output]
        task_type = found[0] if found else "normal"

    print("task_type:", task_type)
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
    fall_response = state.get("user_input", "").strip()
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
