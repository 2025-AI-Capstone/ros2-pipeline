from agent.modules.AgentState import AgentState
from typing import Dict
import re

def task_selector(state: AgentState) -> Dict:
    chain = state["agent_components"]["task_selector_chain"]
    response = chain.invoke({"user_input": state["input"]})
    print("[Raw LLM output]:", response.content)
    
    # LLM 응답 후처리
    raw_output = response.content.strip().lower()
    task_type = re.sub(r'[^a-z_]', '', raw_output)  # 마침표, 따옴표, 종결어미 제거 등
    
    # 안정성 검증 및 fallback 처리
    if task_type not in ["call_weather", "call_news", "call_routine", "normal"]:
        task_type = "normal"
    
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
