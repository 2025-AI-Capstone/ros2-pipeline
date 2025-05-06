from AgentState import AgentState
from typing import Dict

def task_selector(state: AgentState) -> Dict:
    chain = state["agent_components"]["task_selector_chain"]
    response = chain.invoke({"user_input": state["input"]})
    print(response.content)
    task_type = response.content.strip()  
    if task_type not in ["call_weather", "call_news", "call_db", "normal"]:
        task_type = "normal"             

    print("task_type:", task_type)
    state["task_type"] = task_type
    return state


def check_routine_edge(state: AgentState) -> Dict:
    check_routine_chain = state["agent_components"].get("check_routine_chain")
    if not check_routine_chain:
        raise ValueError("check_routine_chain not found in agent_components")

    response = check_routine_chain.invoke({"user_input": state["input"]})
    state["check_routine"] = response
    print(response)
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
