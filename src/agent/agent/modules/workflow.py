from typing import Dict, Any
from langgraph.graph import StateGraph, END
from agent.modules.AgentState import AgentState
from agent.modules.nodes import (
    generator, get_weather, get_news, post_routine,
     send_emergency_report
)
from dotenv import load_dotenv
from agent.modules.edges import await_voice_response,task_selector,check_routine_edge

def run_workflow(input: str, llm: Any, fall_alert: bool = False, agent_components: Dict[str, Any] = None) -> str:
    workflow = StateGraph(AgentState)

    # 노드 정의
    workflow.add_node("task_selector", task_selector)
    workflow.add_node("get_weather", get_weather)
    workflow.add_node("get_news", get_news)
    workflow.add_node("post_routine", post_routine)
    workflow.add_node("check_routine_edge", check_routine_edge)
    workflow.add_node("generator", generator)
    workflow.add_node("await_voice_response", await_voice_response)
    workflow.add_node("send_emergency_report", send_emergency_report)

    # 시작 지점
    workflow.set_entry_point("task_selector")

    # task_selector 결과 기반 분기 처리
    workflow.add_conditional_edges(
        "task_selector",
        lambda state: state["task_type"],
        {
            "call_weather": "get_weather",
            "call_news": "get_news",
            "call_routine": "check_routine_edge",
            "normal": "generator",
            "emergency_voice_check": "await_voice_response"  # ✅ fall_alert 처리 분기
        }
    )

    # 일반 작업 처리 후 generator로 이동
    workflow.add_edge("get_weather", "generator")
    workflow.add_edge("get_news", "generator")
    workflow.add_edge("post_routine", "generator")

    # 루틴 등록 여부 판단 분기
    workflow.add_conditional_edges(
        "check_routine_edge",
        lambda state: "reject" if state["check_routine"] == "reject" else "call_db",
        {
            "call_db": "post_routine",
            "reject": "generator"
        }
    )

    workflow.add_edge("generator", END)

    # 낙상 후 음성 응답 결과 처리
    workflow.add_conditional_edges(
        "await_voice_response",
        lambda state: state["voice_response"],
        {
            "ok": END,
            "report": "send_emergency_report",
            "no_response": "send_emergency_report"
        }
    )

    workflow.add_edge("send_emergency_report", END)

    # 실행
    app = workflow.compile()

    initial_state = {
        "input": input,
        "llm": llm,
        "fall_alert": fall_alert,
        "agent_components": agent_components or {},
        "weather_info": "",
        "news_info": "",
        "db_info": False,
        "check_routine": "",
        "routine_data": "",
        "final_answer": "",
        "routine_alarm": {},
        "voice_input": "",
        "voice_response": ""
    }

    result = app.invoke(initial_state)

    return result["final_answer"]
