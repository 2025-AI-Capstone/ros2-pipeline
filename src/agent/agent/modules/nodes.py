from typing import Dict, Any
from agent.modules.AgentState import AgentState
from dotenv import load_dotenv
import requests
import json
import os
import urllib

def get_weather(state: AgentState) -> Dict[str, Any]:
    api_key = os.getenv("WEATHER_API_KEY")
    if not api_key:
        state["weather_info"] = ""
        return state

    url = (
        "https://api.openweathermap.org/data/2.5/weather"
        "?q=Seoul,KR"
        f"&appid={api_key}"
        "&units=metric"
    )
    try:
        response = requests.get(url, timeout=5)
        response.raise_for_status()
        data = response.json()

        city = data.get("name", "Unknown")
        main_data = data.get("main", {})
        weather_arr = data.get("weather", [])
        weather_main = weather_arr[0].get("main", "") if weather_arr else ""
        temp = main_data.get("temp", "N/A")

        weather_info = {
            "city": city,
            "temp": temp,
            "weather": weather_main
        }
        state["weather_info"] = str(weather_info)
        return state

    except requests.RequestException as e:
        state["weather_info"] = f"날씨 API 오류: {e}"
        return state


def get_news(state: AgentState) -> Dict[str, Any]:
    load_dotenv()
    client_id = os.getenv("CLIENT_ID")
    client_secret = os.getenv("CLIENT_SECRET")

    agent_response = state.get("agent_response", "")
    encText = urllib.parse.quote(agent_response)

    url = 'https://openapi.naver.com/v1/search/news?query=' + encText
    request = urllib.request.Request(url)
    request.add_header("X-Naver-Client-id", client_id)
    request.add_header("X-Naver-Client-Secret", client_secret)
    response = urllib.request.urlopen(request)
    rescode = response.getcode()

    if rescode == 200:
        response_body = response.read()
        search_response = response_body.decode('utf-8')
        state["news_info"] = search_response
    else:
        state["news_info"] = ""
    return state


def get_db(state: AgentState) -> Dict[str, Any]:
    backend_url = "http://localhost:8080/routines"
    routine_payload = state.get("routine_data")

    if not routine_payload:
        state["db_info"] = False
        return state

    try:
        response = requests.post(backend_url, json=routine_payload, timeout=3)
        response.raise_for_status()
        state["db_info"] = True
    except requests.exceptions.RequestException as e:
        state["db_info"] = False
    return state


def generator(state: AgentState) -> Dict[str, Any]:
    if state.get("fall_alert"):
        state["final_answer"] = "낙상이 감지되었습니다. 즉시 확인이 필요합니다. 괜찮으신가요?"
        return state

    generator_chain = state["agent_components"]["generator_chain"]
    response = generator_chain.invoke({
        "user_input": state.get("input", ""),
        "weather_info": state.get("weather_info", ""),
        "news_info": state.get("news_info", ""),
        "check_routine": str(state.get("check_routine", "")),
        "db_info": str(state.get("db_info", False)),
        "fall_alert": str(state.get("fall_alert", False)),
    })
    state["final_answer"] = response
    return state


def send_emergency_report(state: AgentState) -> Dict[str, Any]:
    report_data = {
        "user_id": 1,
        "event": "fall_detected",
        "status": state.get("voice_response", "unknown"),
        "timestamp": "to-be-filled",
        "details": "응답 없음 또는 신고 요청 감지로 인한 자동 신고"
    }
    backend_url = state["agent_components"].get("backend_url", "http://localhost:8080")

    try:
        requests.post(f"{backend_url}/emergency/report", json=report_data, timeout=3)
        state["final_answer"] = "응급 신고가 전송되었습니다."
    except Exception as e:
        state["final_answer"] = f"신고 요청 중 오류 발생: {e}"

    return state
