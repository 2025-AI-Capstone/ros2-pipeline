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
    try:
        load_dotenv()
        client_id = os.getenv("CLIENT_ID")
        client_secret = os.getenv("CLIENT_SECRET")

        if not client_id or not client_secret:
            state["news_info"] = "Error: Missing API credentials"
            return state

        user_input = state.get("input", "")
        if not user_input:
            state["news_info"] = "Error: Missing search query"
            return state

        encText = urllib.parse.quote(user_input)
        url = f"https://openapi.naver.com/v1/search/news?query={encText}"

        request = urllib.request.Request(url)
        request.add_header("X-Naver-Client-Id", client_id)
        request.add_header("X-Naver-Client-Secret", client_secret)
        request.add_header("User-Agent", "Mozilla/5.0")

        response = urllib.request.urlopen(request)
        rescode = response.getcode()

        if rescode == 200:
            import re, json
            from html import unescape

            def clean_html(raw):
                return unescape(re.sub('<.*?>', '', raw))

            response_body = response.read().decode('utf-8')
            data = json.loads(response_body)
            titles = [clean_html(item["title"]) for item in data.get("items", [])[:3]]
            summary = "\n".join(titles)
            state["news_info"] = summary
        else:
            state["news_info"] = f"Error: API returned code {rescode}"
    except Exception as e:
        state["news_info"] = f"Error: {str(e)}"

    return state


def get_db(state: AgentState) -> Dict[str, Any]:
    login_url = "http://localhost:8000/login"
    routine_url = "http://localhost:8000/routines"

    routine_payload = state.get("routine_data")
    credentials = {"name": "홍길동", "password": "1234"}

    print(f"get_db Routine payload received: {routine_payload}")

    if not routine_payload or "name" not in credentials or "password" not in credentials:
        state["db_info"] = False
        state["routine_result_message"] = "루틴 등록에 필요한 정보가 부족합니다."
        print("get_db Missing routine payload or credentials.")
        return state

    try:
        # 1. 로그인 요청
        print("get_db Attempting login...")
        login_res = requests.post(login_url, json=credentials, timeout=5)
        login_res.raise_for_status()
        session_id = login_res.cookies.get("session_id")

        if not session_id:
            state["db_info"] = False
            state["routine_result_message"] = "로그인에 실패했습니다: 세션 ID 없음."
            print("get_db Login failed: No session ID.")
            return state
        print("get_db Login successful.")

        # 2. 루틴 등록 요청
        headers = {
            "Cookie": f"session_id={session_id}"
        }
        print("get_db Attempting routine registration...")
        routine_res = requests.post(routine_url, json=routine_payload, headers=headers, timeout=5)
        routine_res.raise_for_status()

        print(f"get_db Routine registration response status: {routine_res.status_code}")
        print(f"get_db Routine registration response body: {routine_res.text}")

        state["db_info"] = True
        state["routine_result_message"] = "루틴이 성공적으로 등록되었습니다."
        print("get_db Routine registration successful.")
    except requests.exceptions.RequestException as e:
        state["db_info"] = False
        state["routine_result_message"] = f"루틴 등록 중 오류 발생: {e}"
        print(f"get_db Routine registration failed due to request error: {e}")
    except Exception as e: # Catch any other unexpected errors
        state["db_info"] = False
        state["routine_result_message"] = f"알 수 없는 오류로 루틴 등록에 실패했습니다: {e}"
        print(f"get_db Routine registration failed due to unknown error: {e}")
    return state



def generator(state: AgentState) -> Dict[str, Any]:
    # Check if this is a routine flow and a message is already set
    if state.get("is_routine_flow") and state.get("routine_result_message"):
        state["final_answer"] = state["routine_result_message"]
        print(f"generator Returning routine result message: {state["final_answer"]}")
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
    state["final_answer"] = response.content
    return state


def send_emergency_report(state: AgentState) -> Dict[str, Any]:
    report_data = {
        "message":"FOCUS:낙상이 감지되어 자동으로 신고되었습니다. 안전을 위해 빠른 확인을 부탁드립니다."
    }
    backend_url = state["agent_components"].get("backend_url", "http://localhost:8080")

    try:
        requests.post(f"{backend_url}/emergency/send-alert", json=report_data, timeout=3)
        state["final_answer"] = "응급 신고가 전송되었습니다."
    except Exception as e:
        state["final_answer"] = f"신고 요청 중 오류 발생: {e}"

    return state
