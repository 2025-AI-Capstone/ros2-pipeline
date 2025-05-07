from langchain.prompts import PromptTemplate, ChatPromptTemplate
import datetime
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from langchain_community.llms import HuggingFacePipeline

def initialize_agent_components(llm):

    # 루틴 등록 여부 확인
    check_routine_prompt = PromptTemplate.from_template("""
시스템: 당신은 루틴 등록 요청을 감지하는 분석기입니다. 사용자가 일정이나 루틴을 등록하기 원하면 아래와 같은 정확한 JSON 형식으로 변환하고, 아니면 "reject"만 출력하세요. 변환한 json 데이터만 반환하세요.

{{
  "title": "약먹기",
  "alarm_time": "09:00:00",
  "repeat_type": "daily",
  "user_id": 1
}}

입력: {user_input}
""")

    task_selector_prompt = PromptTemplate.from_template("""
    당신은 사용자의 요청을 분류하는 역할입니다. 요청을 읽고 다음 중 하나의 작업 유형만 정확하게 출력하십시오. 
    다른 설명, 문장, 마침표 등은 포함하지 마십시오.
    call_weather: 사용자가 날씨 관련 정보를 물어볼 때
    call_news: 사용자가 실시간 정보 또는 뉴스에 관련한 질문을 물어볼 때
    call_routine: 사용자가 일정 또는 루틴을 저장하거나 조회하길 원할 때
    normal:그외
    작업 유형:
    - call_weather
    - call_news
    - call_routine
    - normal

    출력 예시:
    call_weather
    오답: call_weather입니다 / call_weather. / 이 요청은 call_weather입니다

    입력: {user_input}
    """)

    # 홈 어시스턴트 응답 생성
    generator_prompt = ChatPromptTemplate.from_messages([
    ("system", """
    당신은 한국어 스마트 홈 어시스턴트입니다. 다음 규칙을 엄격히 따르세요:
    1. 제공된 정보 중 값이 있는 항목만 언급하세요
    2. 값이 비어있거나 없는 항목은 절대 언급하지 마세요
    3. 응답은 정보 전달에만 집중하고 불필요한 설명이나 추가 문구를 포함하지 마세요
    4. 정보를 사실적으로만 전달하고 추가적인 제안이나 질문을 하지 마세요
    5. 뉴스 정보의 경우, 기사 제목의 내용만 간단히 취합해 정리하세요
    """),
    ("human", "{user_input}"),
    ("system", """
    현재 정보:
    날씨: {weather_info}
    뉴스: {news_info}
    루틴: {check_routine}
    DB: {db_info}
    낙상알림: {fall_alert}
    
    위 정보 중 값이 있는 항목만 사용해 응답하세요. 
    정보를 있는 그대로만 전달하고, 추가 질문이나 제안, 인사말, 마무리 문구를 붙이지 마세요.
    응답은 필요한 정보만 포함하고 다른 내용은 제외하세요.
    """)
])

    # 낙상 후 음성 응답 평가
    check_emergency_prompt = PromptTemplate.from_template("""
    다음 음성 내용을 보고, 신고 여부를 판단하세요.

    입력: "{fall_response}"

    다음 중 하나만 출력:
    - "report"
    - "ok"
    - "no response"
    """)

    check_routine_chain = check_routine_prompt | llm.bind(temperature=0.0)
    generator_chain = generator_prompt | llm.bind(temperature=0.3)
    check_emergency_chain = check_emergency_prompt | llm.bind(temperature=0.2)
    task_selector_chain = task_selector_prompt | llm.bind(temperature=0.0)

    
    return {
        "check_routine_chain":check_routine_chain,
        "generator_chain":generator_chain,
        "check_emergency_chain":check_emergency_chain,
        "task_selector_chain":task_selector_chain
    }


