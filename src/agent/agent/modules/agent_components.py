from langchain.prompts import PromptTemplate, ChatPromptTemplate
import datetime
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from langchain_community.llms import HuggingFacePipeline

def initialize_agent_components(llm):

    # 루틴 등록 여부 확인
    check_routine_prompt = PromptTemplate.from_template("""
    시스템: 당신은 사용자의 루틴 등록 요청을 감지하는 분석기입니다.
    만약 사용자의 입력이 루틴 등록 요청이면 아래의 JSON 형식으로 정확히 변환하세요.
    그 외의 경우에는 단순히 "reject"만 출력하세요.
    아래는 반환 예시입니다:

    {{
    "user_id": 1,
    "title": "복약 알림",
    "description": "매일 아침 9시에 약 복용",
    "alarm_time": "09:00:00",
    "repeat_type": "daily"
    }}

    입력: {user_input}
    """)

    task_selector_prompt = PromptTemplate.from_template("""
    당신은 사용자의 요청을 다음 4가지 중 하나로 정확히 분류해야 합니다. 반드시 아래 단어 중 하나만 출력하세요:

    - call_weather
    - call_news
    - call_routine
    - normal

    절대 다른 문장, 마침표, 설명 없이 위 단어 중 하나만 출력하세요.

    ## 각 유형 설명:

    - call_weather: 날씨, 기온, 강수량, 예보 관련 질문
    예: "오늘 날씨 어때?", "비 와?", "기온 알려줘"

    - call_news: 최근 뉴스, 사회/경제/정치/연예/스포츠 등 정보 요청
    키워드: 뉴스, 소식, 최근, 요즘, 현재, 주가, 사건, 연예인, 경기 결과 등
    예: "오늘 뉴스 뭐 있어?", "삼성 주가 어때?", "코로나 소식 알려줘"

    - call_routine: 일정, 루틴, 알람 설정 등 시간+행동 요청
    키워드: 등록, 예약, 설정, 추가, 알람, 루틴, 스케줄, 시간표 + 오전/오후/내일/매일 등
    예: "내일 오전 9시에 알람 맞춰줘", "매일 운동 루틴 등록해줘"

    - normal: 일반 인사, 감정 표현, 대화 등 위 3가지에 해당하지 않는 경우
    예: "안녕", "고마워", "기분이 안 좋아"

    사용자 입력: {user_input}
    """)

    # 홈 어시스턴트 응답 생성
    generator_prompt = ChatPromptTemplate.from_messages([
    ("system", """
    당신은 한국어 스마트 홈 어시스턴트입니다. 다음 규칙을 엄격히 따르세요:
    1. 제공된 정보 중 값이 있는 항목만 언급하세요
    2. 값이 비어있거나 없는 항목은 절대 언급하지 마세요
    3. 응답은 정보 전달에만 집중하고 불필요한 설명이나 추가 문구를 포함하지 마세요
    4. 정보를 사실적으로만 전달하고 추가적인 제안이나 질문을 하지 마세요
    5. 뉴스 정보의 경우, 기사 제목의 내용를 취합해 구어체로 정리하세요
    """),
    ("human", "{user_input}"),
    ("system", """
    현재 정보:
    날씨: {weather_info}
    뉴스: {news_info}
    낙상알림: {fall_alert}
    
    위 정보 중 값이 있는 항목만 사용해 응답하세요. 
    정보를 있는 그대로만 전달하고, 추가 질문이나 제안, 인사말, 마무리 문구를 붙이지 마세요.
    응답은 필요한 정보만 포함하고 다른 내용은 제외하세요.
    """)
])

    # 낙상 후 음성 응답 평가
    check_emergency_prompt = PromptTemplate.from_template("""
    다음 음성 내용을 보고, 신고 여부를 판단하세요. 
    사용자가 괜찮다고 하거나 신고가 필요하다고 하지 않는 경우에는 ok를, 신고를 요청하거나 응답이 이상한 경우에는 report를
    입력: "{fall_response}"

    다음 중 하나만 출력:
    - "report"
    - "ok"
    - "no response"
    """)

    check_routine_chain = check_routine_prompt | llm.bind(temperature=0.1)
    generator_chain = generator_prompt | llm.bind(temperature=0.3)
    check_emergency_chain = check_emergency_prompt | llm.bind(temperature=0.2)
    task_selector_chain = task_selector_prompt | llm.bind(temperature=0.2)

    
    return {
        "check_routine_chain":check_routine_chain,
        "generator_chain":generator_chain,
        "check_emergency_chain":check_emergency_chain,
        "task_selector_chain":task_selector_chain
    }


