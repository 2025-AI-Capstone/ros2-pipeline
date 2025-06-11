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

    ---

    ## 유형별 기준 (의도 + 주요 키워드 중심):

    ### call_weather
    - 날씨, 기온, 온도, 습도, 강수량, 비, 눈, 흐림, 맑음, 예보 등 **날씨 현상/예측**을 묻는 경우
    - 주요 키워드: 날씨, 기온, 온도, 습도, 비, 눈, 바람, 예보, 미세먼지, 황사
    - 예: "오늘 날씨 어때?", "내일 비 와?", "지금 기온이 몇 도야?", "미세먼지 많아?"

    ### call_news
    - **최근 정보**, 시사 뉴스, 이슈, 사건, 사회/경제/정치/연예/스포츠/과학 등 **실시간 정보 요청**
    - 주요 키워드: 뉴스, 소식, 최근, 요즘, 지금, 오늘, 어제, 이번 주, 사건, 사고, 이슈, 헤드라인  
    + 대통령, 국회, 정책, 전쟁, 경제, 환율, 주식, 부동산, 연예인, 아이돌, 월드컵, 경기 결과, 지진
    - 예: "오늘 뉴스 뭐 있어?", "요즘 사건 많아?", "삼성 주가 알려줘", "우크라이나 전쟁 상황은?"

    ### call_routine
    - **일정/알람/루틴/시간표 설정** 등 시간+행동 요청
    - 주요 키워드: 등록, 예약, 설정, 추가, 저장, 알림, 알람, 루틴, 스케줄, 시간표, 일과, 반복  
    + 오전, 오후, 몇 시, 내일, 오늘, 모레, 매일, 매주, 시간 표현 포함
    - 예: "내일 오전 8시에 약속 추가해줘", "매일 밤 10시에 스트레칭 알람 설정", "루틴 목록 보여줘"

    ### normal
    - 인사말, 감정 표현, 일반 질문, 일상 대화 등 **정보나 요청 없이 감정/소통 중심**인 경우
    - 주요 키워드: 안녕, 반가워, 고마워, 기분, 생각, 감정, 힘들어, 도와줘, 심심해
    - 예: "안녕하세요", "요즘 기분이 좀 그래", "고마워", "심심해", "도움이 필요해"

    ---

    ## 출력 예시:

    입력: "내일 비 올까?" → call_weather  
    입력: "요즘 뉴스 뭐 있어?" → call_news  
    입력: "오전 9시에 회의 예약해줘" → call_routine  
    입력: "루틴 등록해줘" → call_routine  
    입력: "기분이 안 좋아" → normal  
    입력: "삼성전자 주가 어때?" → call_news  
    입력: "이번 주 일정 알려줘" → call_routine  
    입력: "미세먼지 많아?" → call_weather  
    입력: "오늘 뉴스 알려줘" → call_news  
    입력: "도와줘" → normal

    ---

    사용자 입력: {user_input}
    """)


    # 홈 어시스턴트 응답 생성
    generator_prompt = ChatPromptTemplate.from_messages([
        ("system", """
    당신은 한국어 스마트 홈 어시스턴트입니다. 다음 규칙을 반드시 따르세요:

    1. 제공된 정보 중 실제로 값이 존재하는 항목만 말하세요.
    2. 값이 없는 항목(빈 문자열, 없음 등)은 절대 언급하지 마세요.
    3. 응답은 간결하게, 정보 전달에만 집중하세요. 인삿말, 마무리 문구, 제안은 금지입니다.
    4. 뉴스는 기사 제목들을 구어체로 요약해서 전달하세요. 기사 제목을 그대로 나열하지 마세요.
    """),
        ("human", "{user_input}"),
        ("system", """
    현재 정보:
    - 날씨: {weather_info}
    - 뉴스: {news_info}
    - 낙상 알림: {fall_alert}

    위 항목 중 **실제로 값이 존재하는 정보만** 사용해 응답하세요.
    정보를 **있는 그대로만** 전달하고, 그 외 다른 문장(예: 질문, 제안, 인사말, 마무리 등)은 절대 포함하지 마세요.
    """)
    ])

    # 낙상 후 음성 응답 평가
    check_emergency_prompt = PromptTemplate.from_template("""
    다음은 낙상 후 사용자의 음성 응답입니다. 이 응답을 바탕으로 **신고가 필요한지 여부**를 판단하세요.

    판단 기준:
    - 사용자가 "신고", "도와줘", "응급" 등 **도움 요청**을 명확히 표현하면: `"report"`
    - 사용자가 "괜찮아요", "문제없어요" 등 **괜찮다고 말하면**: `"ok"`
    - 아무 말도 없거나, 의미를 이해할 수 없으면: `"no response"`

    입력:
    "{fall_response}"

    다음 중 정확히 하나만 출력하세요 (따옴표 포함):
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


