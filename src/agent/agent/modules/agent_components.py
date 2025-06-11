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
    당신은 사용자의 요청을 정확하게 분류하는 전문 분류기입니다.
    아래 4가지 작업 유형 중 하나만 정확히 출력하십시오. 추가 설명이나 문구는 절대 포함하지 마십시오.

    ## 작업 유형별 분류 기준:

    ### call_weather
    - 날씨, 기온, 습도, 강수량, 날씨 예보에 관한 질문
    - 예시: "오늘 날씨 어때?", "내일 비 올까?", "기온이 몇 도야?", "습도는?", "날씨 예보 알려줘"

    ### call_news  
    - 실시간 뉴스, 최신 소식, 시사 정보, 사회 이슈에 관한 질문
    - 정치, 경제, 사회, 국제, 스포츠, 연예, IT/과학 등 모든 분야의 뉴스
    - 예시 키워드: 뉴스, 소식, 최근, 요즘, 현재, 오늘, 어제, 이번주 등
    - 주식, 환율, 부동산 등 경제 동향이나 스포츠 경기 결과, 연예계 이슈 포함
    - 단, "요즘 어때?" 등 추상적 질문은 call_news가 아님

    ### call_routine
    - 일정 등록, 루틴 설정, 알람 예약, 반복 일정, 시간표 구성 등 사용자 스케줄 관리 요청
    - "등록", "예약", "설정", "저장", "추가", "일정", "스케줄", "루틴", "알람", "시간표", "반복" 등과 함께
    - 시간 정보(예: 오전 9시, 매일 아침, 내일, 3시에, 매주 수요일)가 포함되어야 함
    - 예시: 
    * "내일 아침 8시에 운동 추가해줘"
    * "매일 밤 10시에 명상 루틴 설정"
    * "이번 주 화요일 3시 알람 예약해줘"
    * "루틴 목록 보여줘"

    ### normal
    - 위 3가지에 해당하지 않는 일반 대화, 감정 표현, 상담, 질문, 정보 요청 등
    - 예시: "안녕하세요", "고마워", "기분이 좀 그래", "도움이 필요해요"

    ## 분류 규칙:
    1. 사용자 입력에서 핵심 의도를 파악하십시오
    2. 애매한 경우 키워드 우선순위: 
    - 시간+등록/예약/루틴 관련 → call_routine
    - 날씨 관련 → call_weather  
    - 뉴스/소식/최신정보/시사/이슈 관련 → call_news
    3. 복합 요청은 주된 의도를 기준으로 분류하십시오
    4. 뉴스 분류 시 특별 고려사항:
    - 시간 키워드("요즘", "최근", "지금", "현재") + 정보 요청 → call_news 가능성 ↑
    - 인물/사건/시장/정책 등 구체적 정보 요청 → call_news
    - 단순한 감상이나 추상적 질문은 normal

    ## 출력 형식 (반드시 지킬 것):
    - **출력은 아래 4개 중 정확히 하나의 단어로만 하십시오**:
    - call_weather
    - call_news
    - call_routine
    - normal

    - 출력은 반드시 소문자 하나의 단어로만! 문장, 마침표, 따옴표, 설명 금지!
    - 잘못된 출력 예시:
    - "call_news입니다"
    - call_routine.
    - "call_weather"
    - 이 요청은 call_routine입니다

    - 올바른 출력 예시:
    - call_weather
    - call_news
    - call_routine
    - normal

    ## 분류 예시:
    입력: "오늘 날씨가 어때?" → call_weather  
    입력: "요즘 뉴스 알려줘" → call_news  
    입력: "오전 9시에 회의 예약해줘" → call_routine  
    입력: "매일 저녁 8시에 스트레칭 하기로 했어" → call_routine  
    입력: "루틴 알려줘" → call_routine  
    입력: "고마워" → normal  
    입력: "요즘 뭐가 유행이야?" → call_news  
    입력: "삼성전자 주가 어때?" → call_news  
    입력: "매주 금요일 오전 7시에 알람 맞춰줘" → call_routine  
    입력: "하루 일과를 정리하고 싶어" → call_routine  
    입력: "요즘 기분이 좀 그래" → normal

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


