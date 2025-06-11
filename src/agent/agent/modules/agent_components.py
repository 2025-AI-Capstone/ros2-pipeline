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
    - 오늘의 뉴스, 헤드라인, 속보, 사건사고, 이슈, 동향, 현황 등
    - 특정 인물, 기업, 정책, 사건에 대한 최신 정보 요청
    - 주식, 환율, 부동산 등 경제 동향 및 시장 정보
    - 스포츠 경기 결과, 연예계 소식, 화제의 인물 관련 질문
    - "뉴스", "소식", "최근", "요즘", "현재", "지금", "오늘", "어제", "이번주" 등의 시간 키워드와 함께 나오는 정보 요청
    - 예시: 
    * 일반 뉴스: "오늘 뉴스 알려줘", "최근 소식은?", "헤드라인 뉴스", "무슨 일이 일어났어?"
    * 정치: "대통령이 뭐라고 했어?", "정부 정책 어떻게 바뀌었어?", "선거 결과는?"
    * 경제: "주식 어떻게 되고 있어?", "환율 요즘 어때?", "부동산 시장 동향은?"
    * 사회: "코로나 상황은?", "교통사고 소식 있어?", "사건사고 뉴스"
    * 국제: "미국에서 무슨 일이?", "일본 지진 소식", "우크라이나 전쟁 현황"
    * 스포츠: "축구 경기 결과", "올림픽 소식", "야구 순위는?"
    * 연예: "○○○ 최근 소식", "드라마 시청률", "가수 컴백 소식"
    * IT/과학: "애플 신제품 소식", "ChatGPT 업데이트", "우주 탐사 뉴스"

    ### call_routine
    - 일정 등록, 루틴 설정, 알람 예약, 스케줄 관리 요청
    - "등록", "예약", "설정", "저장", "추가", "스케줄" 등의 키워드와 시간 정보 포함
    - 예시: "오늘 9시에 운동 일정 등록해줘", "내일 약속 저장해줘", "매일 아침 7시 알람 설정"

    ### normal
    - 위 3가지에 해당하지 않는 모든 요청
    - 일반 대화, 질문, 상담, 정보 요청 등
    - 예시: "안녕하세요", "도움이 필요해요", "어떻게 지내세요?", "감사합니다"

    ## 분류 규칙:
    1. 사용자 입력에서 핵심 의도를 파악하십시오
    2. 애매한 경우 키워드 우선순위: 
    - 시간+등록/예약 관련 → call_routine
    - 날씨 관련 → call_weather  
    - 뉴스/소식/최신정보/시사/이슈 관련 → call_news
    3. 복합 요청의 경우 주된 의도를 기준으로 분류하십시오
    4. 뉴스 분류 시 특별 고려사항:
    - "요즘", "최근", "지금", "현재" 등 시간 키워드 + 정보 요청 = call_news 가능성 높음
    - 특정 인물/기업/사건명 + 정보 요청 = call_news 가능성 높음  
    - 시장/경제 동향, 사회 이슈, 정치 상황 관련 = call_news
    5. 정확히 4가지 중 하나만 출력하십시오: call_weather, call_news, call_routine, normal

    ## 출력 형식:
    - 올바른 출력: call_weather
    - 잘못된 출력: "call_weather입니다", "call_weather.", "이 요청은 call_weather 유형입니다"

    ## 분류 예시:
    입력: "오늘 날씨가 어때?" → call_weather
    입력: "내일 뉴스 알려줘" → call_news  
    입력: "오후 3시에 회의 일정 등록해줘" → call_routine
    입력: "고마워" → normal
    입력: "날씨 좋으니까 산책하러 가자" → call_weather (날씨가 주요 키워드)
    입력: "뉴스에서 본 그 사건 어떻게 생각해?" → call_news (뉴스가 주요 키워드)
    입력: "요즘 주식 어떻게 되고 있어?" → call_news (시장 동향 정보)
    입력: "대통령이 뭐라고 했어?" → call_news (정치 이슈)
    입력: "코로나 상황 어때?" → call_news (사회 이슈)
    입력: "삼성전자 주가 어때?" → call_news (경제 정보)
    입력: "월드컵 결과 알려줘" → call_news (스포츠 뉴스)
    입력: "BTS 최근 소식 있어?" → call_news (연예 뉴스)
    입력: "애플 신제품 언제 나와?" → call_news (IT 뉴스)
    입력: "부동산 시장 어떻게 되고 있어?" → call_news (경제 동향)
    입력: "교통사고 소식 들었어?" → call_news (사회 이슈)

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


