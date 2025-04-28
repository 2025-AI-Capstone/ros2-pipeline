from langchain.prompts import PromptTemplate, ChatPromptTemplate
import datetime
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from langchain_community.llms import HuggingFacePipeline
def initialize_agent_components(llm):

    # 루틴 등록 여부 확인
    check_routine_prompt = PromptTemplate.from_template("""
    입력이 루틴 등록 요청이면 아래 형식의 JSON으로 추출하고,
    그 외에는 "reject"만 출력하세요.

    예시 형식:
    {{
    "title": "약먹을시간",
    "alarm_time": "09:00:00",
    "repeat_type": "daily",
    "user_id": 1
    }}

    입력: {user_input}
    """)

    # 홈 어시스턴트 응답 생성
    generator_prompt = ChatPromptTemplate.from_messages([
        ("system", "너는 한국어 스마트 홈 어시스턴트야."),
        ("human", """
    다음 정보를 참고해서 사용자 질문에 간단히 대답해줘.

    - 날씨: {weather_info}
    - 뉴스: {news_info}
    - 루틴 등록됨?: {check_routine}
    - DB 정보 있음?: {db_info}
    - 낙상 감지됨?: {fall_alert}
    - 사용자 질문: {user_input}

    주의: 낙상 감지가 True면 제일 먼저 "⚠️ 경고!" 문구를 포함해.
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

    check_routine_chain = check_routine_prompt | llm.bind(temperature=0.4)
    generator_chain = generator_prompt | llm.bind(temperature=0.3)
    check_emergency_chain = check_emergency_prompt | llm.bind(temperature=0.2)
    
    return {
        "check_routine_chain":check_routine_chain,
        "generator_chain":generator_chain,
        "check_emergency_chain":check_emergency_chain
    }



def load_llm(model_id):

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id)

    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=60,
        do_sample=True,
        top_k=50,
        top_p=0.95,
        temperature=0.7
    )

    llm = HuggingFacePipeline(pipeline=pipe)
    return llm

