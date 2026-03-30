from ..core.schemas import AgentState
from ..core.config import config
from loguru import logger
import json
from langchain_ollama import ChatOllama
from langchain_core.messages import SystemMessage, HumanMessage

def multimodal_intent_router(state: AgentState) -> dict:
    """
    (Step 1) 사용자의 입력이 이미지인지 텍스트인지 판별하여 분기합니다. (Ollama 사용)
    """
    input_type = state.get("input_type", "text")
    logger.info(f"Router received input type: {input_type}")
    
    if input_type == "image":
        return {"intent": "image_processing"}
    else:
        # SpeakNode 방식의 Ollama JSON 포맷 라우팅 적용
        llm = ChatOllama(
            model=config.OLLAMA_MODEL,
            base_url=config.OLLAMA_BASE_URL,
            temperature=0.0,
            format="json"
        )
        
        prompt = state.get("user_prompt", "")
        sys_prompt = '''You are a router. Analyze the user request about a shoe.
Return exactly this JSON format: {"intent": "<intent>"}
If asking to modify an existing design -> "text_modification"
Otherwise -> "text_generation"'''
        
        try:
            response = llm.invoke([
                SystemMessage(content=sys_prompt),
                HumanMessage(content=prompt)
            ])
            parsed = json.loads(response.content.strip())
            intent = parsed.get("intent", "text_generation")
        except Exception as e:
            logger.warning(f"Ollama JSON parse failed: {e}")
            intent = "text_generation"
            
        return {"intent": intent}

def intent_router_condition(state: AgentState) -> str:
    """ LangGraph 분기 결정을 리턴 """
    intent = state.get("intent", "")
    if intent == "image_processing":
        return "synthesizer" # (수정) preprocessor 삭제됨. 이미지 입력 시 3D 변환을 위해 곧장 다각도 생성기로 감.
    elif intent in ["text_generation", "text_modification"]:
        return "rag_retriever"
    else:
        return "end"
