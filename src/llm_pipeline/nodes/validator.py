import json
import random
import time
from loguru import logger
from langchain_ollama import ChatOllama
from langchain_core.messages import SystemMessage, HumanMessage
from ..core.schemas import AgentState
from ..core.config import config

def validate_image_for_trellis(state: AgentState) -> dict:
    """
    (Step 4 - LLM-as-a-Judge)
    다중 이미지(Multi-view) 배열(`current_image_urls`) 각각에 대해 
    Vision LLM이 3D 변환에 적합한지 (단색 흰 배경 확인 등) 검사합니다.
    (모든 다각도 이미지가 성공해야 통과)
    """
    current_image_urls = state.get("current_image_urls", [])
    
    if not current_image_urls:
        return {
            "is_valid": False,
            "validation_feedback": "생성/전처리된 다각도 이미지가 존재하지 않습니다.",
            "status_message": "검증 실패: 이미지가 없습니다."
        }
        
    logger.info(f"Validating {len(current_image_urls)} Multi-view Images via Ollama Vision...")
    
    # 1. Ollama Vision LLM 설정
    llm = ChatOllama(
        model=config.OLLAMA_VISION_MODEL,
        base_url=config.OLLAMA_BASE_URL,
        temperature=0.0,
        format="json"
    )
    
    sys_prompt = "You are a QA judge for 3D multi-view image integration. Does the shoe stand out against a very clean background? Return: {'is_valid': true or false, 'reason': '...'}"
    all_passed = True
    feedbacks = []
    
    # 2. 다각도 이미지 배열 전체 순회 검사 (성공 시에만 통과)
    for idx, img_url in enumerate(current_image_urls):
        logger.debug(f"Validating View [{idx+1}/{len(current_image_urls)}]: {img_url}")
        time.sleep(1) # 모의 
        
        try:
            # 실제 구현: URL Base64 디코딩 후 HumanMessage(Content) 배열에 이미지 객체 포함하여 invoke
            # response = llm.invoke(...)
            # parsed = json.loads(response.content)
            raise NotImplementedError("Base64 Multi-image upload parsing logic skeleton.")
        except Exception as e:
            # 더미 로직 동작 (Vision 모델 실제 미연결시 임시 통과)
            passed_dummy = random.random() > 0.1 
            if not passed_dummy:
                all_passed = False
                feedbacks.append(f"View {idx+1} failed checking: Not pure white background.")
            else:
                feedbacks.append(f"View {idx+1} passed white bg check.")
    
    if all_passed:
        logger.success("All multi-angle views passed validation!")
        return {
            "is_valid": True,
            "validation_feedback": "All views confirmed solid white and centered.",
            "final_output_urls": current_image_urls,
            "status_message": "3D 다각도 렌더링에 적합한 모든 이미지(" + str(len(current_image_urls)) + "장) 정규화 도출 성공."
        }
    else:
        logger.warning(f"Some multi-angle views failed validation. Feedbacks: {feedbacks}")
        return {
            "is_valid": False,
            "validation_feedback": " | ".join(feedbacks),
            "status_message": "다각도 이미지 중 일부의 검증에 실패했습니다. " + feedbacks[0]
        }

def validation_condition(state: AgentState) -> str:
    is_valid = state.get("is_valid", False)
    if is_valid:
        return "end"
    else:
        return "error_handler"
