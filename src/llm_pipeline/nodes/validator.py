import base64
import json
import logging

import requests
from langchain_core.messages import HumanMessage
from langchain_ollama import ChatOllama

from ..core.config import config
from ..core.schemas import AgentState

logger = logging.getLogger(__name__)

_BACKGROUND_CONTRAST_POLICY = (
    "The background must be a single solid color that is clearly complementary or otherwise "
    "strongly contrasting to the ring's dominant material color for clean alpha matting and "
    "inner-hole separation. Reject white/platinum/silver rings on white or light gray, "
    "yellow/rose gold rings on beige, cream, peach, or warm gold backgrounds, and black or "
    "gunmetal rings on black or charcoal backgrounds. Prefer black/charcoal/navy for white "
    "metals, cool blue/cyan/teal for yellow or rose gold, and pale icy gray or white for dark "
    "metals."
)


def _validation_result(is_valid: bool, reason: str, result_type: str = "judged") -> dict:
    return {"is_valid": is_valid, "reason": reason, "result_type": result_type}


def _handle_validation_error(reason: str) -> dict:
    if config.ALLOW_VALIDATION_BYPASS:
        logger.warning(f"Validation bypassed: {reason}")
        return _validation_result(True, f"Validation bypassed: {reason}", result_type="system_error_bypassed")
    return _validation_result(False, reason, result_type="system_error")


def _encode_image_from_url(image_url: str) -> str:
    """ComfyUI에서 생성된 이미지를 다운로드하여 Base64로 변환"""
    if not image_url:
        return ""

    try:
        response = requests.get(image_url, timeout=10)
        response.raise_for_status()
        return base64.b64encode(response.content).decode("utf-8")
    except Exception as exc:
        logger.warning(f"Failed to download image for validation: {exc}")
        return ""


def _call_vision_judge(image_url: str, prompt: str) -> dict:
    """공통 Vision LLM 호출 유틸"""
    img_base64 = _encode_image_from_url(image_url)
    if not img_base64:
        return _handle_validation_error("검수용 이미지를 다운로드하지 못했습니다.")

    llm = ChatOllama(
        model=config.OLLAMA_MODEL,
        base_url=config.OLLAMA_BASE_URL,
        temperature=0.0,
        format="json",
    )

    try:
        message_content = [
            {"type": "text", "text": prompt},
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_base64}"}},
        ]

        resp = llm.invoke([HumanMessage(content=message_content)])
        raw_content = resp.content
        if isinstance(raw_content, list):
            raw_content = "".join(
                item.get("text", "") if isinstance(item, dict) else str(item) for item in raw_content
            )

        parsed = json.loads(str(raw_content).strip())
        return _validation_result(bool(parsed.get("is_valid", False)), parsed.get("reason", "검수 이유 없음"))
    except Exception as exc:
        logger.warning(f"Gemma 4 Vision validation failed: {exc}")
        return _handle_validation_error(f"Vision LLM 검수에 실패했습니다. ({exc})")


def _status_message(prefix: str, reason: str, fallback: str) -> str:
    detail = reason or fallback
    return f"{prefix}: {detail}"


def _merge_customization_directive(existing_prompt: str, directive: str) -> str:
    existing_prompt = (existing_prompt or "").strip()
    directive = (directive or "").strip()

    if existing_prompt and directive:
        return f"{existing_prompt}. Also ensure: {directive}"
    return existing_prompt or directive


def validate_base_image(state: AgentState) -> dict:
    """프롬프트에 맞는 퀄리티의 베이스 반지가 생성되었는지 판별"""
    target_img = state.get("base_ring_image_url", "")
    user_prompt = state.get("user_prompt", "")
    synthesized_prompt = state.get("synthesized_prompt", "")
    retry_count = state.get("retry_count", 0)

    sys_prompt = (
        "You are a jewelry generation judge. "
        "Does the generated ring in the image accurately reflect the user's request: "
        f"'{user_prompt}'? "
        f"Generated prompt hint: '{synthesized_prompt}'. "
        f"{_BACKGROUND_CONTRAST_POLICY} "
        "Confirm the ring silhouette and inner hole are clearly separated from the background, "
        "and the ring itself is high quality. "
        "Return JSON {'is_valid': true/false, 'reason': '...'}."
    )

    logger.info(f"Validating base ring generation (Retry: {retry_count})...")
    result = _call_vision_judge(target_img, sys_prompt)

    is_valid = result.get("is_valid", False)
    reason = result.get("reason", "")
    return {
        "is_valid": is_valid,
        "validation_reason": reason,
        "retry_count": retry_count + 1 if not is_valid else retry_count,
        "status_message": "" if is_valid else _status_message("베이스 이미지 검수 실패", reason, "요구사항 반영이 부족합니다."),
    }


def validate_edited_image(state: AgentState) -> dict:
    """사용자가 지시한 커스텀(각인/보석)이 제대로 합성되었는지 판별"""
    target_img = state.get("edited_ring_image_url", "")
    custom_prompt = state.get("customization_prompt") or state.get("user_prompt", "")
    retry_count = state.get("retry_count", 0)

    sys_prompt = (
        "Does the image accurately apply the requested modifications: "
        f"'{custom_prompt}'? Check engravings when text is requested, gems when stones are "
        "requested, and background corrections when background changes are requested. "
        "Return JSON {'is_valid': true/false, 'reason': '...'}."
    )

    logger.info(f"Validating custom edit application (Retry: {retry_count})...")
    result = _call_vision_judge(target_img, sys_prompt)

    is_valid = result.get("is_valid", False)
    reason = result.get("reason", "")
    return {
        "is_valid": is_valid,
        "validation_reason": reason,
        "retry_count": retry_count + 1 if not is_valid else retry_count,
        "status_message": "" if is_valid else _status_message("커스텀 이미지 검수 실패", reason, "수정 요청 반영이 부족합니다."),
    }


def validate_rembg(state: AgentState) -> dict:
    """다각도 분리 및 반지 안쪽 빈 공간 투명화가 완벽한지 판별"""
    urls = state.get("current_image_urls", [])
    retry_count = state.get("retry_count", 0)

    sys_prompt = (
        "You are a TRELLIS preparation judge. Look at this ring image. "
        "Is the background completely removed (transparent alpha), and more importantly, "
        "is the inner hole of the ring completely hollowed out without background artifact "
        "remaining? Return JSON {'is_valid': true/false, 'reason': '...'}."
    )

    logger.info(f"Validating rembg (Retry: {retry_count})...")

    if not urls:
        reason = "다각도 결과 이미지가 비어 있습니다."
        return {
            "is_valid": False if not config.ALLOW_VALIDATION_BYPASS else True,
            "validation_reason": reason,
            "retry_count": retry_count + (0 if config.ALLOW_VALIDATION_BYPASS else 1),
            "final_output_urls": urls if config.ALLOW_VALIDATION_BYPASS else [],
            "status_message": _status_message("다각도 검수 실패", reason, reason),
        }

    is_valid = True
    reason = "All multi-views passed rembg validation."

    for url in urls:
        res = _call_vision_judge(url, sys_prompt)
        if not res.get("is_valid", False):
            is_valid = False
            reason = res.get("reason", "Failed on one of the multi-views.")
            break

    return {
        "is_valid": is_valid,
        "validation_reason": reason,
        "retry_count": retry_count + 1 if not is_valid else retry_count,
        "final_output_urls": urls if is_valid else [],
        "status_message": "" if is_valid else _status_message("다각도 검수 실패", reason, "누끼 품질이 부족합니다."),
    }


def validate_input_image(state: AgentState) -> dict:
    """시나리오 2/3에서 업로드된 시안 이미지의 배경을 사전 검사"""
    target_img = state.get("base_ring_image_url", "")

    sys_prompt = (
        "You are a pre-processing judge. Check the image. "
        f"{_BACKGROUND_CONTRAST_POLICY} "
        "If it's too similar (for example white ring on white background), output "
        "is_valid=false, and in 'reason', write exactly a short directive for an "
        "inpainting model to fix it, like 'Change the background to solid pitch black'. "
        "Return JSON {'is_valid': true/false, 'reason': '...'}."
    )

    logger.info("Guarding: validating input image contrast for scenarios 2/3...")
    result = _call_vision_judge(target_img, sys_prompt)

    is_valid = result.get("is_valid", False)
    reason = result.get("reason", "Good contrast.")
    result_type = result.get("result_type", "judged")

    if result_type == "system_error":
        return {
            "is_valid": False,
            "guardrail_result": "system_error",
            "validation_reason": reason,
            "status_message": _status_message(
                "입력 이미지 사전 검수 시스템 오류",
                reason,
                "입력 이미지 가드레일 검수 중 시스템 오류가 발생했습니다.",
            ),
        }

    if result_type == "system_error_bypassed":
        return {
            "is_valid": True,
            "guardrail_result": "pass",
            "validation_reason": reason,
            "status_message": "",
        }

    update_dict = {
        "is_valid": is_valid,
        "guardrail_result": "pass" if is_valid else "repair_required",
        "validation_reason": reason,
        "status_message": "" if is_valid else _status_message("입력 이미지 사전 검수 실패", reason, "배경 대비가 부족합니다."),
    }

    if not is_valid:
        current_prompt = state.get("customization_prompt") or state.get("user_prompt", "")
        update_dict["customization_prompt"] = _merge_customization_directive(current_prompt, reason)

    return update_dict
