import base64
import json
import logging
import re
from io import BytesIO
from urllib.parse import quote

import requests
from PIL import Image, UnidentifiedImageError

from ..core.config import config
from ..core.schemas import AgentState
from ..core.vllm_client import invoke_multimodal_json

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

_BASE_SURFACE_REJECTION_POLICY = (
    "Reject the image if there is any visible ground plane, support surface, tabletop, pedestal, "
    "studio sweep curve, textured floor, gradient backdrop, floor reflection, cast shadow, "
    "contact shadow, drop shadow, or ambient shadow directly beneath the ring."
)


def _select_validation_urls(urls: list[str], limit: int) -> list[str]:
    if limit <= 0 or len(urls) <= limit:
        return urls
    if limit == 1:
        return [urls[0]]

    sampled = [urls[0], urls[-1]]
    if limit > 2:
        middle_indexes = [round(i * (len(urls) - 1) / (limit - 1)) for i in range(1, limit - 1)]
        for idx in middle_indexes:
            sampled.append(urls[idx])

    deduped: list[str] = []
    for url in sampled:
        if url not in deduped:
            deduped.append(url)
    return deduped[:limit]


def _truncate_for_log(text: str, max_len: int = 400) -> str:
    compact = " ".join((text or "").split())
    if len(compact) <= max_len:
        return compact
    return f"{compact[:max_len]}..."


def _validation_result(is_valid: bool, reason: str, result_type: str = "judged") -> dict:
    return {"is_valid": is_valid, "reason": reason, "result_type": result_type}


def _handle_validation_error(reason: str) -> dict:
    if config.ALLOW_VALIDATION_BYPASS:
        logger.warning(f"Validation bypassed: {reason}")
        return _validation_result(True, f"Validation bypassed: {reason}", result_type="system_error_bypassed")
    return _validation_result(False, reason, result_type="system_error")


def _parse_json_object(raw_content: object) -> dict:
    text = str(raw_content or "").strip()
    if not text:
        raise ValueError("Vision response was empty.")

    fenced_match = re.match(r"^```(?:json)?\s*(.*?)\s*```$", text, flags=re.IGNORECASE | re.DOTALL)
    if fenced_match:
        text = fenced_match.group(1).strip()

    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end == -1 or start >= end:
            raise
        parsed = json.loads(text[start : end + 1])

    if not isinstance(parsed, dict):
        raise ValueError("Vision response JSON must be an object.")

    return parsed


def _detect_image_mime_type(image_bytes: bytes, response_content_type: str = "") -> str:
    normalized = (response_content_type or "").split(";")[0].strip().lower()
    if normalized.startswith("image/"):
        return normalized

    try:
        with Image.open(BytesIO(image_bytes)) as image:
            return (Image.MIME.get(image.format) or "image/png").lower()
    except (UnidentifiedImageError, OSError, ValueError):
        return "image/png"


def _encode_image_from_url(image_url: str) -> str:
    """ComfyUI에서 생성된 이미지를 다운로드하여 Base64로 변환"""
    if not image_url:
        return ""

    try:
        resolved_url = image_url
        if not image_url.startswith(("http://", "https://")):
            # Uploaded ComfyUI input images are often passed around as plain filenames.
            resolved_url = f"{config.COMFYUI_URL.rstrip('/')}/view?filename={quote(image_url)}&type=input"

        response = requests.get(resolved_url, timeout=config.IMAGE_DOWNLOAD_TIMEOUT_SECONDS)
        response.raise_for_status()
        mime_type = _detect_image_mime_type(response.content, response.headers.get("Content-Type", ""))
        encoded = base64.b64encode(response.content).decode("utf-8")
        return f"data:{mime_type};base64,{encoded}"
    except Exception as exc:
        logger.warning(f"Failed to download image for validation: {exc}")
        return ""


def _call_vision_judge(image_url: str, prompt: str) -> dict:
    """공통 Vision LLM 호출 유틸"""
    image_data_url = _encode_image_from_url(image_url)
    if not image_data_url:
        return _handle_validation_error("검수용 이미지를 다운로드하지 못했습니다.")

    try:
        raw_content = invoke_multimodal_json(
            prompt=prompt,
            image_data_url=image_data_url,
            max_tokens=config.VLLM_VALIDATOR_MAX_TOKENS,
        )

        logger.info(f"Vision raw response: {_truncate_for_log(str(raw_content))}")
        parsed = _parse_json_object(raw_content)
        result = _validation_result(bool(parsed.get("is_valid", False)), parsed.get("reason", "검수 이유 없음"))
        logger.info(f"Vision parsed result: {result}")
        return result
    except Exception as exc:
        logger.warning(f"vLLM vision validation failed: {exc}")
        return _handle_validation_error(f"vLLM Vision 검수에 실패했습니다. ({exc})")


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
    generation_result = state.get("generation_result", "")

    if generation_result == "system_error" or not target_img:
        reason = state.get("status_message", "베이스 이미지 생성 시스템 오류가 발생했습니다.")
        return {
            "is_valid": False,
            "generation_result": "system_error",
            "validation_reason": reason,
            "retry_count": retry_count,
            "status_message": reason,
        }

    sys_prompt = (
        "You are a jewelry generation judge. "
        "Does the generated ring in the image accurately reflect the user's request: "
        f"'{user_prompt}'? "
        f"Generated prompt hint: '{synthesized_prompt}'. "
        f"{_BACKGROUND_CONTRAST_POLICY} "
        f"{_BASE_SURFACE_REJECTION_POLICY} "
        "Confirm the ring silhouette and inner hole are clearly separated from the background, "
        "the requested ring count and user-specified details are preserved, and the ring itself is high quality. "
        "If the image is invalid, explain the dominant corrective changes succinctly so the next retry can fix them. "
        "Return JSON {'is_valid': true/false, 'reason': '...'}."
    )

    logger.info(f"Validating base ring generation (Retry: {retry_count})...")
    result = _call_vision_judge(target_img, sys_prompt)

    is_valid = result.get("is_valid", False)
    reason = result.get("reason", "")
    logger.info(f"Base validation outcome: is_valid={is_valid}, reason={_truncate_for_log(reason)}")
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
    customization_context = state.get("customization_context", "")
    customization_kind = state.get("customization_kind", "")
    expected_engraving_text = state.get("expected_engraving_text", "")
    retry_count = state.get("retry_count", 0)
    generation_result = state.get("generation_result", "")

    if generation_result == "system_error" or not target_img:
        reason = state.get("status_message", "커스텀 이미지 생성 시스템 오류가 발생했습니다.")
        return {
            "is_valid": False,
            "generation_result": "system_error",
            "validation_reason": reason,
            "retry_count": retry_count,
            "status_message": reason,
        }

    if customization_kind == "engraving" and expected_engraving_text:
        sys_prompt = (
            "You are a strict jewelry engraving judge. "
            f"The ring must contain only the exact engraving text '{expected_engraving_text}'. "
            "Fail the image if any extra letters, request words, filler text, or malformed text appear. "
            "Fail the image if the text looks printed, floating, pasted, or drawn on top instead of carved into the metal. "
            "The engraving must follow the ring curvature, inherit material highlights and shadows, "
            "and look physically integrated into the band. "
            f"User request: '{custom_prompt}'. "
            f"Edit guidance used: '{customization_context}'. "
            "Return JSON {'is_valid': true/false, 'reason': '...'}."
        )
    else:
        sys_prompt = (
            "You are a strict jewelry edit judge. "
            f"User request: '{custom_prompt}'. "
            f"Edit guidance used: '{customization_context}'. "
            "Confirm that the requested modification is applied cleanly and looks physically integrated into the ring. "
            "Fail the image if the new detail looks pasted on, floating, or visually disconnected from the ring surface. "
            "Return JSON {'is_valid': true/false, 'reason': '...'}."
        )

    logger.info(f"Validating custom edit application (Retry: {retry_count})...")
    result = _call_vision_judge(target_img, sys_prompt)

    is_valid = result.get("is_valid", False)
    reason = result.get("reason", "")
    logger.info(f"Edit validation outcome: is_valid={is_valid}, reason={_truncate_for_log(reason)}")
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
    generation_result = state.get("generation_result", "")

    if generation_result == "system_error":
        reason = state.get("status_message", "다각도 생성 시스템 오류가 발생했습니다.")
        return {
            "is_valid": False,
            "generation_result": "system_error",
            "validation_reason": reason,
            "retry_count": retry_count,
            "final_output_urls": [],
            "status_message": reason,
        }

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

    sampled_urls = _select_validation_urls(urls, max(int(config.MULTI_VIEW_VALIDATION_SAMPLE_COUNT), 1))
    logger.info(f"Sampling {len(sampled_urls)} multi-view images for rembg validation out of {len(urls)} total.")

    for url in sampled_urls:
        res = _call_vision_judge(url, sys_prompt)
        if not res.get("is_valid", False):
            is_valid = False
            reason = res.get("reason", "Failed on one of the multi-views.")
            break

    logger.info(f"Rembg validation outcome: is_valid={is_valid}, reason={_truncate_for_log(reason)}")

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
    logger.info(
        "Input image guardrail outcome: "
        f"result_type={result_type}, is_valid={is_valid}, reason={_truncate_for_log(reason)}"
    )

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
