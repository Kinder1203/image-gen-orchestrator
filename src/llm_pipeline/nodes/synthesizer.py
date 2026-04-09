import json
import logging
import random
import re
import time
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlencode, urlparse
from typing import Optional

import requests
from langchain_core.messages import HumanMessage
from langchain_ollama import ChatOllama

from ..core.config import config
from ..core.schemas import AgentState
from .rag import retrieve_rules_for_query

logger = logging.getLogger(__name__)

COMFY_URL = config.COMFYUI_URL.rstrip("/")


def _resolve_template_path(*candidates: str) -> Path:
    for candidate in candidates:
        path = Path(candidate)
        if path.exists():
            return path
    raise FileNotFoundError(f"None of the ComfyUI template files were found: {', '.join(candidates)}")


BASE_TEMPLATE_PATH = _resolve_template_path("image_z_image_turbo (2).json", "image_z_image_turbo.json")
EDIT_TEMPLATE_PATH = _resolve_template_path("image_qwen_image_edit_2509.json")
MULTI_VIEW_TEMPLATE_PATH = _resolve_template_path("templates-1_click_multiple_character_angles-v1.0 (3) (1).json")


def _truncate_text(text: str, max_len: int = 400) -> str:
    compact = " ".join((text or "").split())
    if len(compact) <= max_len:
        return compact
    return f"{compact[:max_len]}..."


def _comfy_result(image_urls: Optional[list[str]] = None, error_message: str = "") -> dict:
    return {
        "image_urls": image_urls or [],
        "error_message": error_message,
    }


def _sync_call_comfyui(payload: dict) -> dict:
    """
    ComfyUI에 프롬프트를 전송하고, 완료된 산출물의 이미지 URL 목록 또는 명시적 오류를 반환합니다.
    """
    if not payload:
        error_message = "ComfyUI payload is empty."
        logger.error(error_message)
        return _comfy_result(error_message=error_message)

    try:
        response = requests.post(f"{COMFY_URL}/prompt", json=payload, timeout=10)
        response.raise_for_status()
        prompt_id = response.json().get("prompt_id")

        if not prompt_id:
            error_message = "ComfyUI /prompt succeeded but prompt_id was missing from the response."
            logger.error(error_message)
            return _comfy_result(error_message=error_message)

        logger.info(f"Sent workflow to ComfyUI. Waiting for generation... (Prompt ID: {prompt_id})")
        started_at = time.monotonic()
        timeout_seconds = max(int(config.COMFYUI_HISTORY_TIMEOUT_SECONDS), 1)

        while True:
            elapsed = time.monotonic() - started_at
            if elapsed > timeout_seconds:
                error_message = (
                    f"ComfyUI history polling timed out after {timeout_seconds}s "
                    f"while waiting for prompt_id={prompt_id}."
                )
                logger.error(error_message)
                return _comfy_result(error_message=error_message)

            hist_res = requests.get(f"{COMFY_URL}/history/{prompt_id}", timeout=10)
            hist_res.raise_for_status()
            hist_data = hist_res.json()

            if prompt_id in hist_data:
                image_urls: list[str] = []
                outputs = hist_data[prompt_id].get("outputs", {})

                for out_data in outputs.values():
                    for img in out_data.get("images", []):
                        query = urlencode(
                            {
                                "filename": img["filename"],
                                "subfolder": img.get("subfolder", ""),
                                "type": img.get("type", "output"),
                            }
                        )
                        image_urls.append(f"{COMFY_URL}/view?{query}")

                if not image_urls:
                    error_message = "ComfyUI completed the workflow but returned no image outputs in /history."
                    logger.error(error_message)
                    return _comfy_result(error_message=error_message)

                return _comfy_result(image_urls=image_urls)

            time.sleep(2.0)

    except requests.HTTPError as exc:
        status_code = exc.response.status_code if exc.response is not None else "unknown"
        response_body = _truncate_text(exc.response.text if exc.response is not None else "")
        error_message = f"ComfyUI request failed with HTTP {status_code}"
        if response_body:
            error_message += f": {response_body}"
        logger.error(error_message)
        return _comfy_result(error_message=error_message)
    except requests.RequestException as exc:
        error_message = f"ComfyUI request failed: {exc}"
        logger.error(error_message)
        return _comfy_result(error_message=error_message)
    except Exception as exc:
        error_message = f"Unexpected ComfyUI execution error: {exc}"
        logger.error(error_message)
        return _comfy_result(error_message=error_message)


def _load_workflow_template(template_path: Path) -> dict:
    with template_path.open("r", encoding="utf-8") as file:
        return json.load(file)


def _infer_background_spec(text: str) -> str:
    lower = (text or "").lower()
    if any(token in lower for token in ("white gold", "platinum", "silver", "white ring", "하얀", "화이트", "백금", "실버")):
        return "pure pitch-black"
    if any(token in lower for token in ("yellow gold", "gold", "골드", "노란")):
        return "pure cool cyan"
    if any(token in lower for token in ("rose gold", "로즈골드", "핑크골드", "rose")):
        return "pure teal-cyan"
    if any(token in lower for token in ("black", "gunmetal", "titanium", "블랙", "검은", "티타늄")):
        return "pure icy white"
    return "pure pitch-black"


def _enforce_background_contrast(prompt: str, source_text: str) -> str:
    background_spec = _infer_background_spec(source_text)
    enforced = (
        f"{prompt}, perfectly isolated ring, extremely strong silhouette separation, "
        f"flat {background_spec} background, no gradient, no texture, no props, "
        "no background reflections, no cast objects, crisp inner-hole separation"
    )
    return " ".join(enforced.split())


def _extract_engraving_text(custom_prompt: str) -> str:
    text = (custom_prompt or "").strip()
    if not text:
        return ""

    quoted_patterns = [
        r"'([^']+)'",
        r'"([^"]+)"',
        r"“([^”]+)”",
        r"‘([^’]+)’",
    ]
    for pattern in quoted_patterns:
        match = re.search(pattern, text)
        if match and match.group(1).strip():
            return match.group(1).strip()

    regex_patterns = [
        r"([A-Za-z0-9가-힣&+\-_/]{1,20})라고\s*각인",
        r"([A-Za-z0-9가-힣&+\-_/]{1,20})로\s*각인",
        r"각인(?:은|을)?\s*([A-Za-z0-9가-힣&+\-_/]{1,20})",
        r"engrave\s+([A-Za-z0-9가-힣&+\-_/]{1,20})",
    ]
    for pattern in regex_patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match and match.group(1).strip():
            return match.group(1).strip()

    return ""


def _detect_customization_kind(custom_prompt: str) -> str:
    lower = (custom_prompt or "").lower()
    if any(token in lower for token in ("각인", "engrave", "engraving", "문구", "텍스트", "lettering", "이니셜")):
        return "engraving"
    if any(token in lower for token in ("큐빅", "보석", "gem", "diamond", "stone", "스톤")):
        return "gemstone"
    return "general"


def _build_customization_context(state: AgentState) -> tuple[str, str, str]:
    custom_prompt = state.get("customization_prompt") or state.get("user_prompt", "")
    base_prompt = state.get("synthesized_prompt", "")
    query = custom_prompt or base_prompt or state.get("user_prompt", "")
    rag_context = retrieve_rules_for_query(query, top_k=4)
    customization_kind = _detect_customization_kind(custom_prompt)
    engraving_text = _extract_engraving_text(custom_prompt) if customization_kind == "engraving" else ""
    return rag_context, customization_kind, engraving_text


def _compose_edit_prompt(state: AgentState, rag_context: str, customization_kind: str, engraving_text: str) -> str:
    custom_prompt = (state.get("customization_prompt") or state.get("user_prompt", "")).strip()
    base_prompt = (state.get("synthesized_prompt") or "").strip()
    base_descriptor = f"Base ring description: {base_prompt}. " if base_prompt else ""

    if customization_kind == "engraving" and engraving_text:
        return (
            f"{base_descriptor}"
            f"Preserve the original ring design, material, thickness, geometry, reflections, and camera angle. "
            f"Engrave only the exact text '{engraving_text}' on the visible front outer band unless the user explicitly requested inner-band placement. "
            "The engraving must be physically carved into the metal surface, follow the ring curvature naturally, "
            "share the same lighting, bevel, reflections, and shadow logic as the band, and feel like real jewelry craftsmanship. "
            "No floating letters, no sticker text, no printed overlay, no painted text, no extra characters, no extra words, no additional inscriptions anywhere else. "
            f"Relevant ring-editing rules: {rag_context}"
        )

    if customization_kind == "gemstone":
        return (
            f"{base_descriptor}"
            f"Apply only this requested ring customization: {custom_prompt}. "
            "Integrate the gemstone naturally into the ring structure, preserving the existing band shape and material unless explicitly changed. "
            "The added stone must look physically mounted, proportionate, and harmonized with the ring design, not pasted on top. "
            f"Relevant ring-editing rules: {rag_context}"
        )

    return (
        f"{base_descriptor}"
        f"Apply only this requested ring customization: {custom_prompt}. "
        "Preserve the original ring geometry, material, finish, and overall composition unless the user explicitly asked to change them. "
        "Any new detail must look integrated into the ring itself, never floating or overlaid. "
        f"Relevant ring-editing rules: {rag_context}"
    )


def _upload_image_bytes_to_comfyui(image_bytes: bytes, filename_hint: str) -> str:
    files = {
        "image": (filename_hint or "comfy_bridge.png", image_bytes, "application/octet-stream"),
    }
    response = requests.post(f"{COMFY_URL}/upload/image", files=files, timeout=20)
    response.raise_for_status()
    uploaded_name = response.json().get("name")
    if not uploaded_name:
        raise ValueError("ComfyUI /upload/image succeeded but did not return a file name.")
    return uploaded_name


def _normalize_comfy_image_reference(image_ref: str) -> str:
    if not image_ref:
        return ""

    if not image_ref.startswith(("http://", "https://")):
        return image_ref

    parsed = urlparse(image_ref)
    query = parse_qs(parsed.query)
    filename = unquote(query.get("filename", [Path(parsed.path).name or "comfy_bridge.png"])[0])
    subfolder = unquote(query.get("subfolder", [""])[0])
    image_type = query.get("type", [""])[0]
    comfy_host = urlparse(COMFY_URL).netloc

    if parsed.netloc == comfy_host and image_type == "input" and not subfolder and filename:
        return filename

    logger.info(f"Bridging image reference into ComfyUI input upload: {image_ref}")
    download_response = requests.get(image_ref, timeout=20)
    download_response.raise_for_status()
    uploaded_name = _upload_image_bytes_to_comfyui(download_response.content, filename or "comfy_bridge.png")
    logger.info(f"Bridged ComfyUI image reference as input file: {uploaded_name}")
    return uploaded_name


def _safe_chainable_image_ref(image_ref: str) -> str:
    if not image_ref:
        return ""
    try:
        return _normalize_comfy_image_reference(image_ref)
    except Exception as exc:
        logger.warning(f"Failed to prepare chainable ComfyUI image reference: {exc}")
        return ""


def _is_api_prompt_template(workflow: dict) -> bool:
    if not isinstance(workflow, dict) or not workflow:
        return False
    return all(
        isinstance(node, dict) and "class_type" in node and "inputs" in node
        for node in workflow.values()
    )


def _require_api_prompt_template(workflow: dict, template_name: str) -> dict:
    if _is_api_prompt_template(workflow):
        return workflow
    raise ValueError(
        f"{template_name} is not a ComfyUI API-format prompt JSON. "
        "Expected top-level node ids mapped to {class_type, inputs}."
    )


def _replace_placeholders(node: object, replacements: dict[str, str]) -> object:
    if isinstance(node, dict):
        return {key: _replace_placeholders(value, replacements) for key, value in node.items()}
    if isinstance(node, list):
        return [_replace_placeholders(item, replacements) for item in node]
    if isinstance(node, str):
        updated = node
        for old, new in replacements.items():
            updated = updated.replace(old, new)
        return updated
    return node


def _randomize_seeds(node: object) -> object:
    if isinstance(node, dict):
        randomized = {}
        for key, value in node.items():
            if key in {"seed", "noise_seed"} and isinstance(value, int):
                randomized[key] = random.randint(1, 2147483647)
            else:
                randomized[key] = _randomize_seeds(value)
        return randomized
    if isinstance(node, list):
        return [_randomize_seeds(item) for item in node]
    return node


def _collect_load_image_nodes(workflow: dict) -> list[tuple[str, dict]]:
    return [
        (node_id, node)
        for node_id, node in workflow.items()
        if isinstance(node, dict) and node.get("class_type") == "LoadImage"
    ]


def _select_edit_load_image_node(workflow: dict) -> tuple[str, dict]:
    load_nodes = _collect_load_image_nodes(workflow)
    if len(load_nodes) == 1:
        return load_nodes[0]
    raise ValueError(f"Edit workflow requires exactly one LoadImage node, found {len(load_nodes)}.")


def _select_multi_view_load_image_node(workflow: dict) -> tuple[str, dict]:
    load_nodes = _collect_load_image_nodes(workflow)
    titled_nodes = [
        (node_id, node)
        for node_id, node in load_nodes
        if ((node.get("_meta") or {}).get("title") or "") == "Load Character Image"
    ]

    if len(titled_nodes) == 1:
        return titled_nodes[0]
    if not titled_nodes and len(load_nodes) == 1:
        return load_nodes[0]
    if len(titled_nodes) > 1:
        raise ValueError("Multi-view workflow has multiple 'Load Character Image' nodes.")
    raise ValueError(f"Multi-view workflow has ambiguous LoadImage nodes: {len(load_nodes)} found.")


def _set_load_image_value(node: dict, image_value: str) -> None:
    inputs = node.get("inputs")
    if not isinstance(inputs, dict) or "image" not in inputs:
        raise ValueError("Selected LoadImage node does not expose inputs.image.")
    inputs["image"] = image_value


def _build_base_payload(enhanced_prompt: str) -> dict:
    workflow = _load_workflow_template(BASE_TEMPLATE_PATH)
    workflow = _require_api_prompt_template(workflow, BASE_TEMPLATE_PATH.name)
    workflow = _replace_placeholders(workflow, {"___USER_PROMPT___": enhanced_prompt})
    workflow = _randomize_seeds(workflow)
    return {
        "client_id": "llm_backend",
        "prompt": workflow,
    }


def _build_edit_payload(base_image: str, custom_prompt: str) -> dict:
    workflow = _load_workflow_template(EDIT_TEMPLATE_PATH)
    workflow = _require_api_prompt_template(workflow, EDIT_TEMPLATE_PATH.name)
    workflow = _replace_placeholders(workflow, {"___CUSTOM_PROMPT___": custom_prompt})
    _, load_node = _select_edit_load_image_node(workflow)
    _set_load_image_value(load_node, _normalize_comfy_image_reference(base_image))
    workflow = _randomize_seeds(workflow)
    return {
        "client_id": "llm_backend",
        "prompt": workflow,
    }


def _build_multi_view_payload(target_image: str) -> dict:
    workflow = _load_workflow_template(MULTI_VIEW_TEMPLATE_PATH)
    workflow = _require_api_prompt_template(workflow, MULTI_VIEW_TEMPLATE_PATH.name)
    _, load_node = _select_multi_view_load_image_node(workflow)
    _set_load_image_value(load_node, _normalize_comfy_image_reference(target_image))
    workflow = _randomize_seeds(workflow)
    return {
        "client_id": "llm_backend",
        "prompt": workflow,
    }


def generate_base_image(state: AgentState) -> dict:
    """
    RAG 컨텍스트를 활용해 베이스 반지 이미지를 신규 생성합니다.
    """
    user_prompt = state.get("user_prompt", "")
    rag_context = state.get("rag_context", "")

    logger.info("Enhancing prompt using Gemma 4 & RAG rules...")

    llm = ChatOllama(
        model=config.OLLAMA_MODEL,
        base_url=config.OLLAMA_BASE_URL,
        temperature=0.3,
    )

    sys_prompt = f"""
You are an expert jewelry prompt engineer for Stable Diffusion.
User requested: '{user_prompt}'
RAG Rules to follow: '{rag_context}'

Your task:
1. Identify the ring's design and color/material from the user's request.
2. Based on the RAG rules, calculate the exact complementary background color.
3. The background must be a flat, single-color, non-textured studio background optimized for clean alpha matting.
4. Output only a comma-separated keywords prompt. It must end with 'solid [COLOR] background'.
No conversational text and no quotes.
"""

    try:
        resp = llm.invoke([HumanMessage(content=sys_prompt)])
        enhanced_prompt = resp.content.strip()
    except Exception as exc:
        logger.warning(f"Prompt enhancement failed: {exc}. Using original prompt.")
        enhanced_prompt = f"{user_prompt}, highly detailed, solid dark background"

    enhanced_prompt = _enforce_background_contrast(enhanced_prompt, f"{user_prompt} {rag_context}")
    logger.info(f"Gemma 4 enhanced prompt: {enhanced_prompt}")

    try:
        comfyui_payload = _build_base_payload(enhanced_prompt)
    except Exception as exc:
        error_message = f"베이스 생성용 ComfyUI 템플릿 로드에 실패했습니다. ({exc})"
        logger.error(error_message)
        return {
            "base_ring_image_url": "",
            "synthesized_prompt": enhanced_prompt,
            "generation_result": "system_error",
            "status_message": error_message,
        }

    comfy_result = _sync_call_comfyui(comfyui_payload)
    result_urls = comfy_result["image_urls"]
    final_url = result_urls[0] if result_urls else ""
    error_message = comfy_result["error_message"]

    return {
        "base_ring_image_url": final_url,
        "base_ring_image_ref": _safe_chainable_image_ref(final_url),
        "synthesized_prompt": enhanced_prompt,
        "generation_result": "system_error" if error_message else "success",
        "status_message": error_message if error_message else "",
    }


def edit_image(state: AgentState) -> dict:
    """
    기존 이미지 또는 이전 수정본을 기반으로 커스텀을 반영합니다.
    """
    base_image = (
        state.get("edited_ring_image_ref")
        or state.get("base_ring_image_ref")
        or state.get("edited_ring_image_url")
        or state.get("base_ring_image_url", "")
    )
    rag_context, customization_kind, engraving_text = _build_customization_context(state)
    custom_prompt = _compose_edit_prompt(state, rag_context, customization_kind, engraving_text)

    logger.info(f"Applying customization to image: {base_image}...")

    if not base_image:
        error_message = "커스텀 편집에 사용할 입력 이미지가 없습니다."
        return {
            "edited_ring_image_url": "",
            "edited_ring_image_ref": "",
            "synthesized_prompt": custom_prompt,
            "customization_context": rag_context,
            "customization_kind": customization_kind,
            "expected_engraving_text": engraving_text,
            "generation_result": "system_error",
            "status_message": error_message,
        }

    try:
        comfyui_payload = _build_edit_payload(base_image, custom_prompt)
    except Exception as exc:
        error_message = f"커스텀 편집용 ComfyUI payload 구성에 실패했습니다. ({exc})"
        logger.error(error_message)
        return {
            "edited_ring_image_url": "",
            "edited_ring_image_ref": "",
            "synthesized_prompt": custom_prompt,
            "customization_context": rag_context,
            "customization_kind": customization_kind,
            "expected_engraving_text": engraving_text,
            "generation_result": "system_error",
            "status_message": error_message,
        }

    comfy_result = _sync_call_comfyui(comfyui_payload)
    result_urls = comfy_result["image_urls"]
    final_url = result_urls[0] if result_urls else ""
    error_message = comfy_result["error_message"]

    return {
        "edited_ring_image_url": final_url,
        "edited_ring_image_ref": _safe_chainable_image_ref(final_url),
        "synthesized_prompt": custom_prompt,
        "customization_context": rag_context,
        "customization_kind": customization_kind,
        "expected_engraving_text": engraving_text,
        "generation_result": "system_error" if error_message else "success",
        "status_message": error_message if error_message else "",
    }


def generate_multi_view(state: AgentState) -> dict:
    """
    최종 채택된 이미지로 다각도 생성과 배경 제거를 수행합니다.
    """
    target_image = (
        state.get("edited_ring_image_ref")
        or state.get("base_ring_image_ref")
        or state.get("edited_ring_image_url", "")
        or state.get("base_ring_image_url", "")
    )

    logger.info(f"Extracting multi-views and applying Birefnet rembg. Target: {target_image}")

    if not target_image:
        error_message = "다각도 생성에 사용할 입력 이미지가 없습니다."
        return {
            "current_image_urls": [],
            "generation_result": "system_error",
            "status_message": error_message,
        }

    try:
        comfyui_payload = _build_multi_view_payload(target_image)
    except Exception as exc:
        error_message = f"다각도용 ComfyUI payload 구성에 실패했습니다. ({exc})"
        logger.error(error_message)
        return {
            "current_image_urls": [],
            "generation_result": "system_error",
            "status_message": error_message,
        }

    comfy_result = _sync_call_comfyui(comfyui_payload)
    result_urls = comfy_result["image_urls"]
    error_message = comfy_result["error_message"]

    return {
        "current_image_urls": result_urls,
        "generation_result": "system_error" if error_message else "success",
        "status_message": error_message if error_message else "",
    }
