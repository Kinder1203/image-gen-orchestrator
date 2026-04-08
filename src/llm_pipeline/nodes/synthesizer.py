import json
import logging
import random
import time
from pathlib import Path
from typing import Optional

import requests
from langchain_core.messages import HumanMessage
from langchain_ollama import ChatOllama

from ..core.config import config
from ..core.schemas import AgentState

logger = logging.getLogger(__name__)

COMFY_URL = config.COMFYUI_URL.rstrip("/")

BASE_TEMPLATE_PATH = Path("image_z_image_turbo.json")
EDIT_TEMPLATE_PATH = Path("image_qwen_image_edit_2509.json")
MULTI_VIEW_TEMPLATE_PATH = Path("templates-1_click_multiple_character_angles-v1.0 (3).json")


def _sync_call_comfyui(payload: dict) -> list[str]:
    """
    ComfyUI에 프롬프트를 전송하고, 완료된 산출물의 이미지 URL 목록을 반환합니다.
    """
    if not payload:
        logger.error("ComfyUI payload is empty.")
        return []

    try:
        response = requests.post(f"{COMFY_URL}/prompt", json=payload, timeout=10)
        response.raise_for_status()
        prompt_id = response.json().get("prompt_id")

        if not prompt_id:
            logger.error("Failed to get prompt_id from ComfyUI.")
            return []

        logger.info(f"Sent workflow to ComfyUI. Waiting for generation... (Prompt ID: {prompt_id})")

        while True:
            hist_res = requests.get(f"{COMFY_URL}/history/{prompt_id}", timeout=10)
            hist_res.raise_for_status()
            hist_data = hist_res.json()

            if prompt_id in hist_data:
                image_urls: list[str] = []
                outputs = hist_data[prompt_id].get("outputs", {})

                for out_data in outputs.values():
                    for img in out_data.get("images", []):
                        filename = img["filename"]
                        image_urls.append(f"{COMFY_URL}/view?filename={filename}")

                return image_urls

            time.sleep(2.0)

    except Exception as exc:
        logger.error(f"ComfyUI Polling Error: {exc}")
        return []


def _load_workflow_template(template_path: Path) -> dict:
    with template_path.open("r", encoding="utf-8") as file:
        return json.load(file)


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


def _collect_load_image_nodes(workflow: dict) -> list[dict]:
    return [node for node in workflow.get("nodes", []) if node.get("type") == "LoadImage"]


def _select_edit_load_image_node(workflow: dict) -> dict:
    load_nodes = _collect_load_image_nodes(workflow)
    if len(load_nodes) == 1:
        return load_nodes[0]
    raise ValueError(f"Edit workflow requires exactly one LoadImage node, found {len(load_nodes)}.")


def _select_multi_view_load_image_node(workflow: dict) -> dict:
    load_nodes = _collect_load_image_nodes(workflow)
    titled_nodes = [node for node in load_nodes if (node.get("title") or "") == "Load Character Image"]

    if len(titled_nodes) == 1:
        return titled_nodes[0]
    if not titled_nodes and len(load_nodes) == 1:
        return load_nodes[0]
    if len(titled_nodes) > 1:
        raise ValueError("Multi-view workflow has multiple 'Load Character Image' nodes.")
    raise ValueError(f"Multi-view workflow has ambiguous LoadImage nodes: {len(load_nodes)} found.")


def _set_load_image_value(node: dict, image_value: str) -> None:
    widgets = node.get("widgets_values")
    if not isinstance(widgets, list) or not widgets:
        raise ValueError("Selected LoadImage node does not expose widgets_values[0].")
    widgets[0] = image_value


def _build_base_payload(enhanced_prompt: str) -> dict:
    workflow = _load_workflow_template(BASE_TEMPLATE_PATH)
    workflow = _replace_placeholders(workflow, {"___USER_PROMPT___": enhanced_prompt})
    workflow = _randomize_seeds(workflow)
    return {
        "client_id": "llm_backend",
        "prompt": workflow,
    }


def _build_edit_payload(base_image: str, custom_prompt: str) -> dict:
    workflow = _load_workflow_template(EDIT_TEMPLATE_PATH)
    workflow = _replace_placeholders(workflow, {"___CUSTOM_PROMPT___": custom_prompt})
    load_node = _select_edit_load_image_node(workflow)
    _set_load_image_value(load_node, base_image)
    workflow = _randomize_seeds(workflow)
    return {
        "client_id": "llm_backend",
        "prompt": workflow,
    }


def _build_multi_view_payload(target_image: str) -> dict:
    workflow = _load_workflow_template(MULTI_VIEW_TEMPLATE_PATH)
    load_node = _select_multi_view_load_image_node(workflow)
    _set_load_image_value(load_node, target_image)
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
        logger.info(f"Gemma 4 enhanced prompt: {enhanced_prompt}")
    except Exception as exc:
        logger.warning(f"Prompt enhancement failed: {exc}. Using original prompt.")
        enhanced_prompt = f"{user_prompt}, highly detailed, solid dark background"

    try:
        comfyui_payload = _build_base_payload(enhanced_prompt)
    except Exception as exc:
        logger.error(f"Failed to load base ComfyUI template: {exc}")
        return {
            "base_ring_image_url": "",
            "synthesized_prompt": enhanced_prompt,
            "status_message": "베이스 생성 템플릿 로드에 실패했습니다.",
        }

    result_urls = _sync_call_comfyui(comfyui_payload)
    final_url = result_urls[0] if result_urls else ""

    return {
        "base_ring_image_url": final_url,
        "synthesized_prompt": enhanced_prompt,
        "status_message": "베이스 이미지 생성에 실패했습니다." if not final_url else "",
    }


def edit_image(state: AgentState) -> dict:
    """
    기존 이미지 또는 이전 수정본을 기반으로 커스텀을 반영합니다.
    """
    base_image = state.get("edited_ring_image_url") or state.get("base_ring_image_url", "")
    custom_prompt = state.get("customization_prompt") or state.get("user_prompt", "")

    logger.info(f"Applying customization to image: {base_image}...")

    if not base_image:
        return {
            "edited_ring_image_url": "",
            "synthesized_prompt": custom_prompt,
            "status_message": "커스텀 편집에 사용할 입력 이미지가 없습니다.",
        }

    try:
        comfyui_payload = _build_edit_payload(base_image, custom_prompt)
    except Exception as exc:
        logger.error(f"Failed to build edit ComfyUI payload: {exc}")
        return {
            "edited_ring_image_url": "",
            "synthesized_prompt": custom_prompt,
            "status_message": f"커스텀 편집용 LoadImage 주입에 실패했습니다. ({exc})",
        }

    result_urls = _sync_call_comfyui(comfyui_payload)
    final_url = result_urls[0] if result_urls else ""

    return {
        "edited_ring_image_url": final_url,
        "synthesized_prompt": custom_prompt,
        "status_message": "커스텀 이미지 생성에 실패했습니다." if not final_url else "",
    }


def generate_multi_view(state: AgentState) -> dict:
    """
    최종 채택된 이미지로 다각도 생성과 배경 제거를 수행합니다.
    """
    target_image = state.get("edited_ring_image_url", "") or state.get("base_ring_image_url", "")

    logger.info(f"Extracting multi-views and applying Birefnet rembg. Target: {target_image}")

    if not target_image:
        return {
            "current_image_urls": [],
            "status_message": "다각도 생성에 사용할 입력 이미지가 없습니다.",
        }

    try:
        comfyui_payload = _build_multi_view_payload(target_image)
    except Exception as exc:
        logger.error(f"Failed to build multi-view ComfyUI payload: {exc}")
        return {
            "current_image_urls": [],
            "status_message": f"다각도용 LoadImage 주입에 실패했습니다. ({exc})",
        }

    result_urls = _sync_call_comfyui(comfyui_payload)

    return {
        "current_image_urls": result_urls,
        "status_message": "다각도 이미지 생성에 실패했습니다." if not result_urls else "",
    }
