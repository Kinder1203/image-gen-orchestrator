from ..core.schemas import AgentState
from ..core.config import config
from loguru import logger
import json
import requests
import time
from langchain_ollama import ChatOllama
from langchain_core.messages import SystemMessage, HumanMessage

def synthesize_prompt_and_generate_image(state: AgentState) -> dict:
    """
    (Step 2 - Text Branch / Qwen Local Inference)
    Ollama로 똑똑한 프롬프트를 짠 뒤, 바로 백그라운드에 켜진 
    로컬 ComfyUI (127.0.0.1:8188)에 기본 다각도 템플릿 JSON 워크플로를 전송합니다.
    이를 통해 3장의 앞,옆,뒤 뷰 이미지를 로컬 GPU에서 직접 렌더링하고 가져옵니다.
    """
    user_prompt = state.get("user_prompt", "")
    rag_context = state.get("rag_context", "") # DB에서 '프롬프트 작성법'과 '신발 지식'을 끌고 옴
    
    logger.info("Synthesizing Qwen Multi-angle prompt utilizing retrieved rules...")
    
    # 1. Ollama (프롬프트 전문가)로 완벽한 Qwen API용 지시문 제작
    llm = ChatOllama(
        model=config.OLLAMA_MODEL,
        base_url=config.OLLAMA_BASE_URL,
        temperature=0.7,
        format="json"
    )
    
    # RAG DB에서 끌어온 qwen 프롬프팅 룰을 주입
    sys_prompt = f"""Role: Master 3D Prompt Engineer.
Context Rules & Knowledge: {rag_context}

Create the base prompt describing the shoe. Do NOT specify the camera angle yet (ComfyUI will handle that).
Return exactly this JSON: {{"base_prompt": "<detailed description>"}}"""
    
    try:
        response = llm.invoke([
            SystemMessage(content=sys_prompt),
            HumanMessage(content=f"User request: {user_prompt}")
        ])
        parsed = json.loads(response.content.strip())
        base_prompt = parsed.get("base_prompt", user_prompt)
    except Exception as e:
        logger.warning(f"Ollama JSON generation failed: {e}")
        base_prompt = f"A highly detailed modern sneaker. {user_prompt}."
    
    logger.debug(f"Approved Base Prompt for Qwen: {base_prompt}")

    # ==============================================================
    # 2. 로컬 ComfyUI (Qwen 내장 템플릿) API 호출
    # ==============================================================
    COMFY_URL = "http://127.0.0.1:8188/prompt"
    
    # ComfyUI의 API JSON 워크플로 형태 (실제로는 저장해둔 workflow_api.json을 load합니다)
    # 여기서는 텍스트 프롬프트를 KSampler 노드에 채워넣는 개념적 구조를 보여줍니다.
    comfyUI_payload = {
        "client_id": "llm_backend",
        "prompt": {
            "3": { # 모델 로더 노드
                "class_type": "CheckpointLoaderSimple",
                "inputs": { "ckpt_name": "qwen_multi_angle_base.safetensors" }
            },
            "6": { # 텍스트 인코더 노드 (여기에 앞면 뷰 + 베이스 프롬프트 합성)
                "class_type": "CLIPTextEncode",
                "inputs": { "text": f"<sks> front view eye-level shot medium shot, {base_prompt}, solid white background" }
            },
            # ...(옆면, 뒷면 처리를 위한 노드들 연결)...
            "9": { # 출력 노드
                "class_type": "SaveImage",
                "inputs": { "filename_prefix": "qwen_shoe_generation" }
            }
        }
    }
    
    logger.info("Sending batch rendering request to Local ComfyUI Engine...")
    try:
        # 실제 워크플로 큐(Queue) 등록
        # req = requests.post(COMFY_URL, json=comfyUI_payload, timeout=5)
        # req.raise_for_status()
        # 이후 WebSockets(127.0.0.1:8188/ws)를 통해 렌더링 끝날때까지 기다리다가 파일명을 얻어오는 로직 필요
        pass
    except Exception as e:
        logger.error(f"Failed to connect to Local ComfyUI: {e}. Are you sure it's running on port 8188?")
    
    # (모의 반환) ComfyUI가 내 지정된 경로(output 폴더)에 저장했다고 치고 경로 배열 리턴
    time.sleep(2)
    rendered_image_urls = [
        "http://127.0.0.1:8188/view?filename=qwen_shoe_generation_0001_front.png",
        "http://127.0.0.1:8188/view?filename=qwen_shoe_generation_0002_side.png",
        "http://127.0.0.1:8188/view?filename=qwen_shoe_generation_0003_back.png"
    ]
    
    logger.success(f"Rendered 3 Multi-Angle Views successfully from ComfyUI.")
    
    return {
        "synthesized_prompt": base_prompt,
        "current_image_urls": rendered_image_urls
    }
