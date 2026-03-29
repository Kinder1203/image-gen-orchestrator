# 👟 커스텀 신발 다각도(Multi-Angle) 3D 생성 서비스 LLM (최종 통합본)

> **업데이트 포인트 (2026.03.29):** 모든 외부 API 의존성(Qwen 등)을 걷어내고 내 컴퓨터(Local) 환경에서 LangGraph + Ollama + ComfyUI가 직접 맞물려 돌아가도록 완전히 재작성되었습니다. 단일 이미지가 아닌, TRELLIS 3D 렌더링에 최적화된 **'다각도(앞,옆,뒤) 3중 이미지 배열'** 반환 아키텍처로 진화했습니다.

---

## 1. 노드 다이어그램 및 설계 의도 (Architecture)

우리의 백엔드는 오직 LangGraph를 뼈대로 삼은 오케스트레이터입니다. 크게 5단계로 나뉩니다.

*   `Entrypoint (pipelines.py)`: FastAPI 등 메인 서버가 `PipelineRequest` 객체를 쏴주면, 이 파이프라인이 구동을 시작함.
*   **[Step 1] `Router`**: 이미지 업로드인지 문자 기반 프롬프트 생성인지 Ollama (JSON format)가 의도를 파악하고 양갈래 길로 나눔.
*   **[Step 2] `ShoeHybridRAG`**: 사용자가 "스니커즈" 라고 대충 말해도, Qwen 모델의 '다각도 생성 필수 프롬프트 지침서(Azimuth 제어)'와 '신발 소재 사전'을 DB에서 꺼내와 똑똑하게 살을 붙임.
*   **[Step 3] `Synthesizer (ComfyUI Qwen)`**: Ollama가 DB 꿀팁을 활용해 완벽한 생김새 묘사 프롬프트를 짜내면, 백그라운드에 켜져 있는 나만의 ComfyUI(Qwen 다각도 템플릿) 서버에 일감을 밀어 넣어서 3장의 이미지(`front.png, side.png, back.png`)를 받아옴.
*   **[Step 4] `Preprocessor (RemBG)`**: 방금 얻은 3장의 이미지(혹은 유저가 직접 올린 사진)에서 복잡한 배경을 파이썬 스크립트(`rembg`)로 잘라내고, 순백색의 도화지 정중앙에 합성. (TRELLIS 필수 구동 조건)
*   **[Step 5] `Validator`**: 방금 누끼를 딴 3장의 사진을 **Ollama Vision(LLaVA 등)** 이 전부 확인하여, "배경이 완전 흰색인가? 잘린 데 없이 깔끔한가?"를 검사. 합격판정 시 배열 리턴!

---

## 2. 🚨 코드 로직 정합성 평가 및 잔존 결함 (Troubleshooting)

현재 작성된 `src/llm_pipeline` 모듈은 뼈대(Skeleton) 수준의 논리 구조로, 실제 프로덕션 서버로 가기 전에 **아래의 구조적 결함(Flaws) 3가지를 반드시 메워야(수정해야)** 합니다.

### 결함 1: ComfyUI 서버 통신은 `비동기 WebSocket`이어야 함 (현재는 모의)
*   **위치:** `nodes/synthesizer.py`
*   **배경:** 현재 `requests.post("http://127.0.0.1:8188/prompt")` 만 코딩되어 있습니다. 하지만 ComfyUI는 구조상 "응, 큐(Queue)에 등록했어" 라고만 하고 연결을 끊어버립니다(비동기). 
*   **해결 과제:** `websockets` 라이브러리를 통해 `ws://127.0.0.1:8188/ws`에 붙은 다음, 해당 `prompt_id`의 렌더링이 `executed`(완료) 신호를 보낼 때까지 `while`문으로 기다리다가 파일 이름을 빼오는 로직을 추가 작성해야 합니다.

### 결함 2: Ollama Vision 노드의 Base64 인코딩 누락
*   **위치:** `nodes/validator.py`
*   **배경:** 현재 Vision LLM을 호출하도록 설정은 해 두었지만, 로컬 디렉터리에 저장된 `.png` 파일 이미지를 LLaVA에게 읽히려면 `Base64` 문자열로 변환해서 `HumanMessage` 배열에 끼워 넣어야 합니다. 코드에 `NotImplementedError`로 남겨 두었으니 해당 구간의 파일 I/O를 짜야 합니다.

### 결함 3: RAG 피더의 진짜 디비 연결 누락
*   **위치:** `scripts/db_feeder.py`
*   **배경:** Qwen 다각도 제어 가이드 텍스트 데이터 딕셔너리는 잘 만들어져 있으나, 정작 KuzuDB나 Chroma로 밀어 넣는 라이브러리 `add_texts()` 코드가 주석 처리되어 있습니다. `SpeakNode`의 Kuzu 코드를 가져와서 Import만 시켜주시면 작동합니다.

---

## 3. 🚀 실행 가이드라인 (How to Run)

모든 작업이 로컬에 독립적으로 엮였습니다. 아래 순서대로 구동 환경을 맞춰 주십시오.

### Step 1: 환경 세팅 및 의존성 설치
1. 명령 프롬프트(CLI)를 열어, 백엔드 서버 디렉터리로 들어갑니다.
2. `pip install -r src/llm_pipeline/requirements.txt` 로 필수 라이브러리를 한방에 설치합니다. (NVIDIA GPU 환경인 경우 `pytorch`가 CUDA 버전에 맞게 깔려있는지 확인하세요)
3. `.env` 파일을 만들고 아래처럼 세팅합니다.
   ```env
   OLLAMA_MODEL=llama3  
   OLLAMA_VISION_MODEL=llava
   OLLAMA_BASE_URL=http://localhost:11434
   ```

### Step 2: 초기화 및 엔진 가동
1. **Ollama 켜두기:** 터미널을 열고 `ollama run llama3`, `ollama run llava` 한 번씩 쳐서 로컬 언어/시각 뇌를 활성화해 둡니다.
2. **Qwen DB 주입:** `python src/llm_pipeline/scripts/db_feeder.py`를 실행해서 신발 지식 검색 구조를 만들어 놓으세요. (결함 3번 해결 필요)
3. **ComfyUI 켜두기:** 컴퓨터 다른 쪽에 ComfyUI 설치 폴더의 `run_nvidia_gpu.bat`을 실행해두고, Qwen 다각도 JSON 템플릿을 열어두세요 (보통 `127.0.0.1:8188` 모드로 대기 중이어야 파이프라인이 접속합니다).

### Step 3: 백엔드 서버에서 파이프라인 호출
본 파이프라인은 진입점(Entrypoint)이 매우 깔끔하게 포장되어 있습니다. 향후 FastAPI의 `app.py` 라우팅 구간 안에서 이렇게만 치시면 코드가 빙글빙글 돌아갑니다.

```python
from llm_pipeline.pipelines import process_generation_request
from llm_pipeline.core.schemas import PipelineRequest

# 1. 사용자 입력을 Pydantic 구조로 포장
req_box = PipelineRequest(
    input_type="text",
    prompt="청바지에 어울리는 하늘색 매쉬소재 런닝화"
)

# 2. 버튼 하나로 RAG -> Ollama -> ComfyUI -> Rembg -> Vision Validator 끝!
result = process_generation_request(req_box)

# 3. 만약 5단계를 통과하고 true가 떴다면?
if result.status == "success":
    print("완벽히 분리된 배경의 3장 이미지:", result.optimized_image_urls)
    # >>> 이제 이 3장의 리스트를 팀원의 TRELLIS 3D서버 "다각도 지원 API"로 날리면 됩니다!
```