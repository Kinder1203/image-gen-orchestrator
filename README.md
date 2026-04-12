# 반지 커스텀 다각도 3D 생성용 LLM 백엔드

LangGraph, `vLLM`, ComfyUI, Vision 검수를 조합해 반지 시안 생성부터 커스텀 수정, 다각도 추출까지 오케스트레이션하는 졸업작품용 LLM 백엔드입니다. 현재 기준 도메인은 `반지`이며, 추론 백엔드는 `vLLM OpenAI-compatible endpoint`를 기준으로 합니다. 코드 기본값은 `gemma4-e4b` 채팅·비전 모델과 `BAAI/bge-m3` 임베딩 모델이지만, 실제 계약은 환경변수에 주입된 raw 모델 문자열을 그대로 사용합니다.

## 프로젝트 개요
- `text`: 텍스트만 받아 베이스 반지를 생성하고, 사용자 승인 후 다각도 이미지를 만듭니다.
- `image_and_text`: 기존 반지 이미지를 입력받아 텍스트 수정 요청을 반영한 뒤, 사용자 승인 후 다각도 이미지를 만듭니다.
- `image_only`: 완성된 반지 이미지를 입력받아 다각도 이미지와 배경 제거 결과를 바로 만듭니다.
- 입력 이미지 가드레일과 Vision 검수는 생성 품질을 지키기 위한 안전장치이며, 시스템 오류는 보정으로 위장하지 않고 즉시 실패 처리합니다.

상세 파이프라인 계약, 액션 이름, 상태 전이, 검수 정책은 `src/llm_pipeline/README.md`를 단일 기준 문서로 봅니다.

## 개발 방식
- 이 프로젝트는 하네스 엔지니어링에 가깝게, 무거운 실제 추론 통합 테스트보다 계약과 경계면을 먼저 고정하는 방식으로 관리합니다.
- 사람용 온보딩은 루트 `README.md`, Codex 작업 규칙은 `AGENTS.md`, 파이프라인 의미 계약은 `src/llm_pipeline/README.md`, HTTP transport 계약은 `server/README.md`에 분리합니다.
- 계약이 바뀌면 `src/llm_pipeline/README.md`와 `tests.test_pipeline_contract`를 먼저 같이 맞춘 뒤 코드와 데모 스크립트를 수정합니다.
- ComfyUI JSON은 코드가 런타임에 값을 주입하는 실행 템플릿으로 취급하며, 저장소에서 직접 편집하는 운영 방식을 기준으로 하지 않습니다.

## 하네스 운영 원칙
- 1차 고정 대상은 모델 품질 평가가 아니라 계약입니다. 입력 shape 정규화, 액션 이름, 상태 이름, 체크포인터 재개 조건, ComfyUI 템플릿 shape 같은 경계면을 먼저 맞춥니다.
- `python -m unittest tests.test_pipeline_contract`는 템플릿 로더, 스키마 정규화, FastAPI wrapper, LangGraph pause/resume, RAG feeder 같은 핵심 경계면이 문서와 어긋나지 않는지 확인하는 lightweight 하네스입니다.
- 실제 `vLLM`/ComfyUI 품질, 지연 시간, 운영 환경 연결 상태까지 이 테스트 하나로 대체하지는 않습니다. 그런 부분은 `python test_run.py`나 실제 서버 연동 smoke run으로 따로 확인합니다.

## 전체 아키텍처 요약
- `src/llm_pipeline/pipelines.py`: 외부 엔트리포인트 `process_generation_request()`를 제공합니다.
- `src/llm_pipeline/agent.py`: LangGraph 상태 그래프와 휴게소 인터럽트 지점을 정의합니다.
- `src/llm_pipeline/nodes/`: 라우팅, RAG 조회, ComfyUI 생성, Vision 검수를 담당합니다.
- `src/llm_pipeline/core/`: 설정, Pydantic 스키마, vLLM OpenAI-compatible 클라이언트를 둡니다.
- `src/llm_pipeline/scripts/`: `db_feeder.py` 같은 보조 실행 스크립트를 둡니다.

## 설치
```bash
pip install -r requirements.txt
```

## 환경 변수
`.env` 또는 시스템 환경 변수로 값을 설정할 수 있습니다. 아래 블록은 로컬 데모와 문서 확인용으로 바로 쓰기 쉬운 예시이며, 모든 값이 코드 기본값과 1:1로 같다는 뜻은 아닙니다.

```env
VLLM_CHAT_BASE_URL=http://127.0.0.1:8000/v1
VLLM_CHAT_MODEL=gemma4-e4b
VLLM_CHAT_API_KEY=EMPTY
VLLM_EMBED_BASE_URL=http://127.0.0.1:8002/v1
VLLM_EMBED_MODEL=BAAI/bge-m3
VLLM_EMBED_API_KEY=EMPTY
WEBHOOK_URL=NONE
COMFYUI_URL=http://127.0.0.1:8188
VECTOR_DB_PATH=./data/chroma_db
LANGGRAPH_CHECKPOINT_DB_PATH=./data/langgraph_checkpoints.sqlite
ALLOW_VALIDATION_BYPASS=false
```

환경별로 바뀌는 계약 키는 `.env`에서 관리하고, 내부 튜닝값과 유지보수용 기본값은 `src/llm_pipeline/core/config.py`를 기준으로 관리합니다.

- `VLLM_CHAT_MODEL`, `VLLM_EMBED_MODEL`은 현재 환경에 설정된 문자열을 그대로 사용합니다. 코드 기본값처럼 전체 모델 ID를 둘 수도 있고, 로컬/배포 환경에서는 `bge-m3` 같은 alias를 둘 수도 있습니다.
- `/healthz`는 위 값을 별도 정규화하지 않고 그대로 반환하므로, 환경마다 전체 모델명 또는 alias가 그대로 보일 수 있습니다.
- `VLLM_CHAT_API_KEY`, `VLLM_EMBED_API_KEY`는 로컬 무인증 endpoint라면 `EMPTY` 예시를 쓸 수 있지만, gateway나 reverse proxy가 앞단에 있는 배포 환경에서는 실제 토큰이 필요할 수 있습니다.
- `WEBHOOK_URL=NONE`은 로컬 데모용 안전 예시입니다. 현재 코드 기본값은 실제 백엔드 endpoint를 가리킬 수 있으므로, 외부 전송을 막고 싶다면 `.env`에서 명시적으로 `NONE`으로 두는 편이 안전합니다.
- `test_run.py`는 데모 실행 시작 시 `WEBHOOK_URL`을 강제로 `NONE`으로 바꿔 외부 전송을 막습니다.

- `ALLOW_VALIDATION_BYPASS=true`는 Vision 검수 오류를 개발용으로만 우회합니다.
- `LANGGRAPH_CHECKPOINT_DB_PATH`는 HITL 상태를 저장할 SQLite 체크포인트 파일 경로입니다.
- 아래 항목은 `.env` 필수 계약이라기보다 현재 `src/llm_pipeline/core/config.py`가 관리하는 내부 튜닝 기본값 설명입니다.
- `MULTI_VIEW_VALIDATION_SAMPLE_COUNT`는 다각도 결과 중 Vision 검수에 사용할 대표 샘플 수입니다.
- `VLLM_VALIDATOR_MAX_TOKENS`는 검수 JSON 응답 길이를 짧게 제한해 반복 검수 지연을 줄입니다.
- `VLLM_PROMPT_MAX_TOKENS`는 베이스 프롬프트 보강 응답 길이를 제한합니다.

## 실행 준비
1. 채팅·비전 추론용 `vLLM` 인스턴스를 실행합니다.
2. 임베딩 전용 `vLLM` 인스턴스를 실행합니다.
3. ComfyUI를 로컬에서 실행합니다.
4. 최초 1회 또는 임베딩 모델 변경 시 벡터 DB를 다시 적재합니다.

```bash
python -m src.llm_pipeline.scripts.db_feeder
```

## 실행 방법
애플리케이션 서버나 별도 호출 코드에서는 `process_generation_request()`를 직접 사용합니다. LangGraph 그래프와 SQLite 체크포인터는 첫 요청 시점에 lazy 초기화됩니다.

```python
from src.llm_pipeline.core.schemas import PipelineRequest
from src.llm_pipeline.pipelines import process_generation_request

request = PipelineRequest(
    thread_id="demo-thread",
    action="start",
    input_type="text",
    prompt="18k 화이트골드에 얇은 곡선 각인이 들어간 커플링",
)

result = process_generation_request(request)

if result.status == "waiting_for_user":
    print(result.base_image_url)
elif result.status == "success":
    print(result.optimized_image_urls)
else:
    print(result.message)
```

로컬 수동 시연은 아래 명령으로 실행할 수 있습니다.

```bash
python test_run.py
```

`test_run.py`는 데모 전용 스크립트이며 시작 시 `WEBHOOK_URL`을 `NONE`으로 바꿔 외부 전송을 막습니다.

`thread_id`는 모든 요청에서 반드시 명시해야 합니다. 특히 `accept_base`와 `request_customization` 같은 후속 액션은 최초 `start`와 동일한 `thread_id`를 그대로 사용해야 합니다.

백엔드 연동용으로는 FastAPI wrapper도 제공합니다.

```bash
uvicorn server.app:app --host 0.0.0.0 --port 8080
```

HTTP 엔드포인트와 요청/응답 예시는 [server/README.md](server/README.md)를 기준으로 봅니다.

## 검증
문서 계약과 런타임 핵심 가정은 아래 계약 테스트로 빠르게 확인할 수 있습니다.

```bash
python -m unittest tests.test_pipeline_contract
```

이 테스트는 ComfyUI 템플릿 shape, 요청/응답 스키마, LangGraph 휴게소 상태, `vLLM`/Chroma 연결 계약을 함께 확인합니다.
전체 `vLLM`/ComfyUI 추론을 매번 다 돌리는 무거운 E2E 테스트가 아니라, 문서와 코드의 핵심 계약이 어긋나지 않았는지 먼저 확인하는 lightweight 하네스입니다.

- 문서와 직접 맞물리는 주요 검증 대상은 ComfyUI 템플릿 경로/shape, `PipelineRequest` 정규화, `/healthz`와 `/pipeline` wrapper, pause/resume 상태, RAG feeder 연결입니다.
- 실제 이미지 품질이나 외부 서비스 가용성은 이 계약 테스트 범위 밖이므로, 필요하면 `python test_run.py`나 실서버 smoke run을 추가로 수행합니다.

리소스가 많이 부족한 환경에서는 아래처럼 스키마/API smoke test만 먼저 실행해도 됩니다.

```bash
python -m unittest \
  tests.test_pipeline_contract.SchemaContractTests.test_input_type_is_normalized_from_payload_shape \
  tests.test_pipeline_contract.ApiWrapperTests.test_healthz_returns_runtime_metadata
```

## 폴더 구조
```text
.
|-- README.md
|-- AGENTS.md
|-- requirements.txt
|-- test_run.py
|-- comfyui_workflow/
|-- server/
|-- tests/
|-- input_images/
|-- output_images/
`-- src/
    `-- llm_pipeline/
        |-- README.md
        |-- agent.py
        |-- pipelines.py
        |-- core/
        |-- nodes/
        `-- scripts/
```

실행 중에는 `VECTOR_DB_PATH`, `LANGGRAPH_CHECKPOINT_DB_PATH` 설정에 따라 `data/` 같은 런타임 폴더가 생성될 수 있습니다.
`input_images/`와 `output_images/`는 로컬 데모 실행용 폴더이며, 실제 이미지 산출물은 Git에 포함하지 않습니다.

## 문서 맵
- `README.md`: 사람용 프로젝트 개요, 설치, 실행, 폴더 구조
- `server/README.md`: 백엔드 연동용 HTTP 엔드포인트 명세
- `AGENTS.md`: Codex 작업 원칙, 우선순위, 테스트, 출력 형식
- `src/llm_pipeline/README.md`: 파이프라인 계약의 단일 기준 문서
- `src/llm_pipeline/scripts/README.md`: `db_feeder.py` 운영 메모

## 사용 중 주의사항
- ComfyUI 템플릿 JSON은 저장소에서 직접 수정하지 않고 런타임에 필요한 값만 주입하는 전제로 관리합니다.
- 입력 타입, 액션, 상태 이름은 사람이 읽기 쉬운 표현보다 API 계약 이름을 기준으로 맞추는 것이 안전합니다.
- 문서 중 파이프라인 흐름 관련 세부 규칙은 루트가 아니라 `src/llm_pipeline/README.md`에서 관리합니다.
- 추론 백엔드는 `vLLM OpenAI-compatible endpoint`를 기준으로 하며, 배포 시 양자화 모델 선택은 환경변수에 주입된 모델명 책임으로 둡니다.
