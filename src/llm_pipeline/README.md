# llm_pipeline 패키지 계약

`src/llm_pipeline/`는 반지 커스텀 파이프라인의 실제 실행 패키지입니다. 이 문서는 파이프라인 흐름, 입력 계약, 상태 전이, 검수 정책의 단일 기준 문서입니다.

계약이 바뀌면 `tests.test_pipeline_contract`도 함께 맞춰 문서와 자동 검증이 같은 기준을 보도록 유지합니다.

## 계약 운영 방식
- 이 문서는 파이프라인 의미 계약의 단일 기준 문서입니다. 액션 의미, 상태 전이, 검수 정책이 바뀌면 여기부터 수정합니다.
- `tests.test_pipeline_contract`는 이 계약을 가볍게 회귀 확인하는 하네스입니다. 기본 목표는 실제 모델을 매번 끝까지 돌리는 것이 아니라, 스키마, 상태 흐름, 템플릿 shape, wrapper 경계가 문서와 어긋나지 않게 유지하는 것입니다.
- 하네스는 대략 다섯 층을 나눠 봅니다. `ExportedTemplateContractTests`는 ComfyUI 템플릿 경로/shape를, `SchemaContractTests`는 요청·응답 정규화를, `ApiWrapperTests`는 HTTP wrapper를, `PipelineRuntimeTests`는 LangGraph 재개·웹훅·fallback 동작을, `DbFeederContractTests`는 RAG 적재 계약을 확인합니다.
- 이 하네스는 실제 `vLLM`/ComfyUI 서버가 켜진 상태의 이미지 품질까지 대신 보장하지 않습니다. 그런 검증은 `python test_run.py`나 실환경 smoke run으로 별도 확인합니다.
- 루트 `README.md`는 온보딩과 실행 방법만 요약하고, `server/README.md`는 HTTP transport 계약만 설명합니다. 세부 시나리오 의미를 다른 문서에 다시 길게 복제하지 않습니다.
- `src/llm_pipeline/scripts/README.md`는 운영 스크립트 메모만 담당하며, 파이프라인 본문 계약은 이 문서를 기준으로 삼습니다.

## 패키지 역할
- `pipelines.py`: 외부 엔트리포인트 `process_generation_request()`와 동기 응답 포맷을 담당합니다.
- `agent.py`: LangGraph 상태 그래프, 인터럽트 지점, 조건부 분기를 정의합니다.
- `core/`: 설정(`config.py`), 공용 스키마(`schemas.py`), vLLM OpenAI-compatible 추론 클라이언트(`vllm_client.py`)를 둡니다.
- `nodes/`: 라우팅, RAG, ComfyUI 생성, Vision 검수를 수행합니다.
- `scripts/`: DB 적재 같은 보조 운영 스크립트를 둡니다.

HTTP wrapper는 패키지 밖 [server/app.py](../../server/app.py) / [server/api.py](../../server/api.py)에서 관리합니다.

## 추론 백엔드 전제
- 채팅·비전 추론은 `VLLM_CHAT_*` 설정을 사용하는 `vLLM` 인스턴스 1개를 기준으로 합니다.
- 임베딩은 `VLLM_EMBED_*` 설정을 사용하는 별도 `vLLM` 인스턴스 1개를 기준으로 합니다.
- HITL 체크포인터는 기본적으로 `LANGGRAPH_CHECKPOINT_DB_PATH`의 SQLite 파일을 사용합니다.
- 그래프와 체크포인터는 모듈 import 시점이 아니라 첫 `process_generation_request()` 호출 시점에 초기화됩니다.
- validator 호출은 stateless 1회 요청 구조를 유지합니다.
- 검수 병목 완화를 위해 validator와 prompt enhancement는 각각 짧은 `max_tokens` 제한을 둡니다.

## 사용자 표현과 API 액션 매핑
- 시작: `start`
- 승인, 합격, 베이스 수락: `accept_base`
- 각인 수정, 커스텀 요청, 재수정: `request_customization`

UI나 데모에서는 사람 친화적인 표현을 써도 되지만, 실제 계약 이름은 위 액션 값을 기준으로 맞춥니다.

## 입력 계약
### 1. `PipelineRequest`

| 필드 | 설명 |
| --- | --- |
| `thread_id` | LangGraph 세션 스레드 ID. 모든 요청에서 명시해야 하며 후속 액션은 최초 요청과 동일한 값을 사용해야 함 |
| `action` | `start`, `accept_base`, `request_customization` 중 하나 |
| `input_type` | 외부에서 받는 입력 타입. 내부에서는 canonical 값으로 정규화됨 |
| `prompt` | 반지 설명 또는 초기 수정 지시 |
| `image_url` | 기존 반지 시안 또는 ComfyUI 업로드 이미지명 |
| `customization_prompt` | 후속 수정 요청 시 사용할 프롬프트 |

### 2. 외부 허용 입력 타입과 내부 canonical 값
외부에서 허용하는 값은 아래 다섯 가지입니다.

- `text`
- `image`
- `modification`
- `image_only`
- `image_and_text`

내부 canonical 값은 아래 세 가지뿐입니다.

- `text`
- `image_only`
- `image_and_text`

정규화는 raw label이 아니라 payload shape 기준으로 이뤄집니다.

| 조건 | 내부 canonical 값 |
| --- | --- |
| `image_url` 없음 + `prompt` 있음 | `text` |
| `image_url` 있음 + `prompt` 없음 | `image_only` |
| `image_url` 있음 + `prompt` 있음 | `image_and_text` |

`image`, `modification`은 legacy 입력 명칭으로만 허용되며, 실제 동작은 위 payload shape 정규화 결과를 따릅니다.

### 3. 액션별 필수 조건
- `action="start"`: `prompt` 또는 `image_url` 중 하나 이상이 반드시 있어야 합니다.
- `action="accept_base"`: `thread_id`만 있으면 됩니다.
- `action="request_customization"`: `thread_id`와 비어 있지 않은 `customization_prompt`가 반드시 있어야 합니다.
- 빈 문자열 thread ID는 허용하지 않습니다.

## 상태 계약
`process_generation_request()`는 모든 액션에 대해 동기적으로 `PipelineResponse`를 반환합니다.

| 상태 | 의미 | 주요 필드 | 다음 액션 |
| --- | --- | --- | --- |
| `waiting_for_user` | 베이스 반지 시안 검토 대기 | `base_image_url`, `message` | `accept_base` 또는 `request_customization` |
| `waiting_for_user_edit` | 커스텀 반영본 검토 대기 | `base_image_url`, `message` | `accept_base` 또는 `request_customization` |
| `success` | 최종 다각도 이미지 생성 완료 | `optimized_image_urls`, `message` | 없음 |
| `failed` | 생성 또는 검수 실패 | `message`, 필요 시 `base_image_url` | 없음 |

보조 규칙:
- `waiting_for_user`는 1차 베이스 반지 휴게소입니다.
- `waiting_for_user_edit`는 수정 반영본 확인 휴게소입니다.
- `success`일 때만 `optimized_image_urls`가 채워집니다.
- `failed`는 Vision 검수 실패뿐 아니라 ComfyUI 시스템 오류도 포함합니다.

## Canonical Scenarios
### 1. `text`
1. `start` + `prompt`
2. `router -> rag -> generate_base_image -> validate_base_image`
3. 검수 통과 시 `waiting_for_user`
4. `accept_base`면 `generate_multi_view -> validate_rembg -> success`
5. `request_customization`이면 `edit_image -> validate_edited_image -> waiting_for_user_edit`
6. 2차 휴게소에서 다시 `accept_base` 또는 `request_customization`

### 2. `image_and_text`
1. `start` + `image_url` + `prompt`
2. `router -> validate_input_image -> edit_image -> validate_edited_image`
3. 검수 통과 시 `waiting_for_user_edit`
4. `accept_base`면 `generate_multi_view -> validate_rembg -> success`
5. `request_customization`이면 다시 `edit_image`

`image_and_text`의 시작 요청에서는 초기 `prompt`가 첫 커스텀 지시로 사용됩니다.

### 3. `image_only`
1. `start` + `image_url`
2. `router -> validate_input_image -> generate_multi_view -> validate_rembg`
3. 통과 시 바로 `success`
4. `validate_input_image`가 `repair_required`이면 내부적으로만 `edit_image -> validate_edited_image -> generate_multi_view`
5. 이 내부 보정은 사용자 휴게소를 만들지 않습니다.

## 검수 및 실패 정책
### 입력 이미지 가드레일
`validate_input_image`는 아래 세 결과만 반환합니다.

- `pass`: 원래 시나리오대로 진행
- `repair_required`: 배경 대비가 부족해 내부 보정 필요
- `system_error`: 이미지 다운로드 실패, Vision LLM 오류 등으로 즉시 실패

시스템 오류는 배경 보정처럼 위장하지 않습니다.

### 베이스/수정본 검수
- `validate_base_image`는 요청 반영 여부뿐 아니라 배경이 반지 재질과 충분히 대비되는지도 함께 검수합니다.
- `validate_edited_image`는 커스텀 반영 여부를 검수합니다.
- 생성 시스템 오류가 이미 발생했다면 Vision 검수로 넘기지 않고 즉시 실패 처리합니다.

### 다각도 검수
- `validate_rembg`는 전체 다각도를 모두 검사하지 않고 `MULTI_VIEW_VALIDATION_SAMPLE_COUNT` 개의 대표 샘플만 검사합니다.
- 최종 성공은 `is_valid=True`이고 결과 이미지가 1장 이상 있을 때만 인정합니다.

### 개발용 우회
- `ALLOW_VALIDATION_BYPASS=true`일 때만 Vision 검수 오류를 개발용으로 우회할 수 있습니다.

## vLLM 설정
주요 설정 키는 아래와 같습니다.

- `VLLM_CHAT_BASE_URL`
- `VLLM_CHAT_MODEL`
- `VLLM_CHAT_API_KEY`
- `VLLM_EMBED_BASE_URL`
- `VLLM_EMBED_MODEL`
- `VLLM_EMBED_API_KEY`

`VLLM_CHAT_API_KEY`, `VLLM_EMBED_API_KEY`는 로컬 무인증 endpoint라면 `EMPTY` 예시를 쓸 수 있지만, 실제 배포에서 gateway나 reverse proxy가 앞단에 있으면 유효한 토큰이 필요할 수 있습니다.
`VLLM_CHAT_MODEL`은 배포 환경에서 실제 서빙하는 Gemma 4 계열 모델 alias를 주입합니다. 양자화 여부나 `E4B`/다른 variant 선택은 코드가 아니라 배포 설정 책임입니다.
`VLLM_EMBED_MODEL`도 같은 방식으로 raw 설정 문자열을 그대로 사용합니다. 코드 기본값은 `BAAI/bge-m3`지만, 실제 로컬/배포 환경에서는 `bge-m3` 같은 alias를 그대로 둘 수 있습니다.
토큰 제한, 타임아웃, 재시도 상한, RAG 조회 개수, 기본 프롬프트 보강값은 환경 정보가 아니라 내부 튜닝값이므로 `src/llm_pipeline/core/config.py` 기본값으로 관리합니다.

## ComfyUI 템플릿 정책
현재 저장소에서 사용하는 ComfyUI 템플릿 파일은 아래 세 개입니다.

- `comfyui_workflow/image_z_image_turbo (2).json`
- `comfyui_workflow/image_qwen_image_edit_2509.json`
- `comfyui_workflow/templates-1_click_multiple_character_angles-v1.0 (3) (1).json`

운영 원칙:
- 저장소의 JSON 파일은 직접 수정 대상이 아니라 실행용 템플릿입니다.
- Python 런타임이 템플릿을 읽은 뒤, 메모리상에서 필요한 값만 주입해 ComfyUI `/prompt`로 전송합니다.
- 텍스트 프롬프트는 placeholder 치환으로 주입합니다.
- 입력 이미지는 `LoadImage.inputs.image`를 교체하는 방식으로 주입합니다.
- 중간 생성 결과가 output URL이면 다음 단계 전에 ComfyUI input 파일로 다시 브리지합니다.
- 템플릿 구조가 예상과 다르면 조용히 진행하지 않고 즉시 실패 처리합니다.

## 웹훅과 HITL 흐름
- 기본 호출 단위는 `process_generation_request()`의 동기 응답입니다.
- 후속 액션(`accept_base`, `request_customization`)은 저장된 checkpoint가 남아 있는 `thread_id`에서만 유효합니다.
- 기본 SQLite 체크포인터를 쓰면 같은 checkpoint DB 파일을 유지하는 한 프로세스 재시작 뒤에도 pause 상태를 이어갈 수 있습니다.
- `langgraph-checkpoint-sqlite`가 빠진 환경에서는 경고 후 `MemorySaver`로 임시 fallback합니다.
- `success`가 나왔고 `WEBHOOK_URL`이 설정되어 있으며 `NONE`이 아니면, 추가로 결과를 웹훅으로 POST합니다.
- 현재 성공 웹훅 payload는 `status`, `thread_id`, `images`, `prompt_used`를 포함합니다.
- `waiting_for_user`, `waiting_for_user_edit`, `failed`는 현재 웹훅을 보내지 않고 동기 응답으로만 반환됩니다.

## 벡터 DB 재색인
- 임베딩 모델이 바뀌면 기존 Chroma 인덱스는 재사용하지 않습니다.
- 아래 명령을 다시 실행해 벡터 DB를 재적재합니다.

```bash
python -m src.llm_pipeline.scripts.db_feeder
```

## 빠른 호출 예시
```python
from src.llm_pipeline.core.schemas import PipelineRequest
from src.llm_pipeline.pipelines import process_generation_request

request = PipelineRequest(
    thread_id="demo-thread",
    action="start",
    input_type="text",
    prompt="18k 화이트골드에 얇은 곡선 각인이 들어간 커플링",
)

response = process_generation_request(request)
print(response.status)
```
