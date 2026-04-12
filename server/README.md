# Server README

이 문서는 백엔드가 호출할 HTTP 레이어만 설명합니다.  
파이프라인 흐름, 액션 의미, 상태 전이 세부 규칙은 [src/llm_pipeline/README.md](</c:/Users/user/Desktop/project/src/llm_pipeline/README.md:1>)를 기준으로 봅니다.

## 개요

- 서비스 실행: `uvicorn server.app:app --host 0.0.0.0 --port 8080`
- 기본 문서: `/docs`
- OpenAPI JSON: `/openapi.json`

이 서비스는 프론트엔드가 직접 호출하는 용도가 아니라, 백엔드가 내부 API처럼 호출하는 전제를 둡니다.

## 엔드포인트

### `GET /`

간단한 메타 정보 확인용입니다.

예시 응답:

```json
{
  "service": "ring-llm-pipeline",
  "status": "ok",
  "docs_url": "/docs",
  "openapi_url": "/openapi.json"
}
```

### `GET /healthz`

프로세스 기동 여부와 현재 모델/체크포인트 경로를 확인합니다.

예시 응답:

```json
{
  "status": "ok",
  "service": "ring-llm-pipeline",
  "chat_model": "gemma4-e4b",
  "embed_model": "bge-m3",
  "checkpoint_path": "./data/langgraph_checkpoints.sqlite"
}
```

`chat_model`과 `embed_model`은 서버가 현재 설정에서 읽은 raw 문자열을 그대로 반환합니다.  
따라서 환경에 따라 `BAAI/bge-m3` 같은 전체 모델 ID가 보일 수도 있고, `bge-m3` 같은 배포 alias가 보일 수도 있습니다.

### `POST /pipeline`

파이프라인 시작과 HITL 재개를 모두 처리하는 단일 엔드포인트입니다.

요청 바디는 [PipelineRequest](</c:/Users/user/Desktop/project/src/llm_pipeline/core/schemas.py:23>)와 같고, 응답 바디는 [PipelineResponse](</c:/Users/user/Desktop/project/src/llm_pipeline/core/schemas.py:63>)와 같습니다.

## 요청 필드

| 필드 | 타입 | 필수 | 설명 |
| --- | --- | --- | --- |
| `thread_id` | string | 예 | 같은 세션을 이어가기 위한 ID |
| `action` | string | 예 | `start`, `accept_base`, `request_customization` |
| `prompt` | string \| null | 조건부 | 시작 요청의 텍스트 설명 |
| `image_url` | string \| null | 조건부 | 기존 반지 이미지 URL 또는 ComfyUI 업로드 이미지명 |
| `customization_prompt` | string \| null | 조건부 | 재수정 요청 프롬프트 |
| `input_type` | string | 아니오 | 외부에서 보내도 되지만 서버에서 payload shape 기준으로 정규화 |

## 요청 예시

### 1. 텍스트로 시작

```json
{
  "thread_id": "2c1bfe4d-8b8b-4d50-8cd0-5f72a0f0d9d9",
  "action": "start",
  "prompt": "18k 화이트골드 얇은 커플링",
  "image_url": null
}
```

### 2. 이미지 수정으로 시작

```json
{
  "thread_id": "2c1bfe4d-8b8b-4d50-8cd0-5f72a0f0d9d9",
  "action": "start",
  "prompt": "안쪽에 forever 각인 추가",
  "image_url": "uploaded_ring.png"
}
```

### 3. 베이스 시안 승인

```json
{
  "thread_id": "2c1bfe4d-8b8b-4d50-8cd0-5f72a0f0d9d9",
  "action": "accept_base"
}
```

### 4. 재수정 요청

```json
{
  "thread_id": "2c1bfe4d-8b8b-4d50-8cd0-5f72a0f0d9d9",
  "action": "request_customization",
  "customization_prompt": "보석은 빼고 안쪽 각인만 남겨줘"
}
```

## 응답 상태

| `status` | 의미 | 백엔드 처리 |
| --- | --- | --- |
| `waiting_for_user` | 베이스 시안 검토 대기 | `base_image_url` 저장 후 승인/수정 UI 제공 |
| `waiting_for_user_edit` | 수정 반영본 검토 대기 | `base_image_url` 저장 후 승인/재수정 UI 제공 |
| `success` | 최종 다각도 결과 완료 | `optimized_image_urls` 저장 후 완료 처리 |
| `failed` | 실패 | `message`를 사용자용 또는 운영 로그로 전달 |

## 응답 예시

### `waiting_for_user`

```json
{
  "status": "waiting_for_user",
  "optimized_image_urls": [],
  "message": "베이스 반지 시안이 생성되었습니다. 승인하거나 수정 요청을 보내주세요.",
  "base_image_url": "http://127.0.0.1:8188/view?filename=ComfyUI_00001_.png&type=output"
}
```

### `success`

```json
{
  "status": "success",
  "optimized_image_urls": [
    "http://127.0.0.1:8188/view?filename=final_01.png&type=output",
    "http://127.0.0.1:8188/view?filename=final_02.png&type=output"
  ],
  "message": "최종 결과가 생성되었습니다.",
  "base_image_url": "http://127.0.0.1:8188/view?filename=edited.png&type=output"
}
```

## 백엔드 연동 규칙

1. 프론트엔드는 이 서비스가 아니라 백엔드를 호출합니다.
2. 백엔드는 사용자별 `thread_id`를 저장하고 후속 액션에도 같은 값을 재사용합니다.
3. 이미지 업로드가 있는 경우 백엔드는 `image_url`에 접근 가능한 URL이나 ComfyUI 업로드 이미지명을 전달합니다.
4. `waiting_for_user`와 `waiting_for_user_edit`는 동기 응답으로만 오므로, 백엔드는 그 응답을 기준으로 화면 상태를 제어하면 됩니다.
5. `success` 시 최종 결과는 동기 응답으로도 오고, 환경에 따라 별도 웹훅 전송도 발생할 수 있습니다.

## 실행 순서

1. `vLLM` chat 서버 기동
2. `vLLM` embedding 서버 기동
3. ComfyUI 기동
4. `python -m src.llm_pipeline.scripts.db_feeder`
5. `uvicorn server.app:app --host 0.0.0.0 --port 8080`
