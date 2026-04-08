# Ring Pipeline Architecture

반지 커스텀 파이프라인의 실제 실행 흐름을 정리한 문서입니다. 현재 구현은 `gemma4:26b` + LangGraph + ComfyUI 조합을 기준으로 동작합니다.

## Canonical Scenarios

### 1. `text`
- `router -> rag -> generate_base_image -> validate_base_image`
- 검수 성공 시 `wait_for_user_approval`
- `accept_base` 선택 시 `generate_multi_view -> validate_rembg -> end`
- `request_customization` 선택 시 `edit_image -> validate_edited_image -> wait_for_edit_approval`
- 2차 휴게소에서 `accept_base` 면 `generate_multi_view`
- 2차 휴게소에서 `request_customization` 이면 다시 `edit_image`

### 2. `image_and_text`
- `router -> validate_input_image -> edit_image -> validate_edited_image`
- 검수 성공 시 `wait_for_edit_approval`
- `accept_base` 선택 시 `generate_multi_view -> validate_rembg -> end`
- `request_customization` 선택 시 다시 `edit_image`

### 3. `image_only`
- `router -> validate_input_image -> generate_multi_view -> validate_rembg -> end`
- 기본적으로 interrupt 없음
- 입력 이미지 가드레일이 `repair_required` 이면 내부적으로만 `edit_image -> validate_edited_image -> generate_multi_view`
- 이 내부 보정은 사용자 휴게소를 만들지 않음

## Guardrail Semantics

`validate_input_image` 는 아래 3가지 결과만 반환합니다.

- `pass`: 원래 시나리오대로 진행
- `repair_required`: 배경 대비가 부족해서 내부 보정 필요
- `system_error`: 다운로드 실패, Vision LLM 오류 등 시스템 문제로 즉시 실패

즉 내부 `edit_image` 보정은 `배경 대비 불량` 일 때만 발생합니다. 시스템 오류를 배경 수정처럼 처리하지 않습니다.

## Request Contract

외부 입력으로는 `text`, `image`, `modification`, `image_only`, `image_and_text` 를 모두 받습니다. 내부 canonical 값은 `text`, `image_only`, `image_and_text` 입니다.

정규화 규칙은 payload shape 우선입니다.

- `image_url` 없음 + `prompt` 있음: `text`
- `image_url` 있음 + `prompt` 없음: `image_only`
- `image_url` 있음 + `prompt` 있음: `image_and_text`

추가 제약은 다음과 같습니다.

- `action="start"` 는 `prompt` 나 `image_url` 중 하나가 반드시 있어야 함
- `action="request_customization"` 는 `customization_prompt` 가 반드시 있어야 함

## ComfyUI Integration

저장소의 ComfyUI JSON 파일은 `repo에서 직접 수정하는 대상` 이 아니라 `실행용 템플릿` 입니다. 현재 구현은 ComfyUI API format JSON 을 읽은 뒤, 메모리상에서만 필요한 값을 주입해 `/prompt` 로 전송합니다.

- base template: `image_z_image_turbo (2).json` 우선 사용, 없으면 `image_z_image_turbo.json` fallback
- edit template: `image_qwen_image_edit_2509.json`
- multi-view template: `templates-1_click_multiple_character_angles-v1.0 (3).json`

런타임 주입 규칙:

- 텍스트 placeholder 는 문자열 치환으로 주입
- 입력 이미지는 `LoadImage.inputs.image` 를 교체
- edit workflow 는 `LoadImage` 가 정확히 1개여야 함
- multi-view workflow 는 `_meta.title == "Load Character Image"` 인 `LoadImage` 를 우선 사용
- 후보가 애매하면 조용히 진행하지 않고 즉시 실패

## Failure Policy

- ComfyUI `/prompt` 또는 `/history` 오류는 `generation_result=system_error` 로 즉시 실패
- Vision 검수 오류도 기본은 실패
- `ALLOW_VALIDATION_BYPASS=true` 일 때만 개발용 우회 허용
- 최종 성공은 `is_valid=True` 이고 결과 이미지가 1장 이상 있을 때만 인정
