# Ring Customization Pipeline Architecture

이 문서는 현재 저장소의 반지 커스텀 파이프라인을 기준으로 LangGraph 오케스트레이션과 ComfyUI 연동 방식을 설명합니다.

## 시나리오별 주 흐름
### 1. 텍스트 온리
- `router -> rag -> generate_base_image -> validate_base_image`
- `validate_base_image` 는 사용자 요청 반영 여부와 보색/고대비 배경 적합성을 함께 검수
- 검수 성공 시 `wait_for_user_approval`
- `accept_base` 이면 `generate_multi_view -> validate_rembg -> end`
- `request_customization` 이면 `edit_image -> validate_edited_image -> wait_for_edit_approval`
- 2차 휴게소에서 `accept_base` 이면 `generate_multi_view -> validate_rembg -> end`
- 2차 휴게소에서 `request_customization` 이면 다시 `edit_image`

### 2. 이미지 기반 커스텀
- `router -> validate_input_image -> edit_image -> validate_edited_image`
- 검수 성공 시 `wait_for_edit_approval`
- `accept_base` 이면 `generate_multi_view -> validate_rembg -> end`
- `request_customization` 이면 다시 `edit_image`

### 3. 다각도 즉시 추출
- `router -> validate_input_image -> generate_multi_view -> validate_rembg -> end`
- 기본적으로 휴게소 없음
- 다만 `validate_input_image` 가 `배경 대비 문제` 로 실패하면 내부 보정용으로 `edit_image -> validate_edited_image -> generate_multi_view` 를 타고, 이 경로에서도 사용자 interrupt 는 만들지 않습니다.
- `validate_input_image` 가 이미지 다운로드 실패나 Vision 검수 오류 같은 `시스템 오류` 로 실패하면 내부 보정을 시도하지 않고 즉시 실패 처리합니다.

## 내부 가드레일
- 시나리오 1의 `validate_base_image` 와 시나리오 2/3의 `validate_input_image` 는 같은 보색/고대비 배경 원칙을 공유합니다.
- `validate_input_image` 는 시나리오 2/3 에서만 동작합니다.
- `validate_input_image` 의 결과는 `pass`, `repair_required`, `system_error` 세 가지 의미로 해석합니다.
- 시나리오 2 에서는 보정 edit 가 발생해도 최종적으로 `wait_for_edit_approval` 로 갑니다.
- 시나리오 3 에서는 보정 edit 가 발생해도 곧바로 `generate_multi_view` 로 복귀합니다.

## 요청/응답 계약
- 입력 타입은 외부에서 `text`, `image`, `modification`, `image_only`, `image_and_text` 를 허용합니다.
- 내부 canonical 값은 `text`, `image_only`, `image_and_text` 입니다.
- `action=start` 는 `prompt` 또는 `image_url` 중 하나 이상이 필요합니다.
- `action=request_customization` 는 비어 있지 않은 `customization_prompt` 가 필요합니다.
- 휴게소 상태는 `waiting_for_user`, `waiting_for_user_edit` 두 종류입니다.
- 버튼 의미는 고정입니다.
  - `합격 -> accept_base`
  - `각인 수정/재수정 -> request_customization`

## ComfyUI 연동 원칙
- JSON 파일은 ComfyUI 에서 export 한 artifact 로 취급합니다.
- 런타임은 workflow JSON 을 읽은 뒤 메모리상 객체만 수정합니다.
- prompt 는 기존 텍스트 치환 방식으로 주입합니다.
- 이미지 입력은 `LoadImage.widgets_values[0]` 를 런타임에서 교체합니다.
- edit workflow 는 `LoadImage` 노드가 정확히 1개여야 합니다.
- multi-view workflow 는 title 이 `Load Character Image` 인 `LoadImage` 노드를 우선 사용하고, 없으면 `LoadImage` 가 1개일 때만 사용합니다.
- 후보가 여러 개인데 고를 수 없으면 명시적으로 실패합니다.

## 검수 정책
- 기본 정책은 fail-closed 입니다.
- 배경 검수는 반지 재질과 배경색의 분리 가능성을 직접 확인합니다. 흰 반지 on 흰 배경 같은 케이스는 명시적으로 실패 처리합니다.
- 입력 이미지 가드레일에서 시스템 오류가 발생하면 내부 보정 경로로 넘기지 않고 즉시 실패 처리합니다. 개발 중에만 `ALLOW_VALIDATION_BYPASS=true` 로 우회할 수 있습니다.
- 이미지 다운로드 실패, Vision LLM 오류, 빈 multi-view 결과는 기본적으로 실패 처리합니다.
- `.env` 에서 `ALLOW_VALIDATION_BYPASS=true` 를 줄 때만 Vision 검수 오류를 개발용으로 우회합니다.
- 최종 성공 응답은 `is_valid=True` 이고 결과 이미지가 1장 이상일 때만 반환합니다.
