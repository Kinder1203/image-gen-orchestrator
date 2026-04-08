# Nodes Module

`nodes/` 는 파이프라인의 실제 실행 단계를 담당합니다.

## Files

- `router.py`: 입력 형태를 보고 시나리오를 결정
- `rag.py`: 반지 규칙용 Vector RAG 조회
- `synthesizer.py`: ComfyUI API format JSON 에 prompt / image 값을 주입하고 생성 실행
- `validator.py`: Vision 검수와 입력 이미지 가드레일 수행

## Current Contract

- 시나리오 3의 내부 보정 edit 는 성공해도 `wait_for_edit_approval` 로 가지 않고 바로 `generate_multi_view` 로 복귀합니다.
- `validate_input_image` 는 `pass`, `repair_required`, `system_error` 를 명확히 구분합니다.
- JSON 파일은 repo에서 직접 수정하지 않고, 런타임에서만 `LoadImage.inputs.image` 를 교체합니다.
- ComfyUI 생성 시스템 오류는 Vision 검수로 넘기지 않고 즉시 실패 처리합니다.
