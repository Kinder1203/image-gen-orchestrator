# Core 디렉토리 명세

`core/` 는 반지 커스텀 파이프라인의 전역 설정과 공용 스키마를 담당합니다.

## 구성
- `config.py`
  - `gemma4:26b` 기본 모델
  - 로컬 ComfyUI 주소
  - Chroma DB 경로
  - `ALLOW_VALIDATION_BYPASS` 검수 정책
- `schemas.py`
  - `PipelineRequest`
  - `PipelineResponse`
  - `AgentState`

## 규칙
- 비즈니스 로직은 두지 않습니다.
- 외부 계약과 환경 설정만 이 레이어에 둡니다.
- `PipelineRequest.input_type` 는 외부 입력을 받아 내부 canonical 값으로 정규화합니다.
