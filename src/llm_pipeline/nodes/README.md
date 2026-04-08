# Nodes Module

`nodes/` 는 반지 커스텀 파이프라인의 실제 비즈니스 로직 레이어입니다.

## 노드 목록
- `router.py`: 요청 형태를 보고 흐름 분기
- `rag.py`: 반지 도메인 Vector RAG
- `synthesizer.py`: exported ComfyUI JSON 을 읽어 런타임에서 prompt/image 를 주입하고 호출
- `validator.py`: `gemma4` 기반 Vision 검수와 입력 이미지 가드레일, 보색 배경 검수

## 현재 설계 원칙
- 노드끼리 직접 호출하지 않고 `agent.py` 의 LangGraph 가 흐름을 제어합니다.
- 시나리오 3의 내부 가드레일 보정은 사용자 휴게소를 만들지 않습니다.
- JSON workflow 파일은 repo 에서 수정하지 않고 메모리상에서만 `LoadImage.widgets_values[0]` 를 교체합니다.
- 시나리오 1 생성 검수와 시나리오 2/3 입력 검수는 같은 배경 대비 원칙을 공유합니다.
- 입력 이미지 가드레일은 `배경 보정 필요` 와 `검수 시스템 오류` 를 구분하고, 시스템 오류는 보정 edit 로 우회하지 않습니다.
