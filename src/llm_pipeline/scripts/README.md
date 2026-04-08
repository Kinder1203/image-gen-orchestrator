# Scripts 디렉토리 명세

`scripts/` 는 파이프라인 본체와 분리된 독립 실행 도구를 둡니다.

## 현재 스크립트
- `db_feeder.py`
  - 반지 재질, 보색 배경, 각인, 수정, 다각도, rembg 검수 규칙을 Chroma DB 에 적재합니다.
  - 전용 컬렉션을 새로 채우는 방식이라 다시 실행해도 같은 규칙이 중복으로 쌓이지 않습니다.
  - 초기 세팅이나 지식 변경 시 수동으로 다시 실행합니다.

## 사용 예시
```bash
python -m src.llm_pipeline.scripts.db_feeder
```
