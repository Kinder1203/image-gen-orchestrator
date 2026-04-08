# 반지 커스텀 다각도 3D 생성용 LLM 백엔드

LangGraph 기반으로 반지 시안 생성, 커스텀 편집, 다각도 추출, Vision 검수를 오케스트레이션하는 백엔드 워커입니다. 현재 프로젝트의 기준 도메인은 `반지` 이고, 기본 모델은 `gemma4:26b` 입니다.

## 핵심 시나리오
- `text`
  - `RAG -> 1차 생성 -> 검수 -> 1차 휴게소`
  - 유저가 `합격` 하면 `다각도 생성 -> 검수 -> 종료`
  - 유저가 `각인 수정` 을 누르면 `수정 -> 검수 -> 2차 휴게소`
  - 2차 휴게소에서 `합격` 시 `다각도 생성 -> 검수 -> 종료`
- `image_and_text`
  - `1단계 스킵 -> 수정 -> 검수 -> 2차 휴게소`
  - 유저가 `합격` 하면 `다각도 생성 -> 검수 -> 종료`
  - 유저가 `재수정` 하면 다시 수정 루프로 진입
- `image_only`
  - `1, 2단계 스킵 -> 다각도 생성 -> 검수 -> 종료`
  - 기본적으로 휴게소 없음

`validate_input_image` 같은 배경/누끼 가드레일은 시나리오 2/3 진입 전에만 동작하는 내부 보정 단계입니다. 여기서 내부 보정은 배경 대비 문제가 명확할 때만 수행하고, 이미지 다운로드 실패나 Vision LLM 오류 같은 시스템 오류는 보정으로 위장하지 않고 즉시 실패 처리합니다. 특히 `image_only` 에서 내부 보정용 `edit_image` 를 타더라도 사용자 휴게소를 추가하지 않습니다.
시나리오 1의 `validate_base_image` 도 이제 반지 요청 반영 여부뿐 아니라, 배경이 반지 재질과 충분히 보색/고대비인지 함께 검수합니다.

## 설치
```bash
pip install -r requirements.txt
```

## 환경 변수
```env
OLLAMA_MODEL=gemma4:26b
OLLAMA_BASE_URL=http://localhost:11434
COMFYUI_URL=http://127.0.0.1:8188
VECTOR_DB_PATH=./data/chroma_db
WEBHOOK_URL=https://graduation-work-backend.onrender.com/api/model-result
ALLOW_VALIDATION_BYPASS=false
```

`ALLOW_VALIDATION_BYPASS=true` 를 주면 Vision 검수 오류를 개발용으로만 우회할 수 있습니다. 기본값은 `false` 입니다.

## 백엔드 연동
- 파이프라인 본체는 최종 `success` 응답 시 `WEBHOOK_URL` 로 결과를 POST 하도록 구현되어 있습니다.
- 기본 설정은 `https://graduation-work-backend.onrender.com/api/model-result` 입니다.
- 데모용 `test_run.py` 는 로컬 확인 전용이라 실행 시작 시 `config.WEBHOOK_URL = "NONE"` 으로 바꿔서 전송을 막습니다.
- 실제 백엔드 연동 테스트를 하려면 `test_run.py` 를 쓰지 말고, 앱 서버나 별도 호출 코드에서 `process_generation_request()` 를 사용하거나 `test_run.py` 의 해당 줄을 비활성화해야 합니다.

## ComfyUI JSON 취급 원칙
- `image_z_image_turbo.json`
- `image_qwen_image_edit_2509.json`
- `templates-1_click_multiple_character_angles-v1.0 (3).json`

이 파일들은 ComfyUI 에서 export 한 workflow snapshot 으로 취급합니다. 저장소에서는 JSON 자체를 수정하지 않고, Python 런타임이 파일을 읽은 뒤 메모리상에서 prompt 값과 `LoadImage.widgets_values[0]` 만 동적으로 주입해서 ComfyUI 로 전송합니다.

## 실행 준비
1. Ollama 에 `gemma4:26b` 를 준비합니다.
2. ComfyUI 를 로컬에서 띄웁니다.
3. 처음 1회는 벡터 DB 를 적재합니다.

```bash
python -m src.llm_pipeline.scripts.db_feeder
```

`db_feeder.py` 는 전용 Chroma 컬렉션을 새로 채우는 방식으로 동작하므로, 다시 실행해도 같은 규칙이 중복 적재되지 않습니다. 현재는 반지 재질, 보색 배경, 각인, 수정, 다각도, rembg 검수 규칙을 묶어서 적재합니다.

## 사용 예시
```python
from src.llm_pipeline.core.schemas import PipelineRequest
from src.llm_pipeline.pipelines import process_generation_request

request = PipelineRequest(
    input_type="text",
    prompt="18k 화이트골드에 얇은 곡선 각인이 들어간 커플링",
)

result = process_generation_request(request)

if result.status == "success":
    print(result.optimized_image_urls)
elif result.status == "waiting_for_user":
    print(result.base_image_url)
```

## 입력 타입 정규화
외부에서는 아래 값을 모두 받을 수 있지만 내부에서는 payload 형태 기준으로 정규화됩니다.

- `text`
- `image`
- `modification`
- `image_only`
- `image_and_text`

정규화 규칙은 아래와 같습니다.

- `image_url` 없음 + `prompt` 있음: `text`
- `image_url` 있음 + `prompt` 없음: `image_only`
- `image_url` 있음 + `prompt` 있음: `image_and_text`

추가로 입력 계약은 아래를 따릅니다.

- `action="start"` 는 `prompt` 또는 `image_url` 중 하나 이상이 반드시 있어야 합니다.
- `action="request_customization"` 는 `customization_prompt` 가 반드시 있어야 합니다.

## Human-in-the-loop 상태
- `waiting_for_user`: 베이스 반지 시안 검토 대기
- `waiting_for_user_edit`: 커스텀 반영본 검토 대기
- `success`: 최종 다각도 이미지 생성 완료
- `failed`: 생성 또는 검수 실패
