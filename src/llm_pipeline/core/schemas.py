from typing import List, Literal, Optional

from pydantic import BaseModel, Field, model_validator
from typing_extensions import TypedDict


RequestInputType = Literal["text", "image", "modification", "image_only", "image_and_text"]
CanonicalInputType = Literal["text", "image_only", "image_and_text"]
ResponseStatus = Literal["success", "failed", "waiting_for_user", "waiting_for_user_edit"]


def _has_content(value: Optional[str]) -> bool:
    return bool((value or "").strip())


def normalize_input_type(
    prompt: Optional[str],
    image_url: Optional[str],
) -> CanonicalInputType:
    has_prompt = _has_content(prompt)
    has_image = _has_content(image_url)

    if has_image and has_prompt:
        return "image_and_text"
    if has_image:
        return "image_only"
    return "text"


class PipelineRequest(BaseModel):
    input_type: RequestInputType = Field(
        "text",
        description="입력 타입. 구형 값도 허용하지만 내부에서는 canonical 값으로 정규화됩니다.",
    )
    prompt: Optional[str] = Field(None, description="반지 디자인 설명 본문")
    image_url: Optional[str] = Field(None, description="기존 반지 시안 또는 ComfyUI 업로드 이미지명")

    thread_id: str = Field("default_thread", description="LangGraph 세션 스레드 ID")
    action: Literal["start", "accept_base", "request_customization"] = Field(
        "start",
        description="실행 액션 (처음 시작, 승인, 또는 커스텀 요청)",
    )
    customization_prompt: Optional[str] = Field(
        None,
        description="수정 요청 시 추가 각인/보석 디테일 설명",
    )

    @model_validator(mode="after")
    def canonicalize_input_type(self) -> "PipelineRequest":
        self.input_type = normalize_input_type(self.prompt, self.image_url)

        if self.action == "start" and not (_has_content(self.prompt) or _has_content(self.image_url)):
            raise ValueError("start action requires either a prompt, an image_url, or both.")

        if self.action == "request_customization" and not _has_content(self.customization_prompt):
            raise ValueError("request_customization action requires customization_prompt.")

        return self


class PipelineResponse(BaseModel):
    status: ResponseStatus
    optimized_image_urls: List[str] = Field(
        default_factory=list,
        description="TRELLIS multi-view 용으로 변환된 다각도 이미지 배열",
    )
    message: str = Field(..., description="사용자 안내 메시지")
    base_image_url: Optional[str] = None


class AgentState(TypedDict):
    input_type: str
    user_prompt: str
    user_image: str
    intent: str
    rag_context: str
    synthesized_prompt: str

    base_ring_image_url: str
    customization_prompt: str
    edited_ring_image_url: str

    validation_reason: str
    guardrail_result: str
    generation_result: str
    retry_count: int

    current_image_urls: List[str]

    is_valid: bool
    validation_feedback: str

    final_output_urls: List[str]
    status_message: str
