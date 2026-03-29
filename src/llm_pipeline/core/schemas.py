from typing import Optional, Dict, Any, Literal, List
from pydantic import BaseModel, Field
from typing_extensions import TypedDict

# 외부에서 파이프라인으로 들어오는 Request
class PipelineRequest(BaseModel):
    input_type: Literal["text", "image"] = Field(..., description="입력 타입: 텍스트 또는 2D 이미지")
    prompt: Optional[str] = Field(None, description="텍스트 묘사 또는 보조 설명")
    image_url: Optional[str] = Field(None, description="단일 원본 2D 이미지 URL (Qwen 확장 또는 직접 3D 처리용)")

# 파이프라인이 외부로 내보내는 Response (다각도 배열 반환)
class PipelineResponse(BaseModel):
    status: Literal["success", "needs_user_action", "error"]
    optimized_image_urls: List[str] = Field(default_factory=list, description="TRELLIS Multi-view용으로 변환된 다각도 이미지 배열")
    message: str = Field(..., description="사용자 안내 메시지")

# LangGraph 내부 State (Current Image URL이 List[str]로 진화)
class AgentState(TypedDict):
    input_type: str
    user_prompt: str
    user_image: str
    intent: str
    rag_context: str
    synthesized_prompt: str
    
    # 처리 중인 다각도 이미지 리스트 (1장~N장 수용)
    current_image_urls: List[str]
    
    # Validation (검증)
    is_valid: bool
    validation_feedback: str
    
    # 최종 결과물 배열
    final_output_urls: List[str]
    status_message: str
