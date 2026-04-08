from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

from .core.schemas import AgentState

from .nodes.router import multimodal_intent_router, intent_router_condition
from .nodes.rag import retrieve_ring_context

from .nodes.synthesizer import generate_base_image, edit_image, generate_multi_view
from .nodes.validator import validate_base_image, validate_edited_image, validate_rembg, validate_input_image

def check_base_validation(state: AgentState) -> str:
    is_valid = state.get("is_valid", False)
    retries = state.get("retry_count", 0)
    generation_result = state.get("generation_result", "")

    if generation_result == "system_error":
        return "end"
    
    if is_valid:
        # 합격하면 사용자 승인 대기 전용 빈 노드로 이동
        return "wait_for_user_approval"
    
    if retries >= 3:
        return "end"
    return "generate_base_image"

def check_edit_validation(state: AgentState) -> str:
    is_valid = state.get("is_valid", False)
    retries = state.get("retry_count", 0)
    intent = state.get("intent", "")
    generation_result = state.get("generation_result", "")

    if generation_result == "system_error":
        return "end"
    
    if is_valid:
        # 시나리오 3에서 guardrail 때문에 내부 edit를 탄 경우에는 휴게소 없이 곧바로 다각도로 복귀
        if intent == "multi_view_only":
            return "generate_multi_view"
        return "wait_for_edit_approval"
    
    if retries >= 3:
        return "end"
    return "edit_image"

def check_rembg_validation(state: AgentState) -> str:
    is_valid = state.get("is_valid", False)
    retries = state.get("retry_count", 0)
    generation_result = state.get("generation_result", "")

    if generation_result == "system_error":
        return "end"
    
    if is_valid:
        return "end"
    
    if retries >= 3:
        return "end"
    return "generate_multi_view"

def wait_for_user_approval(state: AgentState) -> dict:
    """ 사용자 승인을 받기 위한 정지(Interrupt) 지점을 형성하는 1차 빈 노드 """
    return {}

def wait_for_edit_approval(state: AgentState) -> dict:
    """ 편집된 시안에 대해 사용자 승인을 받기 위한 정지(Interrupt) 지점 2차 빈 노드 """
    return {}

def route_after_approval(state: AgentState) -> str:
    """ 사용자의 응답(action) 결과에 맞춰서 최종적으로 분기를 나눠줍니다. """
    intent = state.get("intent", "")
    if intent == "user_requested_customization":
        return "edit_image"
    return "generate_multi_view"

def check_input_image_processing(state: AgentState) -> str:
    """ 시나리오 2 & 3 에서 업로드된 이미지의 시각적 검증 통과 결과에 따라 분기 """
    original_intent = state.get("intent", "")
    guardrail_result = state.get("guardrail_result", "pass")
    
    # 시스템 오류는 배경 보정으로 위장하지 않고 즉시 실패 처리한다.
    if guardrail_result == "system_error":
        return "end"

    # 배경 대비 문제가 명확할 때만 내부 보정 edit로 보낸다.
    if guardrail_result == "repair_required":
        return "edit_image"
        
    # 정상 통과 또는 개발용 우회라면 원래 유저가 원하던 분기로 보낸다.
    if original_intent == "partial_modification":
        return "edit_image"
    return "generate_multi_view"

def build_ring_generation_graph():
    """ LangGraph 빌드 (조건부 분기 및 Human-in-the-loop 적용) """
    workflow = StateGraph(AgentState)
    
    # 1. 메모리 세이버 (Interrupt 지원)
    checkpointer = MemorySaver()
    
    # 2. 노드 등록
    workflow.add_node("intent_router", multimodal_intent_router)
    workflow.add_node("rag_retriever", retrieve_ring_context)
    workflow.add_node("validate_input_image", validate_input_image) # 시나리오 2, 3 사전 검열관
    
    workflow.add_node("generate_base_image", generate_base_image)
    workflow.add_node("validate_base_image", validate_base_image)
    
    workflow.add_node("wait_for_user_approval", wait_for_user_approval) # 추가된 휴게소 노드
    
    workflow.add_node("edit_image", edit_image)
    workflow.add_node("validate_edited_image", validate_edited_image)
    
    workflow.add_node("wait_for_edit_approval", wait_for_edit_approval) # 2차 커스텀 확인 휴게소
    
    workflow.add_node("generate_multi_view", generate_multi_view)
    workflow.add_node("validate_rembg", validate_rembg)
    
    # 3. 엣지 연결 (흐름 제어)
    workflow.set_entry_point("intent_router")
    
    # 라우터 조건부 분기 (시나리오 2와 3은 곧바로 validate_input_image 로 가로챔)
    workflow.add_conditional_edges(
        "intent_router",
        intent_router_condition,
        {
            "rag_retriever": "rag_retriever",
            "edit_image": "validate_input_image",
            "generate_multi_view": "validate_input_image"
        }
    )
    
    # Input Image Guardrail 분기 (검증 후 원래 길로 가거나, 강제 에딧으로 가거나)
    workflow.add_conditional_edges(
        "validate_input_image",
        check_input_image_processing,
        {
            "edit_image": "edit_image",
            "generate_multi_view": "generate_multi_view",
            "end": END,
        }
    )
    
    # Text-to-Image 생성 분기
    workflow.add_edge("rag_retriever", "generate_base_image")
    workflow.add_edge("generate_base_image", "validate_base_image")
    workflow.add_conditional_edges(
        "validate_base_image",
        check_base_validation,
        {
            "wait_for_user_approval": "wait_for_user_approval",
            "generate_base_image": "generate_base_image",
            "end": END
        }
    )
    
    # 사용자 승인 후의 분기 (휴게소에서 출발)
    workflow.add_conditional_edges(
        "wait_for_user_approval",
        route_after_approval,
        {
            "edit_image": "edit_image",
            "generate_multi_view": "generate_multi_view"
        }
    )
    
    # Image Edit 생성 분기
    workflow.add_edge("edit_image", "validate_edited_image")
    workflow.add_conditional_edges(
        "validate_edited_image",
        check_edit_validation,
        {
            "wait_for_edit_approval": "wait_for_edit_approval",
            "generate_multi_view": "generate_multi_view",
            "edit_image": "edit_image",
            "end": END
        }
    )
    
    # 2차 승인 후 루트
    workflow.add_conditional_edges(
        "wait_for_edit_approval",
        route_after_approval,
        {
            "edit_image": "edit_image",
            "generate_multi_view": "generate_multi_view"
        }
    )
    
    # 다각도 + Rembg 분기
    workflow.add_edge("generate_multi_view", "validate_rembg")
    workflow.add_conditional_edges(
        "validate_rembg",
        check_rembg_validation,
        {
            "generate_multi_view": "generate_multi_view",
            "end": END
        }
    )
    
    # 4. 컴파일 (사용자 피드백을 받기 전 일시정지할 지점 선언)
    # 1차 বে이스 완성 후(wait_for_user_approval), 그리고 수정본 완성 후(wait_for_edit_approval) 2번 정지합니다.
    app = workflow.compile(
        checkpointer=checkpointer,
        interrupt_before=["wait_for_user_approval", "wait_for_edit_approval"]
    )
    
    return app
