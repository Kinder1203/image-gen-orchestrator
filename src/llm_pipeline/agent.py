from langgraph.graph import StateGraph, END

# Import schemas
from .core.schemas import AgentState

# Import node functions (Omitted preprocessor completely per the new architecture)
from .nodes.router import multimodal_intent_router, intent_router_condition
from .nodes.rag import retrieve_shoe_context
from .nodes.synthesizer import synthesize_prompt_and_generate_image
from .nodes.validator import validate_image_for_trellis

def build_shoe_generation_graph():
    """ LangGraph 빌드 (4단계 최적화 아키텍처) """
    
    # 1. State 기반 그래프 정의
    workflow = StateGraph(AgentState)
    
    # 2. 노드(Node) 등록 (Preprocessor 삭제됨)
    workflow.add_node("intent_router", multimodal_intent_router)
    workflow.add_node("rag_retriever", retrieve_shoe_context)
    workflow.add_node("synthesizer", synthesize_prompt_and_generate_image)
    workflow.add_node("validator", validate_image_for_trellis)
    
    # 에러 핸들링 로직 (검증 실패 시 Self-Healing 역할 수행)
    def error_handler(state):
        return {"status_message": "에러 발생: " + state.get("validation_feedback", "")}
    
    workflow.add_node("error_handler", error_handler)
    
    # 3. 엣지(Edge/Routing) 연결
    workflow.set_entry_point("intent_router")
    
    # 라우터에서 Text/Image 분기
    # (이미지 입력도 RAG 없이 바로 3D 생성용 Synthesizer로 합류하거나, 혹은 Validator로 직행해도 됨)
    # 구조 최적화를 위해 이미지 입력은 곧바로 Synthesizer (다각도 변환)로 넘김.
    workflow.add_conditional_edges(
        "intent_router",
        intent_router_condition,
        {
            "synthesizer": "synthesizer", # Image 입력시 더 이상 누끼를 따지 않고 ComfyUI 다각도 생성으로 넘김
            "rag_retriever": "rag_retriever", # Text 입력시 RAG를 거쳐서 넘김
            "end": END
        }
    )
    
    # Text 파이프라인 흐름: RAG -> Synthesizer
    workflow.add_edge("rag_retriever", "synthesizer")
    
    # 생성 후 모두 검증 단계(Validator)로 향함
    workflow.add_edge("synthesizer", "validator")
    
    # 검증 파이프라인 흐름 (Self-Healing Loop)
    workflow.add_conditional_edges(
        "validator",
        lambda x: "end" if x.get("is_valid", False) else "error_handler",
        {
            "end": END,
            "error_handler": "error_handler"
        }
    )
    
    # 현재는 에러 시 끝내지만, 추후 error_handler -> synthesizer 로 Self-Healing 구현 가능.
    workflow.add_edge("error_handler", END)
    
    # 4. 컴파일 후 반환
    return workflow.compile()
