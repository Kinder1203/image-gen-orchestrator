from langgraph.graph import StateGraph, END

# Import schemas
from .core.schemas import AgentState

# Import node functions
from .nodes.router import multimodal_intent_router, intent_router_condition
from .nodes.rag import retrieve_shoe_context
from .nodes.synthesizer import synthesize_prompt_and_generate_image
from .nodes.preprocessor import process_uploaded_image
from .nodes.validator import validate_image_for_trellis

def build_shoe_generation_graph():
    """ LangGraph 빌드 (SpeakNode의 Agent 구조 확장) """
    
    # 1. State 기반 그래프 정의
    workflow = StateGraph(AgentState)
    
    # 2. 노드(Node) 등록
    workflow.add_node("intent_router", multimodal_intent_router)
    workflow.add_node("rag_retriever", retrieve_shoe_context)
    workflow.add_node("synthesizer", synthesize_prompt_and_generate_image)
    workflow.add_node("preprocessor", process_uploaded_image)
    workflow.add_node("validator", validate_image_for_trellis)
    
    # 에러 핸들링 더미 노드 (재생성 로직 등)
    def error_handler(state):
        return {"status_message": "에러 발생: " + state.get("validation_feedback", "")}
    
    workflow.add_node("error_handler", error_handler)
    
    # 3. 엣지(Edge/Routing) 연결
    workflow.set_entry_point("intent_router")
    
    # 라우터에서 Text/Image/Error 분기
    workflow.add_conditional_edges(
        "intent_router",
        intent_router_condition,
        {
            "preprocessor": "preprocessor", # Image 입력시
            "rag_retriever": "rag_retriever", # Text 입력시
            "end": END
        }
    )
    
    # Text 파이프라인 흐름: RAG -> Synthesizer -> Validator
    workflow.add_edge("rag_retriever", "synthesizer")
    workflow.add_edge("synthesizer", "validator")
    
    # Image 파이프라인 흐름: Preprocessor -> Validator
    workflow.add_edge("preprocessor", "validator")
    
    # 검증 파이프라인 흐름
    workflow.add_conditional_edges(
        "validator",
        lambda x: "end" if x.get("is_valid", False) else "error_handler",
        {
            "end": END,
            "error_handler": "error_handler"
        }
    )
    workflow.add_edge("error_handler", END)
    
    # 4. 컴파일 후 반환
    return workflow.compile()
