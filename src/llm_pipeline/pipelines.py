from .core.schemas import PipelineRequest, PipelineResponse
from .agent import build_shoe_generation_graph
from loguru import logger

# 싱글톤 형태로 그래프 초기화
app_graph = build_shoe_generation_graph()

def process_generation_request(request: PipelineRequest) -> PipelineResponse:
    """
    (Entrypoint) 외부 서버가 멀티모달(다각도 지원) 파이프라인을 호출하는 접점
    """
    logger.info(f"Received new request: Type={request.input_type}")
    
    # 1. 초기 State 설정
    initial_state = {
        "input_type": request.input_type,
        "user_prompt": request.prompt if request.prompt else "",
        "user_image": request.image_url if request.image_url else "",
        "intent": "",
        "rag_context": "",
        "synthesized_prompt": "",
        "current_image_urls": [], # 배열 초기화
        "final_output_urls": [],
        "is_valid": False,
        "validation_feedback": "",
        "status_message": "처리 대기중..."
    }
    
    # 2. 파이프라인(LangGraph) 시작
    logger.debug("Starting Qwen Multi-view LangGraph execution...")
    try:
        final_state = app_graph.invoke(initial_state)
        is_valid = final_state.get("is_valid", False)
        
        if is_valid:
            output_urls = final_state.get("final_output_urls", [])
            log_msg = f"렌더링 최적화 성공! (제공된 이미지 수: {len(output_urls)}장)"
            return PipelineResponse(
                status="success",
                optimized_image_urls=output_urls,
                message=final_state.get("status_message", log_msg)
            )
        else:
            return PipelineResponse(
                status="needs_user_action",
                optimized_image_urls=[],
                message=final_state.get("status_message", "품질 검증 실패로 3D 렌더링이 중단되었습니다.")
            )
            
    except Exception as e:
        logger.error(f"Error during execution: {str(e)}")
        return PipelineResponse(
            status="error",
            optimized_image_urls=[],
            message=f"LLM 파이프라인 에러: {str(e)}"
        )
