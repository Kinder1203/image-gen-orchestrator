import requests
from loguru import logger

from .agent import build_ring_generation_graph
from .core.config import config
from .core.schemas import PipelineRequest, PipelineResponse

# 싱글톤 형태로 그래프 초기화
app_graph = build_ring_generation_graph()


def _build_initial_state(request: PipelineRequest) -> dict:
    customization_prompt = request.customization_prompt or ""
    if request.input_type == "image_and_text" and not customization_prompt:
        customization_prompt = request.prompt or ""

    return {
        "user_prompt": request.prompt or "",
        "input_type": request.input_type,
        "base_ring_image_url": request.image_url or "",
        "retry_count": 0,
        "intent": "",
        "customization_prompt": customization_prompt,
        "status_message": "",
    }


def _failed_response(message: str, base_image_url: str = "") -> PipelineResponse:
    return PipelineResponse(
        status="failed",
        optimized_image_urls=[],
        message=message,
        base_image_url=base_image_url,
    )


def process_generation_request(request: PipelineRequest) -> PipelineResponse:
    """
    외부 연동용 엔트리포인트.
    Action 에 따라 LangGraph 모델을 시작하거나 이어서 진행시킵니다.
    """
    if request.action == "start":
        logger.info(f"User Request Input Type: {request.input_type}")
    else:
        logger.info("Follow-up action received without a new start payload.")
    logger.info(f"Action: {request.action} on Thread: {request.thread_id}")

    thread_config = {"configurable": {"thread_id": request.thread_id}}

    try:
        if request.action == "start":
            app_graph.invoke(_build_initial_state(request), config=thread_config)

        elif request.action == "accept_base":
            app_graph.update_state(thread_config, {"intent": "approved_base_only"})
            app_graph.invoke(None, config=thread_config)

        elif request.action == "request_customization":
            app_graph.update_state(
                thread_config,
                {
                    "intent": "user_requested_customization",
                    "customization_prompt": request.customization_prompt or "",
                    "retry_count": 0,
                    "status_message": "",
                },
            )
            app_graph.invoke(None, config=thread_config)

        current_state_obj = app_graph.get_state(thread_config)
        final_state = current_state_obj.values
        next_nodes = current_state_obj.next

        if next_nodes:
            logger.info(f"Pipeline paused. Waiting for human. Next nodes: {next_nodes}")

            if "wait_for_edit_approval" in next_nodes:
                return PipelineResponse(
                    status="waiting_for_user_edit",
                    optimized_image_urls=[],
                    base_image_url=final_state.get("edited_ring_image_url", ""),
                    message="요청하신 커스텀 디자인이 적용되었습니다. 확정하시겠습니까, 아니면 다시 수정하시겠습니까?",
                )

            return PipelineResponse(
                status="waiting_for_user",
                optimized_image_urls=[],
                base_image_url=final_state.get("base_ring_image_url", ""),
                message="기본 반지가 준비되었습니다. 승인하시겠습니까, 아니면 커스텀(각인/보석)을 진행하시겠습니까?",
            )

        is_valid = final_state.get("is_valid", False)
        output_urls = final_state.get("final_output_urls", [])

        if is_valid and output_urls:
            log_msg = f"렌더링 최적화 성공! (이미지 수: {len(output_urls)}장)"

            if config.WEBHOOK_URL and config.WEBHOOK_URL != "NONE":
                try:
                    payload = {
                        "status": "success",
                        "images": output_urls,
                        "prompt_used": final_state.get("synthesized_prompt", ""),
                    }
                    logger.info(f"Sending Webhook to backend: {config.WEBHOOK_URL}")
                    requests.post(config.WEBHOOK_URL, json=payload, timeout=5)
                except Exception as exc:
                    logger.warning(f"Webhook 발송 무시됨: 백엔드 서버에 아직 연결되지 않았습니다. ({exc})")

            return PipelineResponse(
                status="success",
                optimized_image_urls=output_urls,
                message=log_msg,
                base_image_url=final_state.get("edited_ring_image_url", "")
                or final_state.get("base_ring_image_url", ""),
            )

        if is_valid and not output_urls:
            return _failed_response(
                final_state.get("status_message", "검수는 통과했지만 결과 이미지가 비어 있습니다."),
                final_state.get("edited_ring_image_url", "") or final_state.get("base_ring_image_url", ""),
            )

        return _failed_response(
            final_state.get("status_message", "검수 과정에서 최종 불합격 처리되었습니다."),
            final_state.get("edited_ring_image_url", "") or final_state.get("base_ring_image_url", ""),
        )

    except Exception as exc:
        logger.error(f"Pipeline Error: {exc}")
        return _failed_response("서버 오류로 인해 파이프라인이 중단되었습니다.")
