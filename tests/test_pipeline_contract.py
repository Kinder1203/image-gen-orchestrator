import importlib
import importlib.util
import json
import unittest
from pathlib import Path


HAS_PYDANTIC = importlib.util.find_spec("pydantic") is not None
HAS_RUNTIME_DEPS = all(
    importlib.util.find_spec(module_name) is not None
    for module_name in ("pydantic", "langgraph", "langchain_ollama")
)
HAS_DB_DEPS = all(
    importlib.util.find_spec(module_name) is not None
    for module_name in ("pydantic", "langchain_chroma", "langchain_ollama")
)


class ExportedTemplateContractTests(unittest.TestCase):
    def test_edit_template_keeps_exported_load_image_shape(self):
        template = json.loads(Path("image_qwen_image_edit_2509.json").read_text(encoding="utf-8"))
        load_nodes = [node for node in template["nodes"] if node.get("type") == "LoadImage"]

        self.assertEqual(len(load_nodes), 1)
        self.assertNotEqual(load_nodes[0].get("widgets_values", [None])[0], "___BASE_IMAGE___")

    def test_multi_view_template_keeps_exported_load_image_shape(self):
        template = json.loads(
            Path("templates-1_click_multiple_character_angles-v1.0 (3).json").read_text(encoding="utf-8")
        )
        load_nodes = [node for node in template["nodes"] if node.get("type") == "LoadImage"]
        titled_nodes = [node for node in load_nodes if (node.get("title") or "") == "Load Character Image"]

        self.assertGreaterEqual(len(load_nodes), 1)
        self.assertEqual(len(titled_nodes), 1)
        self.assertNotEqual(titled_nodes[0].get("widgets_values", [None])[0], "___TARGET_IMAGE___")


@unittest.skipUnless(HAS_PYDANTIC, "pydantic is required for schema tests")
class SchemaContractTests(unittest.TestCase):
    def test_input_type_is_normalized_from_payload_shape(self):
        from src.llm_pipeline.core.schemas import PipelineRequest

        self.assertEqual(PipelineRequest(prompt="ring idea").input_type, "text")
        self.assertEqual(PipelineRequest(input_type="image", image_url="sample.png").input_type, "image_only")
        self.assertEqual(
            PipelineRequest(input_type="modification", prompt="add engraving", image_url="sample.png").input_type,
            "image_and_text",
        )
        self.assertEqual(PipelineRequest(input_type="image_only", image_url="sample.png").input_type, "image_only")
        self.assertEqual(
            PipelineRequest(input_type="image_and_text", prompt="add gem", image_url="sample.png").input_type,
            "image_and_text",
        )

    def test_waiting_for_user_edit_is_valid_response_status(self):
        from src.llm_pipeline.core.schemas import PipelineResponse

        response = PipelineResponse(
            status="waiting_for_user_edit",
            optimized_image_urls=[],
            message="review edit",
            base_image_url="edited.png",
        )

        self.assertEqual(response.status, "waiting_for_user_edit")

    def test_start_requires_prompt_or_image(self):
        from src.llm_pipeline.core.schemas import PipelineRequest

        with self.assertRaises(ValueError):
            PipelineRequest(action="start")

    def test_request_customization_requires_non_empty_prompt(self):
        from src.llm_pipeline.core.schemas import PipelineRequest

        with self.assertRaises(ValueError):
            PipelineRequest(action="request_customization", customization_prompt="")


class FakeState:
    def __init__(self, values, next_nodes):
        self.values = values
        self.next = next_nodes


class FakeGraph:
    def __init__(self, state):
        self._state = state
        self.invocations = []
        self.updated_state = []

    def invoke(self, payload, config=None):
        self.invocations.append((payload, config))

    def update_state(self, config, values):
        self.updated_state.append((config, values))

    def get_state(self, config):
        return self._state


@unittest.skipUnless(HAS_RUNTIME_DEPS, "runtime dependencies are required for pipeline tests")
class PipelineRuntimeTests(unittest.TestCase):
    def test_input_image_guardrail_routes_system_error_to_end(self):
        agent = importlib.import_module("src.llm_pipeline.agent")

        self.assertEqual(
            agent.check_input_image_processing({"guardrail_result": "system_error", "intent": "partial_modification"}),
            "end",
        )
        self.assertEqual(
            agent.check_input_image_processing({"guardrail_result": "repair_required", "intent": "multi_view_only"}),
            "edit_image",
        )
        self.assertEqual(
            agent.check_input_image_processing({"guardrail_result": "pass", "intent": "multi_view_only"}),
            "generate_multi_view",
        )

    def test_validate_base_image_checks_background_contrast(self):
        validator = importlib.import_module("src.llm_pipeline.nodes.validator")

        seen = {}
        original_call = validator._call_vision_judge

        try:
            validator._call_vision_judge = lambda image_url, prompt: (
                seen.update({"image_url": image_url, "prompt": prompt}) or {"is_valid": True, "reason": "ok"}
            )
            result = validator.validate_base_image(
                {
                    "base_ring_image_url": "base.png",
                    "user_prompt": "thin platinum ring with inside engraving",
                    "synthesized_prompt": "platinum ring, solid black background",
                    "retry_count": 0,
                }
            )
        finally:
            validator._call_vision_judge = original_call

        self.assertTrue(result["is_valid"])
        self.assertEqual(seen["image_url"], "base.png")
        self.assertIn("complementary or otherwise strongly contrasting", seen["prompt"])
        self.assertIn("white/platinum/silver", seen["prompt"])

    def test_validate_input_image_distinguishes_system_error_from_repair(self):
        validator = importlib.import_module("src.llm_pipeline.nodes.validator")

        original_call = validator._call_vision_judge

        try:
            validator._call_vision_judge = lambda image_url, prompt: {
                "is_valid": False,
                "reason": "vision unavailable",
                "result_type": "system_error",
            }
            result = validator.validate_input_image({"base_ring_image_url": "input.png"})

            self.assertEqual(result["guardrail_result"], "system_error")
            self.assertFalse(result["is_valid"])

            validator._call_vision_judge = lambda image_url, prompt: {
                "is_valid": False,
                "reason": "Change the background to solid pitch black",
                "result_type": "judged",
            }
            repaired = validator.validate_input_image(
                {
                    "base_ring_image_url": "input.png",
                    "user_prompt": "simple platinum ring",
                    "customization_prompt": "",
                }
            )
        finally:
            validator._call_vision_judge = original_call

        self.assertEqual(repaired["guardrail_result"], "repair_required")
        self.assertIn("solid pitch black", repaired["customization_prompt"])

    def test_multi_view_only_edit_success_skips_second_wait(self):
        agent = importlib.import_module("src.llm_pipeline.agent")

        self.assertEqual(
            agent.check_edit_validation({"is_valid": True, "retry_count": 0, "intent": "multi_view_only"}),
            "generate_multi_view",
        )
        self.assertEqual(
            agent.check_edit_validation({"is_valid": True, "retry_count": 0, "intent": "partial_modification"}),
            "wait_for_edit_approval",
        )

    def test_start_request_copies_prompt_into_customization_for_image_and_text(self):
        schemas = importlib.import_module("src.llm_pipeline.core.schemas")
        pipelines = importlib.import_module("src.llm_pipeline.pipelines")

        fake_graph = FakeGraph(FakeState({"edited_ring_image_url": "edited.png"}, ("wait_for_edit_approval",)))
        original_graph = pipelines.app_graph
        pipelines.app_graph = fake_graph

        try:
            request = schemas.PipelineRequest(
                input_type="image_and_text",
                prompt="inside engraving forever",
                image_url="uploaded.png",
            )
            pipelines.process_generation_request(request)
        finally:
            pipelines.app_graph = original_graph

        initial_state = fake_graph.invocations[0][0]
        self.assertEqual(initial_state["customization_prompt"], "inside engraving forever")
        self.assertEqual(initial_state["input_type"], "image_and_text")

    def test_success_requires_non_empty_output_urls(self):
        schemas = importlib.import_module("src.llm_pipeline.core.schemas")
        pipelines = importlib.import_module("src.llm_pipeline.pipelines")

        fake_state = FakeState(
            {
                "is_valid": True,
                "final_output_urls": [],
                "status_message": "empty outputs",
                "base_ring_image_url": "base.png",
            },
            (),
        )
        fake_graph = FakeGraph(fake_state)
        original_graph = pipelines.app_graph
        pipelines.app_graph = fake_graph

        try:
            request = schemas.PipelineRequest(input_type="text", prompt="slim platinum ring")
            response = pipelines.process_generation_request(request)
        finally:
            pipelines.app_graph = original_graph

        self.assertEqual(response.status, "failed")
        self.assertEqual(response.message, "empty outputs")


@unittest.skipUnless(HAS_DB_DEPS, "db feeder dependencies are required for feeder tests")
class DbFeederContractTests(unittest.TestCase):
    def test_curated_rules_are_rich_and_stable(self):
        db_feeder = importlib.import_module("src.llm_pipeline.scripts.db_feeder")

        docs = db_feeder.CURATED_RULES
        ids = [doc["id"] for doc in docs]
        categories = {doc["category"] for doc in docs}

        self.assertGreaterEqual(len(docs), 20)
        self.assertEqual(len(ids), len(set(ids)))
        self.assertIn("Validation_and_Rembg", categories)
        self.assertIn("Ring_Material", categories)


if __name__ == "__main__":
    unittest.main()
