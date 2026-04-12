from pydantic_settings import BaseSettings, SettingsConfigDict


class EnvironmentSettings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # vLLM chat / multimodal inference
    VLLM_CHAT_BASE_URL: str = "http://127.0.0.1:8000/v1"
    VLLM_CHAT_MODEL: str = "gemma4-e4b"
    VLLM_CHAT_API_KEY: str = "EMPTY"

    # vLLM embedding inference
    VLLM_EMBED_BASE_URL: str = "http://127.0.0.1:8002/v1"
    VLLM_EMBED_MODEL: str = "BAAI/bge-m3"
    VLLM_EMBED_API_KEY: str = "EMPTY"

    # External services and environment paths
    WEBHOOK_URL: str = "https://graduation-work-backend.onrender.com/api/model-result"
    COMFYUI_URL: str = "http://127.0.0.1:8188"
    VECTOR_DB_PATH: str = "./data/chroma_db"
    LANGGRAPH_CHECKPOINT_DB_PATH: str = "./data/langgraph_checkpoints.sqlite"

    # Environment mode flag
    ALLOW_VALIDATION_BYPASS: bool = False


class InternalConfigDefaults:
    # Internal tuning defaults. These are maintained in code and are not part of
    # the primary environment contract.
    VLLM_VALIDATOR_MAX_TOKENS = 400
    VLLM_PROMPT_MAX_TOKENS = 256
    VLLM_PROMPT_TEMPERATURE = 0.2
    WEBHOOK_TIMEOUT_SECONDS = 5
    COMFYUI_HISTORY_TIMEOUT_SECONDS = 300
    COMFYUI_REQUEST_TIMEOUT_SECONDS = 10
    COMFYUI_UPLOAD_TIMEOUT_SECONDS = 20
    COMFYUI_POLL_INTERVAL_SECONDS = 2.0
    IMAGE_DOWNLOAD_TIMEOUT_SECONDS = 10
    IMAGE_BRIDGE_DOWNLOAD_TIMEOUT_SECONDS = 20

    # TRELLIS-style prompt additions kept neutral to avoid conflicting with
    # the complementary-background rule used before rembg.
    TRELLIS_REQUIRED_PROMPT = (
        ", isolated product render, high resolution, clean silhouette, "
        "studio lighting, orthographic view, highly detailed"
    )

    # Validation policy / pipeline tuning
    MULTI_VIEW_VALIDATION_SAMPLE_COUNT = 2
    BASE_VALIDATION_MAX_RETRIES = 3
    EDIT_VALIDATION_MAX_RETRIES = 3
    REMBG_VALIDATION_MAX_RETRIES = 3
    RAG_DEFAULT_TOP_K = 3
    CUSTOMIZATION_RAG_TOP_K = 4


class Config:
    def __init__(self, env_settings: EnvironmentSettings | None = None):
        env = env_settings or EnvironmentSettings()

        # Environment-backed values
        self.VLLM_CHAT_BASE_URL = env.VLLM_CHAT_BASE_URL
        self.VLLM_CHAT_MODEL = env.VLLM_CHAT_MODEL
        self.VLLM_CHAT_API_KEY = env.VLLM_CHAT_API_KEY

        self.VLLM_EMBED_BASE_URL = env.VLLM_EMBED_BASE_URL
        self.VLLM_EMBED_MODEL = env.VLLM_EMBED_MODEL
        self.VLLM_EMBED_API_KEY = env.VLLM_EMBED_API_KEY

        self.WEBHOOK_URL = env.WEBHOOK_URL
        self.COMFYUI_URL = env.COMFYUI_URL
        self.VECTOR_DB_PATH = env.VECTOR_DB_PATH
        self.LANGGRAPH_CHECKPOINT_DB_PATH = env.LANGGRAPH_CHECKPOINT_DB_PATH
        self.ALLOW_VALIDATION_BYPASS = env.ALLOW_VALIDATION_BYPASS

        # Code-backed defaults
        self.VLLM_VALIDATOR_MAX_TOKENS = InternalConfigDefaults.VLLM_VALIDATOR_MAX_TOKENS
        self.VLLM_PROMPT_MAX_TOKENS = InternalConfigDefaults.VLLM_PROMPT_MAX_TOKENS
        self.VLLM_PROMPT_TEMPERATURE = InternalConfigDefaults.VLLM_PROMPT_TEMPERATURE
        self.WEBHOOK_TIMEOUT_SECONDS = InternalConfigDefaults.WEBHOOK_TIMEOUT_SECONDS
        self.COMFYUI_HISTORY_TIMEOUT_SECONDS = InternalConfigDefaults.COMFYUI_HISTORY_TIMEOUT_SECONDS
        self.COMFYUI_REQUEST_TIMEOUT_SECONDS = InternalConfigDefaults.COMFYUI_REQUEST_TIMEOUT_SECONDS
        self.COMFYUI_UPLOAD_TIMEOUT_SECONDS = InternalConfigDefaults.COMFYUI_UPLOAD_TIMEOUT_SECONDS
        self.COMFYUI_POLL_INTERVAL_SECONDS = InternalConfigDefaults.COMFYUI_POLL_INTERVAL_SECONDS
        self.IMAGE_DOWNLOAD_TIMEOUT_SECONDS = InternalConfigDefaults.IMAGE_DOWNLOAD_TIMEOUT_SECONDS
        self.IMAGE_BRIDGE_DOWNLOAD_TIMEOUT_SECONDS = InternalConfigDefaults.IMAGE_BRIDGE_DOWNLOAD_TIMEOUT_SECONDS
        self.TRELLIS_REQUIRED_PROMPT = InternalConfigDefaults.TRELLIS_REQUIRED_PROMPT
        self.MULTI_VIEW_VALIDATION_SAMPLE_COUNT = InternalConfigDefaults.MULTI_VIEW_VALIDATION_SAMPLE_COUNT
        self.BASE_VALIDATION_MAX_RETRIES = InternalConfigDefaults.BASE_VALIDATION_MAX_RETRIES
        self.EDIT_VALIDATION_MAX_RETRIES = InternalConfigDefaults.EDIT_VALIDATION_MAX_RETRIES
        self.REMBG_VALIDATION_MAX_RETRIES = InternalConfigDefaults.REMBG_VALIDATION_MAX_RETRIES
        self.RAG_DEFAULT_TOP_K = InternalConfigDefaults.RAG_DEFAULT_TOP_K
        self.CUSTOMIZATION_RAG_TOP_K = InternalConfigDefaults.CUSTOMIZATION_RAG_TOP_K


config = Config()
