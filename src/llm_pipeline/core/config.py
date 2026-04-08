from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # LLM Settings
    OLLAMA_MODEL: str = "gemma4:26b"
    OLLAMA_BASE_URL: str = "http://localhost:11434"

    # Backend Webhook
    WEBHOOK_URL: str = "https://graduation-work-backend.onrender.com/api/model-result"

    # Local ComfyUI Endpoint
    COMFYUI_URL: str = "http://127.0.0.1:8188"

    # TRELLIS-style prompt additions kept neutral to avoid conflicting with
    # the complementary-background rule used before rembg.
    TRELLIS_REQUIRED_PROMPT: str = (
        ", isolated product render, high resolution, clean silhouette, "
        "studio lighting, orthographic view, highly detailed"
    )

    # Vector DB Settings
    VECTOR_DB_PATH: str = "./data/chroma_db"

    # Validation policy
    ALLOW_VALIDATION_BYPASS: bool = False


config = Settings()
