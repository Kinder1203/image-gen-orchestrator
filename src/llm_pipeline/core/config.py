import os
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    # LLM Settings (Ollama 기반으로 변경)
    OLLAMA_MODEL: str = os.getenv("OLLAMA_MODEL", "llama3")
    OLLAMA_VISION_MODEL: str = os.getenv("OLLAMA_VISION_MODEL", "llava")
    OLLAMA_BASE_URL: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    
    # TRELLIS Prompt Additions
    TRELLIS_REQUIRED_PROMPT: str = ", solid white background, high resolution, isolated on white, studio lighting, orthographic view, highly detailed"
    
    # DB Settings (기존 SpeakNode KuzuDB 연동용)
    VECTOR_DB_PATH: str = "./data/chroma_db"
    GRAPH_DB_PATH: str = "./data/kuzu_db"
    
    class Config:
        env_file = ".env"

config = Settings()
