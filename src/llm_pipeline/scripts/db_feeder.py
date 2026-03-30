import os
import json
from loguru import logger
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from ..core.config import config

# [B] 신발 디자인 도메인 전문 지식 (Shoe Terminology)
SHOE_KNOWLEDGE_BASE = [
    {
        "category": "Shoe_Design",
        "title": "Upper Materials",
        "content": "Leather provides structure and durability. Mesh provides breathability. Suede offers a premium matte finish but requires careful lighting. Flyknit/Primeknit offers a sock-like seamless fit."
    },
    {
        "category": "Shoe_Design",
        "title": "Midsole and Outsole Types",
        "content": "A 'cupsole' is commonly found in lifestyle and skate shoes (e.g., Dunk, Air Force 1), providing a rigid, flat base. 'EVA foam' is used in running shoes for cushioning. A 'chunky sole' or 'platform sole' drastically increases the height of the shoe profile."
    }
]

# [A] Qwen Multi-angle 공식 프롬프트 가이드 (템플릿 기반 통제 규칙)
QWEN_PROMPT_GUIDES = [
    {
        "category": "Qwen_Multi_Angle_Prompting",
        "title": "Azimuth (Horizontal Angle) Control",
        "content": "To generate multiple views correctly with Qwen Multi-Angle, you must specify the camera horizontal angle in the prompt. Use exact keywords: 'front view' for 0 degrees, 'right side view' for 90 degrees, 'back view' for 180 degrees, and 'left side view' for 270 degrees. Do not use ambiguous terms like 'from the side'."
    },
    {
        "category": "Qwen_Multi_Angle_Prompting",
        "title": "Elevation (Vertical Angle) Control",
        "content": "For 3D object rendering like shoes, default to 'eye-level shot' (0 degrees elevation) to prevent perspective distortion. Only use 'high-angle shot' if explicitly requested by the user to show the top of the shoe (e.g. the tongue or insole)."
    },
    {
        "category": "Qwen_Multi_Angle_Prompting",
        "title": "Distance and Background",
        "content": "Always append 'medium shot, isolated on solid white background, highly detailed' to the end of your Qwen prompt. This is mandatory for the TRELLIS 3D conversion pipeline."
    }
]

def init_vector_db():
    """
    위의 딕셔너리 지식들을 Chroma Vector DB에 임베딩하여 로컬에 밀어 넣는 초기화 함수입니다.
    기존 Kuzu(GraphDB) 오버엔지니어링 코드를 제거하고 순수 Vector RAG로 다이어트했습니다.
    """
    db_path = config.VECTOR_DB_PATH
    logger.info(f"Initializing Lightweight Chroma DB at {db_path}...")
    
    # 1. 임베딩 모델 준비 (로컬 Ollama 임베딩)
    embedder = OllamaEmbeddings(model="nomic-embed-text", base_url=config.OLLAMA_BASE_URL)
    
    # 임시 디렉토리 생성
    db_dir = os.path.dirname(db_path)
    if not os.path.exists(db_dir):
        os.makedirs(db_dir, exist_ok=True)
        
    # 2. Chroma DB 객체 연결
    vector_store = Chroma(
        collection_name="shoe_qwen_rules",
        embedding_function=embedder,
        persist_directory=db_path
    )
    
    # 3. 문서화 및 임베딩 
    all_docs = QWEN_PROMPT_GUIDES + SHOE_KNOWLEDGE_BASE
    texts = []
    metadatas = []
    
    for doc in all_docs:
        text = f"[{doc['category']}] {doc['title']}: {doc['content']}"
        texts.append(text)
        metadatas.append({"category": doc["category"], "title": doc["title"]})
        
    logger.debug(f"Ingesting {len(texts)} logic entries into Vector DB. This may take a moment...")
    
    # 4. DB에 일괄 삽입
    vector_store.add_texts(texts=texts, metadatas=metadatas)
    
    logger.success(f"Successfully ingested {len(all_docs)} foundational rules into ChromaDB.")
    logger.info("Now the LLM Synthesizer will quickly fetch and exactly know how to prompt the Qwen model!")

if __name__ == "__main__":
    init_vector_db()
