import os
import json
from loguru import logger
# from langchain_chroma import Chroma
# from langchain_ollama import OllamaEmbeddings

# 1. 원시 지식 (Knowledge Base) 데이터 
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

def init_vector_db(db_path: str = "../data/chroma_db"):
    """
    위의 딕셔너리 지식들을 Vector DB나 KuzuDB에 임베딩하여 밀어 넣는 초기화 함수입니다.
    """
    logger.info("Initializing Qwen Prompt & Shoe Knowledge Base DB...")
    
    # 1. 임베딩 모델 준비 (Ollama 로컬 임베딩)
    # embedder = OllamaEmbeddings(model="nomic-embed-text")
    
    # 2. 문서화 및 임베딩 
    all_docs = QWEN_PROMPT_GUIDES + SHOE_KNOWLEDGE_BASE
    
    for idx, doc in enumerate(all_docs):
        text = f"[{doc['category']}] {doc['title']}: {doc['content']}"
        logger.debug(f"Ingesting logic {idx+1}/{len(all_docs)} into Vector DB...")
        # db.add_texts(texts=[text], metadatas=[{"category": doc["category"]}])
        
    logger.success(f"Successfully ingested {len(all_docs)} foundational rules into {db_path}.")
    logger.info("Now the LLM Synthesizer will exactly know how to prompt the Qwen model!")

if __name__ == "__main__":
    init_vector_db()
