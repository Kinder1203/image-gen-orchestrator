import logging
import os
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from ..core.schemas import AgentState
from ..core.config import config

logger = logging.getLogger(__name__)

# 기존 SpeakNode의 무거운 HybridRAG를 버리고, 오직 유사도 검색에 특화된 가벼운 Vector 검색기
class ShoeVectorRAG:
    """
    가벼운 단일 구조의 RAG 검색기. (오버엔지니어링된 Cypher 생성 로직 50줄 통삭제)
    단지 Chroma를 활용하여, 사용자 질문에 맞는 Qwen 통제 가이드 텍스트만 0.1초만에 찾아옴.
    """
    def __init__(self):
        self.embeddings = OllamaEmbeddings(
            model="nomic-embed-text", 
            base_url=config.OLLAMA_BASE_URL
        )
        
        # db_feeder.py가 먼저 실행되었다는 가정 하에 로드
        if os.path.exists(config.VECTOR_DB_PATH):
            self.vector_store = Chroma(
                collection_name="shoe_qwen_rules",
                embedding_function=self.embeddings,
                persist_directory=config.VECTOR_DB_PATH
            )
        else:
            self.vector_store = None
            logger.warning("Vector DB not found! Please run `scripts/db_feeder.py` first.")

    def search_shoe_rules(self, query: str, top_k: int = 3) -> str:
        """가장 연관성이 높은 Qwen 프롬프트 팁이나 신발 소재 지식을 문자열로 반환"""
        if not self.vector_store:
            return "No specific instructions found. Follow standard multi-angle prompt logic."
            
        logger.info(f"ShoeVectorRAG searching for exact logic matching: '{query}'")
        
        try:
            # 단순 Vector Similarity Search 수행
            results = self.vector_store.similarity_search(query, k=top_k)
            
            context_parts = []
            for doc in results:
                cat = doc.metadata.get("category", "General")
                context_parts.append(f"[{cat}] {doc.page_content}")
                
            return "\n\n".join(context_parts)
        except Exception as e:
            logger.error(f"Vector search failed: {e}")
            return "Failure during context retrieval."

# 노드 래퍼 함수 (State 반영용)
def retrieve_shoe_context(state: AgentState) -> dict:
    """
    (Step 2 - Text Branch) 순수 Vector DB(Chroma)를 이용해 3D 규격용 맥락 주입.
    """
    prompt = state.get("user_prompt", "")
    logger.info("Executing Lightweight Vector RAG for Generative Rules...")
    
    rag_engine = ShoeVectorRAG()
    real_context = rag_engine.search_shoe_rules(prompt)
    
    logger.success(f"Retrieved 3D Control Rules Length: {len(real_context)}")
    return {"rag_context": real_context}
