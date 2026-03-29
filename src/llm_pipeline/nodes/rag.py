import json
import logging
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_ollama import ChatOllama
from ..core.schemas import AgentState
from ..core.config import config

logger = logging.getLogger(__name__)

# 기존 SpeakNode의 HybridRAG를 신발 도메인으로 자립(물리 내장)시킨 객체입니다.
class ShoeHybridRAG:
    """
    KuzuDB(Graph) + Vector DB 복합 검색기 복제본.
    (SpeakNode 원본에서 Task/Meeting을 걷어낸 신발 특화 구조)
    """
    def __init__(self):
        self._cypher_llm = ChatOllama(
            model=config.OLLAMA_MODEL,
            base_url=config.OLLAMA_BASE_URL,
            temperature=0.0,
            format="json",
        )
    
    def _generate_shoe_cypher(self, question: str, limit: int) -> tuple[str, dict]:
        """자연어를 읽고 신발 디자인/엔티티를 검색하는 Cypher 변환기"""
        prompt = """You are a Cypher query generator for a Shoe Knowledge Graph.
Return JSON only:
{"query": "<cypher>", "params": { ... }}

Schema:
- Brand(name, history)
- DesignElement(name, material, style_category)
- Subculture(name, era)

Relations:
- (Brand)-[:PRODUCED]->(DesignElement)
- (DesignElement)-[:INSPIRED_BY]->(Subculture)

Rules:
ONLY MATCH/RETURN statements. No CREATE/DELETE. Keep under LIMIT.
"""
        response = self._cypher_llm.invoke([
            SystemMessage(content=prompt),
            HumanMessage(content=f"Question: {question}\nLimit: {limit}")
        ])
        
        try:
            parsed = json.loads(response.content.strip())
            return parsed.get("query", ""), parsed.get("params", {})
        except Exception:
            return "", {}

    def search_shoe_context(self, query: str) -> str:
        """벡터 검색과 Cypher 검색 결과를 하나로 융합합니다 (기존 hybrid_search 함수 역할)"""
        # 실제 구현시엔 KuzuManager(db_path=...)가 들어갑니다.
        logger.info(f"ShoeHybridRAG processing query: '{query}'")
        
        # 1. 쿼리 생성 테스트 수행 (내부 모듈 활용)
        cypher_q, params = self._generate_shoe_cypher(query, 5)
        logger.debug(f"Generated Cypher from SpeakNode Engine: {cypher_q}")
        
        # 2. 결과 머지 (더미 결과 반환)
        context_parts = []
        if "조던" in query.lower() or "jordan" in query.lower():
            context_parts.append("## 브랜드/역사 (Graph Result)")
            context_parts.append("- Brand: Nike (Produced: Air Jordan 1)")
            context_parts.append("- Inspiration: 80s Basketball Culture")
            context_parts.append("\n## 유사 신발 디자인 (Vector Result)")
            context_parts.append("- Feature: High-top, durable leather overlay, perforated toe box.")
        else:
            context_parts.append("## 카테고리 (Generic Shoe Context)")
            context_parts.append("- Standard multi-layered mesh running shoe.")
            
        return "\n".join(context_parts)

# 노드 래퍼 함수 (State 반영용)
def retrieve_shoe_context(state: AgentState) -> dict:
    """
    (Step 2 - Text Branch) Hybrid RAG / VectorDB를 이용해 구체적 맥락 덧붙임.
    SpeakNode 코드를 응용한 ShoeHybridRAG를 직접 인스턴스화 하여 처리.
    """
    prompt = state.get("user_prompt", "")
    logger.info("Executing built-in Hybrid RAG for Shoe Knowledge...")
    
    rag_engine = ShoeHybridRAG()
    real_context = rag_engine.search_shoe_context(prompt)
    
    logger.success(f"RAG Retrieved Context Length: {len(real_context)}")
    return {"rag_context": real_context}
