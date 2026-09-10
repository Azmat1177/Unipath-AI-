"""
UniPath AI - Base Agent
=======================
Shared functionality for all specialized agents: retrieve relevant
KB chunks (RAG), format them as a CONTEXT block with citations, call
the LLM with a system prompt, and return both the AI text and the
raw retrieved chunks (so Streamlit can show "sources" if desired).
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from llm.groq_client import GroqClient
from rag.vector_store import Chunk, get_knowledge_base


@dataclass
class AgentResult:
    agent_name: str
    answer: str
    sources: List[Dict[str, Any]]   # [{id, type, source_url, last_updated}]
    raw_chunks: List[Chunk]


def format_context(chunks: List[Chunk]) -> str:
    lines = []
    for i, c in enumerate(chunks, start=1):
        lines.append(
            f"[{i}] ({c.record_type}) {c.text} "
            f"(source: {c.source or 'N/A'}, last updated: {c.last_updated or 'N/A'})"
        )
    return "\n".join(lines) if lines else "(no matching records found in knowledge base)"


class BaseAgent:
    agent_name: str = "base_agent"
    system_prompt: str = ""

    def __init__(self, llm_client: Optional[GroqClient] = None):
        self.llm = llm_client or GroqClient()
        self.kb = get_knowledge_base()

    def retrieve(
        self,
        query: str,
        top_k: int = 4,
        record_type: Optional[str] = None,
        country: Optional[str] = None,
    ) -> List[Chunk]:
        return self.kb.search(query, top_k=top_k, record_type=record_type, country=country)

    def build_user_prompt(self, profile: Dict[str, Any], context: str, extra: str = "") -> str:
        return (
            f"STUDENT PROFILE:\n{self._profile_to_text(profile)}\n\n"
            f"CONTEXT (verified knowledge base excerpts):\n{context}\n\n"
            f"{extra}\n\n"
            "Write your response now, grounded strictly in the CONTEXT above."
        )

    @staticmethod
    def _profile_to_text(profile: Dict[str, Any]) -> str:
        parts = []
        for k, v in profile.items():
            if v not in (None, "", []):
                parts.append(f"- {k}: {v}")
        return "\n".join(parts) if parts else "(no profile details provided)"

    def _sources_from_chunks(self, chunks: List[Chunk]) -> List[Dict[str, Any]]:
        return [
            {
                "id": c.chunk_id,
                "type": c.record_type,
                "source": c.source,
                "last_updated": c.last_updated,
            }
            for c in chunks
        ]

    def run(self, profile: Dict[str, Any]) -> AgentResult:
        raise NotImplementedError
