"""
UniPath AI - Main Module Entry Point
=====================================
This is the ONLY file your friend needs to import into Streamlit.

    from main import UniPathAI

    ai = UniPathAI()
    roadmap = ai.generate_roadmap(profile_dict)
    answer = ai.chat("Can I work part-time in Malaysia?", profile_dict)

See README.md for the full profile schema and a Streamlit usage example.
"""

from typing import Any, Dict, List, Optional

from agents.planner_agent import PlannerAgent
from llm.groq_client import GroqClient
from prompts.prompts import CHAT_AGENT_SYSTEM
from rag.vector_store import get_knowledge_base
from config import MOCK_MODE


class UniPathAI:
    """AI / RAG intelligence layer for UniPath AI.

    Wraps the full Agentic AI + RAG pipeline described in the project
    brief: Student Profile -> University Matching -> Eligibility Check
    -> Scholarship Search -> Budget Planning -> Accommodation ->
    Part-Time Work -> Admission Roadmap.
    """

    def __init__(self):
        self.llm = GroqClient()
        self.planner = PlannerAgent(self.llm)
        self.kb = get_knowledge_base()
        self.mock_mode = MOCK_MODE

    # ---------- main pipeline ----------

    def generate_roadmap(self, profile: Dict[str, Any]) -> Dict[str, Any]:
        """Run the full multi-agent pipeline for a student profile.

        Expected profile keys (all optional but more = better results):
            academic_qualification: str        e.g. "FSc Pre-Engineering"
            cgpa: float                        e.g. 3.2
            percentage: float                  e.g. 78  (use instead of cgpa if applicable)
            field_of_study: str                e.g. "Computer Science"
            preferred_country: str             e.g. "Malaysia"  (Pakistan | Malaysia | UAE in this MVP)
            budget_usd: float                  e.g. 5000         (yearly budget)
            monthly_living_estimate_usd: float optional override for living costs
            ielts_score: float                 e.g. 5.5
            accommodation_preference: str       e.g. "shared room"
            interested_in_part_time_work: bool

        Returns a dict:
            {
              "profile": {...},
              "final_roadmap": "<AI-written personalized roadmap text>",
              "agents": {
                 "university": {"answer": ..., "sources": [...]},
                 "eligibility": {"answer": ..., "sources": [...], "structured_checks": [...]},
                 "scholarship": {...},
                 "accommodation": {...},
                 "work": {...},
                 "budget": {..., "structured_breakdown": [...]},
              }
            }
        """
        return self.planner.run(profile)

    # ---------- conversational follow-up ----------

    def chat(
        self,
        query: str,
        profile: Optional[Dict[str, Any]] = None,
        top_k: int = 5,
    ) -> Dict[str, Any]:
        """Answer a free-form follow-up question using RAG over the
        verified knowledge base, optionally personalized with the
        student's profile. Use this for the conversational Generative
        AI experience (e.g. chat box below the roadmap).
        """
        profile = profile or {}
        country = profile.get("preferred_country") or None
        chunks = self.kb.search(query, top_k=top_k, country=country)

        from agents.base_agent import format_context  # local import avoids circularity at module load
        context = format_context(chunks)

        profile_text = "\n".join(f"- {k}: {v}" for k, v in profile.items() if v not in (None, "", []))
        user_prompt = (
            f"STUDENT PROFILE:\n{profile_text or '(not provided)'}\n\n"
            f"CONTEXT (verified knowledge base excerpts):\n{context}\n\n"
            f"STUDENT QUESTION:\n{query}\n\n"
            "Answer the student's question now, grounded strictly in the CONTEXT above."
        )
        answer = self.llm.chat(CHAT_AGENT_SYSTEM, user_prompt)

        return {
            "answer": answer,
            "sources": [
                {"id": c.chunk_id, "type": c.record_type, "source": c.source, "last_updated": c.last_updated}
                for c in chunks
            ],
        }

    # ---------- utility for Streamlit dropdowns etc. ----------

    def list_countries(self) -> List[str]:
        countries = {c.record.get("country") for c in self.kb.get_by_type("university")}
        return sorted(c for c in countries if c)

    def list_fields(self) -> List[str]:
        fields = set()
        for c in self.kb.get_by_type("university"):
            fields.update(c.record.get("programs", []))
        return sorted(fields)
