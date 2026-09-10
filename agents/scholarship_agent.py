from typing import Any, Dict

from agents.base_agent import AgentResult, BaseAgent, format_context
from prompts.prompts import SCHOLARSHIP_AGENT_SYSTEM


class ScholarshipAgent(BaseAgent):
    agent_name = "scholarship_agent"
    system_prompt = SCHOLARSHIP_AGENT_SYSTEM

    def run(self, profile: Dict[str, Any]) -> AgentResult:
        query = (
            f"scholarship for {profile.get('field_of_study', '')} student in "
            f"{profile.get('preferred_country', '')} budget {profile.get('budget_usd', '')}"
        )
        chunks = self.retrieve(
            query,
            top_k=6,
            record_type="scholarship",
            country=profile.get("preferred_country") or None,
        )
        context = format_context(chunks)
        user_prompt = self.build_user_prompt(
            profile, context,
            extra="Rank the scholarships by how well they fit this student's profile and budget need."
        )
        answer = self.llm.chat(self.system_prompt, user_prompt)
        return AgentResult(self.agent_name, answer, self._sources_from_chunks(chunks), chunks)
