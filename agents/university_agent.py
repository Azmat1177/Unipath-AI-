from typing import Any, Dict

from agents.base_agent import AgentResult, BaseAgent, format_context
from prompts.prompts import UNIVERSITY_AGENT_SYSTEM


class UniversityAgent(BaseAgent):
    agent_name = "university_agent"
    system_prompt = UNIVERSITY_AGENT_SYSTEM

    def run(self, profile: Dict[str, Any]) -> AgentResult:
        query = (
            f"{profile.get('field_of_study', '')} programs in "
            f"{profile.get('preferred_country', '')} for a student with "
            f"CGPA {profile.get('cgpa', '')}"
        )
        chunks = self.retrieve(
            query,
            top_k=6,
            record_type="university",
            country=profile.get("preferred_country") or None,
        )
        context = format_context(chunks)
        user_prompt = self.build_user_prompt(
            profile, context,
            extra="Categorize each university as Best Match, Possible Match, or Ambitious Option."
        )
        answer = self.llm.chat(self.system_prompt, user_prompt)
        return AgentResult(self.agent_name, answer, self._sources_from_chunks(chunks), chunks)
