from typing import Any, Dict

from agents.base_agent import AgentResult, BaseAgent, format_context
from prompts.prompts import ACCOMMODATION_AGENT_SYSTEM


class AccommodationAgent(BaseAgent):
    agent_name = "accommodation_agent"
    system_prompt = ACCOMMODATION_AGENT_SYSTEM

    def run(self, profile: Dict[str, Any]) -> AgentResult:
        query = (
            f"accommodation hostel options in {profile.get('preferred_country', '')} "
            f"{profile.get('accommodation_preference', '')}"
        )
        chunks = self.retrieve(
            query,
            top_k=5,
            record_type="accommodation",
            country=profile.get("preferred_country") or None,
        )
        context = format_context(chunks)
        user_prompt = self.build_user_prompt(profile, context)
        answer = self.llm.chat(self.system_prompt, user_prompt)
        return AgentResult(self.agent_name, answer, self._sources_from_chunks(chunks), chunks)
