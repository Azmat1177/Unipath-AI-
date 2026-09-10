from typing import Any, Dict

from agents.base_agent import AgentResult, BaseAgent, format_context
from prompts.prompts import WORK_AGENT_SYSTEM


class WorkAgent(BaseAgent):
    agent_name = "work_agent"
    system_prompt = WORK_AGENT_SYSTEM

    def run(self, profile: Dict[str, Any]) -> AgentResult:
        if not profile.get("interested_in_part_time_work"):
            return AgentResult(self.agent_name, "", [], [])

        query = f"part time work rules for students in {profile.get('preferred_country', '')}"
        chunks = self.retrieve(
            query,
            top_k=3,
            record_type="part_time_work",
            country=profile.get("preferred_country") or None,
        )
        context = format_context(chunks)
        user_prompt = self.build_user_prompt(profile, context)
        answer = self.llm.chat(self.system_prompt, user_prompt)
        return AgentResult(self.agent_name, answer, self._sources_from_chunks(chunks), chunks)
