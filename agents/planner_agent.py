"""
Planner / Recommendation Agent
===============================
The orchestrator. Runs all specialized agents against the student's
profile, then asks the LLM to synthesize their outputs into one
coherent, personalized roadmap + explanation.

Agent Flow (per project design):
Student -> Planner Agent -> Specialized Agents -> RAG retrieval ->
Analysis -> Final Personalized Roadmap
"""

from typing import Any, Dict

from agents.accommodation_agent import AccommodationAgent
from agents.base_agent import BaseAgent
from agents.budget_agent import BudgetAgent
from agents.eligibility_agent import EligibilityAgent
from agents.scholarship_agent import ScholarshipAgent
from agents.university_agent import UniversityAgent
from agents.work_agent import WorkAgent
from prompts.prompts import PLANNER_AGENT_SYSTEM


class PlannerAgent(BaseAgent):
    agent_name = "planner_agent"
    system_prompt = PLANNER_AGENT_SYSTEM

    def __init__(self, llm_client=None):
        super().__init__(llm_client)
        self.university_agent = UniversityAgent(self.llm)
        self.eligibility_agent = EligibilityAgent(self.llm)
        self.scholarship_agent = ScholarshipAgent(self.llm)
        self.accommodation_agent = AccommodationAgent(self.llm)
        self.work_agent = WorkAgent(self.llm)
        self.budget_agent = BudgetAgent(self.llm)

    def run(self, profile: Dict[str, Any]) -> Dict[str, Any]:
        """Runs the full agent pipeline and returns a structured roadmap dict
        ready to be rendered in Streamlit.
        """
        university_result = self.university_agent.run(profile)
        eligibility_result = self.eligibility_agent.run(profile)
        scholarship_result = self.scholarship_agent.run(profile)
        accommodation_result = self.accommodation_agent.run(profile)
        work_result = self.work_agent.run(profile)
        budget_result = self.budget_agent.run(
            profile,
            university_chunks=university_result.raw_chunks,
            accommodation_chunks=accommodation_result.raw_chunks,
        )

        agent_outputs_text = (
            f"UNIVERSITY AGENT OUTPUT:\n{university_result.answer}\n\n"
            f"ELIGIBILITY AGENT OUTPUT:\n{eligibility_result.answer}\n\n"
            f"SCHOLARSHIP AGENT OUTPUT:\n{scholarship_result.answer}\n\n"
            f"ACCOMMODATION AGENT OUTPUT:\n{accommodation_result.answer}\n\n"
            f"WORK AGENT OUTPUT:\n{work_result.answer or '(student not interested in part-time work)'}\n\n"
            f"BUDGET AGENT OUTPUT:\n{budget_result.answer}\n"
        )

        user_prompt = (
            f"STUDENT PROFILE:\n{self._profile_to_text(profile)}\n\n"
            f"{agent_outputs_text}\n\n"
            "Synthesize the above into the final personalized roadmap as instructed."
        )
        final_roadmap_text = self.llm.chat(self.system_prompt, user_prompt, max_tokens=1200)

        return {
            "profile": profile,
            "final_roadmap": final_roadmap_text,
            "agents": {
                "university": {
                    "answer": university_result.answer,
                    "sources": university_result.sources,
                },
                "eligibility": {
                    "answer": eligibility_result.answer,
                    "sources": eligibility_result.sources,
                    "structured_checks": getattr(eligibility_result, "structured_checks", []),
                },
                "scholarship": {
                    "answer": scholarship_result.answer,
                    "sources": scholarship_result.sources,
                },
                "accommodation": {
                    "answer": accommodation_result.answer,
                    "sources": accommodation_result.sources,
                },
                "work": {
                    "answer": work_result.answer,
                    "sources": work_result.sources,
                },
                "budget": {
                    "answer": budget_result.answer,
                    "sources": budget_result.sources,
                    "structured_breakdown": getattr(budget_result, "structured_breakdown", []),
                },
            },
        }
