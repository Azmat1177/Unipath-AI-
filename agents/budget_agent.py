"""
Budget Agent
============
Pulls tuition (from matched universities) and accommodation costs (from
the accommodation KB), adds a configurable food/transport/misc estimate,
and computes a deterministic cost breakdown. The LLM is used only to
present this clearly - all numbers come from code, not generation.
"""

from typing import Any, Dict, List

from agents.base_agent import AgentResult, BaseAgent
from prompts.prompts import BUDGET_AGENT_SYSTEM
from rag.vector_store import Chunk

# Rough default monthly living costs (food + transport + misc) by country,
# used only when the student hasn't supplied their own estimate.
DEFAULT_MONTHLY_LIVING_USD = {
    "pakistan": 120,
    "malaysia": 220,
    "uae": 450,
}


class BudgetAgent(BaseAgent):
    agent_name = "budget_agent"
    system_prompt = BUDGET_AGENT_SYSTEM

    def compute_breakdown(
        self,
        profile: Dict[str, Any],
        university_chunks: List[Chunk],
        accommodation_chunks: List[Chunk],
        scholarship_coverage_pct: float = 0.0,
    ) -> List[Dict[str, Any]]:
        country = (profile.get("preferred_country") or "").lower()
        monthly_living = float(
            profile.get("monthly_living_estimate_usd")
            or DEFAULT_MONTHLY_LIVING_USD.get(country, 200)
        )

        breakdown = []
        for uc in university_chunks:
            tuition_year = float(uc.record.get("tuition_fee_usd_per_year", 0))
            tuition_month = tuition_year / 12

            # find a matching accommodation record (by country, or by university id)
            acc_month = None
            for ac in accommodation_chunks:
                if ac.record.get("applies_to_university_id") == uc.chunk_id:
                    acc_month = float(ac.record.get("shared_room_cost_usd_per_month", 0))
                    break
            if acc_month is None:
                same_country = [
                    ac for ac in accommodation_chunks
                    if str(ac.record.get("country", "")).lower() == country
                ]
                acc_month = (
                    float(same_country[0].record.get("shared_room_cost_usd_per_month", 0))
                    if same_country else 0.0
                )

            monthly_total = tuition_month + acc_month + monthly_living
            yearly_total = monthly_total * 12
            scholarship_adjusted_yearly = yearly_total - (tuition_year * scholarship_coverage_pct / 100)

            breakdown.append({
                "university": uc.record.get("name"),
                "tuition_usd_per_year": tuition_year,
                "accommodation_usd_per_month": acc_month,
                "living_usd_per_month": monthly_living,
                "monthly_total_usd": round(monthly_total, 2),
                "yearly_total_usd": round(yearly_total, 2),
                "scholarship_adjusted_yearly_usd": round(scholarship_adjusted_yearly, 2),
            })
        return breakdown

    def run(self, profile: Dict[str, Any], university_chunks=None, accommodation_chunks=None) -> AgentResult:
        university_chunks = university_chunks or self.retrieve(
            f"{profile.get('field_of_study','')} {profile.get('preferred_country','')}",
            top_k=5, record_type="university", country=profile.get("preferred_country") or None,
        )
        accommodation_chunks = accommodation_chunks or self.retrieve(
            f"accommodation {profile.get('preferred_country','')}",
            top_k=5, record_type="accommodation", country=profile.get("preferred_country") or None,
        )

        breakdown = self.compute_breakdown(profile, university_chunks, accommodation_chunks)
        breakdown_text = "\n".join(
            f"- {b['university']}: tuition ${b['tuition_usd_per_year']}/yr, "
            f"accommodation ${b['accommodation_usd_per_month']}/mo, "
            f"living ${b['living_usd_per_month']}/mo => "
            f"total ${b['monthly_total_usd']}/mo (${b['yearly_total_usd']}/yr)"
            for b in breakdown
        ) or "(no matching universities found to compute a budget for)"

        user_prompt = (
            f"STUDENT PROFILE:\n{self._profile_to_text(profile)}\n\n"
            f"COMPUTED BUDGET BREAKDOWN (already calculated - present this clearly, "
            f"do not alter the numbers):\n{breakdown_text}\n\n"
            "Summarize this budget for the student in a clear, easy to scan way."
        )
        answer = self.llm.chat(self.system_prompt, user_prompt)

        result = AgentResult(
            self.agent_name, answer,
            self._sources_from_chunks(university_chunks + accommodation_chunks),
            university_chunks + accommodation_chunks,
        )
        result.structured_breakdown = breakdown  # type: ignore[attr-defined]
        return result
