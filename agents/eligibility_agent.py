"""
Eligibility Agent
=================
Eligibility is checked with DETERMINISTIC rules first (comparing the
student's numeric profile against each university's stated requirements),
then the LLM is used only to explain the results in natural language.

This two-step design (rules -> LLM explanation) avoids the LLM hallucinating
eligibility verdicts, which matters since this directly affects a student's
application decisions.
"""

from typing import Any, Dict, List

from agents.base_agent import AgentResult, BaseAgent, format_context
from prompts.prompts import ELIGIBILITY_AGENT_SYSTEM
from rag.vector_store import Chunk


def _to_float(x, default=None):
    try:
        return float(x)
    except (TypeError, ValueError):
        return default


def check_eligibility(profile: Dict[str, Any], uni_record: Dict[str, Any]) -> Dict[str, Any]:
    """Deterministic rule check. Returns a verdict + reasons list."""
    reasons: List[str] = []
    verdict = "Eligible"

    student_cgpa = _to_float(profile.get("cgpa"))
    student_pct = _to_float(profile.get("percentage"))
    student_ielts = _to_float(profile.get("ielts_score"))

    min_cgpa = _to_float(uni_record.get("min_cgpa"))
    min_pct = _to_float(uni_record.get("min_percentage"))
    ielts_required = uni_record.get("ielts_required", False)
    min_ielts = _to_float(uni_record.get("min_ielts"))

    # Academic requirement: satisfied if EITHER CGPA or percentage clears the bar
    academic_ok = None
    if student_cgpa is not None and min_cgpa is not None:
        academic_ok = student_cgpa >= min_cgpa
        reasons.append(
            f"CGPA {student_cgpa} vs required {min_cgpa}: "
            f"{'meets' if academic_ok else 'below'} requirement."
        )
    elif student_pct is not None and min_pct is not None:
        academic_ok = student_pct >= min_pct
        reasons.append(
            f"Percentage {student_pct}% vs required {min_pct}%: "
            f"{'meets' if academic_ok else 'below'} requirement."
        )
    else:
        reasons.append("No CGPA/percentage provided by student - cannot fully verify academic requirement.")

    # IELTS requirement
    ielts_ok = True
    if ielts_required:
        if student_ielts is not None and min_ielts is not None:
            ielts_ok = student_ielts >= min_ielts
            reasons.append(
                f"IELTS {student_ielts} vs required {min_ielts}: "
                f"{'meets' if ielts_ok else 'below'} requirement."
            )
        else:
            ielts_ok = None
            reasons.append("University requires IELTS but student's IELTS score/status was not provided.")

    # Determine verdict
    if academic_ok is False or ielts_ok is False:
        verdict = "Not Eligible"
    elif academic_ok is None or ielts_ok is None:
        verdict = "Borderline / Needs More Info"
    else:
        verdict = "Eligible"

    return {
        "university_id": uni_record.get("id"),
        "university_name": uni_record.get("name"),
        "verdict": verdict,
        "reasons": reasons,
    }


class EligibilityAgent(BaseAgent):
    agent_name = "eligibility_agent"
    system_prompt = ELIGIBILITY_AGENT_SYSTEM

    def run(self, profile: Dict[str, Any]) -> AgentResult:
        query = (
            f"{profile.get('field_of_study', '')} programs in "
            f"{profile.get('preferred_country', '')}"
        )
        chunks: List[Chunk] = self.retrieve(
            query, top_k=6, record_type="university",
            country=profile.get("preferred_country") or None,
        )

        checks = [check_eligibility(profile, c.record) for c in chunks]
        checks_text = "\n".join(
            f"- {ch['university_name']}: {ch['verdict']} "
            f"({'; '.join(ch['reasons'])})"
            for ch in checks
        )

        context = format_context(chunks)
        user_prompt = self.build_user_prompt(
            profile, context,
            extra=(
                "RULE-BASED ELIGIBILITY CHECK RESULTS (already computed - "
                "explain these to the student, do not recompute or contradict them):\n"
                f"{checks_text}"
            ),
        )
        answer = self.llm.chat(self.system_prompt, user_prompt)

        result = AgentResult(self.agent_name, answer, self._sources_from_chunks(chunks), chunks)
        # Attach structured verdicts for programmatic use (e.g. Streamlit badges)
        result.structured_checks = checks  # type: ignore[attr-defined]
        return result
