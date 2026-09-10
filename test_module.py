"""
UniPath AI - Module Test Script
================================
Runs the full pipeline end-to-end against a few sample student profiles.
Works with NO internet / NO API key (falls back to MOCK MODE so your
friend can verify the plumbing - retrieval, agents, orchestration -
before wiring in a real GROQ_API_KEY).

Run:
    python test_module.py
"""

import json

from main import UniPathAI


def print_header(title: str) -> None:
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


def test_knowledge_base_loading(ai: UniPathAI) -> None:
    print_header("TEST 1: Knowledge base loads correctly")
    countries = ai.list_countries()
    fields = ai.list_fields()
    print(f"Countries in KB: {countries}")
    print(f"Fields/programs in KB: {fields}")
    assert len(countries) >= 2, "Expected at least 2 countries in KB"
    assert len(fields) >= 1, "Expected at least 1 field of study in KB"
    print("PASSED")


def test_rag_retrieval(ai: UniPathAI) -> None:
    print_header("TEST 2: RAG retrieval returns relevant, cited chunks")
    chunks = ai.kb.search("computer science scholarship Malaysia", top_k=3)
    assert len(chunks) > 0, "Expected at least 1 retrieved chunk"
    for c in chunks:
        print(f"- [{c.record_type}] {c.record.get('name')} | source: {c.source}")
        assert c.source, "Every retrieved chunk must carry a source for citation"
    print("PASSED")


def test_eligibility_logic(ai: UniPathAI) -> None:
    print_header("TEST 3: Deterministic eligibility rules behave correctly")
    from agents.eligibility_agent import check_eligibility

    strong_profile = {"cgpa": 3.6, "ielts_score": 6.5}
    weak_profile = {"cgpa": 2.0, "ielts_score": 4.0}
    uni = ai.kb.get_record("uni_003")  # UTM - min_cgpa 2.7, ielts required 5.5
    assert uni is not None

    strong_check = check_eligibility(strong_profile, uni)
    weak_check = check_eligibility(weak_profile, uni)
    print("Strong profile verdict:", strong_check["verdict"], strong_check["reasons"])
    print("Weak profile verdict:  ", weak_check["verdict"], weak_check["reasons"])

    assert strong_check["verdict"] == "Eligible"
    assert weak_check["verdict"] == "Not Eligible"
    print("PASSED")


def test_budget_math(ai: UniPathAI) -> None:
    print_header("TEST 4: Budget calculation arithmetic is correct")
    from agents.budget_agent import BudgetAgent

    profile = {"preferred_country": "Pakistan", "monthly_living_estimate_usd": 100}
    uni_chunks = ai.kb.search("computer science", top_k=2, record_type="university", country="Pakistan")
    acc_chunks = ai.kb.search("accommodation", top_k=2, record_type="accommodation", country="Pakistan")

    agent = BudgetAgent(ai.llm)
    breakdown = agent.compute_breakdown(profile, uni_chunks, acc_chunks)
    assert len(breakdown) > 0
    for b in breakdown:
        expected_monthly = round(
            b["tuition_usd_per_year"] / 12 + b["accommodation_usd_per_month"] + b["living_usd_per_month"], 2
        )
        print(f"- {b['university']}: computed monthly=${b['monthly_total_usd']} expected=${expected_monthly}")
        assert abs(b["monthly_total_usd"] - expected_monthly) < 0.01
    print("PASSED")


def test_full_roadmap_pipeline(ai: UniPathAI) -> None:
    print_header("TEST 5: Full multi-agent roadmap pipeline runs end-to-end")
    profile = {
        "academic_qualification": "FSc Pre-Engineering",
        "cgpa": 3.2,
        "field_of_study": "Computer Science",
        "preferred_country": "Malaysia",
        "budget_usd": 6000,
        "ielts_score": 5.5,
        "accommodation_preference": "shared room",
        "interested_in_part_time_work": True,
    }
    result = ai.generate_roadmap(profile)

    assert "final_roadmap" in result
    assert result["final_roadmap"]
    for agent_name in ["university", "eligibility", "scholarship", "accommodation", "work", "budget"]:
        assert agent_name in result["agents"], f"Missing agent output: {agent_name}"

    print("Final roadmap (first 400 chars):")
    print(result["final_roadmap"][:400])
    print("\nEligibility structured checks:")
    print(json.dumps(result["agents"]["eligibility"]["structured_checks"], indent=2)[:600])
    print("\nBudget structured breakdown:")
    print(json.dumps(result["agents"]["budget"]["structured_breakdown"], indent=2)[:600])
    print("PASSED")


def test_chat_followup(ai: UniPathAI) -> None:
    print_header("TEST 6: Conversational follow-up (RAG chat) works")
    profile = {"preferred_country": "Malaysia", "cgpa": 3.2, "field_of_study": "IT"}
    result = ai.chat("My CGPA is 3.2 and my budget is limited, which universities should I consider?", profile)
    assert result["answer"]
    print("Answer (first 300 chars):", result["answer"][:300])
    print("Sources used:", [s["id"] for s in result["sources"]])
    print("PASSED")


if __name__ == "__main__":
    ai = UniPathAI()
    print(f"MOCK MODE: {ai.mock_mode} "
          f"({'no GROQ_API_KEY set - using offline placeholder responses' if ai.mock_mode else 'live Groq calls enabled'})")

    test_knowledge_base_loading(ai)
    test_rag_retrieval(ai)
    test_eligibility_logic(ai)
    test_budget_math(ai)
    test_full_roadmap_pipeline(ai)
    test_chat_followup(ai)

    print_header("ALL TESTS PASSED")
