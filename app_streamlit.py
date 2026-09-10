"""
Minimal example of how to wire this AI module into Streamlit.
Your friend can copy this into their real app.py and build the real UI
around it - this just proves the integration points.

Run:
    export GROQ_API_KEY="gsk_..."   # optional, works in mock mode without it
    streamlit run app_example.py
"""

import streamlit as st

from main import UniPathAI

st.set_page_config(page_title="UniPath AI", page_icon="🎓")


@st.cache_resource
def load_ai():
    return UniPathAI()


ai = load_ai()

if ai.mock_mode:
    st.warning(
        "Running in MOCK MODE (no GROQ_API_KEY set). The pipeline works "
        "end-to-end, but AI text is a placeholder. Set GROQ_API_KEY to "
        "get real Groq-generated answers."
    )

st.title("🎓 UniPath AI — Your Journey to the Right University")

with st.form("profile_form"):
    col1, col2 = st.columns(2)
    with col1:
        cgpa = st.number_input("CGPA (out of 4.0)", 0.0, 4.0, 3.0, 0.1)
        field = st.selectbox("Field of study", ai.list_fields())
        country = st.selectbox("Preferred country", ai.list_countries())
    with col2:
        ielts = st.number_input("IELTS score (0 if not taken)", 0.0, 9.0, 0.0, 0.5)
        budget = st.number_input("Yearly budget (USD)", 0, 50000, 6000, 500)
        work = st.checkbox("Interested in part-time work?", value=True)

    submitted = st.form_submit_button("Generate my roadmap")

if submitted:
    profile = {
        "cgpa": cgpa,
        "field_of_study": field,
        "preferred_country": country,
        "ielts_score": ielts if ielts > 0 else None,
        "budget_usd": budget,
        "interested_in_part_time_work": work,
    }
    with st.spinner("Running the agent pipeline (university, eligibility, "
                     "scholarship, accommodation, work, budget)..."):
        result = ai.generate_roadmap(profile)

    st.subheader("📍 Your Personalized Roadmap")
    st.write(result["final_roadmap"])

    with st.expander("🏫 University matches"):
        st.write(result["agents"]["university"]["answer"])

    with st.expander("✅ Eligibility check"):
        st.write(result["agents"]["eligibility"]["answer"])
        st.json(result["agents"]["eligibility"]["structured_checks"])

    with st.expander("💰 Scholarships"):
        st.write(result["agents"]["scholarship"]["answer"])

    with st.expander("🏠 Accommodation"):
        st.write(result["agents"]["accommodation"]["answer"])

    with st.expander("💼 Part-time work"):
        st.write(result["agents"]["work"]["answer"] or "Not requested.")

    with st.expander("📊 Budget breakdown"):
        st.write(result["agents"]["budget"]["answer"])
        st.json(result["agents"]["budget"]["structured_breakdown"])

    st.session_state["profile"] = profile

st.divider()
st.subheader("💬 Ask a follow-up question")
question = st.text_input("e.g. Can I work part-time in Malaysia?")
if st.button("Ask") and question:
    profile = st.session_state.get("profile", {})
    with st.spinner("Thinking..."):
        chat_result = ai.chat(question, profile)
    st.write(chat_result["answer"])
    st.caption(f"Sources: {[s['id'] for s in chat_result['sources']]}")
