# UniPath AI — Intelligence Layer (RAG + Agentic AI)

This is the **AI/RAG module** for UniPath AI. It is a self-contained Python
package your teammate can `import` directly into a Streamlit app — no
notebook glue code needed.

It implements everything from the "AI intelligence layer" scope:

- RAG over a verified knowledge base (universities, scholarships,
  accommodation, part-time work rules)
- Groq API integration (with an offline **mock mode** for demoing/testing
  without a key)
- An agentic workflow: 6 specialized agents + 1 planner/orchestrator
- Deterministic eligibility and budget logic (not left to the LLM to
  "guess" numbers)
- Source citation + "last updated" on every fact, since fees/deadlines
  change
- A single `UniPathAI` class as the integration point for Streamlit
- A test script that proves the whole pipeline end-to-end

---

## 1. Folder structure

```
unipath_ai/
├── main.py                  <- import this in Streamlit (UniPathAI class)
├── config.py                 <- reads env vars, no secrets hardcoded
├── requirements.txt
├── .env.example
├── test_module.py            <- run this first to verify everything works
├── app_example.py            <- minimal working Streamlit demo
├── knowledge_base/
│   ├── universities.json
│   ├── scholarships.json
│   ├── accommodation.json
│   └── part_time_work.json
├── rag/
│   └── vector_store.py       <- loads KB, TF-IDF retrieval, get_knowledge_base()
├── llm/
│   └── groq_client.py        <- Groq API wrapper + mock mode
├── prompts/
│   └── prompts.py            <- all system prompts, one place to tune
└── agents/
    ├── base_agent.py
    ├── university_agent.py
    ├── eligibility_agent.py
    ├── scholarship_agent.py
    ├── accommodation_agent.py
    ├── work_agent.py
    ├── budget_agent.py
    └── planner_agent.py      <- orchestrates all of the above
```

---

## 2. Quick start

```bash
cd unipath_ai
pip install -r requirements.txt

# Optional but recommended: get a free key at https://console.groq.com/keys
cp .env.example .env
# edit .env and paste your key, then:
export GROQ_API_KEY="gsk_..."

# verify everything works end-to-end
python test_module.py

# try the demo UI
streamlit run app_example.py
```

**No API key yet?** Everything still runs. The module falls back to
**MOCK MODE**: retrieval, eligibility rules, and budget math all run for
real — only the natural-language write-up is a placeholder. This lets
your team build/demo the Streamlit UI in parallel before the Groq key is
wired in, and `test_module.py` passes with zero network access.

---

## 3. How to integrate into Streamlit

```python
from main import UniPathAI

ai = UniPathAI()   # loads KB + builds RAG index once (cache with st.cache_resource)

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

result["final_roadmap"]                          # personalized roadmap text
result["agents"]["university"]["answer"]          # university matches
result["agents"]["eligibility"]["structured_checks"]  # [{verdict, reasons, ...}]
result["agents"]["budget"]["structured_breakdown"]     # [{monthly_total_usd, ...}]

# Conversational follow-up (RAG chat, e.g. "My CGPA is 3.2, which
# universities should I consider?")
chat = ai.chat("Can I work part-time in Malaysia?", profile)
chat["answer"]
chat["sources"]
```

See `app_example.py` for a full working Streamlit page (form → roadmap →
expandable per-agent sections → follow-up chat box). Copy it into the real
app and restyle.

### Profile schema (all fields optional — more filled in = better results)

| Field | Type | Example |
|---|---|---|
| `academic_qualification` | str | `"FSc Pre-Engineering"` |
| `cgpa` | float | `3.2` |
| `percentage` | float | `78` (use if no CGPA) |
| `field_of_study` | str | `"Computer Science"` |
| `preferred_country` | str | `"Malaysia"` (MVP covers Pakistan / Malaysia / UAE) |
| `budget_usd` | float | `6000` |
| `monthly_living_estimate_usd` | float | overrides the built-in default |
| `ielts_score` | float | `5.5` |
| `accommodation_preference` | str | `"shared room"` |
| `interested_in_part_time_work` | bool | `True` |

---

## 4. Agentic AI workflow

```
Student → Planner Agent → Specialized Agents → RAG retrieval → Analysis → Final Roadmap
```

| Agent | Responsibility | How it avoids hallucination |
|---|---|---|
| **University Agent** | Matches programs by field/country/CGPA, tags Best/Possible/Ambitious | LLM only writes from retrieved KB chunks |
| **Eligibility Agent** | Eligible / Borderline / Not Eligible per university | **Rule-based** comparison in code first; LLM only explains the verdicts |
| **Scholarship Agent** | Ranks scholarships by fit to profile + budget | LLM only writes from retrieved KB chunks |
| **Accommodation Agent** | Hostel/shared/private room costs near shortlisted unis | LLM only writes from retrieved KB chunks |
| **Work Agent** | Part-time work legality/hours/earnings, separates official rules from suggestions | LLM only writes from retrieved KB chunks |
| **Budget Agent** | Tuition + accommodation + living cost, monthly/yearly, scholarship-adjusted | **All math done in code** (`budget_agent.compute_breakdown`); LLM only formats it |
| **Planner Agent** | Combines all of the above into one roadmap + explanation | Synthesizes only the other agents' already-grounded outputs |

---

## 5. RAG design

- Verified facts live in `knowledge_base/*.json` (edit/extend these files
  directly — no re-indexing step needed, the index rebuilds automatically
  the first time `UniPathAI()` is created).
- Each record carries `source` (URL) and `last_updated`, and every
  retrieved chunk keeps both, so the UI can show "as of <date>, source:
  <link>" — important since fees/deadlines/policies change.
- Retrieval is **TF-IDF + cosine similarity** (`rag/vector_store.py`) —
  chosen deliberately over a neural embedding model so the module needs
  **zero external downloads** and starts instantly, which matters for a
  hackathon demo. It's swappable: implement your own retriever and pass
  it in if your team later wants real embeddings.
- Every agent prompt (`prompts/prompts.py`) enforces "answer only from the
  provided CONTEXT" to keep fees/deadlines/eligibility grounded in real
  data rather than the LLM's general knowledge.

## 6. Extending the knowledge base

To add more universities/countries, just append objects to
`knowledge_base/universities.json` following the existing schema (same for
scholarships/accommodation/part_time_work). No code changes needed — the
RAG index rebuilds from the JSON files automatically on next run.

## 7. Notes / known limits (be upfront about these in your demo)

- The knowledge base is a **hand-curated MVP sample** (10 universities, 3
  countries, IT/CS-focused) per the hackathon-scope section of the
  project brief — not a live/scraped dataset.
- `GROQ_API_KEY` is required for real AI-generated text; this sandbox
  could not make live calls to `api.groq.com`, so the module ships fully
  tested in **mock mode** — test it live with a real key before the demo.
- Eligibility/budget numbers are deterministic and tested
  (`test_module.py`), but always double-check the sample KB data against
  each university's real current admissions page before presenting it as
  fact — the `source` field on every record is there for exactly that.
