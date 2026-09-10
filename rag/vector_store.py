"""
UniPath AI - RAG Knowledge Base
================================
Loads the verified JSON knowledge base (universities, scholarships,
accommodation, part-time work) and exposes semantic-ish retrieval
over it using TF-IDF + cosine similarity.

Why TF-IDF instead of a neural embedding model?
- Zero external downloads (no HuggingFace/model weights needed),
  so this works offline and starts instantly - ideal for a hackathon.
- It's easy to swap out: see `EmbeddingBackend` below - if your team
  later wants real embeddings (e.g. sentence-transformers, OpenAI,
  Groq's own embedding endpoint), implement that interface and pass
  it into KnowledgeBase(embedding_backend=...).

Every chunk keeps a pointer back to its source record, so answers can
always cite `source` + `last_updated`, per the project's RAG design
(show source + last-updated because data changes).
"""

import glob
import json
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from config import KB_DIR, TOP_K


@dataclass
class Chunk:
    chunk_id: str
    text: str
    record_type: str          # "university" | "scholarship" | "accommodation" | "part_time_work"
    record: Dict[str, Any]     # original JSON record, for structured use by agents
    source: str = ""
    last_updated: str = ""


def _record_to_text(record_type: str, r: Dict[str, Any]) -> str:
    """Flatten a KB JSON record into a natural-language chunk for retrieval."""
    if record_type == "university":
        return (
            f"University: {r['name']} in {r['city']}, {r['country']}. "
            f"Degree level: {r.get('degree_level','')}. "
            f"Programs offered: {', '.join(r.get('programs', []))}. "
            f"Minimum CGPA required: {r.get('min_cgpa','N/A')} "
            f"(or {r.get('min_percentage','N/A')}%). "
            f"IELTS required: {'Yes, minimum ' + str(r.get('min_ielts')) if r.get('ielts_required') else 'No'}. "
            f"Tuition fee: approximately ${r.get('tuition_fee_usd_per_year','N/A')} per year. "
            f"Application deadline: {r.get('application_deadline','N/A')}. "
            f"Required documents: {', '.join(r.get('required_documents', []))}."
        )
    if record_type == "scholarship":
        return (
            f"Scholarship: {r['name']} ({r.get('country','')}). "
            f"Eligibility: {r.get('eligibility','')}. "
            f"Coverage: {r.get('coverage','')}. Amount: {r.get('amount','')}. "
            f"Deadline: {r.get('deadline','N/A')}. "
            f"Required documents: {', '.join(r.get('required_documents', []))}. "
            f"Applies to university id: {r.get('applies_to_university_id','any')}."
        )
    if record_type == "accommodation":
        return (
            f"Accommodation in {r.get('city','')}, {r.get('country','')} "
            f"(near university id {r.get('applies_to_university_id','')}). "
            f"Hostel available: {r.get('hostel_available')}. "
            f"Hostel cost: ${r.get('hostel_cost_usd_per_month','N/A')}/month. "
            f"Shared room: ${r.get('shared_room_cost_usd_per_month','N/A')}/month. "
            f"Private room: ${r.get('private_room_cost_usd_per_month','N/A')}/month. "
            f"Notes: {r.get('notes','')}."
        )
    if record_type == "part_time_work":
        return (
            f"Part-time work rules in {r.get('country','')}: "
            f"Allowed for students: {r.get('allowed_for_students')}. "
            f"Max hours per week: {r.get('max_hours_per_week','N/A')}. "
            f"Official note: {r.get('official_rule_note','')}. "
            f"Common job categories: {', '.join(r.get('common_job_categories', []))}. "
            f"Approx earnings: {r.get('approx_earning_usd_per_month','N/A')} per month."
        )
    return json.dumps(r)


_TYPE_BY_FILENAME = {
    "universities.json": "university",
    "scholarships.json": "scholarship",
    "accommodation.json": "accommodation",
    "part_time_work.json": "part_time_work",
}


class KnowledgeBase:
    """Loads KB JSON files and provides TF-IDF based retrieval (`search`)."""

    def __init__(self, kb_dir: str = KB_DIR):
        self.kb_dir = kb_dir
        self.chunks: List[Chunk] = []
        self._vectorizer: Optional[TfidfVectorizer] = None
        self._matrix = None
        self._load()
        self._build_index()

    # ---------- loading ----------

    def _load(self) -> None:
        self.chunks = []
        for path in sorted(glob.glob(os.path.join(self.kb_dir, "*.json"))):
            filename = os.path.basename(path)
            record_type = _TYPE_BY_FILENAME.get(filename)
            if record_type is None:
                continue
            with open(path, "r", encoding="utf-8") as f:
                records = json.load(f)
            for r in records:
                text = _record_to_text(record_type, r)
                self.chunks.append(
                    Chunk(
                        chunk_id=r.get("id", f"{record_type}_{len(self.chunks)}"),
                        text=text,
                        record_type=record_type,
                        record=r,
                        source=r.get("source", ""),
                        last_updated=r.get("last_updated", ""),
                    )
                )
        if not self.chunks:
            raise RuntimeError(f"No knowledge base records found in {self.kb_dir}")

    def _build_index(self) -> None:
        corpus = [c.text for c in self.chunks]
        self._vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
        self._matrix = self._vectorizer.fit_transform(corpus)

    # ---------- retrieval ----------

    def search(
        self,
        query: str,
        top_k: int = TOP_K,
        record_type: Optional[str] = None,
        country: Optional[str] = None,
    ) -> List[Chunk]:
        """Return the top_k most relevant chunks for `query`.

        Optional filters:
          record_type: restrict to "university" | "scholarship" |
                        "accommodation" | "part_time_work"
          country: restrict to a specific country (case-insensitive)
        """
        if self._vectorizer is None or self._matrix is None:
            return []

        q_vec = self._vectorizer.transform([query])
        sims = cosine_similarity(q_vec, self._matrix).flatten()

        candidates = list(range(len(self.chunks)))
        if record_type:
            candidates = [i for i in candidates if self.chunks[i].record_type == record_type]
        if country:
            candidates = [
                i for i in candidates
                if str(self.chunks[i].record.get("country", "")).lower() == country.lower()
            ]

        candidates.sort(key=lambda i: sims[i], reverse=True)
        top = candidates[:top_k]
        return [self.chunks[i] for i in top]

    def get_by_type(self, record_type: str, country: Optional[str] = None) -> List[Chunk]:
        result = [c for c in self.chunks if c.record_type == record_type]
        if country:
            result = [c for c in result if str(c.record.get("country", "")).lower() == country.lower()]
        return result

    def get_record(self, record_id: str) -> Optional[Dict[str, Any]]:
        for c in self.chunks:
            if c.chunk_id == record_id:
                return c.record
        return None


# Singleton-style accessor so agents can share one loaded/indexed KB.
_kb_instance: Optional[KnowledgeBase] = None


def get_knowledge_base() -> KnowledgeBase:
    global _kb_instance
    if _kb_instance is None:
        _kb_instance = KnowledgeBase()
    return _kb_instance
