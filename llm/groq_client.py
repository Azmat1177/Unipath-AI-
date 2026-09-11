"""
UniPath AI - Groq LLM Client
============================
Thin wrapper around the Groq chat-completions API (OpenAI-compatible
schema). If GROQ_API_KEY is not set, falls back to MOCK MODE so the
rest of the pipeline (retrieval, agents, orchestration) can be built,
tested, and demoed without a live key or internet access.

Your friend just needs to set the environment variable before running
Streamlit:

    export GROQ_API_KEY="gsk_..."
    streamlit run app.py

Get a free key at https://console.groq.com/keys
"""

import json
import re
import time
from typing import List, Dict, Optional

import requests

from config import (
    GROQ_API_KEY,
    GROQ_API_URL,
    GROQ_MODEL,
    LLM_MAX_TOKENS,
    LLM_TEMPERATURE,
    MOCK_MODE,
)

MAX_RETRIES = 2  # on a 429, retry this many times before giving up
DEFAULT_RETRY_WAIT_SECONDS = 5.0


class GroqClient:
    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = None):
        self.api_key = api_key or GROQ_API_KEY
        self.model = model or GROQ_MODEL
        self.mock_mode = self.api_key == ""

    def chat(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = LLM_TEMPERATURE,
        max_tokens: int = LLM_MAX_TOKENS,
        json_mode: bool = False,
    ) -> str:
        """Send a system+user prompt to Groq and return the text response.

        If json_mode=True, asks the model to return raw JSON only
        (caller is still responsible for parsing/validating it).

        On a 429 (rate limit), automatically waits and retries up to
        MAX_RETRIES times - Groq's error message tells us exactly how
        long to wait, so we use that instead of guessing.
        """
        if self.mock_mode:
            return self._mock_response(system_prompt, user_prompt, json_mode)

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if json_mode:
            payload["response_format"] = {"type": "json_object"}

        last_error = None
        for attempt in range(MAX_RETRIES + 1):
            resp = requests.post(GROQ_API_URL, headers=headers, json=payload, timeout=30)

            if resp.ok:
                data = resp.json()
                return data["choices"][0]["message"]["content"]

            try:
                detail = resp.json().get("error", {}).get("message", resp.text)
            except Exception:
                detail = resp.text
            last_error = RuntimeError(
                f"Groq API error ({resp.status_code}) for model '{self.model}': {detail}"
            )

            if resp.status_code == 429 and attempt < MAX_RETRIES:
                wait_seconds = self._parse_retry_wait(detail)
                time.sleep(wait_seconds)
                continue

            # Non-429 error, or retries exhausted: give up now.
            raise last_error

        raise last_error  # pragma: no cover - loop always returns or raises above

    @staticmethod
    def _parse_retry_wait(detail: str) -> float:
        """Extract the 'try again in 14.4s' style hint Groq includes in
        its 429 message body. Falls back to a fixed default if not found.
        """
        match = re.search(r"try again in ([\d.]+)s", detail)
        if match:
            try:
                return float(match.group(1)) + 0.5  # small buffer
            except ValueError:
                pass
        return DEFAULT_RETRY_WAIT_SECONDS

    # ---------- mock mode ----------

    def _mock_response(self, system_prompt: str, user_prompt: str, json_mode: bool) -> str:
        """
        Deterministic offline stand-in so the pipeline is fully testable
        without network access / an API key. Clearly labeled as mock output.
        Real agent logic (retrieval, filtering, scoring) still runs -
        only the natural-language write-up is templated here.
        """
        note = (
            "[MOCK MODE - no GROQ_API_KEY set, this is a placeholder AI response. "
            "Set GROQ_API_KEY to get real Groq-generated answers.]"
        )
        if json_mode:
            return json.dumps({"mock": True, "note": note, "user_prompt_preview": user_prompt[:200]})
        return f"{note}\n\nBased on the retrieved information:\n{user_prompt[:600]}"
