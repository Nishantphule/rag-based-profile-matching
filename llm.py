"""
llm.py
======

Optional LLM-backed reasoning generator for the matcher.

Why an LLM here?
----------------
The retriever and scorer (`HybridRetriever`, `JobMatcher._score`) are fully
deterministic and explainable - they're the right place to make ranking
decisions you'll have to defend in production. We use the LLM **only** to
produce a richer 2-3 sentence justification for the top-K already chosen by
the retriever. Everything stays grounded:

    inputs:  the JD text, the candidate's matched skills, the top retrieved
             excerpts, years of experience and the raw retrieval scores.
    output:  a short natural-language reasoning paragraph.

Usage
-----
    from llm import OpenRouterReasoner
    from job_matcher import JobMatcher

    reasoner = OpenRouterReasoner.from_env()   # reads OPENROUTER_API_KEY
    matcher = JobMatcher(reasoner=reasoner)
    result = matcher.match(jd_text, top_k=5)

If `OPENROUTER_API_KEY` is missing, `JobMatcher` silently falls back to its
deterministic template, so the system always works offline.
"""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass
from typing import Optional

log = logging.getLogger("llm")


# ---------------------------------------------------------------------------
# Optional .env support (only if python-dotenv is installed)
# ---------------------------------------------------------------------------

def _load_dotenv_if_present() -> None:
    try:
        from dotenv import load_dotenv  # type: ignore

        load_dotenv()
    except Exception:
        pass


_load_dotenv_if_present()


# ---------------------------------------------------------------------------
# Reasoner
# ---------------------------------------------------------------------------

DEFAULT_MODEL = "openai/gpt-5.2"
DEFAULT_BASE_URL = "https://openrouter.ai/api/v1"

REASONING_SYSTEM_PROMPT = (
    "You are a senior technical recruiter. Given a job description and a "
    "candidate's resume excerpts, write a concise 2-3 sentence reasoning "
    "that explains WHY this candidate matches (or doesn't) the role. "
    "Ground every claim in the provided excerpts - do NOT invent facts. "
    "Mention concrete strengths first, then any notable gaps. "
    "Be specific (e.g. 'shipped a transformer-based ranker'), not vague "
    "(e.g. 'has ML experience'). Keep it under 80 words."
)


@dataclass
class ReasoningInput:
    """Bag of evidence we hand to the LLM. All fields come from the retriever."""

    job_description: str
    candidate_name: str
    candidate_title: str
    experience_years: float
    matched_skills: list[str]
    excerpts: list[str]
    semantic_score: float
    keyword_score: float


class OpenRouterReasoner:
    """Calls OpenRouter (OpenAI-compatible) to generate the per-match reasoning.

    Failures are non-fatal: if the API call errors out or times out, we return
    None and the caller (JobMatcher) falls back to the template reasoning.
    """

    def __init__(
        self,
        api_key: str,
        model: str = DEFAULT_MODEL,
        base_url: str = DEFAULT_BASE_URL,
        site_url: str | None = None,
        site_name: str | None = None,
        timeout_s: float = 30.0,
        temperature: float = 0.2,
        max_tokens: int = 200,
    ) -> None:
        # Lazy import: keep the rest of the package usable when the openai
        # package isn't installed (e.g. in a stripped-down container).
        from openai import OpenAI

        if not api_key:
            raise ValueError("OpenRouter API key is empty.")

        self.client = OpenAI(base_url=base_url, api_key=api_key, timeout=timeout_s)
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self._extra_headers: dict[str, str] = {}
        if site_url:
            self._extra_headers["HTTP-Referer"] = site_url
        if site_name:
            self._extra_headers["X-OpenRouter-Title"] = site_name

    # ---- factory ------------------------------------------------------

    @classmethod
    def from_env(
        cls,
        env_var: str = "OPENROUTER_API_KEY",
        **kwargs,
    ) -> Optional["OpenRouterReasoner"]:
        """Build a reasoner from env vars, or return None if no key is set."""
        api_key = os.environ.get(env_var, "").strip()
        if not api_key:
            log.info("%s not set - LLM reasoning disabled.", env_var)
            return None
        return cls(
            api_key=api_key,
            model=os.environ.get("OPENROUTER_MODEL", DEFAULT_MODEL),
            base_url=os.environ.get("OPENROUTER_BASE_URL", DEFAULT_BASE_URL),
            site_url=os.environ.get("OPENROUTER_SITE_URL") or None,
            site_name=os.environ.get("OPENROUTER_SITE_NAME") or None,
            **kwargs,
        )

    # ---- main entry ---------------------------------------------------

    def generate(self, evidence: ReasoningInput) -> Optional[str]:
        """Return a reasoning string, or None on any failure."""
        prompt = self._build_user_prompt(evidence)
        t0 = time.perf_counter()
        try:
            resp = self.client.chat.completions.create(
                model=self.model,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                extra_headers=self._extra_headers or None,
                messages=[
                    {"role": "system", "content": REASONING_SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
            )
            text = (resp.choices[0].message.content or "").strip()
            log.debug(
                "LLM reasoning for %s in %.0f ms (%d chars)",
                evidence.candidate_name,
                (time.perf_counter() - t0) * 1000,
                len(text),
            )
            return text or None
        except Exception as exc:  # network, auth, rate-limit, etc.
            log.warning("LLM reasoning failed for %s: %s", evidence.candidate_name, exc)
            return None

    # ---- prompt -------------------------------------------------------

    @staticmethod
    def _build_user_prompt(e: ReasoningInput) -> str:
        excerpts_block = "\n".join(f"- {ex}" for ex in e.excerpts) or "(no excerpts)"
        skills_block = ", ".join(e.matched_skills) if e.matched_skills else "(none matched)"
        return (
            f"## Job Description\n{e.job_description.strip()}\n\n"
            f"## Candidate\n"
            f"Name: {e.candidate_name}\n"
            f"Title: {e.candidate_title}\n"
            f"Experience: {e.experience_years:.0f} years\n"
            f"Matched skills (intersection with JD): {skills_block}\n"
            f"Retrieval signals: semantic={e.semantic_score:.2f}, keyword={e.keyword_score:.2f}\n\n"
            f"## Top retrieved excerpts from this candidate's resume\n"
            f"{excerpts_block}\n\n"
            "Write the reasoning now."
        )
