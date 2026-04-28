"""
llm.py
======

Optional LLM-backed enhancements for the matcher. Three independent capabilities,
each strictly post-retrieval (the retriever stays deterministic):

1. `generate(evidence)`            - per-candidate reasoning paragraph
2. `extract_filters(jd_text)`      - structured-extraction of the JD's hard
                                     requirements (min experience, must-have
                                     skills) so the user doesn't have to type
                                     them on the CLI
3. `shortlist_summary(jd, top)`    - one final paragraph comparing the top-K
                                     and recommending the strongest match

All three are grounded in inputs the retriever already produced, fail
gracefully (`None` on any error), and never alter the deterministic ranking.

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

import json
import logging
import os
import re
import time
from dataclasses import dataclass, field
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

JD_FILTER_SYSTEM_PROMPT = (
    "You are a JD parser. Extract the HARD requirements as strict JSON. "
    "Schema:\n"
    "{\n"
    '  "min_experience_years": <number>,         // 0 if not stated\n'
    '  "required_skills": [<lowercase strings>], // EXACTLY 2 or 3 skills\n'
    '  "nice_to_have_skills": [<lowercase strings>],\n'
    '  "seniority": "junior" | "mid" | "senior" | "staff" | "principal" | "unknown",\n'
    '  "domain": <short phrase>\n'
    "}\n"
    "Rules:\n"
    "- Only return valid JSON. No prose, no markdown fences.\n"
    "- Use lowercase canonical skill names with these specific abbreviations:\n"
    "  'python', 'java', 'aws', 'gcp', 'azure', 'kubernetes' (NOT 'k8s'), "
    "'machine learning', 'deep learning', 'nlp' (NOT 'natural language processing'), "
    "'llms' (NOT 'large language models'), 'computer vision', 'rag', "
    "'vector databases', 'pytorch', 'tensorflow', 'sql', 'spark'.\n"
    "- `required_skills` must contain ONLY 2 or 3 truly disqualifying skills - the "
    "minimum bar for the role. Pick the SINGLE most common skill from any OR-clause "
    "('AWS or GCP' -> just 'aws'; the other goes to `nice_to_have_skills`). Skip "
    "umbrella terms like 'cloud' or 'devops'.\n"
    "- All other JD skills (4th onwards, OR-clause runners-up, preferred bullets) "
    "go in `nice_to_have_skills`.\n"
    "- `min_experience_years` is the smallest explicit threshold (e.g. '5+ years' -> 5)."
)

SHORTLIST_SYSTEM_PROMPT = (
    "You are a senior technical recruiter writing a final hiring recommendation. "
    "Given a job description and a ranked shortlist of candidates (with their "
    "scores, years of experience and matched skills), write a 3-4 sentence "
    "executive summary. Recommend the strongest candidate by name and explain "
    "in one sentence why. Mention the runner-up briefly. Be concrete and "
    "decisive. Do not invent facts not present in the shortlist data. "
    "Keep it under 100 words."
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


@dataclass
class ExtractedFilters:
    """Hard filters parsed out of a JD by the LLM."""

    min_experience_years: float = 0.0
    required_skills: list[str] = field(default_factory=list)
    nice_to_have_skills: list[str] = field(default_factory=list)
    seniority: str = "unknown"
    domain: str = ""

    def to_dict(self) -> dict:
        return {
            "min_experience_years": self.min_experience_years,
            "required_skills": list(self.required_skills),
            "nice_to_have_skills": list(self.nice_to_have_skills),
            "seniority": self.seniority,
            "domain": self.domain,
        }


@dataclass
class ShortlistCandidate:
    """Compact view of a top-K candidate handed to the summary prompt."""

    rank: int
    name: str
    title: str
    experience_years: float
    match_score: int
    matched_skills: list[str]


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

    # ---- JD -> filters ------------------------------------------------

    def extract_filters(self, jd_text: str) -> Optional[ExtractedFilters]:
        """Use the LLM to pull hard requirements out of a JD.

        Returns an `ExtractedFilters` on success, or `None` if the LLM call /
        JSON parsing fails. The caller is expected to fall back to its own
        defaults / regex parser in that case.
        """
        if not jd_text or not jd_text.strip():
            return None
        t0 = time.perf_counter()
        try:
            resp = self.client.chat.completions.create(
                model=self.model,
                temperature=0.0,           # zero temp for structured extraction
                max_tokens=800,
                extra_headers=self._extra_headers or None,
                messages=[
                    {"role": "system", "content": JD_FILTER_SYSTEM_PROMPT},
                    {"role": "user",
                     "content": f"## Job Description\n{jd_text.strip()}\n\nReturn the JSON now."},
                ],
            )
            raw = (resp.choices[0].message.content or "").strip()
            data = _parse_json_loose(raw)
            if not isinstance(data, dict):
                log.warning("Filter extraction returned non-object: %r", raw[:200])
                return None

            filters = ExtractedFilters(
                min_experience_years=float(data.get("min_experience_years") or 0),
                required_skills=_clean_skill_list(data.get("required_skills")),
                nice_to_have_skills=_clean_skill_list(data.get("nice_to_have_skills")),
                seniority=str(data.get("seniority") or "unknown").strip().lower(),
                domain=str(data.get("domain") or "").strip(),
            )
            log.debug(
                "extract_filters: %.0f ms -> %s",
                (time.perf_counter() - t0) * 1000,
                filters.to_dict(),
            )
            return filters
        except Exception as exc:  # network, auth, JSON, anything
            log.warning("extract_filters failed: %s", exc)
            return None

    # ---- shortlist summary --------------------------------------------

    def shortlist_summary(
        self,
        jd_text: str,
        shortlist: list[ShortlistCandidate],
    ) -> Optional[str]:
        """Return a short executive recommendation across the top-K candidates."""
        if not shortlist:
            return None
        candidates_block = "\n".join(
            f"#{c.rank} {c.name}"
            f" - {c.title or 'unknown title'}"
            f" - {c.experience_years:.0f} yrs"
            f" - score {c.match_score}/100"
            f" - skills: {', '.join(c.matched_skills) or '(none)'}"
            for c in shortlist
        )
        prompt = (
            f"## Job Description\n{jd_text.strip()}\n\n"
            f"## Ranked shortlist (by match_score)\n{candidates_block}\n\n"
            "Write the recommendation now."
        )
        t0 = time.perf_counter()
        try:
            resp = self.client.chat.completions.create(
                model=self.model,
                temperature=0.3,
                max_tokens=220,
                extra_headers=self._extra_headers or None,
                messages=[
                    {"role": "system", "content": SHORTLIST_SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
            )
            text = (resp.choices[0].message.content or "").strip()
            log.debug(
                "shortlist_summary in %.0f ms (%d chars)",
                (time.perf_counter() - t0) * 1000, len(text),
            )
            return text or None
        except Exception as exc:
            log.warning("shortlist_summary failed: %s", exc)
            return None


# ---------------------------------------------------------------------------
# JSON parsing helpers (LLMs sometimes wrap JSON in fences or extra prose)
# ---------------------------------------------------------------------------

_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL | re.IGNORECASE)


def _parse_json_loose(text: str) -> object:
    """Parse JSON that may be wrapped in code fences or surrounded by prose."""
    if not text:
        return None
    # Try as-is first
    try:
        return json.loads(text)
    except Exception:
        pass
    # Strip code fences if present
    m = _FENCE_RE.search(text)
    if m:
        try:
            return json.loads(m.group(1))
        except Exception:
            pass
    # Last resort: find the first {...} block
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end > start:
        try:
            return json.loads(text[start : end + 1])
        except Exception:
            pass
    return None


def _clean_skill_list(value: object) -> list[str]:
    """Normalize a list-of-strings field coming back from the LLM."""
    if not isinstance(value, list):
        return []
    out: list[str] = []
    seen: set[str] = set()
    for item in value:
        if not isinstance(item, str):
            continue
        s = re.sub(r"\s+", " ", item.strip().lower())
        if s and s not in seen:
            seen.add(s)
            out.append(s)
    return out
