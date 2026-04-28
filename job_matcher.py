"""
job_matcher.py
==============

Part B: Job Matching Engine.

Given a job description, retrieve the top-K most relevant resumes from the
ChromaDB index built by `resume_rag.py`, score them on a 0-100 scale, and
explain why each one matched.

Features
--------
* **Semantic search** via dense embeddings (cosine similarity).
* **Hybrid search**: blends semantic similarity with a BM25 keyword score
  computed over the same chunks. This is essential for hard skills - if the
  JD requires "Kubernetes", a candidate who literally writes "Kubernetes" on
  their resume should rank above one who only writes "container orchestration".
* **Hard filters** for must-have requirements:
      - Minimum years of experience
      - Required skills (intersection-based)
* **Match reasoning**: the top excerpts and a short generated explanation.

Usage
-----
    # Build the index first (see resume_rag.py)
    python resume_rag.py --build

    # Match a JD on disk:
    python job_matcher.py --jd data/job_descriptions/01_senior_python_ml_engineer.txt

    # Add hard requirements:
    python job_matcher.py --jd <file> --min-experience 5 --required-skills python,aws

    # Or pass the JD inline:
    python job_matcher.py --jd-text "We need a senior Python ML engineer ..."
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterable

import numpy as np
from rank_bm25 import BM25Okapi

from resume_rag import (
    DEFAULT_COLLECTION,
    DEFAULT_DB_PATH,
    DEFAULT_EMBED_MODEL,
    SKILL_VOCAB,
    Embedder,
    ResumeRAG,
)

try:  # optional - LLM reasoning is purely additive
    from llm import OpenRouterReasoner, ReasoningInput
except Exception:  # pragma: no cover
    OpenRouterReasoner = None  # type: ignore[assignment]
    ReasoningInput = None      # type: ignore[assignment]

log = logging.getLogger("job_matcher")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
)
# Quiet third-party noise so the demo output stays clean.
for _noisy in ("httpx", "httpcore", "huggingface_hub", "sentence_transformers",
               "urllib3", "chromadb", "openai"):
    logging.getLogger(_noisy).setLevel(logging.ERROR)


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class CandidateMatch:
    candidate_name: str
    resume_path: str
    match_score: int
    matched_skills: list[str]
    relevant_excerpts: list[str]
    reasoning: str
    # Helpful debug fields - kept out of the canonical assignment schema.
    semantic_score: float = 0.0
    keyword_score: float = 0.0
    experience_years: float = 0.0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_TOKEN_RE = re.compile(r"[A-Za-z0-9.+#-]{2,}")

def tokenize(text: str) -> list[str]:
    """Light, retrieval-friendly tokenizer: lowercase, keep tech tokens like c++."""
    return [t.lower() for t in _TOKEN_RE.findall(text)]


def _normalize_skill(s: str) -> str:
    return re.sub(r"[\s_-]+", " ", s.strip().lower())


def extract_jd_skills(jd_text: str) -> list[str]:
    """Pull skill tokens out of a JD using the same controlled vocabulary."""
    text_lc = jd_text.lower()
    found: list[str] = []
    for skill in SKILL_VOCAB:
        if " " in skill or "." in skill or "+" in skill or "-" in skill:
            if skill in text_lc:
                found.append(skill)
        else:
            if re.search(rf"\b{re.escape(skill)}\b", text_lc):
                found.append(skill)
    return sorted(set(found))


def extract_min_experience_from_jd(jd_text: str) -> float:
    """Best-effort '5+ years' style parsing - used when the user doesn't pass --min-experience."""
    m = re.search(r"(\d+(?:\.\d+)?)\s*\+?\s*years?", jd_text, flags=re.I)
    return float(m.group(1)) if m else 0.0


# ---------------------------------------------------------------------------
# Hybrid retrieval
# ---------------------------------------------------------------------------

class HybridRetriever:
    """Blend dense (Chroma) retrieval with sparse BM25 over the same chunks.

    We pull a wider candidate pool from Chroma (n_dense) and BM25 (n_sparse),
    union the chunks, normalize scores into [0, 1], and combine with weights.
    """

    def __init__(
        self,
        rag: ResumeRAG,
        semantic_weight: float = 0.7,
        keyword_weight: float = 0.3,
        dense_pool: int = 50,
        sparse_pool: int = 50,
    ) -> None:
        if not (0 <= semantic_weight <= 1 and 0 <= keyword_weight <= 1):
            raise ValueError("weights must be in [0, 1]")
        self.rag = rag
        self.w_sem = semantic_weight
        self.w_kw = keyword_weight
        self.dense_pool = dense_pool
        self.sparse_pool = sparse_pool
        self._bm25, self._bm25_ids, self._bm25_docs, self._bm25_meta = self._build_bm25()

    # ---- BM25 corpus --------------------------------------------------

    def _build_bm25(self) -> tuple[BM25Okapi, list[str], list[str], list[dict]]:
        col = self.rag.collection
        # Pull *everything* once for BM25. The corpus is small (resumes), so
        # this is fine. For very large corpora we'd page or use an inverted index.
        all_data = col.get(include=["documents", "metadatas"])
        ids = all_data["ids"]
        docs = all_data["documents"]
        metas = all_data["metadatas"]
        if not docs:
            return BM25Okapi([[""]]), [], [], []
        tokenized = [tokenize(d) for d in docs]
        return BM25Okapi(tokenized), ids, docs, metas

    # ---- search -------------------------------------------------------

    def search(self, query_text: str, where: dict | None = None) -> list[dict]:
        """Return chunk-level results sorted by hybrid score (desc)."""
        # ---- dense ---------------------------------------------------
        dense_res = self.rag.query(query_text, top_k=self.dense_pool, where=where)
        dense_ids: list[str] = (dense_res.get("ids") or [[]])[0]
        dense_docs: list[str] = (dense_res.get("documents") or [[]])[0]
        dense_metas: list[dict] = (dense_res.get("metadatas") or [[]])[0]
        dense_dists: list[float] = (dense_res.get("distances") or [[]])[0]
        # cosine distance -> similarity in [0, 1]
        dense_sims = {i: max(0.0, 1.0 - float(d)) for i, d in zip(dense_ids, dense_dists)}

        # ---- sparse --------------------------------------------------
        if self._bm25_docs:
            scores = self._bm25.get_scores(tokenize(query_text))
            top_sparse_idx = np.argsort(scores)[::-1][: self.sparse_pool]
            max_score = float(scores.max()) if scores.size else 0.0
            sparse_sims = {
                self._bm25_ids[i]: (float(scores[i]) / max_score) if max_score > 0 else 0.0
                for i in top_sparse_idx
            }
        else:
            sparse_sims = {}

        # ---- merge ---------------------------------------------------
        # Build a lookup of doc/meta for everything we've seen.
        doc_lookup: dict[str, str] = {}
        meta_lookup: dict[str, dict] = {}
        for cid, d, m in zip(dense_ids, dense_docs, dense_metas):
            doc_lookup[cid] = d
            meta_lookup[cid] = m
        for cid in sparse_sims:
            if cid not in doc_lookup:
                idx = self._bm25_ids.index(cid)
                doc_lookup[cid] = self._bm25_docs[idx]
                meta_lookup[cid] = self._bm25_meta[idx]

        all_ids = set(dense_sims) | set(sparse_sims)
        results: list[dict] = []
        for cid in all_ids:
            sem = dense_sims.get(cid, 0.0)
            kw = sparse_sims.get(cid, 0.0)
            score = self.w_sem * sem + self.w_kw * kw
            meta = meta_lookup[cid]
            # Apply post-hoc metadata filter for ids that came in only from BM25
            # (Chroma already filtered the dense side via `where=`).
            if where and not _matches_where(meta, where):
                continue
            results.append(
                {
                    "chunk_id": cid,
                    "resume_id": meta.get("resume_id", ""),
                    "section": meta.get("section", ""),
                    "text": doc_lookup[cid],
                    "metadata": meta,
                    "semantic": sem,
                    "keyword": kw,
                    "score": score,
                }
            )

        results.sort(key=lambda r: r["score"], reverse=True)
        return results


def _matches_where(meta: dict, where: dict) -> bool:
    """A tiny subset of Chroma's where-filter, applied client-side to BM25 hits.

    Supports: equality and {'$gte': v} / {'$lte': v}.
    """
    for key, cond in where.items():
        val = meta.get(key)
        if isinstance(cond, dict):
            for op, target in cond.items():
                if op == "$gte" and not (val is not None and val >= target):
                    return False
                if op == "$lte" and not (val is not None and val <= target):
                    return False
                if op == "$eq" and val != target:
                    return False
        else:
            if val != cond:
                return False
    return True


# ---------------------------------------------------------------------------
# Job Matcher
# ---------------------------------------------------------------------------

class JobMatcher:
    """End-to-end matcher: takes a JD, returns ranked candidates with reasoning."""

    def __init__(
        self,
        db_path: str = DEFAULT_DB_PATH,
        collection_name: str = DEFAULT_COLLECTION,
        embed_model: str = DEFAULT_EMBED_MODEL,
        semantic_weight: float = 0.7,
        keyword_weight: float = 0.3,
        reasoner: "OpenRouterReasoner | None" = None,
        llm_top_k: int = 3,
    ) -> None:
        self.rag = ResumeRAG(
            db_path=db_path,
            collection_name=collection_name,
            embed_model=embed_model,
        )
        if self.rag.collection.count() == 0:
            log.warning(
                "Collection '%s' is empty - run `python resume_rag.py --build` first.",
                collection_name,
            )
        self.retriever = HybridRetriever(
            self.rag,
            semantic_weight=semantic_weight,
            keyword_weight=keyword_weight,
        )
        self.reasoner = reasoner
        self.llm_top_k = max(0, int(llm_top_k))

    # ---- main entry ----------------------------------------------------

    def match(
        self,
        job_description: str,
        top_k: int = 10,
        min_experience: float = 0.0,
        required_skills: list[str] | None = None,
    ) -> dict:
        """Run the full matching pipeline.

        Returns the assignment-shaped dict:
            {"job_description": ..., "top_matches": [ ... ]}
        """
        t0 = time.perf_counter()
        required_skills = [_normalize_skill(s) for s in (required_skills or [])]

        where = self._build_where(min_experience)
        chunk_hits = self.retriever.search(job_description, where=where)

        # ---- aggregate to the resume level ---------------------------
        per_resume: dict[str, dict] = {}
        for hit in chunk_hits:
            rid = hit["resume_id"]
            slot = per_resume.setdefault(
                rid,
                {
                    "resume_id": rid,
                    "metadata": hit["metadata"],
                    "chunks": [],
                    "best_section_scores": {},
                    "semantic_max": 0.0,
                    "keyword_max": 0.0,
                },
            )
            slot["chunks"].append(hit)
            slot["semantic_max"] = max(slot["semantic_max"], hit["semantic"])
            slot["keyword_max"] = max(slot["keyword_max"], hit["keyword"])
            sec = hit["section"]
            slot["best_section_scores"][sec] = max(
                slot["best_section_scores"].get(sec, 0.0), hit["score"]
            )

        # Apply required-skills hard filter (intersection).
        jd_skills = extract_jd_skills(job_description)
        candidates: list[CandidateMatch] = []
        for rid, slot in per_resume.items():
            meta = slot["metadata"]
            cand_skills = self._skills_from_meta(meta)

            if required_skills:
                missing = [s for s in required_skills if s not in cand_skills]
                if missing:
                    continue  # hard filter

            matched_skills = sorted(set(jd_skills) & set(cand_skills))

            # Score: best semantic score across this resume's chunks, blended
            # with skill coverage and keyword score, scaled to 0-100.
            score = self._score(
                semantic_max=slot["semantic_max"],
                keyword_max=slot["keyword_max"],
                jd_skills=jd_skills,
                matched_skills=matched_skills,
            )

            excerpts = self._select_excerpts(slot["chunks"], k=3)
            reasoning = self._build_reasoning(
                meta=meta,
                jd_skills=jd_skills,
                matched_skills=matched_skills,
                section_scores=slot["best_section_scores"],
                semantic=slot["semantic_max"],
                keyword=slot["keyword_max"],
            )

            candidates.append(
                CandidateMatch(
                    candidate_name=meta.get("name", rid),
                    resume_path=meta.get("file_path", ""),
                    match_score=int(round(score)),
                    matched_skills=[s.title() if " " not in s else s.title() for s in matched_skills],
                    relevant_excerpts=excerpts,
                    reasoning=reasoning,
                    semantic_score=round(float(slot["semantic_max"]), 4),
                    keyword_score=round(float(slot["keyword_max"]), 4),
                    experience_years=float(meta.get("experience_years", 0.0) or 0.0),
                )
            )

        candidates.sort(key=lambda c: c.match_score, reverse=True)
        top = candidates[:top_k]

        # LLM enrichment - only on the top results we'll actually return.
        # The deterministic reasoning we already wrote stays as a fallback.
        if self.reasoner is not None and self.llm_top_k > 0:
            self._enrich_with_llm(top[: self.llm_top_k], job_description)

        elapsed_ms = int((time.perf_counter() - t0) * 1000)
        log.info("Matched %d / %d candidates in %d ms", len(top), len(candidates), elapsed_ms)

        return {
            "job_description": job_description,
            "filters": {
                "min_experience": min_experience,
                "required_skills": required_skills,
            },
            "latency_ms": elapsed_ms,
            "llm_enabled": self.reasoner is not None,
            "top_matches": [self._public_dict(c) for c in top],
        }

    # ---- LLM enrichment ----------------------------------------------

    def _enrich_with_llm(self, top: list["CandidateMatch"], job_description: str) -> None:
        """Replace the template reasoning with an LLM-generated one, in-place.

        Failures are silent: the deterministic reasoning is kept on any error.
        """
        if not top or self.reasoner is None or ReasoningInput is None:
            return
        for c in top:
            evidence = ReasoningInput(
                job_description=job_description,
                candidate_name=c.candidate_name,
                candidate_title="",
                experience_years=c.experience_years,
                matched_skills=c.matched_skills,
                excerpts=c.relevant_excerpts,
                semantic_score=c.semantic_score,
                keyword_score=c.keyword_score,
            )
            generated = self.reasoner.generate(evidence)
            if generated:
                c.reasoning = generated

    # ---- internals ----------------------------------------------------

    @staticmethod
    def _public_dict(c: CandidateMatch) -> dict:
        """Match the assignment's required JSON schema (extras kept under `_debug`)."""
        full = asdict(c)
        debug = {
            "semantic_score": full.pop("semantic_score"),
            "keyword_score": full.pop("keyword_score"),
            "experience_years": full.pop("experience_years"),
        }
        full["_debug"] = debug
        return full

    @staticmethod
    def _skills_from_meta(meta: dict) -> list[str]:
        s = meta.get("skills_str") or ""
        return [_normalize_skill(x) for x in s.split(",") if x.strip()]

    def _build_where(self, min_experience: float) -> dict | None:
        if min_experience and min_experience > 0:
            return {"experience_years": {"$gte": float(min_experience)}}
        return None

    def _score(
        self,
        semantic_max: float,
        keyword_max: float,
        jd_skills: list[str],
        matched_skills: list[str],
    ) -> float:
        """Combine retrieval signals with skill coverage into a 0-100 score.

        - Retrieval contributes up to ~70 points.
        - Skill coverage contributes up to ~30 points.
        """
        retrieval = self.retriever.w_sem * semantic_max + self.retriever.w_kw * keyword_max
        coverage = (len(matched_skills) / len(jd_skills)) if jd_skills else 0.0
        score = 70.0 * retrieval + 30.0 * coverage
        return max(0.0, min(100.0, score))

    @staticmethod
    def _select_excerpts(chunks: list[dict], k: int = 3) -> list[str]:
        # Diverse top-k by section: at most one excerpt per section unless we run out.
        chunks_sorted = sorted(chunks, key=lambda c: c["score"], reverse=True)
        seen_sections: set[str] = set()
        out: list[str] = []
        for c in chunks_sorted:
            if c["section"] in seen_sections:
                continue
            seen_sections.add(c["section"])
            out.append(_truncate(c["text"], 320))
            if len(out) >= k:
                return out
        # Fill remaining slots if needed.
        for c in chunks_sorted:
            txt = _truncate(c["text"], 320)
            if txt not in out:
                out.append(txt)
            if len(out) >= k:
                break
        return out

    @staticmethod
    def _build_reasoning(
        meta: dict,
        jd_skills: list[str],
        matched_skills: list[str],
        section_scores: dict[str, float],
        semantic: float,
        keyword: float,
    ) -> str:
        # Pick the top 2 sections by score for the reasoning string.
        top_sections = sorted(section_scores.items(), key=lambda x: x[1], reverse=True)[:2]
        section_phrase = (
            ", ".join(s for s, _ in top_sections) if top_sections else "overall resume"
        )

        years = meta.get("experience_years", 0)
        coverage = f"{len(matched_skills)}/{len(jd_skills)}" if jd_skills else "0/0"

        skill_phrase = ", ".join(matched_skills[:5]) if matched_skills else "no listed JD skills"
        title = meta.get("title") or "the candidate"

        return (
            f"{title} ({years:.0f} yrs experience) covers {coverage} JD skills "
            f"({skill_phrase}). Strongest signal came from the "
            f"{section_phrase} section{'s' if len(top_sections) > 1 else ''} "
            f"(semantic={semantic:.2f}, keyword={keyword:.2f})."
        )


def _truncate(text: str, n: int) -> str:
    text = re.sub(r"\s+", " ", text).strip()
    return text if len(text) <= n else text[: n - 1].rstrip() + "…"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Match resumes to a job description (Part B)")
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--jd", help="Path to a job description text file")
    src.add_argument("--jd-text", help="Job description as an inline string")

    p.add_argument("--top-k", type=int, default=10)
    p.add_argument("--min-experience", type=float, default=0.0,
                   help="Filter out candidates with fewer than N years of experience.")
    p.add_argument("--required-skills", default="",
                   help="Comma-separated must-have skills (case-insensitive).")
    p.add_argument("--db-path", default=DEFAULT_DB_PATH)
    p.add_argument("--collection", default=DEFAULT_COLLECTION)
    p.add_argument("--semantic-weight", type=float, default=0.7)
    p.add_argument("--keyword-weight", type=float, default=0.3)

    # LLM reasoning (optional, requires OPENROUTER_API_KEY)
    p.add_argument("--use-llm", action="store_true",
                   help="Replace template reasoning with LLM-generated reasoning for the top candidates.")
    p.add_argument("--llm-top-k", type=int, default=3,
                   help="How many top candidates get LLM-generated reasoning (rest use the template).")
    p.add_argument("--llm-model", default=None,
                   help="Override LLM model (default: openai/gpt-5.2 via OpenRouter).")

    p.add_argument("--json", action="store_true",
                   help="Print raw JSON instead of the pretty rich-formatted output.")
    p.add_argument("--output", help="If given, write the JSON result to this path.")
    return p.parse_args()


def main() -> None:
    args = _parse_args()

    # In pretty mode, silence our own INFO logs so the rich panels render cleanly,
    # and show the banner up-front so users see something immediately while the
    # embedding model warms up.
    if not args.json:
        for _name in ("job_matcher", "resume_rag", "llm"):
            logging.getLogger(_name).setLevel(logging.WARNING)
        try:
            from cli_pretty import banner, console as _console
            banner(subtitle="Loading embedding model and warming up retriever...")
        except Exception:  # pragma: no cover
            pass

    jd_text = Path(args.jd).read_text(encoding="utf-8") if args.jd else args.jd_text
    required_skills = [s.strip() for s in args.required_skills.split(",") if s.strip()]

    reasoner = None
    if args.use_llm:
        if OpenRouterReasoner is None:
            log.error("--use-llm requested but the `openai` package is not installed.")
        else:
            kwargs = {"model": args.llm_model} if args.llm_model else {}
            reasoner = OpenRouterReasoner.from_env(**kwargs)
            if reasoner is None:
                log.error("--use-llm requested but OPENROUTER_API_KEY is not set; "
                          "falling back to template reasoning.")

    matcher = JobMatcher(
        db_path=args.db_path,
        collection_name=args.collection,
        semantic_weight=args.semantic_weight,
        keyword_weight=args.keyword_weight,
        reasoner=reasoner,
        llm_top_k=args.llm_top_k,
    )
    result = matcher.match(
        job_description=jd_text,
        top_k=args.top_k,
        min_experience=args.min_experience,
        required_skills=required_skills,
    )

    if args.json:
        print(json.dumps(result, indent=2, ensure_ascii=False))
    else:
        try:
            from cli_pretty import render_match_result
            render_match_result(result)
        except Exception as exc:  # pragma: no cover - safety net
            log.warning("Pretty renderer failed (%s); falling back to JSON.", exc)
            print(json.dumps(result, indent=2, ensure_ascii=False))

    if not args.json:
        # Restore log level for any teardown messages (no-op for now).
        pass

    if args.output:
        Path(args.output).write_text(
            json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
        )


if __name__ == "__main__":
    main()
