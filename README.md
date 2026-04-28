# RAG-Based Profile Matching System

A production-style Retrieval-Augmented Generation (RAG) pipeline that matches resumes
to job descriptions using **section-aware chunking**, **semantic embeddings**, a
**vector database (ChromaDB)** and **hybrid (semantic + BM25 keyword) search**.

> **For a 5-minute walkthrough,** see [`DEMO.md`](DEMO.md).

## Project Layout

```
rag-based-profile-matching/
├── resume_rag.py              # Part A: document pipeline + vector index
├── job_matcher.py             # Part B: matching engine + CLI
├── llm.py                     # Optional LLM-backed reasoner (OpenRouter)
├── cli_pretty.py              # Rich-based terminal renderer
├── data/
│   ├── resumes/               # 33 sample resumes (.txt)
│   └── job_descriptions/      # 6 sample job descriptions (.txt)
├── notebooks/
│   └── experimentation.ipynb  # Analysis, metrics, latency, charts, LLM demo
├── scripts/
│   └── generate_dataset.py    # Regenerates the resumes / JDs
├── chroma_db/                 # Persistent vector store (auto-created)
├── .env.example               # Copy to .env and set OPENROUTER_API_KEY
├── DEMO.md                    # 5-minute mentor demo runbook
├── requirements.txt
└── README.md
```

## Quick Start

```bash
# On Windows, use --only-binary=:all: so chromadb / hnswlib don't need MSVC.
pip install --only-binary=:all: -r requirements.txt

# (Optional) Regenerate the sample dataset of 33 resumes + 6 JDs
python scripts/generate_dataset.py

# Build the index from data/resumes/
python resume_rag.py --build --resumes-dir data/resumes

# Match a job description (file or string) - pretty colored output by default
python job_matcher.py --jd data/job_descriptions/01_senior_python_ml_engineer.txt --top-k 10

# Same call, raw JSON for downstream tooling
python job_matcher.py --jd data/job_descriptions/01_senior_python_ml_engineer.txt --top-k 10 --json

# Add hard filters (must-have requirements)
python job_matcher.py --jd data/job_descriptions/01_senior_python_ml_engineer.txt ^
                     --min-experience 5 --required-skills python,"machine learning"

# Optional: LLM-generated reasoning (OpenRouter -> openai/gpt-5.2 by default).
# 1) Copy .env.example to .env and set OPENROUTER_API_KEY=...
# 2) Run with --use-llm:
python job_matcher.py --jd data/job_descriptions/02_genai_rag_engineer.txt --use-llm --llm-top-k 3

# LLM auto-extracts hard filters (min_experience + required_skills) from the JD,
# and emits a final shortlist recommendation. Combine with --use-llm for the full
# AI-augmented experience.
python job_matcher.py --jd data/job_descriptions/02_genai_rag_engineer.txt --auto-filter --use-llm

# Reproduce all metrics, ablation and charts
jupyter notebook notebooks/experimentation.ipynb
```

## LLM-Augmented Features (Optional)

The retrieval, ranking and scoring pipeline is **fully deterministic** so you can
audit every score. The LLM is used in three strictly post-retrieval ways, each
gated behind its own flag:

| Flag | What the LLM does | Demo value |
| --- | --- | --- |
| `--use-llm` | Per-candidate **reasoning paragraphs** (top `--llm-top-k`, default 3). | Replaces the template "matched skills X, Y, Z" with grounded, JD-specific prose. |
| `--use-llm` | Final **shortlist recommendation** comparing the top-K. | One-paragraph executive summary at the bottom of pretty output. |
| `--auto-filter` | Parses the JD into structured **hard filters** (`min_experience` + 2-3 must-have skills) before retrieval. | The user just pastes a JD; no need to type `--required-skills python,...` by hand. |

Inputs handed to the LLM are strictly the JD text and the deterministic
retriever's outputs — never the full resume corpus. If the API call fails for
any reason (no key, network, rate-limit, timeout, malformed JSON) the system
silently falls back to deterministic behaviour. Any value the user passes via
`--min-experience` / `--required-skills` always wins over `--auto-filter`.

```bash
# Without the key the same command still works - it just uses the template.
python job_matcher.py --jd <jd_file> --use-llm

# With the key (set via .env or shell env):
$env:OPENROUTER_API_KEY = "sk-or-..."
python job_matcher.py --jd <jd_file> --use-llm --llm-top-k 3
python job_matcher.py --jd <jd_file> --auto-filter            # filters only
python job_matcher.py --jd <jd_file> --auto-filter --use-llm  # full AI mode
```

Defaults (override via `.env`): model `openai/gpt-5.2`, base URL
`https://openrouter.ai/api/v1`, temperature `0.2` (reasoning) / `0.0`
(extraction), 200-800 max output tokens depending on the call.

## Design Decisions

| Concern | Choice | Why |
| --- | --- | --- |
| Embeddings | `sentence-transformers/all-MiniLM-L6-v2` | Free, fast, 384-dim, strong on STS / retrieval |
| Vector DB | **ChromaDB** (persistent) | Zero-ops, local, metadata filtering built-in |
| Chunking | **Section-aware** (Education, Experience, Skills, Projects, …) | Preserves semantic units; better than fixed windows for resumes |
| Keyword search | **BM25** over the same chunks | Catches exact-skill mentions semantic search may miss |
| Hybrid score | `0.7 * semantic + 0.3 * bm25` (configurable) | Balances meaning with literal skill matches |
| Metadata | Name, skills, years of experience, education, file path | Enables hard filters (e.g. `5+ years Python`) |

## Performance (measured in `notebooks/experimentation.ipynb`)

Evaluated on 6 hand-labeled JDs against 33 resumes:

| Metric | Value |
| --- | --- |
| Mean Reciprocal Rank | **1.000** (top-1 is always a labeled ideal candidate) |
| Recall @ 10 | **1.000** (every gold candidate retrieved) |
| Recall @ 5  | 0.958 |
| Retrieval mean latency / JD | **~50 ms** (CPU, after warm-up) |
| Retrieval p95 latency / JD | ~80 ms |
| LLM-reasoning add-on (3 calls) | ~11 s end-to-end (network bound) |

Hybrid retrieval (semantic + BM25) outperforms semantic-only on JDs with
hard skill keywords (e.g. Computer Vision: P@5 0.4 vs 0.2). LLM reasoning is
strictly post-ranking — the retrieval/scoring stay deterministic and fast.

## Output Format

The required assignment schema (always present):

```json
{
  "job_description": "...",
  "top_matches": [
    {
      "candidate_name": "John Doe",
      "resume_path": "data/resumes/john_doe.txt",
      "match_score": 92,
      "matched_skills": ["Python", "Machine Learning"],
      "relevant_excerpts": ["..."],
      "reasoning": "Strong match for ML experience..."
    }
  ]
}
```

The implementation also adds these observability fields on top of the spec
(safe to ignore if you only need the schema above):

- `filters` — the hard filters applied (`min_experience`, `required_skills`).
- `auto_filter` — when `--auto-filter` is on: the structured filter the LLM
  parsed from the JD (`min_experience_years`, `required_skills`,
  `nice_to_have_skills`, `seniority`, `domain`).
- `auto_filter_relaxed` / `auto_filter_relaxed_skills` — `true` if the
  intersection was so strict that nothing passed and the system fell back to a
  relaxed query; the original attempted skills are reported alongside.
- `shortlist_summary` — when `--use-llm` is on: one-paragraph executive
  recommendation across the top-K.
- `latency_ms` — total matcher latency (retrieval + scoring + optional LLM).
- `llm_enabled` — `true` when LLM-augmented reasoning was used.
- `_debug` (per match) — `semantic_score`, `keyword_score`, `experience_years`.
