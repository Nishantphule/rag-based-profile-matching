"""
cli_pretty.py
=============

Terminal renderer for `resume_rag.py` and `job_matcher.py`.

Uses `rich` to print:
  * A title banner with the project name.
  * A panel summarising the job description.
  * A table of hard filters (if any).
  * One color-coded card per match: rank, name, title, score, skill chips,
    top excerpts and the reasoning paragraph.
  * A footer with retrieval / LLM latency.

The renderer reads from the same `result` dict that `JobMatcher.match()`
already returns, so all data is pulled from a single source of truth.
"""

from __future__ import annotations

import shutil
from typing import Any

from rich.box import HEAVY, ROUNDED
from rich.columns import Columns
from rich.console import Console, Group
from rich.padding import Padding
from rich.panel import Panel
from rich.rule import Rule
from rich.table import Table
from rich.text import Text


# ---------------------------------------------------------------------------
# Console
# ---------------------------------------------------------------------------

_TERM_WIDTH = max(80, min(shutil.get_terminal_size((120, 30)).columns, 120))
console = Console(width=_TERM_WIDTH, highlight=False)


# ---------------------------------------------------------------------------
# Banner
# ---------------------------------------------------------------------------

_BANNER_LINES = [
    " ____      _     ____             __ _ _      ",
    "|  _ \\ ___| |_  |  _ \\ _ __ ___  / _(_) | ___ ",
    "| |_) / _ \\ __| | |_) | '__/ _ \\| |_| | |/ _ \\",
    "|  _ <  __/ |_  |  __/| | | (_) |  _| | |  __/",
    "|_| \\_\\___|\\__| |_|   |_|  \\___/|_| |_|_|\\___|",
    "                                               ",
    "       Matching with RAG  ·  Hybrid Retrieval  ",
]


def banner(subtitle: str | None = None) -> None:
    art = Text("\n".join(_BANNER_LINES), style="bold cyan")
    body = Group(art, Text(subtitle or "", style="dim italic")) if subtitle else art
    console.print(Panel(body, border_style="cyan", box=HEAVY, padding=(0, 2)))


def section(title: str) -> None:
    console.print()
    console.print(Rule(Text(title, style="bold magenta"), style="magenta"))


# ---------------------------------------------------------------------------
# Indexer summary (used by resume_rag.py)
# ---------------------------------------------------------------------------

def render_index_stats(stats: dict[str, Any]) -> None:
    table = Table.grid(padding=(0, 2))
    table.add_column(justify="right", style="dim")
    table.add_column(style="bold green")
    table.add_row("Resumes indexed", f"{stats.get('resumes', 0)}")
    table.add_row("Chunks created", f"{stats.get('chunks', 0)}")
    table.add_row("Avg chunks / resume", f"{stats.get('avg_chunks_per_resume', 0)}")
    table.add_row("Build time", f"{stats.get('elapsed_seconds', 0)} s")
    table.add_row("Collection", f"{stats.get('collection', '')}")
    table.add_row("DB path", f"{stats.get('db_path', '')}")
    console.print(
        Panel(
            table,
            title="[bold]Index built[/bold]",
            border_style="green",
            box=ROUNDED,
            padding=(1, 2),
        )
    )


# ---------------------------------------------------------------------------
# Job-match result
# ---------------------------------------------------------------------------

def render_match_result(result: dict[str, Any]) -> None:
    """Render the full match result returned by `JobMatcher.match()`."""
    _render_jd(result.get("job_description", ""))
    _render_auto_filter(result.get("auto_filter"))
    _render_filters(result.get("filters") or {})
    _render_relaxed_notice(
        result.get("auto_filter_relaxed"),
        result.get("auto_filter_relaxed_skills") or [],
    )

    matches = result.get("top_matches") or []
    if not matches:
        console.print(
            Panel(
                Text(
                    "No candidates passed the filters. "
                    "Try relaxing --min-experience or --required-skills.",
                    style="yellow",
                ),
                border_style="yellow",
                title="No matches",
            )
        )
    else:
        section(f"Top {len(matches)} candidates")
        for rank, m in enumerate(matches, start=1):
            console.print(_render_match_card(rank, m))
            console.print()

    _render_shortlist_summary(result.get("shortlist_summary"))
    _render_footer(result)


# ---- pieces -------------------------------------------------------------

def _render_jd(jd_text: str) -> None:
    title = _first_line(jd_text) or "Job Description"
    snippet = _trim(jd_text, max_lines=8, max_chars=600)
    body = Text(snippet, style="white")
    console.print(
        Panel(
            body,
            title=f"[bold]Job Description — {title}[/bold]",
            border_style="blue",
            box=ROUNDED,
            padding=(1, 2),
        )
    )


def _render_filters(filters: dict[str, Any]) -> None:
    min_exp = float(filters.get("min_experience") or 0)
    req = filters.get("required_skills") or []
    if not min_exp and not req:
        return

    chips: list[Text] = []
    if min_exp:
        chips.append(_chip(f"min experience: {min_exp:g} yrs", "yellow"))
    for s in req:
        chips.append(_chip(f"must have: {s}", "yellow"))

    console.print(Padding(Columns(chips, expand=False, padding=(0, 1)), (1, 0, 0, 0)))


def _render_auto_filter(auto: dict[str, Any] | None) -> None:
    """Show what the LLM auto-extracted from the JD."""
    if not auto:
        return

    rows: list[Text] = []
    seniority = (auto.get("seniority") or "").strip()
    domain = (auto.get("domain") or "").strip()
    if seniority and seniority != "unknown":
        rows.append(Text.assemble(("seniority: ", "dim"), (seniority, "bold cyan")))
    if domain:
        rows.append(Text.assemble(("domain: ", "dim"), (domain, "bold cyan")))

    min_exp = float(auto.get("min_experience_years") or 0)
    rows.append(Text.assemble(
        ("min experience: ", "dim"),
        (f"{min_exp:g} yrs" if min_exp else "(none)", "bold cyan"),
    ))

    must = auto.get("required_skills") or []
    if must:
        rows.append(Text("required skills (must-have):", style="dim"))
        rows.append(Padding(Columns([_chip(s, "cyan") for s in must], padding=(0, 1)), (0, 0, 0, 2)))

    nice = auto.get("nice_to_have_skills") or []
    if nice:
        rows.append(Text("nice-to-have skills:", style="dim"))
        rows.append(Padding(Columns([_chip(s, "blue") for s in nice], padding=(0, 1)), (0, 0, 0, 2)))

    console.print(
        Panel(
            Group(*rows),
            title="[bold cyan]LLM-extracted JD requirements[/bold cyan]",
            border_style="cyan",
            box=ROUNDED,
            padding=(1, 2),
        )
    )


def _render_relaxed_notice(relaxed: Any, attempted: list[str]) -> None:
    if not relaxed:
        return
    skills_str = ", ".join(attempted) if attempted else "(none)"
    msg = (
        f"No candidate had ALL {len(attempted)} LLM-extracted must-have skills "
        f"({skills_str}). Showing the strongest matches without the "
        "skill-intersection filter (min-experience still applied)."
    )
    console.print()
    console.print(
        Panel(
            Text(msg, style="yellow"),
            title="[bold yellow]Auto-filter relaxed[/bold yellow]",
            border_style="yellow",
            box=ROUNDED,
            padding=(0, 1),
        )
    )


def _render_shortlist_summary(summary: str | None) -> None:
    if not summary:
        return
    console.print()
    console.print(
        Panel(
            Text(summary, style="white"),
            title="[bold green]Shortlist recommendation (LLM)[/bold green]",
            border_style="green",
            box=ROUNDED,
            padding=(1, 2),
        )
    )


def _render_match_card(rank: int, m: dict[str, Any]) -> Panel:
    score = int(m.get("match_score") or 0)
    score_style = _score_style(score)
    debug = m.get("_debug") or {}
    years = float(debug.get("experience_years") or 0)
    semantic = float(debug.get("semantic_score") or 0)
    keyword = float(debug.get("keyword_score") or 0)

    name = m.get("candidate_name") or "?"
    path = m.get("resume_path") or ""

    # Header row: rank · name · score
    header = Table.grid(expand=True)
    header.add_column(justify="left")
    header.add_column(justify="right")
    header.add_row(
        Text.assemble(
            (f"#{rank}  ", "bold cyan"),
            (name, "bold white"),
        ),
        Text.assemble(
            ("score ", "dim"),
            (f"{score}", f"bold {score_style}"),
            ("/100", "dim"),
        ),
    )

    # Stats row
    stats = Text.assemble(
        (f"{years:.0f} yrs experience", "white"),
        ("  ·  ", "dim"),
        (f"semantic {semantic:.2f}", "dim"),
        ("  ·  ", "dim"),
        (f"keyword {keyword:.2f}", "dim"),
        ("  ·  ", "dim"),
        (path, "dim italic"),
    )

    # Skills chips
    matched_skills = m.get("matched_skills") or []
    skill_label = Text("Matched skills (intersection w/ JD)", style="bold")
    if matched_skills:
        chips = [_chip(s, "green") for s in matched_skills]
        skill_block = Group(skill_label, Padding(Columns(chips, expand=False, padding=(0, 1)), (0, 0, 0, 0)))
    else:
        skill_block = Group(skill_label, Text("  (none)", style="dim italic"))

    # Excerpts
    excerpts = m.get("relevant_excerpts") or []
    excerpts_block: list[Text] = [Text("Relevant excerpts", style="bold")]
    for e in excerpts[:3]:
        excerpts_block.append(Text(f"  • {_trim(e, max_chars=240)}", style="white"))
    if not excerpts:
        excerpts_block.append(Text("  (none)", style="dim italic"))

    # Reasoning
    reasoning = m.get("reasoning") or ""
    reasoning_block = Group(
        Text("Reasoning", style="bold"),
        Panel(
            Text(reasoning, style="italic"),
            border_style="dim",
            box=ROUNDED,
            padding=(0, 1),
        ),
    )

    body = Group(
        header,
        stats,
        Text(""),
        skill_block,
        Text(""),
        Group(*excerpts_block),
        Text(""),
        reasoning_block,
    )

    return Panel(
        body,
        border_style=score_style,
        box=ROUNDED,
        padding=(1, 2),
    )


def _render_footer(result: dict[str, Any]) -> None:
    latency = result.get("latency_ms")
    llm = result.get("llm_enabled")
    bits: list[Text] = []
    if latency is not None:
        bits.append(Text.assemble(("latency: ", "dim"), (f"{latency} ms", "bold green")))
    bits.append(Text.assemble(
        ("LLM reasoning: ", "dim"),
        ("on", "bold green") if llm else ("off (template)", "dim"),
    ))
    n = len(result.get("top_matches") or [])
    bits.append(Text.assemble(("top_k returned: ", "dim"), (f"{n}", "bold")))

    console.print(Rule(style="dim"))
    console.print(Padding(Columns(bits, expand=False, padding=(0, 3)), (0, 1)))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _score_style(score: int) -> str:
    if score >= 80:
        return "bright_green"
    if score >= 65:
        return "green"
    if score >= 50:
        return "yellow"
    return "red"


def _chip(text: str, style: str) -> Text:
    return Text(f" {text} ", style=f"black on {style}")


def _first_line(text: str) -> str:
    for line in (text or "").splitlines():
        line = line.strip()
        if line:
            return line[:80]
    return ""


def _trim(text: str, max_chars: int = 320, max_lines: int | None = None) -> str:
    if not text:
        return ""
    s = text.strip()
    if max_lines is not None:
        lines = s.splitlines()
        if len(lines) > max_lines:
            s = "\n".join(lines[:max_lines]) + "\n…"
    if len(s) > max_chars:
        s = s[: max_chars - 1].rstrip() + "…"
    return s
