from __future__ import annotations

import re
from typing import Iterable

from .openai_client import OpenAIChatClient

_PLACEHOLDER = re.compile(r"\b(TODO|TBD|INSERT)\b|(\[[^\]]{0,60}\])|(<[^>]{1,60}>)", re.I)


def _truncate(s: str, max_chars: int) -> str:
    s = (s or "").strip()
    if max_chars and len(s) > max_chars:
        return s[: max(0, max_chars - 30)] + "...[TRUNCATED]"
    return s


def craft_query(user_instruction: str) -> str:
    ui = (user_instruction or "").strip()
    if not ui:
        return "Extract the key information needed to fulfill the task; prioritize decisions, constraints, key numbers, risks, and next steps."
    return (
        "From the evidence, extract what is needed to fulfill the instruction. "
        "Prioritize decision-making utility: constraints, key numbers, tradeoffs, risks, and actionable next steps."
    )


def _heuristic_anchor_extension(query: str, required_sections: Iterable[str] | None = None) -> str:
    q = (query or "").strip()
    secs = [s.strip() for s in (required_sections or []) if isinstance(s, str) and s.strip()]
    sec_hint = ", ".join(secs[:8]) if secs else ""
    hints = [
        "methodology", "experimental setup", "datasets", "baselines", "ablation", "results", "metrics",
        "limitations", "failure cases", "tables", "figures", "comparisons",
    ]
    if sec_hint:
        hints.append(f"sections: {sec_hint}")
    if not q:
        return "\n".join(["ANCHOR:", "- " + "\n- ".join(hints)])
    return "\n".join([
        "ANCHOR:",
        "- retrieve concrete evidence: numbers, tables, figures, comparisons",
        "- " + "\n- ".join(hints),
    ])


def generate_anchor_query(
    client: OpenAIChatClient,
    model: str,
    temperature: float,
    query: str,
    user_instruction: str | None = None,
    required_sections: list[str] | None = None,
    max_chars: int = 1400,
) -> tuple[str, dict]:
    """
    Anchor = query extension for retrieval (NOT a summary).
    """
    q = (query or "").strip()
    ui = (user_instruction or "").strip()
    sections = [s for s in (required_sections or []) if isinstance(s, str) and s.strip()]

    system = (
        "You are a retrieval specialist for technical PDFs. "
        "Given a user query/instruction, write a short anchor that EXTENDS the query for better retrieval. "
        "Add key entities, synonyms, acronyms, metrics, baselines, datasets, and where to look (tables/figures/sections). "
        "Do NOT answer the question. Do NOT summarize evidence. Do NOT repeat the original query verbatim. "
        "No placeholders (no TODO/TBD/[])."
    )
    user = (
        f"USER_QUERY:\n{q}\n\n"
        + (f"USER_INSTRUCTION:\n{ui}\n\n" if ui and ui != q else "")
        + ("REQUIRED_SECTIONS_HINTS:\n" + "\n".join([f"- {s}" for s in sections]) + "\n\n" if sections else "")
        + "Write an ANCHOR (query extension) in this exact format:\n"
        "ANCHOR:\n"
        "- 4-10 bullet points with additional search intent (entities, synonyms, metrics, baselines, datasets).\n"
        "- Include at least 1 bullet explicitly requesting TABLES/FIGURES.\n"
        "- Keep it concise (<= 1400 characters).\n"
    )

    res = client.chat(model=model, system=system, user=user, temperature=temperature)
    text = _truncate(res.text or "", max_chars)
    meta = {
        "model": res.model,
        "anchor_kind": "query_extension",
        "has_placeholders": bool(_PLACEHOLDER.search(text)),
        "chars": len(text),
    }
    if not text.strip() or text.strip().lower() == "anchor:":
        text = _truncate(_heuristic_anchor_extension(q, required_sections=sections), max_chars)
        meta["fallback"] = "heuristic"
    return text.strip(), meta


def generate_anchor_summary(
    client: OpenAIChatClient,
    model: str,
    temperature: float,
    user_instruction: str,
    required_sections: list[str],
    selected_chunks: list[dict],
    max_chars_per_chunk: int,
) -> tuple[str, dict]:
    rq = craft_query(user_instruction)
    sections = [s for s in required_sections if isinstance(s, str) and s.strip()] or [
        "Overview", "Key Findings", "Evidence", "Risks", "Next Steps"
    ]

    ctx_lines = []
    for ch in selected_chunks:
        cid = ch.get("chunk_id")
        txt = (ch.get("content") or {}).get("text") if isinstance(ch.get("content"), dict) else None
        ut = (ch.get("metadata") or {}).get("unit_type") if isinstance(ch.get("metadata"), dict) else None
        if not isinstance(cid, str):
            continue
        if not isinstance(txt, str):
            txt = ""
        txt = txt.strip()
        if len(txt) > max_chars_per_chunk:
            txt = txt[: max_chars_per_chunk - 30] + "...[TRUNCATED]"
        ctx_lines.append(f"[{cid}] ({ut or 'unit'}): {txt}")

    system = "You write concise, structured anchor summaries grounded in provided evidence. No placeholders. No repetition."
    user = (
        f"USER_INSTRUCTION:\n{user_instruction}\n\n"
        f"CRAFTED_QUERY:\n{rq}\n\n"
        "EVIDENCE (selected chunks):\n"
        + "\n".join(ctx_lines)
        + "\n\n"
        "Write an anchor summary with EXACTLY these sections, in this order. Use bullet points. Cite chunk ids like [chunk_id]. "
        "Avoid repetition and templated filler.\n"
        + "\n".join([f"- {s}" for s in sections])
        + "\n\n"
        "Output format:\n"
        + "\n".join([f"## {s}\n- ..." for s in sections])
    )

    res = client.chat(model=model, system=system, user=user, temperature=temperature)
    text = (res.text or "").strip()
    meta = {"model": res.model, "crafted_query": rq, "has_placeholders": bool(_PLACEHOLDER.search(text))}
    return text, meta
