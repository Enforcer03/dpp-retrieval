from __future__ import annotations

import re
from typing import Iterable

from .openai_client import OpenAIChatClient

_PLACEHOLDER = re.compile(r"\b(TODO|TBD|INSERT)\b|(\[[^\]]{0,60}\])|(<[^>]{1,60}>)", re.I)


def craft_query(user_instruction: str) -> str:
    ui = (user_instruction or "").strip()
    if not ui:
        return "Extract the key information needed to fulfill the task; prioritize decisions, constraints, key numbers, risks, and next steps."
    return (
        "From the evidence, extract what is needed to fulfill the instruction. "
        "Prioritize decision-making utility: constraints, key numbers, tradeoffs, risks, and actionable next steps."
    )


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
    sections = [s for s in required_sections if isinstance(s, str) and s.strip()] or ["Overview", "Key Findings", "Evidence", "Risks", "Next Steps"]

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
