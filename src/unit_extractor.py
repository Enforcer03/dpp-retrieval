from __future__ import annotations

import re

from .openai_client import OpenAIChatClient

_UNIT_RE = re.compile(r"<unit>(.*?)</unit>", re.S | re.I)


def extract_units(
    client: OpenAIChatClient,
    model: str,
    temperature: float,
    user_instruction: str,
    selected_chunks: list[dict],
    wrap_unit_tags: bool,
    max_units: int,
    max_chars_per_input: int,
) -> tuple[str, list[dict]]:
    lines = []
    for ch in selected_chunks:
        cid = ch.get("chunk_id")
        md = ch.get("metadata") if isinstance(ch.get("metadata"), dict) else {}
        ut = md.get("unit_type")
        txt = (ch.get("content") or {}).get("text") if isinstance(ch.get("content"), dict) else ""
        if not isinstance(cid, str) or not isinstance(txt, str):
            continue
        txt = txt.strip()
        if len(txt) > 1200:
            txt = txt[:1170] + "...[TRUNCATED]"
        lines.append(f"[{cid}] ({ut or 'unit'}): {txt}")

    blob = "\n".join(lines)
    if len(blob) > max_chars_per_input:
        blob = blob[: max_chars_per_input - 30] + "...[TRUNCATED]"

    system = "You extract coherent knowledge units from evidence. Preserve tables/figures as separate units when present. No placeholders."
    if wrap_unit_tags:
        fmt = (
            "Output up to {N} units. Each unit MUST be wrapped exactly as:\n"
            "<unit>\nTYPE: ...\nSOURCE_CHUNK_IDS: [id1,id2]\nTEXT: ...\n</unit>\n"
            "Keep TEXT concise but complete. Do not use markdown fences."
        ).format(N=max_units)
    else:
        fmt = (
            "Output up to {N} units. Separate units with a blank line. Each unit begins with:\n"
            "TYPE: ...\nSOURCE_CHUNK_IDS: [id1,id2]\nTEXT: ...\n"
            "Keep TEXT concise but complete."
        ).format(N=max_units)

    user = (
        f"USER_INSTRUCTION:\n{user_instruction}\n\n"
        "EVIDENCE (selected chunks):\n"
        f"{blob}\n\n"
        f"{fmt}"
    )

    raw = client.chat(model=model, system=system, user=user, temperature=temperature).text
    parsed = parse_unit_blocks(raw) if wrap_unit_tags else []
    return raw, parsed


def parse_unit_blocks(raw: str) -> list[dict]:
    out = []
    for i, m in enumerate(_UNIT_RE.findall(raw or ""), start=1):
        txt = (m or "").strip()
        if not txt:
            continue
        out.append({"unit_id": f"u{i:03d}", "text": txt})
    return out
