from __future__ import annotations

import re
from typing import Any

DEFAULT_REQUIRED_SECTIONS = ["Overview", "Key Points", "Evidence", "Risks", "Next Steps"]

_MATH = re.compile(r"(\$[^$]{1,120}\$|\\\(|\\\)|\\\[|\\\])")
_CODE = re.compile(r"```|^\s*(def|class)\s+\w+\s*\(", re.M)
_TABLE = re.compile(r"<table\b|\|\s*-{2,}\s*\|", re.I)
_UNCLEAR = re.compile(r"\[UNCLEAR\]|\bUNCLEAR\b", re.I)
_OCR_BAD = re.compile(r"[��]")


def export_eval_packet(pipeline_output: dict) -> dict:
    run_id = _first(pipeline_output, ["run_id", "id", "run", "meta.run_id"])
    user_instruction = _first(pipeline_output, ["task.user_instruction", "user_instruction", "instruction", "task.prompt", "query"])
    required_sections = _first(pipeline_output, ["task.required_sections", "required_sections"])
    if not isinstance(required_sections, list) or not required_sections:
        required_sections = DEFAULT_REQUIRED_SECTIONS

    selected_ids = _selected_ids(pipeline_output)
    budgets = _first(pipeline_output, ["retrieval.budgets", "budgets", "selection.budgets", "retrieval.budget", "budget"])
    if not isinstance(budgets, dict):
        budgets = {}

    lam = _first(
        pipeline_output,
        ["retrieval.objective.lambda_diversity", "objective.lambda_diversity", "lambda_diversity", "diversity_lambda", "lambda"],
    )
    lam = float(lam) if isinstance(lam, (int, float)) else None

    anchors = _anchors(pipeline_output)
    summarization = _summarization(pipeline_output)

    chunk_map = _collect_chunks(pipeline_output)
    selected_chunks = []
    for cid in selected_ids:
        c = chunk_map.get(cid) or {}
        text = _chunk_text(c)
        selected_chunks.append(
            {
                "chunk_id": cid,
                "content": {"text": _truncate(text, 12000) if isinstance(text, str) else None},
                "metadata": _chunk_meta(c, text),
            }
        )

    pkt = {
        "run_id": run_id if isinstance(run_id, str) else None,
        "task": {"user_instruction": user_instruction if isinstance(user_instruction, str) else None, "required_sections": required_sections},
        "retrieval": {"selected_chunk_ids": selected_ids, "budgets": budgets, "objective": {"lambda_diversity": lam}},
        "chunks": selected_chunks,
    }
    if anchors is not None:
        pkt["anchors"] = anchors
    if summarization is not None:
        pkt["summarization"] = summarization
    return pkt


def _truncate(text: str, max_chars: int) -> str:
    t = text or ""
    if len(t) <= max_chars:
        return t
    return t[: max_chars - 50] + "\n...[TRUNCATED]..."


def _first(obj: Any, paths: list[str]) -> Any:
    for p in paths:
        v = _get(obj, p)
        if v is not None:
            return v
    return None


def _get(obj: Any, path: str) -> Any:
    cur = obj
    for part in path.split("."):
        if isinstance(cur, dict) and part in cur:
            cur = cur[part]
        else:
            return None
    return cur


def _selected_ids(obj: dict) -> list[str]:
    v = _first(obj, ["retrieval.selected_chunk_ids", "selected_chunk_ids", "selection.selected_chunk_ids", "selected_ids"])
    ids = _to_str_list(v)
    if ids:
        return ids
    for k in ("selected", "selected_chunks", "retrieval.selected"):
        lst = _get(obj, k)
        if isinstance(lst, list):
            out = []
            for it in lst:
                if isinstance(it, dict):
                    cid = it.get("chunk_id") or it.get("unit_id") or it.get("id")
                    if isinstance(cid, (int, float)):
                        cid = str(cid)
                    if isinstance(cid, str):
                        out.append(cid)
            if out:
                return out
    return []


def _to_str_list(v: Any) -> list[str]:
    if v is None:
        return []
    if isinstance(v, str):
        return [v]
    if isinstance(v, list):
        out = []
        for x in v:
            if isinstance(x, str):
                out.append(x)
            elif isinstance(x, (int, float)):
                out.append(str(x))
        return out
    return []


def _collect_chunks(obj: dict) -> dict[str, dict]:
    lists = _find_chunk_lists(obj)
    if not lists:
        return {}
    best = max(lists, key=len)
    out: dict[str, dict] = {}
    for it in best:
        if not isinstance(it, dict):
            continue
        cid = it.get("chunk_id") or it.get("unit_id") or it.get("id")
        if isinstance(cid, (int, float)):
            cid = str(cid)
        if isinstance(cid, str):
            out[cid] = it
    return out


def _find_chunk_lists(obj: Any) -> list[list[dict]]:
    found: list[list[dict]] = []
    q = [obj]
    seen = 0
    while q and seen < 4000:
        cur = q.pop(0)
        seen += 1
        if isinstance(cur, dict):
            for v in cur.values():
                q.append(v)
        elif isinstance(cur, list):
            if cur and all(isinstance(x, dict) for x in cur):
                if any(any(k in x for k in ("chunk_id", "unit_id", "id")) for x in cur[:5]):
                    found.append(cur)  # type: ignore
            for v in cur:
                q.append(v)
    return found


def _chunk_text(chunk: dict) -> str | None:
    if not isinstance(chunk, dict):
        return None
    for k in ("text", "content", "context_text", "retrieval_text", "raw_text"):
        v = chunk.get(k)
        if isinstance(v, str) and v.strip():
            return v
        if isinstance(v, dict):
            t = v.get("text")
            if isinstance(t, str) and t.strip():
                return t
    c = chunk.get("content")
    if isinstance(c, dict):
        for kk in ("text", "md", "markdown", "html"):
            vv = c.get(kk)
            if isinstance(vv, str) and vv.strip():
                return vv
    return None


def _chunk_meta(chunk: dict, text: str | None) -> dict:
    m = chunk.get("metadata") if isinstance(chunk.get("metadata"), dict) else {}
    out = {}

    unit_type = m.get("unit_type") or chunk.get("unit_type") or m.get("type") or chunk.get("type") or m.get("kind") or chunk.get("kind")
    out["unit_type"] = unit_type if isinstance(unit_type, str) else None

    cost = m.get("cognitive_cost") or chunk.get("cognitive_cost") or m.get("cost") or chunk.get("cost")
    out["cognitive_cost"] = float(cost) if isinstance(cost, (int, float)) else None

    imp = m.get("importance_score") or chunk.get("importance_score") or m.get("importance") or chunk.get("importance")
    out["importance_score"] = float(imp) if isinstance(imp, (int, float)) else None

    t = text or ""
    ut = (out["unit_type"] or "").lower()
    out["contains_tables"] = bool(_TABLE.search(t) or ("table" in ut))
    out["contains_math"] = bool(_MATH.search(t))
    out["contains_code"] = bool(_CODE.search(t))

    eq = m.get("extraction_quality") or chunk.get("extraction_quality")
    eqv = eq.lower() if isinstance(eq, str) else "unknown"
    if eqv not in ("high", "medium", "low", "unknown"):
        eqv = "unknown"

    flags: list[str] = []
    if _UNCLEAR.search(t):
        flags.append("HAS_UNCLEAR_TOKENS")
    if _OCR_BAD.search(t) or _ocr_hint(chunk, m):
        flags.append("OCR_SUSPECT")
    if out["contains_tables"]:
        flags.append("HAS_TABLE")
    if "figure" in ut:
        flags.append("HAS_FIGURE")

    if eqv == "unknown":
        if "HAS_UNCLEAR_TOKENS" in flags or "OCR_SUSPECT" in flags:
            eqv = "low"

    out["extraction_quality"] = eqv
    out["flags"] = flags
    return out


def _ocr_hint(chunk: dict, meta: dict) -> bool:
    src = meta.get("source") or chunk.get("source") or meta.get("provenance") or chunk.get("provenance")
    if isinstance(src, str) and "ocr" in src.lower():
        return True
    risk = meta.get("risk") or chunk.get("risk")
    if isinstance(risk, str) and "ocr" in risk.lower():
        return True
    return False


def _anchors(obj: dict) -> dict | None:
    a = _first(obj, ["anchors", "anchor", "anchor_summary", "anchor_summary_text"])
    if isinstance(a, dict):
        t = a.get("anchor_summary_text") or a.get("text") or a.get("summary")
        if isinstance(t, str) and t.strip():
            return {"anchor_summary_text": t}
        return None
    if isinstance(a, str) and a.strip():
        return {"anchor_summary_text": a}
    return None


def _summarization(obj: dict) -> dict | None:
    s = _first(obj, ["summarization", "summary", "final_summary"])
    if isinstance(s, dict):
        out = {}
        st = s.get("summary_text") or s.get("text")
        ss = s.get("summary_structured") or s.get("structured")
        if isinstance(st, str) and st.strip():
            out["summary_text"] = st
        if isinstance(ss, dict) and ss:
            out["summary_structured"] = ss
        return out or None
    if isinstance(s, str) and s.strip():
        return {"summary_text": s}
    return None
