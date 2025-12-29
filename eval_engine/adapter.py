# eval_engine/adapter.py
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

_TRUNC = 1600


def _get(d: Any, *keys: str) -> Any:
    cur = d
    for k in keys:
        if not isinstance(cur, dict):
            return None
        cur = cur.get(k)
    return cur


def _first_str(*vals: Any) -> str | None:
    for v in vals:
        if isinstance(v, str) and v.strip():
            return v.strip()
    return None


def _read_text_if_path(p: Any, max_chars: int = 20000) -> str | None:
    if not isinstance(p, str) or not p:
        return None
    try:
        fp = Path(p)
        if fp.exists() and fp.is_file():
            return fp.read_text(encoding="utf-8", errors="ignore")[:max_chars]
    except Exception:
        return None
    return None


def _truncate(s: Any, n: int = _TRUNC) -> str | None:
    if not isinstance(s, str):
        return None
    s = s.strip()
    if not s:
        return None
    return s[:n]


def _extract_requirements(obj: dict) -> list[dict]:
    reqs = _get(obj, "requirements")
    if not isinstance(reqs, list):
        return []
    out: list[dict] = []
    for r in reqs:
        if not isinstance(r, dict):
            continue
        rid = r.get("id")
        desc = r.get("description")
        if isinstance(rid, str) and isinstance(desc, str):
            out.append({"id": rid, "description": desc})
    return out


def _extract_decisions(obj: dict) -> list[str]:
    dec = _get(obj, "task", "decisions")
    if isinstance(dec, list):
        return [d.strip() for d in dec if isinstance(d, str) and d.strip()]
    if isinstance(dec, str) and dec.strip():
        return [dec.strip()]

    instr = _first_str(
        _get(obj, "user_instruction"),
        _get(obj, "query"),
        _get(obj, "instruction"),
        _get(obj, "run", "user_instruction"),
        _get(obj, "run", "query"),
    )
    return [instr] if instr else []


def _extract_selected_ids(obj: dict) -> list[str]:
    ids = _get(obj, "retrieval", "selected_chunk_ids")
    if isinstance(ids, list):
        return [x.strip() for x in ids if isinstance(x, str) and x.strip()]

    ids = obj.get("selected_chunk_ids") or obj.get("selected_unit_ids")
    if isinstance(ids, list):
        return [x.strip() for x in ids if isinstance(x, str) and x.strip()]

    chunks = obj.get("selected_chunks") or obj.get("selected_units") or _get(obj, "retrieval", "selected_chunks")
    if isinstance(chunks, list):
        out: list[str] = []
        for c in chunks:
            if not isinstance(c, dict):
                continue
            cid = c.get("chunk_id") or c.get("unit_id") or c.get("id")
            if isinstance(cid, str) and cid.strip():
                out.append(cid.strip())
        return out

    return []


def _extract_budgets(obj: dict) -> dict:
    b = _get(obj, "retrieval", "budgets")
    if isinstance(b, dict):
        return dict(b)

    b = obj.get("budgets")
    if isinstance(b, dict):
        return dict(b)

    total = obj.get("budgets_total") or obj.get("budget_total") or _get(obj, "selection", "budget_tokens")
    used = obj.get("budgets_used") or obj.get("budget_used") or obj.get("used_budget") or _get(obj, "selection", "used_budget")
    out: dict[str, Any] = {}
    if isinstance(total, (int, float)):
        out["total"] = total
    if isinstance(used, (int, float)):
        out["used"] = used
    return out


def _extract_lambda(obj: dict) -> float | None:
    lam = _get(obj, "retrieval", "objective", "lambda_diversity")
    if isinstance(lam, (int, float)):
        return float(lam)

    lam = _get(obj, "objective", "lambda_diversity")
    if isinstance(lam, (int, float)):
        return float(lam)

    lam = obj.get("lambda_diversity") or _get(obj, "selection", "lambda_diversity") or _get(obj, "selection", "lambda")
    if isinstance(lam, (int, float)):
        return float(lam)
    return None


def _extract_parent_path(obj: dict) -> str | None:
    return _first_str(
        _get(obj, "parent", "path"),
        obj.get("pdf"),
        obj.get("pdf_path"),
        _get(obj, "input", "pdf"),
        _get(obj, "run", "pdf"),
        _get(obj, "run", "pdf_path"),
        _get(obj, "meta", "pdf_path"),
    )


def _extract_run_id(obj: dict) -> str | None:
    return _first_str(
        obj.get("run_id"),
        obj.get("id"),
        _get(obj, "run", "id"),
        _get(obj, "meta", "run_id"),
    )


def _extract_selected_chunks(obj: dict, selected_ids: list[str]) -> list[dict]:
    chunks = obj.get("selected_chunks") or obj.get("selected_units") or _get(obj, "retrieval", "selected_chunks")
    if not isinstance(chunks, list):
        all_chunks = obj.get("chunks") or obj.get("units") or _get(obj, "retrieval", "chunks")
        if isinstance(all_chunks, list) and selected_ids:
            sel = set(selected_ids)
            chunks = [
                c for c in all_chunks
                if isinstance(c, dict) and (c.get("chunk_id") or c.get("unit_id") or c.get("id")) in sel
            ]
        else:
            chunks = []

    out: list[dict] = []
    for c in chunks:
        if not isinstance(c, dict):
            continue
        cid = c.get("chunk_id") or c.get("unit_id") or c.get("id")
        if not isinstance(cid, str) or not cid.strip():
            continue

        text = _truncate(
            _get(c, "content", "text")
            or c.get("text")
            or _get(c, "content", "markdown")
            or c.get("markdown")
        )

        image_paths = _get(c, "content", "image_paths") or c.get("image_paths") or c.get("images")
        if isinstance(image_paths, str):
            image_paths = [image_paths]
        if not isinstance(image_paths, list):
            image_paths = []
        image_paths = [p for p in image_paths if isinstance(p, str) and p.strip()]

        meta = c.get("metadata") if isinstance(c.get("metadata"), dict) else {}
        keep_keys = ["page", "pages", "type", "kind", "role", "is_table", "is_figure", "cognitive_cost", "cost", "token_cost"]
        meta_min = {k: meta.get(k) for k in keep_keys if k in meta}
        for k in keep_keys:
            if k in c and k not in meta_min:
                meta_min[k] = c.get(k)

        out.append(
            {
                "chunk_id": cid.strip(),
                "content": {"text": text, "image_paths": image_paths},
                "metadata": meta_min,
            }
        )
    return out


def _unwrap_hybrid(obj: dict) -> dict:
    """
    Optional: support inputs shaped like:
      {"pipeline": {...pipeline_output...}, "dataset": {...dataset_entry...}}
    or:
      {"pipeline_output": {...}, "dataset_entry": {...}}

    We keep retrieval/chunks from pipeline, and overlay task/requirements/parent/id from dataset.
    """
    pairs = [
        ("pipeline", "dataset"),
        ("pipeline_output", "dataset_entry"),
    ]
    for pk, dk in pairs:
        p = obj.get(pk)
        d = obj.get(dk)
        if isinstance(p, dict) and isinstance(d, dict):
            merged = dict(p)
            for k in ("id", "parent", "task", "requirements", "notes"):
                if k in d:
                    merged[k] = d.get(k)
            return merged
    return obj


def export_eval_packet(pipeline_output_or_dataset_entry: dict) -> dict:
    obj = pipeline_output_or_dataset_entry if isinstance(pipeline_output_or_dataset_entry, dict) else {}
    obj = _unwrap_hybrid(obj)

    run_id = _extract_run_id(obj)
    parent_path = _extract_parent_path(obj)
    decisions = _extract_decisions(obj)
    requirements = _extract_requirements(obj)
    selected_ids = _extract_selected_ids(obj)
    budgets = _extract_budgets(obj)
    lam = _extract_lambda(obj)

    chunks = _extract_selected_chunks(obj, selected_ids)
    if not selected_ids:
        selected_ids = [c["chunk_id"] for c in chunks if isinstance(c.get("chunk_id"), str)]

    anchor_text = _first_str(
        _get(obj, "anchors", "anchor_summary_text"),
        obj.get("anchor_summary_text"),
        obj.get("anchor_summary"),
    )
    summary_text = _first_str(
        _get(obj, "summarization", "summary_text"),
        obj.get("summary_text"),
        obj.get("summary"),
        _get(obj, "final", "summary"),
    )

    anchor_path = _first_str(
        _get(obj, "output", "anchor_summary"),
        _get(obj, "paths", "anchor_summary"),
        obj.get("anchor_summary_path"),
    )
    summary_path = _first_str(
        _get(obj, "output", "summary"),
        _get(obj, "paths", "summary"),
        obj.get("summary_path"),
    )

    if not anchor_text:
        anchor_text = _read_text_if_path(anchor_path)
    if not summary_text:
        summary_text = _read_text_if_path(summary_path)

    packet = {
        "run_id": run_id,
        "parent": {"path": parent_path},
        "task": {"decisions": decisions},
        "requirements": requirements,
        "retrieval": {
            "selected_chunk_ids": selected_ids,
            "budgets": budgets,
            "objective": {"lambda_diversity": lam},
        },
        "chunks": chunks,
        "anchors": {"anchor_summary_text": _truncate(anchor_text, 12000) if anchor_text else None},
        "summarization": {"summary_text": _truncate(summary_text, 12000) if summary_text else None},
    }

    if not packet["run_id"]:
        did = obj.get("id")
        if isinstance(did, str) and did.strip():
            packet["run_id"] = did.strip()

    if not packet["parent"]["path"]:
        p = os.getenv("EVAL_PARENT_PATH")
        if isinstance(p, str) and p.strip():
            packet["parent"]["path"] = p.strip()

    return packet
