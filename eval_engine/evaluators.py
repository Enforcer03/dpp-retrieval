# eval_engine/evaluators.py
from __future__ import annotations

import datetime as _dt
import hashlib
import json
import os
import re
from typing import Any

from .adapter import export_eval_packet
from .judge_client import JudgeClient, JudgeRun
from .prompts import SYSTEM_PROMPT, build_retrieval_prompt, build_summary_prompt

_TABLE_RE = re.compile(r"<table\b|\|\s*-{2,}\s*\|", re.I)
_FIG_RE = re.compile(r"\bfigure\b|\bfig\.\b", re.I)
_PLACEHOLDER_RE = re.compile(r"\bTODO\b|\bTBD\b|\[specific.*?\]|\.\.\.|<\.\.\.>", re.I)
_UNCLEAR_RE = re.compile(r"\[UNCLEAR\]", re.I)


def _detect_placeholders(s: str | None) -> bool:
    return bool(s and _PLACEHOLDER_RE.search(s))


def _detect_unclear(s: str | None) -> bool:
    return bool(s and _UNCLEAR_RE.search(s))


def _count_table_figure(chunks: list[dict]) -> tuple[int, int]:
    n_table = 0
    n_fig = 0
    for c in chunks:
        meta = c.get("metadata") if isinstance(c.get("metadata"), dict) else {}
        content = c.get("content") if isinstance(c.get("content"), dict) else {}
        text = content.get("text") if isinstance(content.get("text"), str) else ""

        typ = meta.get("type") or meta.get("kind") or ""
        if meta.get("is_table") is True or (isinstance(typ, str) and "table" in typ.lower()) or _TABLE_RE.search(text or ""):
            n_table += 1
        if meta.get("is_figure") is True or (isinstance(typ, str) and "figure" in typ.lower()) or _FIG_RE.search(text or ""):
            n_fig += 1
    return n_table, n_fig


def _cost_of_chunk(c: dict) -> float:
    meta = c.get("metadata") if isinstance(c.get("metadata"), dict) else {}
    for k in ("cognitive_cost", "token_cost", "cost"):
        v = meta.get(k)
        if isinstance(v, (int, float)):
            return float(v)
    return 0.0


def _compute_prechecks(packet: dict) -> dict:
    chunks = packet.get("chunks") if isinstance(packet.get("chunks"), list) else []
    anchor = ((packet.get("anchors") or {}) if isinstance(packet.get("anchors"), dict) else {}).get("anchor_summary_text")
    summary = ((packet.get("summarization") or {}) if isinstance(packet.get("summarization"), dict) else {}).get("summary_text")

    n_table, n_fig = _count_table_figure(chunks)

    any_unclear = False
    for c in chunks:
        content = c.get("content") if isinstance(c.get("content"), dict) else {}
        text = content.get("text") if isinstance(content.get("text"), str) else ""
        if _detect_unclear(text):
            any_unclear = True
            break

    retrieval = packet.get("retrieval") if isinstance(packet.get("retrieval"), dict) else {}
    budgets = retrieval.get("budgets") if isinstance(retrieval.get("budgets"), dict) else {}
    used = budgets.get("used") if isinstance(budgets.get("used"), (int, float)) else None
    total = budgets.get("total") if isinstance(budgets.get("total"), (int, float)) else None

    costs = [_cost_of_chunk(c) for c in chunks]
    sum_cost = float(sum(costs)) if costs else 0.0
    total_cost_used = float(used) if isinstance(used, (int, float)) else (sum_cost if sum_cost > 0 else None)

    top2 = sorted(costs, reverse=True)[:2] if costs else []
    top2_cost_concentration = None
    if total_cost_used and total_cost_used > 0 and top2:
        top2_cost_concentration = float(sum(top2) / float(total_cost_used))

    return {
        "placeholder_in_anchor": _detect_placeholders(anchor),
        "placeholder_in_summary": _detect_placeholders(summary),
        "any_unclear_chunks": any_unclear,
        "num_selected_chunks": len(chunks),
        "num_table_chunks": n_table,
        "num_figure_chunks": n_fig,
        "total_cost_used": total_cost_used,
        "budget_total": total,
        "top2_cost_concentration": top2_cost_concentration,
    }


def _schema_version(schema: dict) -> str:
    v = schema.get("schema_version")
    if isinstance(v, str) and v.strip():
        return v.strip()
    sid = schema.get("$id")
    if isinstance(sid, str) and sid.strip():
        return sid.strip()
    return "1.0"


def _hash_obj(o: Any) -> str:
    try:
        s = json.dumps(o, sort_keys=True, ensure_ascii=False)
    except Exception:
        s = str(o)
    return hashlib.sha1(s.encode("utf-8", errors="ignore")).hexdigest()


def _jr_to_schema(jr: JudgeRun, output_obj: Any | None = None) -> dict:
    d = {
        "name": jr.name,
        "model": jr.model,
        "ok": bool(jr.ok),
        "retry_used": bool(jr.repaired),
        "latency_ms": int(round(float(jr.latency_s) * 1000.0)),
    }
    if jr.error:
        d["error"] = jr.error
    if output_obj is not None:
        d["output_hash"] = _hash_obj(output_obj)
        if isinstance(output_obj, dict):
            d["output"] = output_obj
    return d


def _default_retrieval_eval(packet: dict) -> dict:
    decisions = (packet.get("task") or {}).get("decisions") or []
    reqs = packet.get("requirements") or []
    missing_ids = [r.get("id") for r in reqs if isinstance(r, dict) and isinstance(r.get("id"), str)]

    out_decisions: list[dict] = []
    for d in decisions or [""]:
        req_out = []
        for r in reqs:
            rid = r.get("id") if isinstance(r, dict) else None
            req_out.append(
                {
                    "id": rid,
                    "status": "unknown",
                    "evidence": [],
                    "quotes": [],
                    "rationale": "No evidence chunks provided.",
                }
            )
        out_decisions.append(
            {
                "decision": d,
                "overall_verdict": "weak" if reqs else "pass",
                "requirements": req_out,
                "missing_requirements": missing_ids,
                "risks": ["no_selected_chunks"],
            }
        )

    return {
        "requirements_eval": {"decisions": out_decisions},
        "coverage": {
            "overall": "poor" if reqs else "good",
            "notes": "No selected chunks provided; requirement satisfaction cannot be verified.",
            "missing_requirements": missing_ids,
        },
        "redundancy": {"level": "unknown", "notes": "No chunks.", "redundant_pairs": []},
        "budget_efficiency": {"rating": "unknown", "notes": "No chunks.", "top_cost_chunks": []},
        "risk_flags": ["no_selected_chunks"],
    }


def _maybe_wandb_enabled(disable_wandb: bool) -> bool:
    if disable_wandb:
        return False
    v = os.getenv("EVAL_WANDB_ENABLED", "").strip().lower()
    return v in {"1", "true", "yes", "y", "on"}


def _wandb_log(packet: dict, prechecks: dict, judge_runs: list[JudgeRun], evaluation: dict) -> None:
    try:
        import wandb  # type: ignore
    except Exception:
        return

    project = os.getenv("EVAL_WANDB_PROJECT", os.getenv("WANDB_PROJECT", "icb-sum"))
    entity = os.getenv("EVAL_WANDB_ENTITY", os.getenv("WANDB_ENTITY"))
    tags = [t for t in os.getenv("EVAL_WANDB_TAGS", "").split(",") if t.strip()]

    run = wandb.init(project=project, entity=entity, tags=tags or None, reinit=True)

    retrieval = packet.get("retrieval") if isinstance(packet.get("retrieval"), dict) else {}
    lam = ((retrieval.get("objective") or {}) if isinstance(retrieval.get("objective"), dict) else {}).get("lambda_diversity")

    verdict = None
    try:
        reval = evaluation.get("retrieval_eval") or {}
        reqe = (reval.get("requirements_eval") or {}) if isinstance(reval, dict) else {}
        decs = reqe.get("decisions") or []
        if decs and isinstance(decs, list):
            verdict = decs[0].get("overall_verdict")
    except Exception:
        verdict = None

    wandb.log(
        {
            "run_id": packet.get("run_id"),
            "lambda_diversity": lam,
            "budgets_total": prechecks.get("budget_total"),
            "budgets_used": prechecks.get("total_cost_used"),
            "num_selected_chunks": prechecks.get("num_selected_chunks"),
            "num_table_chunks": prechecks.get("num_table_chunks"),
            "num_figure_chunks": prechecks.get("num_figure_chunks"),
            "top2_cost_concentration": prechecks.get("top2_cost_concentration"),
            "placeholder_in_anchor": prechecks.get("placeholder_in_anchor"),
            "placeholder_in_summary": prechecks.get("placeholder_in_summary"),
            "any_unclear_chunks": prechecks.get("any_unclear_chunks"),
            "judge_ok_rate": sum(1 for jr in judge_runs if jr.ok) / max(1, len(judge_runs)),
            "requirements_overall_verdict": verdict,
        }
    )
    run.finish()


def run_evaluation(
    pipeline_output_or_dataset_entry: dict,
    *,
    schema: dict,
    mode: str = "all",
    model: str = "gpt-5.1",
    disable_wandb: bool = False,
) -> dict:
    packet = export_eval_packet(pipeline_output_or_dataset_entry)
    prechecks = _compute_prechecks(packet)

    chunks = packet.get("chunks") if isinstance(packet.get("chunks"), list) else []
    has_chunks = len(chunks) > 0
    anchor = ((packet.get("anchors") or {}) if isinstance(packet.get("anchors"), dict) else {}).get("anchor_summary_text")
    summary = ((packet.get("summarization") or {}) if isinstance(packet.get("summarization"), dict) else {}).get("summary_text")
    has_summary = bool((isinstance(anchor, str) and anchor.strip()) or (isinstance(summary, str) and summary.strip()))

    mode = (mode or "all").strip().lower()
    if mode not in {"requirements", "retrieval", "all"}:
        mode = "all"

    judge_runs: list[JudgeRun] = []
    judge_outputs: dict[str, Any] = {}

    evaluation: dict[str, Any] = {
        "judge_runs": [],
        "retrieval_eval": None,
        "summary_eval": None,
        "decision_eval": None,
    }

    # Always produce retrieval_eval (schema requires it).
    if has_chunks:
        jc = JudgeClient(timeout_s=float(os.getenv("EVAL_OPENAI_TIMEOUT_S", "60")))
        ret_prompt = build_retrieval_prompt(packet, prechecks)
        ret_out, jr = jc.judge_json(name="retrieval_eval", model=model, system_prompt=SYSTEM_PROMPT, user_prompt=ret_prompt, temperature=0.1)
        judge_runs.append(jr)
        judge_outputs["retrieval_eval"] = ret_out
        if isinstance(ret_out, dict) and isinstance(ret_out.get("retrieval_eval"), dict):
            evaluation["retrieval_eval"] = ret_out["retrieval_eval"]
        else:
            evaluation["retrieval_eval"] = {}
    else:
        evaluation["retrieval_eval"] = _default_retrieval_eval(packet)
        judge_runs.append(JudgeRun(name="retrieval_eval", model=model, ok=True, repaired=False, latency_s=0.0, usage={}, error=None))
        judge_outputs["retrieval_eval"] = {"retrieval_eval": evaluation["retrieval_eval"]}

    # Only run summary/decision eval if summary exists and mode == all
    if has_chunks and has_summary and mode == "all":
        jc = jc if "jc" in locals() else JudgeClient(timeout_s=float(os.getenv("EVAL_OPENAI_TIMEOUT_S", "60")))
        sum_prompt = build_summary_prompt(packet, prechecks)
        sum_out, jr2 = jc.judge_json(name="summary_decision_eval", model=model, system_prompt=SYSTEM_PROMPT, user_prompt=sum_prompt, temperature=0.1)
        judge_runs.append(jr2)
        judge_outputs["summary_decision_eval"] = sum_out
        if isinstance(sum_out, dict):
            evaluation["summary_eval"] = sum_out.get("summary_eval")
            evaluation["decision_eval"] = sum_out.get("decision_eval")

    # Conform judge_runs to your schema's required fields
    evaluation["judge_runs"] = [
        _jr_to_schema(jr, output_obj=judge_outputs.get(jr.name))
        for jr in judge_runs
    ]

    bundle = {
        "schema_version": _schema_version(schema),
        "kind": "eval_bundle",
        "run_id": packet.get("run_id"),
        "created_at": _dt.datetime.utcnow().replace(microsecond=0).isoformat() + "Z",
        "evaluation": evaluation,
    }

    if _maybe_wandb_enabled(disable_wandb):
        _wandb_log(packet, prechecks, judge_runs, evaluation)

    return bundle
