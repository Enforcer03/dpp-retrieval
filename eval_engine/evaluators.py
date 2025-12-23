from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
from typing import Any

from .adapter import export_eval_packet
from .io import validate_against_schema
from .judge_client import JudgeClient, JudgeRun
from .prompts import DECISION_EVAL_USER, RETRIEVAL_EVAL_USER, SUMMARY_EVAL_USER, SYSTEM_PROMPT
from .utils import sha1_text

log = logging.getLogger(__name__)

_PLACEHOLDER_PATTERNS = ["[specific work]", "[specific field]", "todo", "tbd", "[insert", "<insert"]


def run_evaluation(pipeline_output: dict, schema: dict, mode: str, model: str, disable_wandb: bool = False) -> dict:
    packet = export_eval_packet(pipeline_output)
    pre = _prechecks(packet)
    wb = _wandb_init(pipeline_output, packet.get("run_id"), model, mode, disable_wandb, pre)

    client = JudgeClient(model=model, system_prompt=SYSTEM_PROMPT)

    packet_json = json.dumps(packet, ensure_ascii=False, separators=(",", ":"), sort_keys=True)

    jr_r = client.run("retrieval_eval", RETRIEVAL_EVAL_USER.replace("{EVAL_PACKET_JSON}", packet_json))
    r_out = _sanitize(jr_r.output) if jr_r.output else None

    jr_s = None
    s_out = None
    jr_d = None
    d_out = None

    if mode != "retrieval" and _summary_text(packet):
        jr_s = client.run("summary_eval", SUMMARY_EVAL_USER.replace("{EVAL_PACKET_JSON}", packet_json))
        s_out = _sanitize(jr_s.output) if jr_s.output else None
        jr_d = client.run("decision_eval", DECISION_EVAL_USER.replace("{EVAL_PACKET_JSON}", packet_json))
        d_out = _sanitize(jr_d.output) if jr_d.output else None

    bundle = _build_bundle(packet.get("run_id"), schema, jr_r, r_out, jr_s, s_out, jr_d, d_out)
    validate_against_schema(bundle, schema)

    _wandb_log(wb, packet, pre, jr_r, r_out, jr_s, s_out, jr_d, d_out)
    _wandb_finish(wb)
    return bundle


def _summary_text(packet: dict) -> str | None:
    s = packet.get("summarization")
    if isinstance(s, dict):
        t = s.get("summary_text")
        if isinstance(t, str) and t.strip():
            return t
    a = packet.get("anchors")
    if isinstance(a, dict):
        t = a.get("anchor_summary_text")
        if isinstance(t, str) and t.strip():
            return t
    return None


def _prechecks(packet: dict) -> dict:
    anchor = packet.get("anchors", {}).get("anchor_summary_text") if isinstance(packet.get("anchors"), dict) else ""
    summ = packet.get("summarization", {}).get("summary_text") if isinstance(packet.get("summarization"), dict) else ""
    placeholders = _detect_placeholders(f"{anchor}\n{summ}")

    chunks = packet.get("chunks") if isinstance(packet.get("chunks"), list) else []
    costs = []
    table_chunks = 0
    unclear_chunks = 0

    for ch in chunks:
        if not isinstance(ch, dict):
            continue
        md = ch.get("metadata") if isinstance(ch.get("metadata"), dict) else {}
        if md.get("contains_tables") is True:
            table_chunks += 1
        txt = ch.get("content", {}).get("text") if isinstance(ch.get("content"), dict) else ""
        if isinstance(txt, str) and "[unclear]" in txt.lower():
            unclear_chunks += 1
        cc = md.get("cognitive_cost")
        if isinstance(cc, (int, float)):
            costs.append(float(cc))

    used = _budget_used(packet, costs)
    top2 = _top2_concentration(costs)
    return {
        "has_placeholders": bool(placeholders),
        "placeholders_found": placeholders,
        "num_table_chunks": table_chunks,
        "num_unclear_chunks": unclear_chunks,
        "total_cost_used": used,
        "top2_cost_concentration": top2,
    }


def _detect_placeholders(text: str) -> list[str]:
    t = (text or "").lower()
    return [p for p in _PLACEHOLDER_PATTERNS if p in t]


def _budget_used(packet: dict, costs: list[float]) -> float | None:
    b = packet.get("retrieval", {}).get("budgets") if isinstance(packet.get("retrieval"), dict) else None
    if isinstance(b, dict):
        used = b.get("used")
        if isinstance(used, (int, float)):
            return float(used)
        tokens = b.get("tokens")
        if isinstance(tokens, dict):
            u = tokens.get("used")
            if isinstance(u, (int, float)):
                return float(u)
    return float(sum(costs)) if costs else None


def _top2_concentration(costs: list[float]) -> float | None:
    if not costs:
        return None
    tot = sum(costs)
    if tot <= 0:
        return None
    top2 = sum(sorted(costs, reverse=True)[:2])
    return float(top2 / tot)


def _sanitize(obj: Any) -> Any:
    if isinstance(obj, dict):
        out = {}
        for k, v in obj.items():
            lk = str(k).lower()
            if "quote" in lk or "excerpt" in lk:
                continue
            out[k] = _sanitize(v)
        return out
    if isinstance(obj, list):
        return [_sanitize(x) for x in obj]
    return obj


def _build_bundle(
    run_id: str | None,
    schema: dict,
    jr_r: JudgeRun,
    r_out: dict | None,
    jr_s: JudgeRun | None,
    s_out: dict | None,
    jr_d: JudgeRun | None,
    d_out: dict | None,
) -> dict:
    schema_version = schema.get("schema_version") if isinstance(schema.get("schema_version"), str) else "unknown"
    created_at = datetime.now(timezone.utc).isoformat()

    judge_runs = [_jr_to_dict(jr_r)]
    if jr_s is not None:
        judge_runs.append(_jr_to_dict(jr_s))
    if jr_d is not None:
        judge_runs.append(_jr_to_dict(jr_d))

    evaluation = {"judge_runs": judge_runs, "retrieval_eval": r_out}
    if s_out is not None:
        evaluation["summary_eval"] = s_out
    if d_out is not None:
        evaluation["decision_eval"] = d_out

    return {"schema_version": schema_version, "kind": "eval_bundle", "run_id": run_id, "created_at": created_at, "evaluation": evaluation}


def _jr_to_dict(jr: JudgeRun) -> dict:
    out = {
        "name": jr.name,
        "model": jr.model,
        "ok": jr.ok,
        "retry_used": jr.retry_used,
        "latency_ms": jr.latency_ms,
    }
    if jr.error:
        out["error"] = jr.error
    if jr.output is not None:
        sanitized = _sanitize(jr.output)
        out["output"] = sanitized
        out["output_hash"] = sha1_text(json.dumps(sanitized, ensure_ascii=False, separators=(",", ":"), sort_keys=True))
    return out


def _wandb_init(pipeline_output: dict, run_id: str | None, model: str, mode: str, disable: bool, pre: dict):
    if disable:
        return None
    wb_cfg = None
    if isinstance(pipeline_output.get("wandb"), dict):
        wb_cfg = pipeline_output["wandb"]
    elif isinstance(pipeline_output.get("config"), dict) and isinstance(pipeline_output["config"].get("wandb"), dict):
        wb_cfg = pipeline_output["config"]["wandb"]
    else:
        wb_cfg = {}

    enabled = wb_cfg.get("enabled")
    if not isinstance(enabled, bool):
        enabled = bool(os.getenv("WANDB_API_KEY")) and bool(os.getenv("WANDB_PROJECT"))
    if not enabled:
        return None

    try:
        import wandb  # type: ignore
    except Exception as e:
        log.warning("wandb unavailable: %s", e)
        return None

    project = wb_cfg.get("project") if isinstance(wb_cfg.get("project"), str) else os.getenv("WANDB_PROJECT", "icb-sum-eval")
    entity = wb_cfg.get("entity") if isinstance(wb_cfg.get("entity"), str) else os.getenv("WANDB_ENTITY")
    tags = wb_cfg.get("tags") if isinstance(wb_cfg.get("tags"), list) else []
    name = run_id or "eval"

    try:
        return wandb.init(project=project, entity=entity, name=name, tags=tags, config={"run_id": run_id, "judge_model": model, "mode": mode, **pre}, reinit=True)
    except Exception as e:
        log.warning("wandb init failed: %s", e)
        return None


def _wandb_log(wb, packet: dict, pre: dict, jr_r: JudgeRun, r_out: dict | None, jr_s: JudgeRun | None, s_out: dict | None, jr_d: JudgeRun | None, d_out: dict | None):
    if wb is None:
        return
    try:
        import wandb  # type: ignore
    except Exception:
        return

    retrieval = packet.get("retrieval") if isinstance(packet.get("retrieval"), dict) else {}
    budgets = retrieval.get("budgets") if isinstance(retrieval.get("budgets"), dict) else {}
    obj = retrieval.get("objective") if isinstance(retrieval.get("objective"), dict) else {}
    lam = obj.get("lambda_diversity")

    sel = retrieval.get("selected_chunk_ids") if isinstance(retrieval.get("selected_chunk_ids"), list) else []
    nsel = len(sel)

    m = {
        "run_id": packet.get("run_id"),
        "lambda_diversity": lam,
        "num_selected_chunks": nsel,
        "num_table_chunks": pre.get("num_table_chunks"),
        "num_unclear_chunks": pre.get("num_unclear_chunks"),
        "has_placeholders": pre.get("has_placeholders"),
        "top2_cost_concentration": pre.get("top2_cost_concentration"),
        "total_cost_used": pre.get("total_cost_used"),
        "budgets": budgets,
        "judge/retrieval_ok": jr_r.ok,
        "judge/retrieval_latency_ms": jr_r.latency_ms,
    }

    if isinstance(r_out, dict) and isinstance(r_out.get("overall"), dict):
        sc = r_out["overall"].get("score")
        vd = r_out["overall"].get("verdict")
        if isinstance(sc, (int, float)):
            m["judge/retrieval_score"] = float(sc)
        if isinstance(vd, str):
            m["judge/retrieval_verdict"] = vd

    if jr_s is not None:
        m["judge/summary_ok"] = jr_s.ok
        m["judge/summary_latency_ms"] = jr_s.latency_ms
        if isinstance(s_out, dict) and isinstance(s_out.get("overall"), dict):
            sc = s_out["overall"].get("score")
            vd = s_out["overall"].get("verdict")
            if isinstance(sc, (int, float)):
                m["judge/summary_score"] = float(sc)
            if isinstance(vd, str):
                m["judge/summary_verdict"] = vd

    if jr_d is not None:
        m["judge/decision_ok"] = jr_d.ok
        m["judge/decision_latency_ms"] = jr_d.latency_ms
        if isinstance(d_out, dict) and isinstance(d_out.get("overall"), dict):
            sc = d_out["overall"].get("match_score")
            vd = d_out["overall"].get("verdict")
            if isinstance(sc, (int, float)):
                m["judge/decision_match_score"] = float(sc)
            if isinstance(vd, str):
                m["judge/decision_verdict"] = vd

    try:
        wandb.log(m)
    except Exception:
        pass


def _wandb_finish(wb) -> None:
    if wb is None:
        return
    try:
        wb.finish()
    except Exception:
        pass
