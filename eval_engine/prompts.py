# eval_engine/prompts.py
from __future__ import annotations

import json

SYSTEM_PROMPT = """You are a strict evaluation judge.
Rules:
- Use ONLY the provided evidence (chunks and optional summaries).
- If evidence is missing, answer "unknown" rather than guessing.
- Every evidence reference MUST cite chunk_id.
- Quotes must be <=25 words each.
- Output JSON ONLY (no markdown, no code fences)."""


def _fmt_chunks(chunks: list[dict]) -> str:
    lines: list[str] = []
    for c in chunks:
        cid = c.get("chunk_id")
        content = c.get("content") if isinstance(c.get("content"), dict) else {}
        text = content.get("text") if isinstance(content.get("text"), str) else ""
        imgs = content.get("image_paths") if isinstance(content.get("image_paths"), list) else []
        meta = c.get("metadata") if isinstance(c.get("metadata"), dict) else {}
        meta_s = ", ".join([f"{k}={meta.get(k)}" for k in ["page", "type", "kind", "is_table", "is_figure", "cognitive_cost", "cost"] if k in meta])
        img_s = f" images={len(imgs)}" if imgs else ""
        lines.append(f"- chunk_id={cid} {meta_s}{img_s}\n  text: {text}")
    return "\n".join(lines)


def build_retrieval_prompt(packet: dict, prechecks: dict) -> str:
    decisions = (packet.get("task") or {}).get("decisions") or []
    reqs = packet.get("requirements") or []
    chunks = packet.get("chunks") or []
    parent_path = ((packet.get("parent") or {}) if isinstance(packet.get("parent"), dict) else {}).get("path")

    retrieval = packet.get("retrieval") if isinstance(packet.get("retrieval"), dict) else {}
    budgets = retrieval.get("budgets") if isinstance(retrieval.get("budgets"), dict) else {}
    lam = ((retrieval.get("objective") or {}) if isinstance(retrieval.get("objective"), dict) else {}).get("lambda_diversity")

    anchor = ((packet.get("anchors") or {}) if isinstance(packet.get("anchors"), dict) else {}).get("anchor_summary_text")
    summary = ((packet.get("summarization") or {}) if isinstance(packet.get("summarization"), dict) else {}).get("summary_text")

    return f"""Evaluate whether the provided evidence satisfies the requirements for each decision, and evaluate retrieval quality.

run_id: {packet.get("run_id")}
parent.path: {parent_path}

decisions:
{json.dumps(decisions, ensure_ascii=False)}

requirements (array of {{id, description}}):
{json.dumps(reqs, ensure_ascii=False)}

retrieval_context:
selected_chunk_ids: {json.dumps(retrieval.get("selected_chunk_ids") if isinstance(retrieval, dict) else [], ensure_ascii=False)}
budgets: {json.dumps(budgets, ensure_ascii=False)}
objective.lambda_diversity: {lam}

cheap_prechecks (signals only; do NOT fabricate facts):
{json.dumps(prechecks, ensure_ascii=False)}

selected_chunks (selected only; may be empty):
{_fmt_chunks(chunks)}

optional_anchor_summary_text:
{anchor}

optional_summary_text:
{summary}

Return JSON with:
{{
  "retrieval_eval": {{
    "requirements_eval": {{
      "decisions": [
        {{
          "decision": "...",
          "overall_verdict": "pass|weak|fail",
          "requirements": [
            {{
              "id": "R1",
              "status": "covered|partial|missing|unknown",
              "evidence": ["chunk_id", "..."],
              "quotes": [{{"chunk_id":"...","quote":"..."}}, ...],
              "rationale": "concise"
            }}
          ],
          "missing_requirements": ["R2", "..."],
          "risks": ["...", "..."]
        }}
      ]
    }},
    "coverage": {{"overall":"good|mixed|poor","notes":"...","missing_requirements":["..."]}},
    "redundancy": {{"level":"low|medium|high|unknown","notes":"...","redundant_pairs":[["id1","id2"]]}},
    "budget_efficiency": {{"rating":"good|mixed|poor|unknown","notes":"...","top_cost_chunks":[{{"chunk_id":"...","cost":0,"yield":"high|medium|low"}}]}},
    "risk_flags": ["...", "..."]
  }}
}}"""


def build_summary_prompt(packet: dict, prechecks: dict) -> str:
    decisions = (packet.get("task") or {}).get("decisions") or []
    reqs = packet.get("requirements") or []
    chunks = packet.get("chunks") or []
    anchor = ((packet.get("anchors") or {}) if isinstance(packet.get("anchors"), dict) else {}).get("anchor_summary_text")
    summary = ((packet.get("summarization") or {}) if isinstance(packet.get("summarization"), dict) else {}).get("summary_text")

    return f"""Evaluate summary grounding and decision QA alignment.

decisions: {json.dumps(decisions, ensure_ascii=False)}
requirements: {json.dumps(reqs, ensure_ascii=False)}

cheap_prechecks:
{json.dumps(prechecks, ensure_ascii=False)}

selected_chunks:
{_fmt_chunks(chunks)}

anchor_summary_text (may be null):
{anchor}

summary_text (may be null):
{summary}

Return JSON with:
{{
  "summary_eval": {{
    "verdict": "pass|weak|fail",
    "grounding": "grounded|partially_grounded|ungrounded|unknown",
    "issues": ["..."],
    "unsupported_claims": ["..."],
    "quotes": [{{"chunk_id":"...","quote":"..."}}, ...]
  }},
  "decision_eval": {{
    "decisions": [
      {{"decision":"...","answer":"...","confidence":0.0,"evidence":["chunk_id"],"notes":"..."}}
    ]
  }}
}}"""
