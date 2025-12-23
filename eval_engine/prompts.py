SYSTEM_PROMPT = (
    "You are a strict evaluation judge for instruction-driven budgeted retrieval summarization. "
    "Output JSON only (no markdown/code fences). Never invent facts or evidence. Use only the provided input. "
    "Do not copy chunk text; paraphrase. Evidence must reference chunk_id (no page numbers). "
    "If you include a quote, keep it <=25 words. If info is missing, use null or [] and say uncertain. "
    "Be concise."
)

RETRIEVAL_EVAL_USER = (
    "Evaluate retrieval quality for instruction-driven cognitive-budget summarization.\n\n"
    "Eval packet JSON:\n{EVAL_PACKET_JSON}\n\n"
    "Judge ONLY selected chunks.\n"
    "Assess: (1) coverage vs task.required_sections, (2) redundancy, (3) budget efficiency (expensive/low-yield), "
    "(4) extraction risk (OCR/UNCLEAR/garbling) incl. whether critical tables/figures are likely missed.\n\n"
    "Return JSON only:\n"
    "{"
    "\"kind\":\"retrieval_eval\","
    "\"overall\":{\"score\":0,\"verdict\":\"pass\"},"
    "\"coverage\":{\"missing_sections\":[],\"notes\":\"\"},"
    "\"redundancy\":{\"redundant_pairs\":[],\"notes\":\"\"},"
    "\"budget\":{\"expensive_low_yield\":[],\"notes\":\"\"},"
    "\"risk\":{\"flagged\":[],\"notes\":\"\"},"
    "\"recommended_fixes\":[]"
    "}"
)

SUMMARY_EVAL_USER = (
    "Evaluate summary quality and grounding.\n\n"
    "Eval packet JSON:\n{EVAL_PACKET_JSON}\n\n"
    "Use only the provided summary fields and selected chunks. No assumptions. "
    "Check: grounding (cite chunk_id), completeness vs instruction/required_sections, hygiene (placeholders, repetition, "
    "unsupported superlatives).\n\n"
    "Return JSON only:\n"
    "{"
    "\"kind\":\"summary_eval\","
    "\"overall\":{\"score\":0,\"verdict\":\"pass\"},"
    "\"grounding\":{\"issues\":[]},"
    "\"completeness\":{\"missing\":[],\"notes\":\"\"},"
    "\"hygiene\":{\"placeholders\":[],\"repetition\":\"low\",\"notes\":\"\"},"
    "\"recommended_fixes\":[]"
    "}"
)

DECISION_EVAL_USER = (
    "Evaluate decision utility via Q/A agreement.\n\n"
    "Eval packet JSON:\n{EVAL_PACKET_JSON}\n\n"
    "Create 5 decision-oriented questions aligned to task.user_instruction. For each question:\n"
    "- Answer using SUMMARY ONLY (if missing, null).\n"
    "- Answer using CHUNKS ONLY (cite chunk_id; no long quotes).\n"
    "- Compare: match|partial|miss.\n\n"
    "Return JSON only:\n"
    "{"
    "\"kind\":\"decision_eval\","
    "\"overall\":{\"match_score\":0,\"verdict\":\"pass\"},"
    "\"questions\":[],"
    "\"recommended_fixes\":[]"
    "}"
)
