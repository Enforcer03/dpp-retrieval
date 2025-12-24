from __future__ import annotations

import argparse
import logging
import re
import time
from pathlib import Path

import numpy as np

from src.anchor_summary import generate_anchor_query, generate_anchor_summary
from src.colpali_embedder import MultiModalEmbedder
from src.config import load_config
from src.cognitive_profiler import CostProfiler, TokenCounter
from src.dpp_optimizer import BudgetedSelector
from src.evidence_builder import EvidenceBuilder
from src.indexing import CombinedRetriever
from src.layout_aware_extractor import PdfLayoutExtractor
from src.ocr_engine import OCREngine
from src.openai_client import OpenAIChatClient
from src.pdf_highlighter import highlight_pdf
from src.utils import ensure_dir, setup_logging, stable_id, write_json
from src.unit_extractor import extract_units
from src.wandb_logger import wandb_finish, wandb_init, wandb_log

log = logging.getLogger(__name__)

_TABLE = re.compile(r"<table\b|\|\s*-{2,}\s*\|", re.I)
_MATH = re.compile(r"(\$[^$]{1,120}\$|\\\(|\\\)|\\\[|\\\])")
_CODE = re.compile(r"```|^\s*(def|class)\s+\w+\s*\(", re.M)
_OCR_BAD = re.compile(r"[��]")


# ---- tqdm (graceful fallback) -------------------------------------------------
try:
    from tqdm.auto import tqdm  # type: ignore
except Exception:  # pragma: no cover
    def tqdm(x, **kwargs):  # type: ignore
        return x


def _doc_id(pdf_path: Path) -> str:
    return stable_id(pdf_path.name, str(pdf_path.stat().st_size))


def _normalize(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v) + 1e-12)
    return (v / n).astype(np.float32)


def _meta_for_unit(unit, cost_total: int, rel_score: float | None, ocr_used: bool) -> dict:
    txt = (unit.context_text or "").strip()
    unit_type = unit.type
    contains_tables = bool(unit_type == "table_text" or _TABLE.search(txt))
    contains_math = bool(_MATH.search(txt))
    contains_code = bool(_CODE.search(txt))

    flags = []
    if "[unclear]" in txt.lower():
        flags.append("HAS_UNCLEAR_TOKENS")
    if ocr_used and (_OCR_BAD.search(txt) or ("HAS_UNCLEAR_TOKENS" in flags)):
        flags.append("OCR_SUSPECT")
    if contains_tables:
        flags.append("HAS_TABLE")
    if unit_type == "figure":
        flags.append("HAS_FIGURE")

    if not txt and unit.image_paths:
        eq = "unknown"
    elif "OCR_SUSPECT" in flags:
        eq = "low"
    elif ocr_used:
        eq = "medium"
    else:
        eq = "high"

    return {
        "unit_type": unit_type,
        "cognitive_cost": float(cost_total),
        "importance_score": float(rel_score) if isinstance(rel_score, (int, float)) else None,
        "contains_tables": contains_tables,
        "contains_math": contains_math,
        "contains_code": contains_code,
        "extraction_quality": eq,
        "flags": flags,
    }


def _top2_concentration(costs: list[float]) -> float | None:
    if not costs:
        return None
    tot = sum(costs)
    if tot <= 0:
        return None
    top2 = sum(sorted(costs, reverse=True)[:2])
    return float(top2 / tot)


def _preview_hits(hits, k: int = 8) -> list[dict]:
    """Small log-friendly preview of retrieval hits."""
    out = []
    for h in (hits or [])[:k]:
        out.append({"unit_id": getattr(h, "unit_id", None), "score": float(getattr(h, "score", 0.0))})
    return out


def _safe_snippet(txt: str | None, n: int = 800) -> str:
    if not txt:
        return ""
    t = txt.strip().replace("\r\n", "\n")
    if len(t) <= n:
        return t
    return t[:n] + "\n…(truncated)…"


def _save_embeddings_npz(
    out_dir: Path,
    unit_vecs: dict[str, np.ndarray],
    page_vecs: dict[int, np.ndarray],
    q_vec: np.ndarray,
) -> Path:
    emb_dir = ensure_dir(out_dir / "embeddings")

    # Units
    u_ids = sorted(unit_vecs.keys())
    u_mat = np.stack([unit_vecs[i] for i in u_ids], axis=0) if u_ids else np.zeros((0, 0), dtype=np.float32)

    # Pages
    p_ids = sorted(page_vecs.keys())
    p_mat = np.stack([page_vecs[i] for i in p_ids], axis=0) if p_ids else np.zeros((0, 0), dtype=np.float32)

    out_path = emb_dir / "embeddings_unit_page_query.npz"
    np.savez_compressed(
        out_path,
        unit_ids=np.array(u_ids, dtype=object),
        unit_vecs=u_mat.astype(np.float32),
        page_ids=np.array(p_ids, dtype=np.int32),
        page_vecs=p_mat.astype(np.float32),
        query_vec=q_vec.astype(np.float32),
    )
    return out_path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default="config/default_config.yaml")
    ap.add_argument("--pdf", type=str, required=True)
    ap.add_argument("--query", type=str, required=True)
    ap.add_argument("--instruction", type=str, default=None)
    ap.add_argument("--required_sections", type=str, default=None)
    ap.add_argument("--run_output", type=str, default=None)

    # Optional debug switches (won’t break existing configs)
    ap.add_argument("--save_embeddings", action="store_true", help="Save query/unit/page embeddings as NPZ")
    ap.add_argument("--log_level", type=str, default=None, help="Override logging level (e.g., INFO, DEBUG)")
    args = ap.parse_args()

    cfg = load_config(args.config)
    setup_logging(cfg.logging.level, cfg.logging.file)

    if args.log_level:
        logging.getLogger().setLevel(getattr(logging, args.log_level.upper(), logging.INFO))

    t_all0 = time.perf_counter()

    pdf_path = Path(args.pdf).resolve()
    run_id = _doc_id(pdf_path)

    ensure_dir(cfg.paths.work_dir)
    ensure_dir(cfg.paths.cache_dir)

    # Create run dirs early so we can save anchor query before retrieval
    run_dir = ensure_dir(Path(cfg.paths.work_dir) / run_id)
    out_dir = ensure_dir(run_dir / "results")

    user_instruction = (args.instruction or args.query).strip()
    required_sections = cfg.summarization.required_sections
    if args.required_sections:
        required_sections = [s.strip() for s in args.required_sections.split(",") if s.strip()]

    log.info("run.start run_id=%s pdf=%s size_bytes=%s", run_id, str(pdf_path), pdf_path.stat().st_size)
    log.info("run.query=%s", _safe_snippet(args.query, 600))
    log.info("run.user_instruction=%s", _safe_snippet(user_instruction, 600))
    log.info("run.required_sections=%s", required_sections)
    log.info(
        "config: extract.mode=%s ocr.enabled=%s embed.backend=%s embed.model=%s selection.budget=%s",
        cfg.extract.mode,
        cfg.extract.ocr.enabled,
        cfg.embed.backend,
        cfg.embed.model_name,
        cfg.selection.budget_tokens,
    )

    # ---- Anchor query extension (for retrieval) --------------------------------
    # Anchor here is an EXTENSION of the query to make retrieval better, not a summary.
    retrieval_query = (args.query or "").strip()
    anchor_query_text = None
    anchor_query_meta = None

    anchor_enabled = True
    if hasattr(cfg, "anchor") and getattr(cfg.anchor, "enabled", None) is not None:
        anchor_enabled = bool(cfg.anchor.enabled)

    if cfg.openai.enabled and anchor_enabled:
        try:
            llm_client_for_anchor = OpenAIChatClient(timeout_s=cfg.openai.timeout_s)
            anchor_query_text, anchor_query_meta = generate_anchor_query(
                client=llm_client_for_anchor,
                model=cfg.openai.model,
                temperature=cfg.openai.temperature,
                query=args.query,
                user_instruction=user_instruction,
                required_sections=required_sections,
                max_chars=getattr(getattr(cfg, "anchor", None), "max_chars", 1400),
            )
            if anchor_query_text:
                retrieval_query = (retrieval_query + "\n\n" + anchor_query_text).strip()

            (out_dir / "anchor_query.txt").write_text(anchor_query_text or "", encoding="utf-8")
            write_json(out_dir / "anchor_query_meta.json", anchor_query_meta or {})
            log.info("anchor.query done chars=%s meta=%s", len(anchor_query_text or ""), anchor_query_meta)
            log.info("anchor.query.snippet:\n%s", _safe_snippet(anchor_query_text, 900))
        except Exception as e:
            log.warning("anchor.query failed err=%s", e)
    else:
        log.info("anchor.query skipped openai.enabled=%s anchor.enabled=%s", cfg.openai.enabled, anchor_enabled)

    ocr = None
    if cfg.extract.ocr.enabled:
        ocr = OCREngine(
            engine=cfg.extract.ocr.engine,
            lang=cfg.extract.ocr.lang,
            min_conf=cfg.extract.ocr.min_conf,
            psm=cfg.extract.ocr.psm,
        )
        log.info(
            "ocr.config engine=%s lang=%s min_conf=%s psm=%s on_pages=%s on_figures=%s",
            cfg.extract.ocr.engine,
            cfg.extract.ocr.lang,
            cfg.extract.ocr.min_conf,
            cfg.extract.ocr.psm,
            cfg.extract.ocr.on_pages,
            cfg.extract.ocr.on_figures,
        )

    # ---- Extract ----------------------------------------------------------------
    t0 = time.perf_counter()
    extractor = PdfLayoutExtractor(cfg)
    doc = extractor.extract(pdf_path, out_dir=cfg.paths.work_dir)

    t_extract = time.perf_counter() - t0
    log.info("stage.extract done pages=%d time_s=%.2f", len(doc.pages), t_extract)

    # ---- Build evidence units ---------------------------------------------------
    t0 = time.perf_counter()
    tc = TokenCounter()
    builder = EvidenceBuilder(
        max_text_chunk_tokens=cfg.extract.max_text_chunk_tokens,
        caption_search_px=cfg.extract.caption_search_px,
        stopwords=set(cfg.selection.stopwords),
    )
    units = builder.build(doc, token_counter=tc)
    t_build = time.perf_counter() - t0

    type_counts: dict[str, int] = {}
    img_units = 0
    empty_text_units = 0
    for u in units:
        type_counts[u.type] = type_counts.get(u.type, 0) + 1
        if u.image_paths:
            img_units += 1
        if not (u.context_text or "").strip():
            empty_text_units += 1

    log.info(
        "stage.build_units done units=%d img_units=%d empty_text_units=%d type_counts=%s time_s=%.2f",
        len(units),
        img_units,
        empty_text_units,
        type_counts,
        t_build,
    )

    # ---- Embed ------------------------------------------------------------------
    t0 = time.perf_counter()
    embedder = MultiModalEmbedder(
        backend=cfg.embed.backend,
        model_name=cfg.embed.model_name,
        device=cfg.embed.device,
        batch_size=cfg.embed.batch_size,
        dim=cfg.embed.dim,
    )

    unit_ids = [u.id for u in units]
    unit_texts = [u.retrieval_text or "" for u in units]
    log.info("embed.text start n=%d batch_size=%s", len(unit_ids), cfg.embed.batch_size)
    text_res = embedder.embed_text(unit_ids, unit_texts)
    text_vecs = {i: v.astype(np.float32) for i, v in zip(text_res.ids, text_res.vecs)}
    log.info("embed.text done vecs=%d dim=%s", len(text_vecs), (next(iter(text_vecs.values())).shape[0] if text_vecs else None))

    img_unit_ids = []
    img_unit_paths = []
    for u in tqdm(units, desc="Collect image units", leave=False):
        if u.image_paths:
            img_unit_ids.append(u.id)
            img_unit_paths.append(u.image_paths[0])

    img_vecs = {}
    if img_unit_paths:
        log.info("embed.images(units) start n=%d", len(img_unit_paths))
        img_res = embedder.embed_images(img_unit_ids, img_unit_paths)
        img_vecs = {i: v.astype(np.float32) for i, v in zip(img_res.ids, img_res.vecs)}
        log.info("embed.images(units) done vecs=%d dim=%s", len(img_vecs), (next(iter(img_vecs.values())).shape[0] if img_vecs else None))
    else:
        log.info("embed.images(units) skipped (no image units)")

    unit_vecs: dict[str, np.ndarray] = {}
    wimg = float(cfg.retrieval.unit_image_weight)
    dim = (
        next(iter(text_vecs.values())).shape[0]
        if text_vecs
        else (next(iter(img_vecs.values())).shape[0] if img_vecs else 0)
    )
    z = np.zeros((dim,), dtype=np.float32) if dim else None

    missing_text = 0
    missing_img = 0
    combined_both = 0
    combined_only_text = 0
    combined_only_img = 0

    for u in tqdm(units, desc="Build unit vectors", leave=False):
        tv = text_vecs.get(u.id, z)
        iv = img_vecs.get(u.id)
        if tv is None and iv is None:
            continue
        if tv is None:
            v = iv
            combined_only_img += 1
        elif iv is None:
            v = tv
            combined_only_text += 1
        else:
            v = tv + (wimg * iv)
            combined_both += 1
        if u.id not in text_vecs:
            missing_text += 1
        if u.id not in img_vecs and u.image_paths:
            missing_img += 1
        unit_vecs[u.id] = _normalize(v)

    log.info(
        "unit_vecs built n=%d dim=%d combine: both=%d only_text=%d only_img=%d missing_text=%d missing_img=%d wimg=%.3f",
        len(unit_vecs),
        dim,
        combined_both,
        combined_only_text,
        combined_only_img,
        missing_text,
        missing_img,
        wimg,
    )

    page_ids = []
    page_imgs = []
    for p in tqdm(doc.pages, desc="Collect page images", leave=False):
        if p.page_image_path:
            page_ids.append(p.page)
            page_imgs.append(p.page_image_path)

    page_vecs = {}
    if page_imgs:
        log.info("embed.images(pages) start n=%d", len(page_imgs))
        img_res = embedder.embed_images([str(i) for i in page_ids], page_imgs)
        page_vecs = {int(i): v.astype(np.float32) for i, v in zip(img_res.ids, img_res.vecs)}
        log.info("embed.images(pages) done vecs=%d dim=%s", len(page_vecs), (next(iter(page_vecs.values())).shape[0] if page_vecs else None))
    else:
        log.info("embed.images(pages) skipped (no page renders)")

    # IMPORTANT: query embedding uses retrieval_query (query + anchor extension)
    q_vec = embedder.embed_text(["q"], [retrieval_query]).vecs[0].astype(np.float32)
    q_vec = _normalize(q_vec)

    t_embed = time.perf_counter() - t0
    log.info("stage.embed done time_s=%.2f", t_embed)

    # ---- Retrieve ---------------------------------------------------------------
    t0 = time.perf_counter()
    retriever = CombinedRetriever(
        units=units,
        unit_vecs=unit_vecs,
        page_vecs=page_vecs,
        rrf_k=cfg.retrieval.rrf_k,
        table_boost=cfg.retrieval.table_boost,
        figure_boost=cfg.retrieval.figure_boost,
        min_table_candidates=cfg.retrieval.min_table_candidates,
        min_figure_candidates=cfg.retrieval.min_figure_candidates,
    )
    hits, pages = retriever.retrieve(
        query=retrieval_query,
        query_vec=q_vec,
        top_pages=cfg.retrieval.top_pages,
        top_bm25=cfg.retrieval.top_units_bm25,
        top_dense=cfg.retrieval.top_units_dense,
    )
    t_retrieve = time.perf_counter() - t0

    log.info(
        "stage.retrieve done hits=%d pages_ranked=%d time_s=%.2f top_hits=%s",
        len(hits),
        len(pages),
        t_retrieve,
        _preview_hits(hits, k=8),
    )

    cand_ids = [h.unit_id for h in hits]
    cand_set = set(cand_ids)
    cand = [u for u in units if u.id in cand_set]
    rel_scores = {h.unit_id: float(h.score) for h in hits}

    # ---- Costing + selection ----------------------------------------------------
    t0 = time.perf_counter()
    profiler = CostProfiler(token_counter=tc, alpha_visual=cfg.selection.alpha_visual)
    costs = {}
    for u in tqdm(units, desc="Compute cognitive costs", leave=False):
        costs[u.id] = profiler.cost(u.context_text, u.image_paths).total

    selector = BudgetedSelector(
        budget_tokens=cfg.selection.budget_tokens,
        redundancy_beta=cfg.selection.redundancy_beta,
        rel_weight=cfg.selection.rel_weight,
        coverage_weight=cfg.selection.coverage_weight,
        min_gain=cfg.selection.min_gain,
        stopwords=set(cfg.selection.stopwords),
    )
    selected = selector.select(
        query=retrieval_query,
        candidates=cand,
        rel_scores=rel_scores,
        costs=costs,
        vecs=unit_vecs,
    )
    t_select = time.perf_counter() - t0

    log.info(
        "stage.select done candidates=%d selected=%d budget=%d lambda=%.3f time_s=%.2f",
        len(cand),
        len(selected),
        cfg.selection.budget_tokens,
        cfg.selection.redundancy_beta,
        t_select,
    )

    # ---- Output assembly --------------------------------------------------------
    selected_units = [s.unit for s in selected]
    selected_ids = [u.id for u in selected_units]

    ocr_used = bool(cfg.extract.ocr.enabled and cfg.extract.mode in ("ocr", "auto"))
    selected_chunks = []
    sel_costs = []
    table_chunks = 0
    unclear_chunks = 0
    figure_chunks = 0

    for s in tqdm(selected, desc="Build selected chunks", leave=False):
        u = s.unit
        c = int(costs.get(u.id, 0))
        sel_costs.append(float(c))
        rel = rel_scores.get(u.id)
        md = _meta_for_unit(u, c, rel, ocr_used=ocr_used)
        txt = u.context_text or ""
        if md.get("contains_tables") is True:
            table_chunks += 1
        if u.type == "figure" or md.get("unit_type") == "figure":
            figure_chunks += 1
        if isinstance(txt, str) and "[unclear]" in txt.lower():
            unclear_chunks += 1

        # IMPORTANT: keep image_paths so tables/figures/page images can be used downstream
        selected_chunks.append(
            {
                "chunk_id": u.id,
                "content": {
                    "text": txt,
                    "image_paths": [str(p) for p in (u.image_paths or [])],
                },
                "metadata": md,
            }
        )

    used_budget = int(sum(sel_costs))
    budgets = {"total": int(cfg.selection.budget_tokens), "used": used_budget}
    top2_conc = _top2_concentration(sel_costs)

    log.info(
        "selection.stats used_budget=%d/%d selected=%d top2_cost_concentration=%s tables=%d figures=%d unclear=%d",
        budgets["used"],
        budgets["total"],
        len(selected_ids),
        top2_conc,
        table_chunks,
        figure_chunks,
        unclear_chunks,
    )

    # ---- wandb init -------------------------------------------------------------
    run_wandb = wandb_init(
        enabled=cfg.wandb.enabled,
        project=cfg.wandb.project,
        entity=cfg.wandb.entity,
        tags=cfg.wandb.tags,
        name=run_id,
        config={
            "run_id": run_id,
            "budget_total": budgets["total"],
            "lambda_diversity": cfg.selection.redundancy_beta,
            "embed_backend": cfg.embed.backend,
            "embed_model_name": cfg.embed.model_name,
            "extract_mode": cfg.extract.mode,
            "ocr_enabled": cfg.extract.ocr.enabled,
        },
    )

    # ---- Summarization (anchor summary) ----------------------------------------
    anchor_text = None
    anchor_meta = None
    llm_client = None

    t_sum = None
    if cfg.openai.enabled and cfg.summarization.enabled:
        t0 = time.perf_counter()
        llm_client = OpenAIChatClient(timeout_s=cfg.openai.timeout_s)
        anchor_text, anchor_meta = generate_anchor_summary(
            client=llm_client,
            model=cfg.openai.model,
            temperature=cfg.openai.temperature,
            user_instruction=user_instruction,
            required_sections=required_sections,
            selected_chunks=selected_chunks,
            max_chars_per_chunk=cfg.summarization.max_chars_per_chunk,
        )
        t_sum = time.perf_counter() - t0
        log.info("anchor.summary done time_s=%.2f meta=%s", t_sum, anchor_meta)
        log.info("anchor.summary.snippet:\n%s", _safe_snippet(anchor_text, 1200))

        # Save full anchor summary
        (out_dir / "anchor_summary.md").write_text(anchor_text or "", encoding="utf-8")
        log.info("output.anchor_summary path=%s", out_dir / "anchor_summary.md")
    else:
        log.info("anchor.summary skipped openai.enabled=%s summarization.enabled=%s", cfg.openai.enabled, cfg.summarization.enabled)

    # ---- LLM extraction ---------------------------------------------------------
    extraction_info = None
    t_llm_extract = None
    if cfg.openai.enabled and cfg.llm_extraction.enabled:
        t0 = time.perf_counter()
        llm_client = llm_client or OpenAIChatClient(timeout_s=cfg.openai.timeout_s)
        raw, parsed = extract_units(
            client=llm_client,
            model=cfg.llm_extraction.model,
            temperature=cfg.llm_extraction.temperature,
            user_instruction=user_instruction,
            selected_chunks=selected_chunks,
            wrap_unit_tags=cfg.llm_extraction.wrap_unit_tags,
            max_units=cfg.llm_extraction.max_units,
            max_chars_per_input=cfg.llm_extraction.max_chars_per_input,
        )
        t_llm_extract = time.perf_counter() - t0

        raw_path = out_dir / "llm_extraction_raw.txt"
        parsed_path = out_dir / "llm_extraction_parsed.json"
        if cfg.llm_extraction.save_raw:
            raw_path.write_text(raw or "", encoding="utf-8")
        write_json(parsed_path, {"units": parsed})
        extraction_info = {
            "enabled": True,
            "wrap_unit_tags": bool(cfg.llm_extraction.wrap_unit_tags),
            "raw_path": str(raw_path) if cfg.llm_extraction.save_raw else None,
            "parsed_path": str(parsed_path),
            "num_units": len(parsed),
            "time_s": float(t_llm_extract) if t_llm_extract is not None else None,
        }

        log.info(
            "llm.extraction done num_units=%d save_raw=%s time_s=%.2f parsed_path=%s",
            len(parsed),
            cfg.llm_extraction.save_raw,
            t_llm_extract,
            parsed_path,
        )
    else:
        log.info("llm.extraction skipped openai.enabled=%s llm_extraction.enabled=%s", cfg.openai.enabled, cfg.llm_extraction.enabled)

    # ---- Save embeddings (optional, but powerful for debugging) -----------------
    cfg_save_emb = bool(getattr(getattr(cfg, "output", object()), "save_embeddings", False))
    save_embeddings = bool(args.save_embeddings or cfg_save_emb)

    emb_path = None
    if save_embeddings:
        t0 = time.perf_counter()
        emb_path = _save_embeddings_npz(out_dir, unit_vecs=unit_vecs, page_vecs=page_vecs, q_vec=q_vec)
        log.info("output.embeddings path=%s time_s=%.2f", emb_path, time.perf_counter() - t0)

    # ---- wandb log --------------------------------------------------------------
    unit_norm_mean = float(np.mean([np.linalg.norm(v) for v in unit_vecs.values()])) if unit_vecs else None
    page_norm_mean = float(np.mean([np.linalg.norm(v) for v in page_vecs.values()])) if page_vecs else None
    q_norm = float(np.linalg.norm(q_vec)) if q_vec is not None else None

    wandb_log(
        run_wandb,
        {
            "run_id": run_id,
            "budgets_total": budgets["total"],
            "budgets_used": budgets["used"],
            "lambda_diversity": cfg.selection.redundancy_beta,
            "num_selected_chunks": len(selected_ids),
            "top2_cost_concentration": top2_conc,
            "num_table_chunks": table_chunks,
            "num_figure_chunks": figure_chunks,
            "num_unclear_chunks": unclear_chunks,
            "pages_ranked_len": len(pages),
            "anchor_has_placeholders": bool(anchor_meta.get("has_placeholders")) if isinstance(anchor_meta, dict) else None,
            "timing_extract_s": t_extract,
            "timing_build_units_s": t_build,
            "timing_embed_s": t_embed,
            "timing_retrieve_s": t_retrieve,
            "timing_select_s": t_select,
            "timing_anchor_summary_s": t_sum,
            "timing_llm_extract_s": t_llm_extract,
            "unit_vecs_n": len(unit_vecs),
            "page_vecs_n": len(page_vecs),
            "unit_vec_norm_mean": unit_norm_mean,
            "page_vec_norm_mean": page_norm_mean,
            "query_vec_norm": q_norm,
            "selected_chunks": selected_chunks,
            "llm_extraction": extraction_info,
            
        },
    )

    # ---- Context JSON / highlighted pdf ----------------------------------------
    if cfg.output.save_context_json:
        ctx_payload = {
            "run_id": run_id,
            "pdf_path": str(pdf_path),
            "query": args.query,
            "retrieval_query": retrieval_query,
            "anchor_query_text": anchor_query_text,
            "pages_ranked": pages,
            "selected": [
                {
                    "chunk_id": s.unit.id,
                    "page": s.unit.page,
                    "bbox": list(s.unit.bbox),
                    "type": s.unit.type,
                    "rel": s.rel,
                    "cost": s.cost,
                    "text": s.unit.context_text,
                    "image_paths": [str(p) for p in s.unit.image_paths],
                }
                for s in selected
            ],
            "debug": {
                "type_counts": type_counts,
                "embed": {
                    "unit_vecs_n": len(unit_vecs),
                    "page_vecs_n": len(page_vecs),
                    "unit_vec_norm_mean": unit_norm_mean,
                    "page_vec_norm_mean": page_norm_mean,
                    "query_vec_norm": q_norm,
                    "embeddings_npz_path": str(emb_path) if emb_path else None,
                },
                "timings_s": {
                    "extract": t_extract,
                    "build_units": t_build,
                    "embed": t_embed,
                    "retrieve": t_retrieve,
                    "select": t_select,
                    "anchor_summary": t_sum,
                    "llm_extraction": t_llm_extract,
                    "total": time.perf_counter() - t_all0,
                },
            },
        }
        write_json(out_dir / "context.json", ctx_payload)
        log.info("output.context_json path=%s", out_dir / "context.json")

    if cfg.output.save_highlighted_pdf:
        out_pdf = out_dir / "highlighted.pdf"
        highlight_pdf(pdf_path, selected_units, out_pdf)
        log.info("output.highlighted_pdf path=%s", out_pdf)

    # ---- Pipeline output --------------------------------------------------------
    pipeline_output = {
        "run_id": run_id,
        "task": {"user_instruction": user_instruction, "required_sections": required_sections},
        "retrieval": {
            "query": args.query,
            "retrieval_query": retrieval_query,
            "anchor_query_text": anchor_query_text,
            "selected_chunk_ids": selected_ids,
            "budgets": budgets,
            "objective": {"lambda_diversity": cfg.selection.redundancy_beta},
        },
        "chunks": selected_chunks,
        "config": {"wandb": {"enabled": cfg.wandb.enabled, "project": cfg.wandb.project, "entity": cfg.wandb.entity, "tags": cfg.wandb.tags}},
    }
    if anchor_text:
        pipeline_output["anchors"] = {"anchor_summary_text": anchor_text, "meta": anchor_meta}
        pipeline_output["summarization"] = {"summary_text": anchor_text}
    if extraction_info:
        pipeline_output["extraction"] = extraction_info

    if save_embeddings and emb_path:
        pipeline_output.setdefault("debug", {})
        pipeline_output["debug"]["embeddings_npz_path"] = str(emb_path)

    if cfg.output.save_run_output_json:
        out_path = Path(args.run_output) if args.run_output else (out_dir / cfg.output.run_output_filename)
        write_json(out_path, pipeline_output)
        log.info("output.pipeline_json path=%s", out_path)

    wandb_finish(run_wandb)

    t_total = time.perf_counter() - t_all0
    log.info("done run_id=%s selected=%d total_time_s=%.2f", run_id, len(selected_ids), t_total)


if __name__ == "__main__":
    main()
