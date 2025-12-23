from __future__ import annotations

import argparse
import logging
import re
from pathlib import Path

import numpy as np

from src.anchor_summary import generate_anchor_summary
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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default="config/default_config.yaml")
    ap.add_argument("--pdf", type=str, required=True)
    ap.add_argument("--query", type=str, required=True)
    ap.add_argument("--instruction", type=str, default=None)
    ap.add_argument("--required_sections", type=str, default=None)
    ap.add_argument("--run_output", type=str, default=None)
    args = ap.parse_args()

    cfg = load_config(args.config)
    setup_logging(cfg.logging.level, cfg.logging.file)

    pdf_path = Path(args.pdf).resolve()
    run_id = _doc_id(pdf_path)

    ensure_dir(cfg.paths.work_dir)
    ensure_dir(cfg.paths.cache_dir)

    user_instruction = (args.instruction or args.query).strip()
    required_sections = cfg.summarization.required_sections
    if args.required_sections:
        required_sections = [s.strip() for s in args.required_sections.split(",") if s.strip()]

    ocr = None
    if cfg.extract.ocr.enabled:
        ocr = OCREngine(
            engine=cfg.extract.ocr.engine,
            lang=cfg.extract.ocr.lang,
            min_conf=cfg.extract.ocr.min_conf,
            psm=cfg.extract.ocr.psm,
        )

    extractor = PdfLayoutExtractor(
        dpi=cfg.extract.dpi,
        work_dir=cfg.paths.work_dir,
        keep_page_renders=cfg.extract.keep_page_renders,
        keep_crops=cfg.extract.keep_crops,
        mode=cfg.extract.mode,
        ocr=ocr,
        ocr_on_pages=cfg.extract.ocr.on_pages,
        ocr_on_figures=cfg.extract.ocr.on_figures,
    )
    doc = extractor.extract(pdf_path, doc_id=run_id)

    tc = TokenCounter()
    builder = EvidenceBuilder(
        max_text_chunk_tokens=cfg.extract.max_text_chunk_tokens,
        caption_search_px=cfg.extract.caption_search_px,
        stopwords=set(cfg.selection.stopwords),
    )
    units = builder.build(doc, token_counter=tc)

    embedder = MultiModalEmbedder(
        backend=cfg.embed.backend,
        model_name=cfg.embed.model_name,
        device=cfg.embed.device,
        batch_size=cfg.embed.batch_size,
        dim=cfg.embed.dim,
    )

    unit_ids = [u.id for u in units]
    unit_texts = [u.retrieval_text or "" for u in units]
    text_res = embedder.embed_text(unit_ids, unit_texts)
    text_vecs = {i: v.astype(np.float32) for i, v in zip(text_res.ids, text_res.vecs)}

    img_unit_ids = []
    img_unit_paths = []
    for u in units:
        if u.image_paths:
            img_unit_ids.append(u.id)
            img_unit_paths.append(u.image_paths[0])
    img_vecs = {}
    if img_unit_paths:
        img_res = embedder.embed_images(img_unit_ids, img_unit_paths)
        img_vecs = {i: v.astype(np.float32) for i, v in zip(img_res.ids, img_res.vecs)}

    unit_vecs: dict[str, np.ndarray] = {}
    wimg = float(cfg.retrieval.unit_image_weight)
    dim = next(iter(text_vecs.values())).shape[0] if text_vecs else (next(iter(img_vecs.values())).shape[0] if img_vecs else 0)
    z = np.zeros((dim,), dtype=np.float32) if dim else None

    for u in units:
        tv = text_vecs.get(u.id, z)
        iv = img_vecs.get(u.id)
        if tv is None and iv is None:
            continue
        if tv is None:
            v = iv
        elif iv is None:
            v = tv
        else:
            v = tv + (wimg * iv)
        unit_vecs[u.id] = _normalize(v)

    page_ids = []
    page_imgs = []
    for p in doc.pages:
        if p.page_image_path:
            page_ids.append(p.page)
            page_imgs.append(p.page_image_path)
    page_vecs = {}
    if page_imgs:
        img_res = embedder.embed_images([str(i) for i in page_ids], page_imgs)
        page_vecs = {int(i): v.astype(np.float32) for i, v in zip(img_res.ids, img_res.vecs)}

    q_vec = embedder.embed_text(["q"], [args.query]).vecs[0].astype(np.float32)
    q_vec = _normalize(q_vec)

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
        query=args.query,
        query_vec=q_vec,
        top_pages=cfg.retrieval.top_pages,
        top_bm25=cfg.retrieval.top_units_bm25,
        top_dense=cfg.retrieval.top_units_dense,
    )

    cand_ids = [h.unit_id for h in hits]
    cand_set = set(cand_ids)
    cand = [u for u in units if u.id in cand_set]
    rel_scores = {h.unit_id: float(h.score) for h in hits}

    profiler = CostProfiler(token_counter=tc, alpha_visual=cfg.selection.alpha_visual)
    costs = {u.id: profiler.cost(u.context_text, u.image_paths).total for u in units}

    selector = BudgetedSelector(
        budget_tokens=cfg.selection.budget_tokens,
        redundancy_beta=cfg.selection.redundancy_beta,
        rel_weight=cfg.selection.rel_weight,
        coverage_weight=cfg.selection.coverage_weight,
        min_gain=cfg.selection.min_gain,
        stopwords=set(cfg.selection.stopwords),
    )
    selected = selector.select(query=args.query, candidates=cand, rel_scores=rel_scores, costs=costs, vecs=unit_vecs)

    out_dir = ensure_dir(cfg.paths.work_dir / run_id / "results")
    selected_units = [s.unit for s in selected]
    selected_ids = [u.id for u in selected_units]

    ocr_used = bool(cfg.extract.ocr.enabled and cfg.extract.mode in ("ocr", "auto"))
    selected_chunks = []
    sel_costs = []
    table_chunks = 0
    unclear_chunks = 0

    for s in selected:
        u = s.unit
        c = int(costs.get(u.id, 0))
        sel_costs.append(float(c))
        rel = rel_scores.get(u.id)
        md = _meta_for_unit(u, c, rel, ocr_used=ocr_used)
        txt = u.context_text or ""
        if md.get("contains_tables") is True:
            table_chunks += 1
        if isinstance(txt, str) and "[unclear]" in txt.lower():
            unclear_chunks += 1
        selected_chunks.append(
            {
                "chunk_id": u.id,
                "content": {"text": txt},
                "metadata": md,
            }
        )

    used_budget = int(sum(sel_costs))
    budgets = {"total": int(cfg.selection.budget_tokens), "used": used_budget}
    top2_conc = _top2_concentration(sel_costs)

    run_wandb = wandb_init(
        enabled=cfg.wandb.enabled,
        project=cfg.wandb.project,
        entity=cfg.wandb.entity,
        tags=cfg.wandb.tags,
        name=run_id,
        config={"run_id": run_id, "budget_total": budgets["total"], "lambda_diversity": cfg.selection.redundancy_beta},
    )

    anchor_text = None
    anchor_meta = None
    llm_client = None

    if cfg.openai.enabled and cfg.summarization.enabled:
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

    extraction_info = None
    if cfg.openai.enabled and cfg.llm_extraction.enabled:
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
        }

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
            "num_unclear_chunks": unclear_chunks,
            "pages_ranked_len": len(pages),
            "anchor_has_placeholders": bool(anchor_meta.get("has_placeholders")) if isinstance(anchor_meta, dict) else None,
        },
    )

    if cfg.output.save_context_json:
        ctx_payload = {
            "run_id": run_id,
            "pdf_path": str(pdf_path),
            "query": args.query,
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
        }
        write_json(out_dir / "context.json", ctx_payload)
        log.info("output.context_json path=%s", out_dir / "context.json")

    if cfg.output.save_highlighted_pdf:
        out_pdf = out_dir / "highlighted.pdf"
        highlight_pdf(pdf_path, selected_units, out_pdf)

    pipeline_output = {
        "run_id": run_id,
        "task": {"user_instruction": user_instruction, "required_sections": required_sections},
        "retrieval": {"selected_chunk_ids": selected_ids, "budgets": budgets, "objective": {"lambda_diversity": cfg.selection.redundancy_beta}},
        "chunks": selected_chunks,
        "config": {"wandb": {"enabled": cfg.wandb.enabled, "project": cfg.wandb.project, "entity": cfg.wandb.entity, "tags": cfg.wandb.tags}},
    }
    if anchor_text:
        pipeline_output["anchors"] = {"anchor_summary_text": anchor_text, "meta": anchor_meta}
        pipeline_output["summarization"] = {"summary_text": anchor_text}
    if extraction_info:
        pipeline_output["extraction"] = extraction_info

    if cfg.output.save_run_output_json:
        out_path = Path(args.run_output) if args.run_output else (out_dir / cfg.output.run_output_filename)
        write_json(out_path, pipeline_output)
        log.info("output.pipeline_json path=%s", out_path)

    wandb_finish(run_wandb)
    log.info("done run_id=%s selected=%d", run_id, len(selected_ids))


if __name__ == "__main__":
    main()
