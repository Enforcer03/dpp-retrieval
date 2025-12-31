"""Simplified main pipeline with clear stage separation."""
from __future__ import annotations

import argparse
import logging
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
from src.openai_client import OpenAIChatClient
from src.pdf_highlighter import highlight_pdf
from src.utils import ensure_dir, setup_logging, stable_id, write_json

log = logging.getLogger(__name__)


def normalize(v: np.ndarray) -> np.ndarray:
    """Normalize vector to unit length."""
    return v / (np.linalg.norm(v) + 1e-12)


def build_unit_vectors(
    units, text_vecs: dict, img_vecs: dict, img_weight: float, dim: int
) -> dict[str, np.ndarray]:
    """Combine text + image embeddings for each unit."""
    unit_vecs = {}
    zero = np.zeros(dim, dtype=np.float32)
    
    for u in units:
        text_vec = text_vecs.get(u.id, zero)
        img_vec = img_vecs.get(u.id)
        
        if img_vec is not None:
            combined = text_vec + (img_weight * img_vec)
        else:
            combined = text_vec
        
        unit_vecs[u.id] = normalize(combined)
    
    return unit_vecs


def main() -> None:
    # Parse args
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/default_config.yaml")
    parser.add_argument("--pdf", required=True)
    parser.add_argument("--query", required=True)
    parser.add_argument("--instruction", default=None)
    parser.add_argument("--output_dir", default=None, help="Override output directory")
    parser.add_argument("--recompute", action="store_true", help="Force recompute extraction")
    args = parser.parse_args()

    # Load config
    cfg = load_config(args.config)
    setup_logging(cfg.logging.level, cfg.logging.file)
    
    if args.recompute:
        log.info("Recompute enabled: bypassing extraction cache")

    # Setup
    pdf_path = Path(args.pdf).resolve()
    run_id = stable_id(pdf_path.name, str(pdf_path.stat().st_size))[:16]
    
    # Use custom output_dir if provided, otherwise use default structure
    if args.output_dir:
        out_dir = ensure_dir(Path(args.output_dir))
    else:
        work_dir = ensure_dir(cfg.paths.work_dir / run_id)
        out_dir = ensure_dir(work_dir / "results")
    
    user_instruction = args.instruction or args.query
    required_sections = cfg.summarization.required_sections
    
    log.info("=== ICB-Sum Pipeline ===")
    log.info("run_id=%s", run_id)
    log.info("pdf=%s (%.1f MB)", pdf_path.name, pdf_path.stat().st_size / 1e6)
    log.info("query=%s", args.query[:100])
    log.info("output=%s", out_dir)
    
    t_start = time.perf_counter()

    # ============================================================================
    # STAGE 1: Anchor Query (for better retrieval)
    # ============================================================================
    retrieval_query = args.query
    
    if cfg.anchor.enabled:
        log.info("\n[1/7] Generating anchor query...")
        client = OpenAIChatClient(timeout_s=cfg.extract.timeout_s)
        anchor_text, anchor_meta = generate_anchor_query(
            client=client,
            model=cfg.openai.model,
            temperature=cfg.openai.temperature,
            query=args.query,
            user_instruction=user_instruction,
            required_sections=required_sections,
            max_chars=cfg.anchor.max_chars,
        )
        retrieval_query = f"{args.query}\n\n{anchor_text}".strip()
        log.info("Anchor query: %d chars", len(anchor_text))
    else:
        log.info("\n[1/7] Skipping anchor query (disabled)")

    # ============================================================================
    # STAGE 2: Extract PDF → Elements
    # ============================================================================
    log.info("\n[2/7] Extracting PDF...")
    t0 = time.perf_counter()
    extractor = PdfLayoutExtractor(cfg, force_recompute=args.recompute)
    doc = extractor.extract(pdf_path, out_dir=cfg.paths.work_dir)
    log.info("Extracted: %d pages, %d elements in %.1fs", 
             len(doc.pages), len(doc.elements), time.perf_counter() - t0)

    # ============================================================================
    # STAGE 3: Build Evidence Units
    # ============================================================================
    log.info("\n[3/7] Building evidence units...")
    t0 = time.perf_counter()
    token_counter = TokenCounter()
    builder = EvidenceBuilder(
        max_text_chunk_tokens=cfg.extract.max_text_chunk_tokens,
        caption_search_px=cfg.extract.caption_search_px,
        stopwords=set(cfg.selection.stopwords),
    )
    units = builder.build(doc, token_counter=token_counter)
    log.info("Built: %d units in %.1fs", len(units), time.perf_counter() - t0)

    # ============================================================================
    # STAGE 4: Embed Units + Pages + Query
    # ============================================================================
    log.info("\n[4/7] Embedding...")
    t0 = time.perf_counter()
    
    embedder = MultiModalEmbedder(
        backend=cfg.embed.backend,
        model_name=cfg.embed.model_name,
        device=cfg.embed.device,
        batch_size=cfg.embed.batch_size,
        dim=cfg.embed.dim,
    )
    
    # Text embeddings
    unit_texts = [u.retrieval_text or "" for u in units]
    text_res = embedder.embed_text([u.id for u in units], unit_texts)
    text_vecs = {i: v.astype(np.float32) for i, v in zip(text_res.ids, text_res.vecs)}
    
    # Image embeddings (for units with images)
    img_units = [(u.id, u.image_paths[0]) for u in units if u.image_paths]
    img_vecs = {}
    if img_units:
        try:
            img_ids, img_paths = zip(*img_units)
            img_res = embedder.embed_images(list(img_ids), list(img_paths))
            img_vecs = {i: v.astype(np.float32) for i, v in zip(img_res.ids, img_res.vecs)}
        except Exception as e:
            log.warning("Image embedding failed: %s", str(e))
    # Page embeddings
    page_imgs = [(p.page, p.page_image_path) for p in doc.pages if p.page_image_path]
    page_vecs = {}
    if page_imgs:
        try:
             page_ids, img_paths = zip(*page_imgs)
             page_res = embedder.embed_images([str(i) for i in page_ids], list(img_paths))
             page_vecs = {int(i): v.astype(np.float32) for i, v in zip(page_res.ids, page_res.vecs)}
        except Exception as e:
            log.warning("Page embedding failed: %s", str(e))
        # page_ids, img_paths = zip(*page_imgs)
        # page_res = embedder.embed_images([str(i) for i in page_ids], list(img_paths))
        # page_vecs = {int(i): v.astype(np.float32) for i, v in zip(page_res.ids, page_res.vecs)}
    
    # Combine text + image for units
    dim = text_vecs[next(iter(text_vecs))].shape[0]
    unit_vecs = build_unit_vectors(units, text_vecs, img_vecs, cfg.retrieval.unit_image_weight, dim)
    
    # Query embedding
    query_vec = normalize(embedder.embed_query(retrieval_query) if hasattr(embedder, 'embed_query') 
                          else embedder.embed_text(["q"], [retrieval_query]).vecs[0])
    
    log.info("Embedded: %d unit_vecs, %d page_vecs, query in %.1fs", 
             len(unit_vecs), len(page_vecs), time.perf_counter() - t0)

    # ============================================================================
    # STAGE 5: Retrieve Candidates
    # ============================================================================
    log.info("\n[5/7] Retrieving candidates...")
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
    
    hits, top_pages = retriever.retrieve(
        query_vec=query_vec,
        top_pages=cfg.retrieval.top_pages,
        top_dense=cfg.retrieval.top_units_dense,
    )
    
    candidates = [u for u in units if u.id in {h.unit_id for h in hits}]
    rel_scores = {h.unit_id: h.score for h in hits}
    
    log.info("Retrieved: %d candidates from %d pages in %.1fs (top score: %.3f)", 
             len(candidates), len(top_pages), time.perf_counter() - t0,
             hits[0].score if hits else 0.0)

    # ============================================================================
    # STAGE 6: Budget-Constrained Selection
    # ============================================================================
    log.info("\n[6/7] Selecting with budget constraint...")
    t0 = time.perf_counter()
    
    # Compute cognitive costs
    profiler = CostProfiler(token_counter=token_counter, alpha_visual=cfg.selection.alpha_visual)
    costs = {u.id: profiler.cost(u.context_text, u.image_paths).total for u in units}
    
    # Greedy selection
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
        candidates=candidates,
        rel_scores=rel_scores,
        costs=costs,
        vecs=unit_vecs,
    )
    
    used_budget = sum(costs[s.unit.id] for s in selected)
    log.info("Selected: %d units, %d/%d budget (%.1f%%) in %.1fs", 
             len(selected), used_budget, cfg.selection.budget_tokens,
             100 * used_budget / cfg.selection.budget_tokens,
             time.perf_counter() - t0)

    # ============================================================================
    # STAGE 7: Generate Summary
    # ============================================================================
    log.info("\n[7/7] Generating summary...")
    t0 = time.perf_counter()
    
    # Prepare chunks for summarization
    selected_chunks = []
    for s in selected:
        u = s.unit
        selected_chunks.append({
            "chunk_id": u.id,
            "content": {
                "text": u.context_text,
                "image_paths": [str(p) for p in u.image_paths],
            },
            "metadata": {
                "unit_type": u.type,
                "cognitive_cost": costs[u.id],
                "importance_score": rel_scores.get(u.id),
            },
        })
    
    # Generate anchor summary
    client = OpenAIChatClient(timeout_s=cfg.extract.timeout_s)
    summary_text, summary_meta = generate_anchor_summary(
        client=client,
        model=cfg.openai.model,
        temperature=cfg.openai.temperature,
        user_instruction=user_instruction,
        required_sections=required_sections,
        selected_chunks=selected_chunks,
        max_chars_per_chunk=cfg.summarization.max_chars_per_chunk,
    )
    
    log.info("Generated summary in %.1fs", time.perf_counter() - t0)

# ============================================================================
    # Save Outputs
    # ============================================================================
    (out_dir / "summary.md").write_text(summary_text, encoding="utf-8")
    
    if cfg.output.save_highlighted_pdf:
        highlight_pdf(pdf_path, [s.unit for s in selected], out_dir / "highlighted.pdf")
    
    if cfg.output.save_context_json:
        # Build full context for eval engine
        selected_for_eval = []
        for s in selected:
            u = s.unit
            selected_for_eval.append({
                "chunk_id": u.id,
                "page": u.page,
                "bbox": list(u.bbox),
                "type": u.type,
                "rel": rel_scores.get(u.id, 0.0),
                "cost": costs[u.id],
                "text": u.context_text,
                "image_paths": [str(p) for p in u.image_paths],
            })
        
        write_json(out_dir / "context.json", {
            "run_id": run_id,
            "pdf_path": str(pdf_path),
            "query": args.query,
            "retrieval_query": retrieval_query,
            "task": {
                "user_instruction": user_instruction,
                "required_sections": required_sections,
            },
            "retrieval": {
                "query": args.query,
                "retrieval_query": retrieval_query,
                "selected_chunk_ids": [u.id for u in [s.unit for s in selected]],
                "budgets": {
                    "total": cfg.selection.budget_tokens,
                    "used": used_budget,
                },
            },
            "chunks": selected_chunks,  # Full chunks with metadata
            "selected": selected_for_eval,  # Simple format for compatibility
            "anchors": {
                "anchor_summary_text": summary_text,
            },
            "summarization": {
                "summary_text": summary_text,
            },
        })
    
    log.info("\n=== Pipeline Complete ===")
    log.info("Total time: %.1fs", time.perf_counter() - t_start)
    log.info("Outputs: %s", out_dir)

if __name__ == "__main__":
    main()