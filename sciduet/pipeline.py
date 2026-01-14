"""
Pipeline execution for SciDuet processing.

This module runs stages 3-7 of the DPP-retrieval pipeline:
- Stage 3: Build evidence units
- Stage 4: Embed units and query
- Stage 5: Retrieve candidates (with optional multi-anchor scoring)
- Stage 6: Select units under budget constraint
- Stage 7: Generate summary
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from src.anchor_summary import generate_anchor_summary
from src.cognitive_profiler import EmbeddingBasedCostProfiler, SimpleCostProfiler, TokenCounter
from src.embedder import MultiModalEmbedder
from src.evidence_builder import EvidenceBuilder
from src.indexing import CombinedRetriever
from src.openai_client import OpenAIChatClient
from src.optimizer import BudgetedSelector
from src.schema import DocumentArtifact

log = logging.getLogger(__name__)


def run_pipeline_on_document(
    doc: DocumentArtifact,
    query: str,
    cfg: Any,
    out_dir: Path,
    user_instruction: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Run pipeline stages 3-7 on a DocumentArtifact.
    
    Args:
        doc: DocumentArtifact with pages and elements
        query: Query string for retrieval
        cfg: Configuration object
        out_dir: Output directory (unused but kept for signature compatibility)
        user_instruction: Optional instruction for summary generation
        
    Returns:
        Dictionary containing:
        - units: All evidence units
        - candidates: Retrieved candidate units
        - selections: Dict[method, selected_units]
        - budgets_used: Dict[method, tokens_used]
        - rel_scores: Dict[unit_id, relevance_score]
        - costs: Dict[unit_id, cognitive_cost]
        - summary_text: Generated summary
        - summary_meta: Summary metadata
        - embedder: Embedder instance (for diagnostics)
    """
    user_instruction = user_instruction or query

    # Stage 3: Build evidence units
    log.info("  Stage 3: Building evidence units...")
    token_counter = TokenCounter()
    builder = EvidenceBuilder(
        max_text_chunk_tokens=cfg.extract.max_text_chunk_tokens,
        caption_search_px=cfg.extract.caption_search_px,
        stopwords=set(),
    )
    units = builder.build(doc, token_counter=token_counter)
    log.info("  Built %d evidence units", len(units))

    # Stage 4: Embed
    log.info("  Stage 4: Embedding...")
    embedder = MultiModalEmbedder(
        backend=cfg.embed.backend,
        model_name=cfg.embed.model_name,
        device=cfg.embed.device,
        batch_size=cfg.embed.batch_size,
        dim=cfg.embed.dim,
    )
    unit_vecs, query_vec = _compute_embeddings(embedder, units, query)
    log.info("  Embedded %d units", len(unit_vecs))

    # Stage 5: Retrieve candidates
    log.info("  Stage 5: Retrieving candidates...")
    rel_scores, candidates = _retrieve_candidates(
        units, unit_vecs, query_vec, query, user_instruction, cfg, embedder
    )
    log.info("  Retrieved %d candidates", len(candidates))

    # Stage 6: Select with budget constraint
    log.info("  Stage 6: Selecting units...")
    costs = _compute_costs(units, token_counter, cfg)
    selections, budgets_used = _run_selection(
        query, candidates, rel_scores, costs, unit_vecs, cfg
    )
    log.info("  Selected: %s", ", ".join(f"{m}={len(s)}" for m, s in selections.items()))

    # Stage 7: Generate summary (using greedy selection)
    log.info("  Stage 7: Generating summary...")
    summary_text, summary_meta = _generate_summary(
        selections["greedy"], rel_scores, user_instruction, cfg
    )
    log.info("  Summary generated (%d chars)", len(summary_text))

    return {
        "units": units,
        "candidates": candidates,
        "selections": selections,
        "budgets_used": budgets_used,
        "rel_scores": rel_scores,
        "costs": costs,
        "summary_text": summary_text,
        "summary_meta": summary_meta,
        "embedder": embedder,
    }


# ============================================================================
# Stage 4: Embedding
# ============================================================================

def _compute_embeddings(
    embedder: MultiModalEmbedder, 
    units: List, 
    query: str
) -> Tuple[Dict[str, np.ndarray], np.ndarray]:
    """
    Compute normalized embeddings for units and query.
    
    Returns:
        Tuple of (unit_vecs dict, query_vec)
    """
    unit_texts = [u.retrieval_text or "" for u in units]
    text_res = embedder.embed_text([u.id for u in units], unit_texts)

    unit_vecs = {}
    for uid, vec in zip(text_res.ids, text_res.vecs):
        v = vec.astype(np.float32)
        unit_vecs[uid] = v / (np.linalg.norm(v) + 1e-12)

    query_vec_raw = embedder.embed_text(["q"], [query]).vecs[0].astype(np.float32)
    query_vec = query_vec_raw / (np.linalg.norm(query_vec_raw) + 1e-12)

    return unit_vecs, query_vec


# ============================================================================
# Stage 5: Retrieval
# ============================================================================

def _retrieve_candidates(
    units: List,
    unit_vecs: Dict[str, np.ndarray],
    query_vec: np.ndarray,
    query: str,
    user_instruction: str,
    cfg: Any,
    embedder: MultiModalEmbedder,
) -> Tuple[Dict[str, float], List]:
    """
    Run retrieval and optionally multi-anchor scoring.
    
    Returns:
        Tuple of (relevance_scores dict, candidate_units list)
    """
    retriever = CombinedRetriever(
        units=units,
        unit_vecs=unit_vecs,
        page_vecs={},  # No page images in SciDuet
        rrf_k=cfg.retrieval.rrf_k,
        table_boost=cfg.retrieval.table_boost,
        figure_boost=cfg.retrieval.figure_boost,
        min_table_candidates=cfg.retrieval.min_table_candidates,
        min_figure_candidates=cfg.retrieval.min_figure_candidates,
    )

    hits, _ = retriever.retrieve(
        query_vec=query_vec,
        top_pages=cfg.retrieval.top_pages,
        top_dense=cfg.retrieval.top_units_dense,
    )

    # Optional: Multi-anchor scoring
    if cfg.multi_anchor.enabled:
        rel_scores = _compute_multi_anchor_scores(
            hits, units, unit_vecs, query, user_instruction, cfg, embedder
        )
    else:
        rel_scores = {h.unit_id: h.score for h in hits}

    candidates = [u for u in units if u.id in rel_scores]
    return rel_scores, candidates


def _compute_multi_anchor_scores(
    hits: List,
    units: List,
    unit_vecs: Dict[str, np.ndarray],
    query: str,
    user_instruction: str,
    cfg: Any,
    embedder: MultiModalEmbedder,
) -> Dict[str, float]:
    """Compute relevance scores using multi-anchor approach."""
    log.info("  Stage 5.5: Computing multi-anchor scores...")
    from src.multi_anchor import generate_multi_anchors, MultiAnchorScorer

    client = OpenAIChatClient(timeout_s=cfg.extract.timeout_s)
    anchor_queries = generate_multi_anchors(
        client=client,
        model=cfg.openai.model,
        temperature=cfg.openai.temperature,
        query=query,
        user_instruction=user_instruction,
        anchor_count=cfg.multi_anchor.anchor_count,
        required_sections=cfg.summarization.required_sections,
    )

    candidate_ids = {h.unit_id for h in hits}
    candidate_vecs = {u.id: unit_vecs[u.id] for u in units if u.id in candidate_ids}

    scorer = MultiAnchorScorer(
        embedder=embedder,
        unit_vecs=candidate_vecs,
        temperature=cfg.multi_anchor.temperature,
    )
    return scorer.compute_scores(anchor_queries or [])


# ============================================================================
# Stage 6: Selection
# ============================================================================

def _compute_costs(
    units: List, 
    token_counter: TokenCounter, 
    cfg: Any
) -> Dict[str, int]:
    """Compute cognitive costs for all units."""
    if cfg.selection.cognitive_cost_mode == "emb":
        profiler = EmbeddingBasedCostProfiler(
            token_counter=token_counter,
            model_name=cfg.selection.cognitive_cost_hf_model,
            device=cfg.selection.cognitive_cost_device,
        )
        return {
            u.id: profiler.cost(u.context_text, u.image_paths, unit_type=u.type).total
            for u in units
        }
    else:
        profiler = SimpleCostProfiler(
            token_counter=token_counter,
            alpha_visual=cfg.selection.alpha_visual,
        )
        return {u.id: profiler.cost(u.context_text, u.image_paths).total for u in units}


def _run_selection(
    query: str,
    candidates: List,
    rel_scores: Dict[str, float],
    costs: Dict[str, int],
    unit_vecs: Dict[str, np.ndarray],
    cfg: Any,
) -> Tuple[Dict[str, List], Dict[str, int]]:
    """
    Run all selection methods.

    Returns:
        Tuple of (selections dict, budgets_used dict)
    """
    from . import METHODS

    selector = BudgetedSelector(
        budget_tokens=cfg.selection.budget_tokens,
        redundancy_beta=cfg.selection.redundancy_beta,
        rel_weight=cfg.selection.rel_weight,
        min_gain=cfg.selection.min_gain,
        coverage_weight=getattr(cfg.selection, "coverage_weight", 0.5),
        stopwords=set(getattr(cfg.selection, "stopwords", [])),
    )

    selections, budgets_used = {}, {}
    for method in METHODS:
        sel = selector.select_method(
            method=method,
            query=query,
            coverage_query=query,
            candidates=candidates,
            rel_scores=rel_scores,
            costs=costs,
            vecs=unit_vecs,
        )
        selections[method] = sel
        budgets_used[method] = int(sum(s.cost for s in sel))

    return selections, budgets_used


# ============================================================================
# Stage 7: Summary Generation
# ============================================================================

def _generate_summary(
    selected: List, 
    rel_scores: Dict[str, float], 
    user_instruction: str, 
    cfg: Any
) -> Tuple[str, Dict]:
    """Generate anchor summary from selected units."""
    selected_chunks = [
        {
            "chunk_id": s.unit.id,
            "content": {"text": s.unit.context_text, "image_paths": []},
            "metadata": {
                "unit_type": s.unit.type,
                "cognitive_cost": int(s.cost),
                "importance_score": rel_scores.get(s.unit.id),
            },
        }
        for s in selected
    ]

    client = OpenAIChatClient(timeout_s=cfg.extract.timeout_s)
    return generate_anchor_summary(
        client=client,
        model=cfg.openai.model,
        temperature=cfg.openai.temperature,
        user_instruction=user_instruction,
        required_sections=cfg.summarization.required_sections,
        selected_chunks=selected_chunks,
        max_chars_per_chunk=cfg.summarization.max_chars_per_chunk,
    )