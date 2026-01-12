#!/usr/bin/env python3
"""
run_sciduet.py - Run DPP-retrieval pipeline on SciDuet dataset

Usage:
  python run_sciduet.py --split train --limit 10
  python run_sciduet.py --split validation --config config/default_config.yaml
"""
from __future__ import annotations

import argparse
import logging
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
from datasets import load_dataset
from tqdm import tqdm

from src.anchor_summary import generate_anchor_summary
from src.cognitive_profiler import EmbeddingBasedCostProfiler, SimpleCostProfiler, TokenCounter
from src.config import load_config
from src.diagnostics import generate_method_comparison_diagnostics
from src.embedder import MultiModalEmbedder
from src.evidence_builder import EvidenceBuilder
from src.indexing import CombinedRetriever
from src.openai_client import OpenAIChatClient
from src.optimizer import BudgetedSelector
from src.schema import DocumentArtifact, Element, PageArtifact
from src.utils import (
    ensure_dir,
    setup_logging,
    stable_id,
    write_json,
    load_json,
    save_json,
    calculate_decision_score,
    display_score_summary,
    run_command,
)
from src.wandb_logger import wandb_finish, wandb_init, wandb_log, wandb_log_artifact

log = logging.getLogger(__name__)


# ============================================================================
# SECTION 1: SciDuet to DocumentArtifact Converter
# ============================================================================
class SciDuetConverter:
    """Convert SciDuet paper data to DocumentArtifact format."""

    def __init__(self, max_section_length: int = 4000, target_chunk_size: int = 1200):
        self.max_section_length = max_section_length
        self.target_chunk_size = target_chunk_size
        self.log = logging.getLogger(__name__)

    def convert(self, sample: Dict[str, Any]) -> DocumentArtifact:
        """
        Convert SciDuet sample to DocumentArtifact.

        Strategy:
        - Each section header defines a "page"
        - Content under each section becomes text elements
        - No actual PDF or images (page_image_path=None)
        - Synthetic bounding boxes for layout
        """
        paper_id = str(sample.get("paper_id", "unknown"))

        # Extract paper content (it's a dict with paper_content_text list)
        paper_content_data = sample.get("paper_content", {})
        if isinstance(paper_content_data, dict):
            content_chunks = paper_content_data.get("paper_content_text", [])
        else:
            content_chunks = []

        # Extract headers (it's a dict with paper_header_content list)
        paper_headers_data = sample.get("paper_headers", {})
        if isinstance(paper_headers_data, dict):
            headers = paper_headers_data.get("paper_header_content", [])
        else:
            headers = []

        # Convert to lists of strings
        content_chunks = [str(c) for c in content_chunks if c]
        headers = [str(h) for h in headers if h]

        # Create elements directly from content chunks
        sections = self._create_sections_from_chunks(content_chunks, headers)

        # Create pages and elements
        pages = []
        elements = []

        for page_num, (header, section_chunks) in enumerate(sections, start=1):
            # Create page artifact (no image)
            pages.append(
                PageArtifact(
                    page=page_num,
                    width=1000.0,  # Synthetic dimensions
                    height=1000.0,
                    page_image_path=None,
                )
            )

            # Create header element
            if header:
                elements.append(
                    Element(
                        id=stable_id(paper_id, page_num, "header", header),
                        page=page_num,
                        bbox=(0.0, 0.0, 1000.0, 50.0),
                        type="header",
                        text=header,
                        image_path=None,
                    )
                )

            # Create text elements from content chunks
            y_pos = 100.0
            for chunk_idx, chunk in enumerate(section_chunks):
                if not chunk or not chunk.strip():
                    continue

                # Split very long chunks if needed
                if len(chunk) > self.max_section_length:
                    sub_chunks = self._chunk_text(chunk, self.max_section_length)
                else:
                    sub_chunks = [chunk]

                for sub_idx, sub_chunk in enumerate(sub_chunks):
                    if not sub_chunk.strip():
                        continue

                    elements.append(
                        Element(
                            id=stable_id(paper_id, page_num, "text", f"{chunk_idx}_{sub_idx}"),
                            page=page_num,
                            bbox=(0.0, y_pos, 1000.0, y_pos + 200.0),
                            type="text",
                            text=sub_chunk,
                            image_path=None,
                        )
                    )
                    y_pos += 220.0

        return DocumentArtifact(
            doc_id=paper_id,
            pdf_path=Path(f"{paper_id}.txt"),  # Virtual path
            pages=pages,
            elements=elements,
            meta={
                "source": "sciduet",
                "paper_title": sample.get("paper_title", ""),
                "paper_abstract": sample.get("paper_abstract", ""),
            },
        )

    def _create_sections_from_chunks(
        self, content_chunks: List[str], headers: List[str]
    ) -> List[tuple[str, List[str]]]:
        """
        Create sections from content chunks.

        Strategy:
        - Merge small consecutive chunks into larger text blocks (~1000-1500 chars)
        - Use headers to identify section breaks
        - Return list of (header, merged_chunks) tuples
        """
        if not content_chunks:
            return [("", [])]

        # Merge small chunks into larger blocks
        merged_chunks = []
        current_text = ""

        for chunk in content_chunks:
            if len(current_text) + len(chunk) < self.target_chunk_size:
                current_text += " " + chunk if current_text else chunk
            else:
                if current_text:
                    merged_chunks.append(current_text)
                current_text = chunk

        if current_text:
            merged_chunks.append(current_text)

        # Now split merged chunks by sections using headers
        if not headers:
            # No headers - create sections by grouping ~5-8 merged chunks per section
            sections = []
            chunks_per_section = 6
            for i in range(0, len(merged_chunks), chunks_per_section):
                section_chunks = merged_chunks[i:i + chunks_per_section]
                sections.append((f"Section {i//chunks_per_section + 1}", section_chunks))
            return sections

        # Find where headers appear in merged chunks
        sections = []
        current_header = "Introduction"
        current_chunks = []
        header_idx = 0

        for chunk in merged_chunks:
            # Check if this chunk contains the next header
            if header_idx < len(headers):
                next_header = headers[header_idx]
                # Simple heuristic: chunk starts with or contains header
                if (next_header.lower() in chunk.lower()[:100] or
                    chunk.lower().strip().startswith(next_header.lower()[:20])):
                    # Save previous section
                    if current_chunks:
                        sections.append((current_header, current_chunks))
                    # Start new section
                    current_header = next_header
                    current_chunks = [chunk]
                    header_idx += 1
                else:
                    current_chunks.append(chunk)
            else:
                current_chunks.append(chunk)

        # Add final section
        if current_chunks:
            sections.append((current_header, current_chunks))

        return sections if sections else [("", merged_chunks)]

    def _chunk_text(self, text: str, max_length: int) -> List[str]:
        """Chunk long text into smaller pieces."""
        if len(text) <= max_length:
            return [text]

        # Simple sentence-aware chunking
        sentences = text.split(". ")
        chunks = []
        current_chunk = []
        current_length = 0

        for sent in sentences:
            sent_len = len(sent) + 2  # Include '. '
            if current_length + sent_len > max_length and current_chunk:
                chunks.append(". ".join(current_chunk) + ".")
                current_chunk = [sent]
                current_length = sent_len
            else:
                current_chunk.append(sent)
                current_length += sent_len

        if current_chunk:
            chunks.append(". ".join(current_chunk) + ".")

        return chunks


# ============================================================================
# SECTION 2: Pipeline Runner (Stages 3-7)
# ============================================================================
def run_pipeline_on_document(
    doc: DocumentArtifact,
    query: str,
    cfg: Any,
    out_dir: Path,
    user_instruction: str = None,
) -> Dict[str, Any]:
    """
    Run pipeline stages 3-7 on a DocumentArtifact.

    Returns: Dictionary with selections, summary, and metadata
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

    # Text embeddings
    unit_texts = [u.retrieval_text or "" for u in units]
    text_res = embedder.embed_text([u.id for u in units], unit_texts)
    text_vecs = {i: v.astype(np.float32) for i, v in zip(text_res.ids, text_res.vecs)}

    # No image embeddings (SciDuet has no images)
    img_vecs = {}
    page_vecs = {}

    # Build unit vectors (text only, normalize)
    dim = text_vecs[next(iter(text_vecs))].shape[0]
    unit_vecs = {}
    for u in units:
        if u.id in text_vecs:
            vec = text_vecs[u.id]
            norm = np.linalg.norm(vec)
            unit_vecs[u.id] = vec / (norm + 1e-12)

    # Query embedding
    query_vec_raw = embedder.embed_text(["q"], [query]).vecs[0].astype(np.float32)
    query_vec = query_vec_raw / (np.linalg.norm(query_vec_raw) + 1e-12)

    log.info("  Embedded %d units (dim=%d)", len(unit_vecs), dim)

    # Stage 5: Retrieve candidates
    log.info("  Stage 5: Retrieving candidates...")
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

    # Multi-anchor scoring override (optional)
    if cfg.multi_anchor.enabled:
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
        rel_scores = scorer.compute_scores(anchor_queries or [])
        log.info(
            "  Multi-anchor scoring: K=%d, score range=[%.4f, %.4f]",
            len(anchor_queries or []),
            min(rel_scores.values()) if rel_scores else 0.0,
            max(rel_scores.values()) if rel_scores else 0.0,
        )
    else:
        rel_scores = {h.unit_id: h.score for h in hits}

    candidates = [u for u in units if u.id in rel_scores]
    log.info("  Retrieved %d candidates", len(candidates))

    # Stage 6: Select with budget constraint
    log.info("  Stage 6: Selecting units...")

    # Compute costs
    if cfg.selection.cognitive_cost_mode == "emb":
        profiler = EmbeddingBasedCostProfiler(
            token_counter=token_counter,
            model_name=cfg.selection.cognitive_cost_hf_model,
            device=cfg.selection.cognitive_cost_device,
        )
        costs = {}
        for u in units:
            costs[u.id] = profiler.cost(
                u.context_text, u.image_paths, unit_type=u.type
            ).total
    else:
        profiler = SimpleCostProfiler(
            token_counter=token_counter,
            alpha_visual=cfg.selection.alpha_visual,
        )
        costs = {u.id: profiler.cost(u.context_text, u.image_paths).total for u in units}

    # Selection
    selector = BudgetedSelector(
        budget_tokens=cfg.selection.budget_tokens,
        redundancy_beta=cfg.selection.redundancy_beta,
        rel_weight=cfg.selection.rel_weight,
        min_gain=cfg.selection.min_gain,
        coverage_weight=getattr(cfg.selection, "coverage_weight", 0.5),
        stopwords=set(getattr(cfg.selection, "stopwords", [])),
    )

    methods = ["greedy", "topk", "greedy_cov", "cost_norm", "dpp"]
    selections = {}
    budgets_used = {}

    for m in methods:
        sel = selector.select_method(
            method=m,
            query=query,
            coverage_query=query,
            candidates=candidates,
            rel_scores=rel_scores,
            costs=costs,
            vecs=unit_vecs,
        )
        selections[m] = sel
        budgets_used[m] = int(sum(s.cost for s in sel))

    log.info(
        "  Selected: greedy=%d, topk=%d, greedy_cov=%d, cost_norm=%d, dpp=%d",
        len(selections["greedy"]),
        len(selections["topk"]),
        len(selections["greedy_cov"]),
        len(selections["cost_norm"]),
        len(selections["dpp"]),
    )

    # Use greedy for summary
    selected = selections["greedy"]

    # Stage 7: Generate summary
    log.info("  Stage 7: Generating summary...")
    selected_chunks = [
        {
            "chunk_id": s.unit.id,
            "content": {
                "text": s.unit.context_text,
                "image_paths": [],
            },
            "metadata": {
                "unit_type": s.unit.type,
                "cognitive_cost": int(s.cost),
                "importance_score": rel_scores.get(s.unit.id),
            },
        }
        for s in selected
    ]

    client = OpenAIChatClient(timeout_s=cfg.extract.timeout_s)
    summary_text, summary_meta = generate_anchor_summary(
        client=client,
        model=cfg.openai.model,
        temperature=cfg.openai.temperature,
        user_instruction=user_instruction,
        required_sections=cfg.summarization.required_sections,
        selected_chunks=selected_chunks,
        max_chars_per_chunk=cfg.summarization.max_chars_per_chunk,
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
# SECTION 3: Batch Processing
# ============================================================================
def process_sciduet_batch(
    dataset_split: str,
    cfg: Any,
    output_base: Path,
    limit: int = None,
    offset: int = 0,
    eval_mode: str = "all",
    eval_model: str = "gpt-5.1",
    metadata_path: Path = None,
) -> Dict[str, Any]:
    """Process SciDuet dataset in batch mode."""

    # Load dataset
    log.info("Loading SciDuet dataset (split=%s)...", dataset_split)
    try:
        ds = load_dataset("GEM/SciDuet", split=dataset_split)
    except Exception as e:
        log.error("Failed to load SciDuet dataset: %s", e)
        return {"success": False, "error": str(e)}

    total_samples = len(ds)
    log.info("Loaded %d samples", total_samples)

    # Load requirements metadata if provided
    requirements_map = {}
    if metadata_path and metadata_path.exists():
        log.info("Loading requirements from: %s", metadata_path)
        try:
            metadata = load_json(metadata_path)
            for entry in metadata:
                entry_id = entry.get("id", "")
                requirements_map[entry_id] = entry.get("requirements", [])
            log.info("Loaded requirements for %d entries", len(requirements_map))
        except Exception as e:
            log.warning("Failed to load requirements metadata: %s", e)
    else:
        log.info("No metadata path provided, using empty requirements")

    # Apply limit and offset
    if offset > 0:
        end_idx = min(offset + (limit or total_samples), total_samples)
        ds = ds.select(range(offset, end_idx))
        log.info("Applied offset=%d, processing %d samples", offset, len(ds))
    elif limit:
        ds = ds.select(range(min(limit, total_samples)))
        log.info("Limited to %d samples", len(ds))

    # Initialize converter
    converter = SciDuetConverter(
        max_section_length=cfg.extract.max_text_chunk_tokens,
        target_chunk_size=getattr(cfg.extract, 'target_chunk_size', 1200)
    )

    # Batch tracking
    batch_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results = {
        "batch_timestamp": batch_timestamp,
        "dataset_split": dataset_split,
        "total_samples": len(ds),
        "processed": 0,
        "successful": 0,
        "failed": 0,
        "failed_ids": [],
        "method_metrics": {method: [] for method in ["greedy", "topk", "greedy_cov", "cost_norm", "dpp"]},
        "pipeline_success": 0,
        "eval_success": 0,
    }

    score_data = []

    # Process each sample
    for idx, sample in enumerate(tqdm(ds, desc="Processing samples", unit="sample")):
        paper_id = str(sample.get("paper_id", f"unknown_{idx}"))
        gem_id = str(sample.get("gem_id", ""))

        # Extract slide number from gem_id for simpler run_id
        # gem_id format: "GEM-SciDuet-train-1#paper-954#slide-0"
        slide_num = gem_id.split("#slide-")[-1] if "#slide-" in gem_id else "0"
        run_id = f"{batch_timestamp}_{paper_id}_slide{slide_num}_d0"

        # Look up requirements for this sample
        # The entry_id format in metadata is like: "954_GEM-SciDuet-train-1#paper-954#slide-0"
        entry_id = f"{paper_id}_{gem_id}"
        requirements = requirements_map.get(entry_id, [])

        log.info("")
        log.info("=" * 80)
        log.info("Processing %d/%d: %s", idx + 1, len(ds), run_id)
        log.info("Entry ID: %s, Requirements: %d", entry_id, len(requirements))

        try:
            # Convert to DocumentArtifact
            log.info("  Converting SciDuet sample to DocumentArtifact...")
            doc = converter.convert(sample)
            log.info(
                "  Created DocumentArtifact: %d pages, %d elements",
                len(doc.pages),
                len(doc.elements),
            )

            # Extract query (use slide title)
            query = str(sample.get("slide_title", "Summarize this paper"))
            log.info("  Query: %s", query[:100] + ("..." if len(query) > 100 else ""))

            # Create output directory
            out_dir = ensure_dir(output_base / run_id)

            # Run pipeline
            log.info("  Running pipeline...")
            pipeline_result = run_pipeline_on_document(
                doc=doc,
                query=query,
                cfg=cfg,
                out_dir=out_dir,
                user_instruction=query,
            )

            # Save outputs
            log.info("  Saving outputs...")
            diagnostic_metrics, eval_success = save_outputs(
                out_dir=out_dir,
                run_id=run_id,
                doc=doc,
                query=query,
                pipeline_result=pipeline_result,
                cfg=cfg,
                sample=sample,
                eval_mode=eval_mode,
                eval_model=eval_model,
                requirements=requirements,
            )

            # Aggregate diagnostic metrics
            if diagnostic_metrics:
                for method, metrics in diagnostic_metrics.items():
                    results["method_metrics"][method].append(metrics)

            # Track success
            results["successful"] += 1
            results["pipeline_success"] += 1
            if eval_success:
                results["eval_success"] += 1

            # Collect scores
            requirements_path = out_dir / "requirements.json"
            requirements_data = load_json(requirements_path) if requirements_path.exists() else None

            for method in ["greedy", "topk", "greedy_cov", "cost_norm", "dpp"]:
                eval_bundle_path = out_dir / f"eval_bundle_{method}.json" if method != "greedy" else out_dir / "eval_bundle.json"
                if not eval_bundle_path.exists():
                    continue

                try:
                    eval_bundle = load_json(eval_bundle_path)
                    score_info = calculate_decision_score(eval_bundle, requirements_data)
                    if score_info:
                        score_info["pdf_name"] = f"{paper_id}.pdf"
                        score_info["paper_id"] = paper_id
                        score_info["gem_id"] = gem_id
                        score_info["run_id"] = run_id
                        score_info["method"] = method
                        score_data.append(score_info)
                except Exception as e:
                    log.warning("Failed to calculate score for %s (%s): %s", run_id, method, e)

            log.info("  ✓ SUCCESS: %s", run_id)

        except Exception as e:
            log.error("  ✗ FAILED: %s - %s", run_id, str(e), exc_info=True)
            results["failed"] += 1
            results["failed_ids"].append(run_id)

        results["processed"] += 1

    # Add score data to results
    results["score_data"] = score_data

    return results


# ============================================================================
# SECTION 4: Output Saving
# ============================================================================
def save_outputs(
    out_dir: Path,
    run_id: str,
    doc: DocumentArtifact,
    query: str,
    pipeline_result: Dict[str, Any],
    cfg: Any,
    sample: Dict[str, Any],
    eval_mode: str = "all",
    eval_model: str = "gpt-5.1",
    requirements: List[Dict[str, Any]] = None,
) -> tuple[Dict[str, Any], bool]:
    """Save all pipeline outputs and run evaluation.

    Returns:
        Tuple of (diagnostic_metrics, eval_success)
    """
    if requirements is None:
        requirements = []

    # Summary
    (out_dir / "summary.md").write_text(
        pipeline_result["summary_text"], encoding="utf-8"
    )

    # Comprehensive diagnostics comparing all methods
    diagnostic_metrics = None
    try:
        diagnostic_metrics = generate_method_comparison_diagnostics(
            all_units=pipeline_result["units"],
            candidates=pipeline_result["candidates"],
            selections=pipeline_result["selections"],
            costs=pipeline_result["costs"],
            rel_scores=pipeline_result["rel_scores"],
            budget_tokens=cfg.selection.budget_tokens,
            embedder=pipeline_result["embedder"],
            output_path=out_dir / "diagnostics_comparison.png",
        )

        # Log metrics for each method
        if diagnostic_metrics:
            log.info("  Diagnostic Metrics Summary:")
            for method, metrics in diagnostic_metrics.items():
                log.info(
                    "    %s: n=%d, rel=%.3f, rel/tok=%.5f, budget=%.1f%%, sim=%.3f, nov=%.3f",
                    method.upper(),
                    metrics['n_selected'],
                    metrics['mean_relevance'],
                    metrics['mean_rel_per_token'],
                    metrics['budget_pct'],
                    metrics['mean_offdiag_sim'],
                    metrics['mean_novelty'],
                )
    except Exception as e:
        log.warning("Failed to generate method comparison diagnostics: %s", e)

    # Save diagnostic metrics to JSON
    if diagnostic_metrics:
        write_json(out_dir / "diagnostic_metrics.json", diagnostic_metrics)

    # Context JSON for each method
    selections = pipeline_result["selections"]
    budgets_used = pipeline_result["budgets_used"]
    rel_scores = pipeline_result["rel_scores"]

    method_files = {
        "greedy": "context.json",
        "topk": "context_topk.json",
        "greedy_cov": "context_greedy_cov.json",
        "cost_norm": "context_cost_norm.json",
        "dpp": "context_dpp.json",
    }

    for method, sel in selections.items():
        context_data = {
            "run_id": run_id,
            "source": "sciduet",
            "paper_id": sample.get("paper_id"),
            "slide_id": sample.get("slide_id"),
            "query": query,
            "method": method,
            "selected": [
                {
                    "chunk_id": s.unit.id,
                    "page": s.unit.page,
                    "type": s.unit.type,
                    "rel": float(rel_scores.get(s.unit.id, 0.0)),
                    "cost": int(s.cost),
                    "text": s.unit.context_text[:200],  # Preview
                }
                for s in sel
            ],
            "budgets": {
                "total": cfg.selection.budget_tokens,
                "used": budgets_used[method],
            },
            "chunks": [
                {
                    "chunk_id": s.unit.id,
                    "content": {
                        "text": s.unit.context_text,
                        "image_paths": [],
                    },
                    "metadata": {
                        "unit_type": s.unit.type,
                        "cognitive_cost": int(s.cost),
                        "importance_score": rel_scores.get(s.unit.id),
                    },
                }
                for s in sel
            ],
            "summarization": {"summary_text": pipeline_result["summary_text"]},
        }
        write_json(out_dir / method_files[method], context_data)

    # Requirements (from metadata or empty)
    requirements_data = {
        "requirements": requirements,
        "notes": f"SciDuet dataset - {len(requirements)} requirements from metadata" if requirements else "SciDuet dataset - no requirements provided",
    }
    requirements_path = out_dir / "requirements.json"
    write_json(requirements_path, requirements_data)

    # ========================================================================
    # Run Evaluation for each method
    # ========================================================================
    method_files = {
        "greedy": ("context.json", "eval_bundle.json"),
        "topk": ("context_topk.json", "eval_bundle_topk.json"),
        "greedy_cov": ("context_greedy_cov.json", "eval_bundle_greedy_cov.json"),
        "cost_norm": ("context_cost_norm.json", "eval_bundle_cost_norm.json"),
        "dpp": ("context_dpp.json", "eval_bundle_dpp.json"),
    }

    eval_success_any = False

    for method, (ctx_name, eval_name) in method_files.items():
        ctx_path = out_dir / ctx_name
        if not ctx_path.exists():
            log.info("Skipping eval (%s): missing %s", method, ctx_name)
            continue

        eval_output_path = out_dir / eval_name

        eval_cmd = [
            sys.executable,
            "-m",
            "eval_engine.main",
            "--input",
            str(ctx_path),
            "--requirements",
            str(requirements_path),
            "--output",
            str(eval_output_path),
            "--mode",
            eval_mode,
            "--model",
            eval_model,
        ]

        # Save command for debugging
        try:
            (out_dir / f"eval_{method}_cmd.txt").write_text(" ".join(eval_cmd), encoding="utf-8")
        except Exception:
            pass

        ok = run_command(eval_cmd, f"Evaluation ({method}) for {run_id}")
        if ok:
            eval_success_any = True
        else:
            log.warning("Evaluation failed for %s (method=%s)", run_id, method)

    return diagnostic_metrics, eval_success_any


# ============================================================================
# SECTION 5: Main Entry Point
# ============================================================================
def main():
    parser = argparse.ArgumentParser(
        description="Run DPP-retrieval pipeline on SciDuet dataset"
    )
    parser.add_argument(
        "--split",
        default="train",
        choices=["train", "validation", "test"],
        help="Dataset split to process",
    )
    parser.add_argument(
        "--config",
        default="config/default_config.yaml",
        help="Path to config file",
    )
    parser.add_argument(
        "--output", default="output/sciduet", help="Output directory"
    )
    parser.add_argument("--limit", type=int, help="Limit to first N samples")
    parser.add_argument(
        "--offset", type=int, default=0, help="Skip first N samples"
    )
    parser.add_argument(
        "--log_level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level",
    )
    parser.add_argument(
        "--no_wandb", action="store_true", help="Disable wandb logging"
    )
    parser.add_argument(
        "--eval_mode",
        default="all",
        choices=["requirements", "retrieval", "all"],
        help="Evaluation mode",
    )
    parser.add_argument(
        "--eval_model", default="gpt-5.1", help="Model for evaluation"
    )
    parser.add_argument(
        "--metadata",
        help="Path to metadata JSON with requirements (e.g., data/sciduet/sciduet_metadata_output/train_metadata.json)",
    )

    args = parser.parse_args()

    # Setup
    output_base = ensure_dir(Path(args.output))
    cfg = load_config(args.config)

    # Logging
    log_file = output_base / "sciduet_run.log"
    setup_logging(args.log_level, log_file)

    log.info("=" * 80)
    log.info("SciDuet Pipeline Runner")
    log.info("=" * 80)
    log.info("Split: %s", args.split)
    log.info("Config: %s", args.config)
    log.info("Output: %s", output_base)
    if args.limit:
        log.info("Limit: %d samples", args.limit)
    if args.offset:
        log.info("Offset: %d samples", args.offset)
    if args.metadata:
        log.info("Metadata: %s", args.metadata)
    log.info("Eval mode: %s", args.eval_mode)
    log.info("Eval model: %s", args.eval_model)

    # Initialize wandb
    wandb_run = None
    if not args.no_wandb:
        try:
            wandb_run = wandb_init(
                enabled=cfg.wandb.enabled,
                project=cfg.wandb.project,
                entity=cfg.wandb.entity,
                tags=["sciduet", f"split_{args.split}"] + cfg.wandb.tags,
                name=f"sciduet_{args.split}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                config={
                    "split": args.split,
                    "limit": args.limit,
                    "offset": args.offset,
                },
            )
        except Exception as e:
            log.warning("Failed to initialize wandb: %s", e)

    # Process batch
    if args.metadata:
        metadata_path = Path(args.metadata)
    else:
        # Auto-detect metadata file based on split
        default_metadata_path = Path(f"data/sciduet/sciduet_metadata_output/{args.split}_metadata.json")
        if default_metadata_path.exists():
            metadata_path = default_metadata_path
            log.info("Auto-detected metadata file: %s", metadata_path)
        else:
            metadata_path = None
            log.warning("No metadata file found at %s, evaluation will use empty requirements", default_metadata_path)

    results = process_sciduet_batch(
        dataset_split=args.split,
        cfg=cfg,
        output_base=output_base,
        limit=args.limit,
        offset=args.offset,
        eval_mode=args.eval_mode,
        eval_model=args.eval_model,
        metadata_path=metadata_path,
    )

    # Compute aggregated metrics across samples
    aggregated_metrics = {}
    for method, metrics_list in results.get("method_metrics", {}).items():
        if not metrics_list:
            continue

        # Average all numeric metrics
        aggregated = {}
        for key in metrics_list[0].keys():
            values = [m[key] for m in metrics_list if not np.isnan(m[key])]
            if values:
                aggregated[f"{method}_{key}"] = float(np.mean(values))
                aggregated[f"{method}_{key}_std"] = float(np.std(values))

        aggregated_metrics.update(aggregated)

    results["aggregated_metrics"] = aggregated_metrics

    # Summary
    log.info("")
    log.info("=" * 80)
    log.info("BATCH PROCESSING COMPLETE")
    log.info("=" * 80)
    log.info("Processed: %d", results.get("processed", 0))
    log.info("Successful: %d", results.get("successful", 0))
    log.info("Pipeline successes: %d", results.get("pipeline_success", 0))
    log.info("Eval successes: %d", results.get("eval_success", 0))
    log.info("Failed: %d", results.get("failed", 0))

    if results.get("failed_ids"):
        log.warning("Failed IDs: %s", ", ".join(results["failed_ids"][:10]))

    # Display and save score data
    score_data = results.get("score_data", [])
    if score_data:
        display_score_summary(score_data)
        scores_path = output_base / "decision_scores.json"
        save_json(scores_path, score_data)
        log.info("Decision scores saved to: %s", scores_path)

    # Log aggregated metrics
    if aggregated_metrics:
        log.info("")
        log.info("=" * 80)
        log.info("AGGREGATED METRICS (mean ± std across %d samples)", results.get("successful", 0))
        log.info("=" * 80)
        methods = ["greedy", "topk", "greedy_cov", "cost_norm", "dpp"]
        for method in methods:
            log.info("  %s:", method.upper())
            for key in ["n_selected", "mean_relevance", "mean_rel_per_token", "budget_pct", "mean_offdiag_sim", "mean_novelty"]:
                mean_key = f"{method}_{key}"
                std_key = f"{method}_{key}_std"
                if mean_key in aggregated_metrics:
                    log.info(
                        "    %s: %.4f ± %.4f",
                        key,
                        aggregated_metrics[mean_key],
                        aggregated_metrics.get(std_key, 0.0)
                    )

    # Save summary
    summary_path = output_base / "batch_summary.json"
    write_json(summary_path, results)
    log.info("Summary saved: %s", summary_path)

    # Wandb logging
    if wandb_run:
        try:
            wandb_data = {
                "processed": results.get("processed", 0),
                "successful": results.get("successful", 0),
                "pipeline_success": results.get("pipeline_success", 0),
                "eval_success": results.get("eval_success", 0),
                "failed": results.get("failed", 0),
                "success_rate": (
                    results.get("successful", 0)
                    / max(results.get("processed", 1), 1)
                ),
                "pipeline_success_rate": (
                    results.get("pipeline_success", 0)
                    / max(results.get("processed", 1), 1)
                ),
                "eval_success_rate": (
                    results.get("eval_success", 0)
                    / max(results.get("processed", 1), 1)
                ),
            }

            # Add aggregated metrics to wandb
            if aggregated_metrics:
                wandb_data.update(aggregated_metrics)

            # Add score statistics
            if score_data:
                all_scores = [s["avg_score"] for s in score_data]
                wandb_data.update({
                    "batch/avg_score_mean": float(np.mean(all_scores)),
                    "batch/avg_score_std": float(np.std(all_scores)),
                    "batch/avg_score_min": float(np.min(all_scores)),
                    "batch/avg_score_max": float(np.max(all_scores)),
                })

            wandb_log(wandb_run, wandb_data)
            wandb_finish(wandb_run)
        except Exception as e:
            log.warning("Failed to log to wandb: %s", e)

    # Exit code
    sys.exit(0 if results.get("failed", 0) == 0 else 1)


if __name__ == "__main__":
    main()
