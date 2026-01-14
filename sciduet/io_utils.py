"""
I/O utilities for SciDuet processing.

This module handles:
- Output saving (context files, diagnostics, requirements)
- Evaluation execution
- Batch processing orchestration
- Score collection and aggregation
"""
from __future__ import annotations

import logging
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from datasets import load_dataset
from tqdm import tqdm

from src.diagnostics import generate_method_comparison_diagnostics
from src.utils import ensure_dir, load_json, save_json, write_json, calculate_decision_score

from .converter import SciDuetConverter, extract_sample_ids, lookup_requirements
from .pipeline import run_pipeline_on_document

log = logging.getLogger(__name__)


# ============================================================================
# Requirements Metadata Loading
# ============================================================================

def load_requirements_metadata(path: Optional[Path]) -> Dict[str, List[Dict[str, Any]]]:
    """
    Load requirements metadata from JSON file.
    
    Args:
        path: Path to metadata JSON file
        
    Returns:
        Dict mapping entry_id -> list of requirements
    """
    if not path or not path.exists():
        return {}
    try:
        log.info("Loading requirements from: %s", path)
        metadata = load_json(path)
        requirements_map = {
            str(entry.get("id", "")) if entry.get("id") else "": entry.get("requirements", [])
            for entry in metadata
        }
        log.info("Loaded requirements for %d entries", len(requirements_map))
        return requirements_map
    except Exception as e:
        log.warning("Failed to load requirements: %s", e)
        return {}


# ============================================================================
# Output Saving
# ============================================================================

def save_outputs(
    out_dir: Path,
    run_id: str,
    doc: Any,
    query: str,
    pipeline_result: Dict[str, Any],
    cfg: Any,
    sample: Dict[str, Any],
    eval_mode: str = "all",
    eval_model: str = "gpt-5.1",
    requirements: Optional[List[Dict[str, Any]]] = None,
    skip_eval: bool = False,
) -> Tuple[Optional[Dict[str, Any]], bool]:
    """
    Save all pipeline outputs and optionally run evaluation.
    
    Returns:
        Tuple of (diagnostic_metrics, eval_success)
    """
    requirements = requirements or []

    # Save summary
    (out_dir / "summary.md").write_text(pipeline_result["summary_text"], encoding="utf-8")

    # Generate and save diagnostics
    diagnostic_metrics = _save_diagnostics(out_dir, pipeline_result, cfg)

    # Save context files for each method
    _save_context_files(out_dir, run_id, query, pipeline_result, cfg, sample)

    # Save requirements
    requirements_path = out_dir / "requirements.json"
    _save_requirements(requirements_path, sample, query, requirements)

    if skip_eval:
        log.info("  Evaluation skipped (--no_eval flag)")
        return diagnostic_metrics, False

    # Run evaluation
    eval_success = _run_evaluations(out_dir, run_id, requirements_path, eval_mode, eval_model)
    return diagnostic_metrics, eval_success


def _save_diagnostics(
    out_dir: Path, 
    pipeline_result: Dict[str, Any], 
    cfg: Any
) -> Optional[Dict[str, Any]]:
    """Generate and save diagnostic metrics and charts."""
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

        if diagnostic_metrics:
            write_json(out_dir / "diagnostic_metrics.json", diagnostic_metrics)

            # Log metrics summary
            for method, m in diagnostic_metrics.items():
                log.info(
                    "    %s: n=%d, rel=%.3f, rel/tok=%.5f, budget=%.1f%%, sim=%.3f, nov=%.3f",
                    method.upper(), m["n_selected"], m["mean_relevance"],
                    m["mean_rel_per_token"], m["budget_pct"],
                    m["mean_offdiag_sim"], m["mean_novelty"],
                )

            # Generate spider chart
            _save_spider_chart(out_dir, diagnostic_metrics)

        return diagnostic_metrics
    except Exception as e:
        log.warning("Failed to generate diagnostics: %s", e)
        return None


def _save_spider_chart(out_dir: Path, diagnostic_metrics: Dict[str, Any]) -> None:
    """Generate and save spider chart."""
    try:
        from src.diagnostics import generate_spider_chart
        import matplotlib.pyplot as plt
        spider_fig = generate_spider_chart(diagnostic_metrics, out_dir / "diagnostics_spider.png")
        if spider_fig:
            plt.close(spider_fig)
    except Exception as e:
        log.warning("Failed to generate spider chart: %s", e)


def _save_context_files(
    out_dir: Path,
    run_id: str,
    query: str,
    pipeline_result: Dict[str, Any],
    cfg: Any,
    sample: Dict[str, Any],
) -> None:
    """Save context.json files for each selection method."""
    from . import METHOD_FILES

    selections = pipeline_result["selections"]
    budgets_used = pipeline_result["budgets_used"]
    rel_scores = pipeline_result["rel_scores"]

    for method, sel in selections.items():
        chunks = _build_chunks_list(sel, rel_scores)
        context_data = _build_context_data(
            run_id, query, method, sel, chunks, budgets_used[method],
            cfg, sample, rel_scores, pipeline_result["summary_text"]
        )
        ctx_filename = METHOD_FILES[method][0]
        write_json(out_dir / ctx_filename, context_data)


def _build_chunks_list(selected: List, rel_scores: Dict[str, float]) -> List[Dict]:
    """Build chunks list from selected units."""
    return [
        {
            "chunk_id": s.unit.id,
            "content": {"text": s.unit.context_text, "image_paths": []},
            "metadata": {
                "unit_type": s.unit.type,
                "cognitive_cost": int(s.cost),
                "importance_score": rel_scores.get(s.unit.id),
                "page": s.unit.page,
            },
        }
        for s in selected
    ]


def _build_context_data(
    run_id: str,
    query: str,
    method: str,
    selected: List,
    chunks: List[Dict],
    budget_used: int,
    cfg: Any,
    sample: Dict[str, Any],
    rel_scores: Dict[str, float],
    summary_text: str,
) -> Dict[str, Any]:
    """Build context data dictionary for saving."""
    budgets = {"total": cfg.selection.budget_tokens, "used": budget_used}
    
    return {
        "run_id": run_id,
        "source": "sciduet",
        "paper_id": sample.get("paper_id"),
        "slide_id": sample.get("slide_id"),
        "gem_id": sample.get("gem_id"),
        "query": query,
        "method": method,
        "selected": [
            {
                "chunk_id": s.unit.id,
                "page": s.unit.page,
                "type": s.unit.type,
                "rel": float(rel_scores.get(s.unit.id, 0.0)),
                "cost": int(s.cost),
                "text": (s.unit.context_text or "")[:200],
            }
            for s in selected
        ],
        "budgets": budgets,
        "chunks": chunks,
        "selected_chunks": chunks,
        "retrieved_chunks": chunks,
        "evidence": chunks,
        "pipeline_output": {
            "run_id": run_id,
            "query": query,
            "method": method,
            "budgets": budgets,
            "chunks": chunks,
            "selected_chunks": chunks,
        },
        "summarization": {"summary_text": summary_text},
    }


def _save_requirements(
    path: Path, 
    sample: Dict[str, Any], 
    query: str, 
    requirements: List[Dict[str, Any]]
) -> None:
    """Save requirements.json file."""
    requirements_data = {
        "id": f"{sample.get('paper_id', '')}_{sample.get('gem_id', '')}",
        "parent": {"path": f"{sample.get('paper_id', 'unknown')}.pdf"},
        "task": {"decisions": [query]},
        "requirements": requirements,
        "notes": {"source": "sciduet", "n_requirements": len(requirements)},
    }
    write_json(path, requirements_data)


# ============================================================================
# Evaluation Execution
# ============================================================================

def _run_evaluations(
    out_dir: Path,
    run_id: str,
    requirements_path: Path,
    eval_mode: str,
    eval_model: str
) -> bool:
    """Run evaluation for all methods in parallel."""
    from . import METHODS, METHOD_FILES

    def run_eval(method: str, ctx_name: str, eval_name: str) -> Tuple[str, bool]:
        ctx_path = out_dir / ctx_name
        if not ctx_path.exists():
            return method, False

        eval_cmd = [
            sys.executable, "-m", "eval_engine.main",
            "--input", str(ctx_path),
            "--requirements", str(requirements_path),
            "--output", str(out_dir / eval_name),
            "--mode", eval_mode,
            "--model", eval_model,
        ]

        try:
            (out_dir / f"eval_{method}_cmd.txt").write_text(" ".join(eval_cmd), encoding="utf-8")
            result = subprocess.run(eval_cmd)
            return method, result.returncode == 0
        except Exception:
            log.exception("Eval subprocess crashed (method=%s)", method)
            return method, False

    tasks = [(m, METHOD_FILES[m][0], METHOD_FILES[m][1]) for m in METHODS]
    eval_success = False

    with ThreadPoolExecutor(max_workers=min(5, len(tasks))) as executor:
        futures = [executor.submit(run_eval, *t) for t in tasks]
        for future in as_completed(futures):
            method, ok = future.result()
            eval_success = eval_success or ok
            if not ok:
                log.warning("Evaluation failed for %s (method=%s)", run_id, method)

    return eval_success


# ============================================================================
# Score Collection
# ============================================================================

def collect_scores(
    out_dir: Path,
    paper_id: str,
    gem_id: str,
    slide_id: str,
    run_id: str,
    entry_id: str,
) -> List[Dict[str, Any]]:
    """
    Collect decision scores from eval bundles.

    Returns:
        List of score info dictionaries
    """
    from . import METHOD_FILES

    score_data = []
    requirements_path = out_dir / "requirements.json"
    requirements_data = load_json(requirements_path) if requirements_path.exists() else None

    for method, (_, eval_name) in METHOD_FILES.items():
        eval_path = out_dir / eval_name
        if not eval_path.exists():
            continue
        try:
            eval_bundle = load_json(eval_path)
            score_info = calculate_decision_score(eval_bundle, requirements_data)
            if score_info:
                score_info.update({
                    "pdf_name": f"{paper_id}.pdf",
                    "paper_id": paper_id,
                    "gem_id": gem_id,
                    "slide_id": slide_id,
                    "run_id": run_id,
                    "method": method,
                    "entry_id": entry_id,
                })
                score_data.append(score_info)
        except Exception as e:
            log.warning("Failed to calculate score for %s (%s): %s", run_id, method, e)

    return score_data


# ============================================================================
# Batch Processing
# ============================================================================

def process_sciduet_batch(
    dataset_split: str,
    cfg: Any,
    output_base: Path,
    limit: Optional[int] = None,
    offset: int = 0,
    eval_mode: str = "all",
    eval_model: str = "gpt-5.1",
    metadata_path: Optional[Path] = None,
    skip_eval: bool = False,
) -> Dict[str, Any]:
    """
    Process SciDuet dataset in batch mode.
    
    Args:
        dataset_split: Dataset split ("train", "validation", "test")
        cfg: Configuration object
        output_base: Base output directory
        limit: Optional limit on number of samples
        offset: Number of samples to skip
        eval_mode: Evaluation mode
        eval_model: Model for evaluation
        metadata_path: Path to requirements metadata
        skip_eval: Whether to skip evaluation
        
    Returns:
        Results dictionary with processing stats and scores
    """
    # Load dataset
    log.info("Loading SciDuet dataset (split=%s)...", dataset_split)
    try:
        ds = load_dataset("GEM/SciDuet", split=dataset_split)
    except Exception as e:
        log.error("Failed to load SciDuet dataset: %s", e)
        return {"success": False, "error": str(e)}

    log.info("Loaded %d samples", len(ds))

    # Load requirements metadata
    requirements_map = load_requirements_metadata(metadata_path)

    # Apply limit/offset
    ds = _apply_dataset_slice(ds, offset, limit)

    # Initialize
    converter = SciDuetConverter(
        max_section_length=cfg.extract.max_text_chunk_tokens,
        target_chunk_size=getattr(cfg.extract, "target_chunk_size", 1200),
    )
    batch_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    results = _init_results(batch_timestamp, dataset_split, len(ds))
    score_data: List[Dict[str, Any]] = []

    # Process each sample
    for idx, sample in enumerate(tqdm(ds, desc="Processing samples", unit="sample")):
        paper_id, gem_id, slide_id, slide_num = extract_sample_ids(sample)
        run_id = f"{batch_timestamp}_{paper_id}_slide{slide_num}_d0"
        entry_id, requirements = lookup_requirements(requirements_map, paper_id, gem_id, slide_id)

        log.info("\n%s\nProcessing %d/%d: %s", "=" * 80, idx + 1, len(ds), run_id)
        log.info("Entry ID: %s, Requirements: %d", entry_id, len(requirements))

        try:
            # Convert and run pipeline
            doc = converter.convert(sample)
            query = str(sample.get("slide_title") or "Summarize this paper")
            out_dir = ensure_dir(output_base / run_id)

            pipeline_result = run_pipeline_on_document(doc, query, cfg, out_dir, query)

            diagnostic_metrics, eval_success = save_outputs(
                out_dir, run_id, doc, query, pipeline_result, cfg, sample,
                eval_mode, eval_model, requirements, skip_eval,
            )

            # Aggregate metrics
            if diagnostic_metrics:
                for method, metrics in diagnostic_metrics.items():
                    results["method_metrics"][method].append(metrics)

            results["successful"] += 1
            results["pipeline_success"] += 1
            if eval_success:
                results["eval_success"] += 1

            # Collect scores
            sample_scores = collect_scores(out_dir, paper_id, gem_id, slide_id, run_id, entry_id)
            score_data.extend(sample_scores)

            log.info("  ✓ SUCCESS: %s", run_id)

        except Exception as e:
            log.error("  ✗ FAILED: %s - %s", run_id, e, exc_info=True)
            results["failed"] += 1
            results["failed_ids"].append(run_id)

        results["processed"] += 1

    results["score_data"] = score_data
    return results


def _init_results(batch_timestamp: str, dataset_split: str, total_samples: int) -> Dict[str, Any]:
    """Initialize results dictionary."""
    from . import METHODS

    return {
        "batch_timestamp": batch_timestamp,
        "dataset_split": dataset_split,
        "total_samples": total_samples,
        "processed": 0,
        "successful": 0,
        "failed": 0,
        "failed_ids": [],
        "method_metrics": {m: [] for m in METHODS},
        "pipeline_success": 0,
        "eval_success": 0,
    }


def _apply_dataset_slice(ds, offset: int, limit: Optional[int]):
    """Apply offset and limit to dataset."""
    total = len(ds)
    if offset > 0:
        end = min(offset + (limit or total), total)
        ds = ds.select(range(offset, end))
        log.info("Applied offset=%d, processing %d samples", offset, len(ds))
    elif limit:
        ds = ds.select(range(min(limit, total)))
        log.info("Limited to %d samples", len(ds))
    return ds


# ============================================================================
# Metrics Aggregation
# ============================================================================

def aggregate_metrics(method_metrics: Dict[str, List[Dict]]) -> Dict[str, float]:
    """
    Aggregate diagnostic metrics across samples.
    
    Returns:
        Dict with mean and std for each metric per method
    """
    aggregated = {}
    for method, metrics_list in method_metrics.items():
        if not metrics_list:
            continue
        for key in metrics_list[0].keys():
            values = [m[key] for m in metrics_list if not np.isnan(m[key])]
            if values:
                aggregated[f"{method}_{key}"] = float(np.mean(values))
                aggregated[f"{method}_{key}_std"] = float(np.std(values))
    return aggregated