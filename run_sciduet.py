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
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import numpy as np

from src.config import load_config
from src.diagnostics import generate_aggregated_visualizations
from src.utils import ensure_dir, setup_logging, write_json, save_json, display_score_summary
from src.wandb_logger import wandb_finish, wandb_init, wandb_log

from sciduet import METHODS, process_sciduet_batch
from sciduet.io_utils import aggregate_metrics

log = logging.getLogger(__name__)


def main():
    args = parse_args()
    
    # Setup
    output_base = ensure_dir(Path(args.output))
    cfg = load_config(args.config)
    setup_logging(args.log_level, output_base / "sciduet_run.log")
    
    log_run_info(args)
    
    # Initialize wandb
    wandb_run = init_wandb(args, cfg) if not args.no_wandb else None
    
    # Resolve metadata path
    metadata_path = resolve_metadata_path(args.metadata, args.split)
    
    # Process batch
    results = process_sciduet_batch(
        dataset_split=args.split,
        cfg=cfg,
        output_base=output_base,
        limit=args.limit,
        offset=args.offset,
        eval_mode=args.eval_mode,
        eval_model=args.eval_model,
        metadata_path=metadata_path,
        skip_eval=args.no_eval,
    )
    
    # Aggregate and report
    results["aggregated_metrics"] = aggregate_metrics(results.get("method_metrics", {}))
    report_results(results, output_base)

    # Generate aggregated visualizations
    generate_aggregated_plots(results, output_base)
    
    # Wandb finalization
    if wandb_run:
        finalize_wandb(wandb_run, results, output_base)
    
    sys.exit(0 if results.get("failed", 0) == 0 else 1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run DPP-retrieval pipeline on SciDuet")
    parser.add_argument("--split", default="train", choices=["train", "validation", "test"])
    parser.add_argument("--config", default="config/sciduet_config.yaml")
    parser.add_argument("--output", default="output/sciduet")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--log_level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    parser.add_argument("--no_wandb", action="store_true")
    parser.add_argument("--no_eval", action="store_true")
    parser.add_argument("--eval_mode", default="all", choices=["requirements", "retrieval", "all"])
    parser.add_argument("--eval_model", default="gpt-5.1")
    parser.add_argument("--metadata", help="Path to metadata JSON with requirements")
    return parser.parse_args()


def log_run_info(args: argparse.Namespace) -> None:
    log.info("=" * 80)
    log.info("SciDuet Pipeline Runner")
    log.info("=" * 80)
    log.info("Split: %s | Config: %s | Output: %s", args.split, args.config, args.output)
    if args.limit:
        log.info("Limit: %d samples", args.limit)
    if args.offset:
        log.info("Offset: %d samples", args.offset)
    if args.metadata:
        log.info("Metadata: %s", args.metadata)
    log.info("Evaluation: %s", "DISABLED" if args.no_eval else f"{args.eval_mode} ({args.eval_model})")


def init_wandb(args: argparse.Namespace, cfg: Any) -> Optional[Any]:
    try:
        return wandb_init(
            enabled=cfg.wandb.enabled,
            project=cfg.wandb.project,
            entity=cfg.wandb.entity,
            tags=["sciduet", f"split_{args.split}"] + cfg.wandb.tags,
            name=f"sciduet_{args.split}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            config={"split": args.split, "limit": args.limit, "offset": args.offset},
        )
    except Exception as e:
        log.warning("Failed to initialize wandb: %s", e)
        return None


def resolve_metadata_path(metadata_arg: Optional[str], split: str) -> Optional[Path]:
    if metadata_arg:
        return Path(metadata_arg)
    
    default = Path(f"data/sciduet/sciduet_metadata_output/{split}_metadata.json")
    if default.exists():
        log.info("Auto-detected metadata: %s", default)
        return default
    
    log.warning("No metadata found at %s", default)
    return None


def report_results(results: dict, output_base: Path) -> None:
    log.info("\n%s\nBATCH PROCESSING COMPLETE\n%s", "=" * 80, "=" * 80)
    log.info("Processed: %d | Successful: %d | Failed: %d",
             results.get("processed", 0), results.get("successful", 0), results.get("failed", 0))
    log.info("Pipeline successes: %d | Eval successes: %d",
             results.get("pipeline_success", 0), results.get("eval_success", 0))
    
    if results.get("failed_ids"):
        log.warning("Failed IDs: %s", ", ".join(results["failed_ids"][:10]))
    
    # Display and save scores
    if results.get("score_data"):
        display_score_summary(results["score_data"])
        save_json(output_base / "decision_scores.json", results["score_data"])
    
    # Log aggregated metrics
    if results.get("aggregated_metrics"):
        log_aggregated_metrics(results["aggregated_metrics"], results.get("successful", 0))
    
    # Save summary
    write_json(output_base / "batch_summary.json", results)


def log_aggregated_metrics(aggregated: dict, n_samples: int) -> None:
    log.info("\n%s\nAGGREGATED METRICS (mean ± std across %d samples)\n%s", "=" * 80, n_samples, "=" * 80)
    metric_keys = ["n_selected", "mean_relevance", "mean_rel_per_token", "budget_pct", "mean_offdiag_sim", "mean_novelty"]

    for method in METHODS:
        log.info("  %s:", method.upper())
        for key in metric_keys:
            mean_key = f"{method}_{key}"
            if mean_key in aggregated:
                log.info("    %s: %.4f ± %.4f", key, aggregated[mean_key], aggregated.get(f"{mean_key}_std", 0.0))


def generate_aggregated_plots(results: dict, output_base: Path) -> None:
    """Generate aggregated visualizations from batch results."""
    if not results.get("aggregated_metrics") or not results.get("successful"):
        log.warning("Insufficient data for aggregated visualizations")
        return

    try:
        viz_dir = output_base / "aggregated_visualizations"
        generate_aggregated_visualizations(
            aggregated_metrics=results["aggregated_metrics"],
            output_dir=viz_dir,
            n_samples=results["successful"],
        )
        log.info("Aggregated visualizations saved to %s", viz_dir)
    except Exception as e:
        log.warning("Failed to generate aggregated visualizations: %s", e)


def finalize_wandb(wandb_run: Any, results: dict, output_base: Path) -> None:
    try:
        wandb_data = {
            "processed": results.get("processed", 0),
            "successful": results.get("successful", 0),
            "failed": results.get("failed", 0),
            "success_rate": results.get("successful", 0) / max(results.get("processed", 1), 1),
        }
        wandb_data.update(results.get("aggregated_metrics", {}))
        
        if results.get("score_data"):
            scores = [s["avg_score"] for s in results["score_data"] if "avg_score" in s]
            if scores:
                wandb_data.update({
                    "batch/avg_score_mean": float(np.mean(scores)),
                    "batch/avg_score_std": float(np.std(scores)),
                })
        
        wandb_log(wandb_run, wandb_data)

        # Log aggregated visualizations if available
        viz_dir = output_base / "aggregated_visualizations"
        if viz_dir.exists():
            from src.wandb_logger import wandb_log_image
            viz_files = {
                "aggregated_spider.png": "Aggregated Spider Chart",
                "aggregated_metrics_bars.png": "Metrics Bar Charts",
                "aggregated_budget.png": "Budget Comparison",
                "aggregated_efficiency_relevance.png": "Efficiency vs Relevance",
                "aggregated_diversity_novelty.png": "Diversity vs Novelty",
            }
            for filename, caption in viz_files.items():
                viz_path = viz_dir / filename
                if viz_path.exists():
                    wandb_log_image(wandb_run, f"aggregated/{filename[:-4]}", str(viz_path), caption)

        wandb_finish(wandb_run)
    except Exception as e:
        log.warning("Wandb finalization failed: %s", e)


if __name__ == "__main__":
    main()