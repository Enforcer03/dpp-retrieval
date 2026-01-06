#!/usr/bin/env python3
"""
Batch processing script for running pipeline and evaluation on multiple PDFs.
Reads metadata.json and processes each entry (PDF + decisions + requirements).
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import logging
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from src.config import load_config
from src.wandb_logger import wandb_finish, wandb_init, wandb_log, wandb_log_artifact

log = logging.getLogger(__name__)

from src.utils import load_json, save_json, display_score_summary, calculate_decision_score, process_entry


def main() -> None:
    parser = argparse.ArgumentParser(description="Batch process multiple PDFs with metadata")
    parser.add_argument("--metadata", default="data/metadata.json", help="Path to metadata JSON")
    parser.add_argument("--config", default="config/default_config.yaml", help="Path to config file")
    parser.add_argument("--data_dir", default="data", help="Directory containing PDFs")
    parser.add_argument("--output", default="output", help="Base output directory")
    parser.add_argument("--eval_mode", default="all", choices=["requirements", "retrieval", "all"])
    parser.add_argument("--eval_model", default="gpt-5.1", help="Model for evaluation")
    parser.add_argument("--log_level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    parser.add_argument("--limit", type=int, help="Limit to first N entries (for testing)")
    parser.add_argument("--recompute", action="store_true", help="Force recompute PDF extraction")

    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(Path(args.output) / "batch_run.log", mode="w"),
        ],
    )

    if args.recompute:
        log.info("Recompute enabled: bypassing PDF extraction cache")

    # Load metadata
    metadata_path = Path(args.metadata)
    if not metadata_path.exists():
        log.error("Metadata file not found: %s", metadata_path)
        sys.exit(1)

    log.info("Loading metadata from: %s", metadata_path)
    metadata = load_json(metadata_path)

    if not isinstance(metadata, list):
        log.error("Metadata must be a JSON array")
        sys.exit(1)

    total_entries = len(metadata)
    if args.limit:
        metadata = metadata[:args.limit]
        log.info("Limited to first %d entries (out of %d total)", args.limit, total_entries)

    log.info("Processing %d entries", len(metadata))

    # Generate batch timestamp
    batch_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log.info("Batch timestamp: %s", batch_timestamp)

    # Setup paths
    data_dir = Path(args.data_dir)
    output_base = Path(args.output)
    output_base.mkdir(parents=True, exist_ok=True)

    # Initialize wandb for batch tracking
    cfg = load_config(args.config)

    # Convert config to dict, handling Path objects
    def _path_to_str(obj):
        """Recursively convert Path objects to strings for wandb compatibility."""
        if isinstance(obj, Path):
            return str(obj)
        elif isinstance(obj, dict):
            return {k: _path_to_str(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [_path_to_str(item) for item in obj]
        return obj

    wandb_config = _path_to_str(asdict(cfg))
    wandb_config.update({
        "total_entries": len(metadata),
        "eval_mode": args.eval_mode,
        "eval_model": args.eval_model,
        "config_path": args.config,
    })

    batch_run = wandb_init(
        enabled=cfg.wandb.enabled,
        project=cfg.wandb.project,
        entity=cfg.wandb.entity,
        tags=["batch", f"batch_{batch_timestamp}"] + cfg.wandb.tags,
        name=f"batch_{batch_timestamp}",
        config=wandb_config
    )

    # Process entries
    results = {
        "batch_timestamp": batch_timestamp,
        "total_entries": len(metadata),
        "total_runs": 0,
        "pipeline_success": 0,
        "eval_success": 0,
        "failed_runs": [],
    }

    score_data = []

    for idx, entry in enumerate(metadata):
        log.info("")
        log.info("=" * 80)
        log.info("Entry %d/%d", idx + 1, len(metadata))

        entry_results = process_entry(
            entry=entry,
            entry_index=idx,
            batch_timestamp=batch_timestamp,
            config_path=args.config,
            data_dir=data_dir,
            output_base=output_base,
            eval_mode=args.eval_mode,
            eval_model=args.eval_model,
            recompute=args.recompute,
        )

        # Collect results
        for pipeline_ok, eval_ok, run_id in entry_results:
            results["total_runs"] += 1

            if pipeline_ok:
                results["pipeline_success"] += 1
            if eval_ok:
                results["eval_success"] += 1
            if not (pipeline_ok and eval_ok):
                results["failed_runs"].append(run_id)

            # Calculate scores for successful evals
            if eval_ok:
                eval_bundle_path = output_base / run_id / "eval_bundle.json"
                requirements_path = output_base / run_id / "requirements.json"
                if eval_bundle_path.exists():
                    try:
                        eval_bundle = load_json(eval_bundle_path)
                        requirements_data = load_json(requirements_path) if requirements_path.exists() else None
                        score_info = calculate_decision_score(eval_bundle, requirements_data)
                        if score_info:
                            parent_path = entry.get("parent", {}).get("path", "unknown.pdf")
                            score_info["pdf_name"] = parent_path
                            score_info["run_id"] = run_id
                            score_info["method"] = "greedy"
                            score_data.append(score_info)

                        # Load top-k eval bundle if it exists
                        eval_topk_path = output_base / run_id / "eval_bundle_topk.json"
                        if eval_topk_path.exists():
                            eval_topk = load_json(eval_topk_path)
                            score_topk = calculate_decision_score(eval_topk, requirements_data)
                            if score_topk:
                                score_topk["pdf_name"] = parent_path
                                score_topk["run_id"] = run_id
                                score_topk["method"] = "topk"
                                score_data.append(score_topk)

                        # Load greedy+coverage eval bundle if it exists
                        eval_greedy_cov_path = output_base / run_id / "eval_bundle_greedy_cov.json"
                        if eval_greedy_cov_path.exists():
                            eval_greedy_cov = load_json(eval_greedy_cov_path)
                            score_greedy_cov = calculate_decision_score(eval_greedy_cov, requirements_data)
                            if score_greedy_cov:
                                score_greedy_cov["pdf_name"] = parent_path
                                score_greedy_cov["run_id"] = run_id
                                score_greedy_cov["method"] = "greedy_cov"
                                score_data.append(score_greedy_cov)
                    except Exception as e:
                        log.warning("Failed to process scores for %s: %s", run_id, e)

                # Log artifacts to wandb
                run_dir = output_base / run_id
                if (run_dir / "diagnostics.png").exists():
                    wandb_log_artifact(batch_run, str(run_dir / "diagnostics.png"), f"{run_id}_diagnostics", "plot")
                if (run_dir / "context.json").exists():
                    wandb_log_artifact(batch_run, str(run_dir / "context.json"), f"{run_id}_context", "config")
                if (run_dir / "eval_bundle.json").exists():
                    wandb_log_artifact(batch_run, str(run_dir / "eval_bundle.json"), f"{run_id}_eval", "evaluation")

                # Log per-run metrics
                if score_info:
                    wandb_log(batch_run, {
                        f"{run_id}/avg_score": score_info["avg_score"],
                        f"{run_id}/num_requirements": score_info["num_requirements"],
                        f"{run_id}/covered": score_info["status_breakdown"]["covered"],
                        f"{run_id}/partial": score_info["status_breakdown"]["partial"],
                        f"{run_id}/missing": score_info["status_breakdown"]["missing"],
                    })

    # Summary
    log.info("")
    log.info("=" * 80)
    log.info("BATCH PROCESSING COMPLETE")
    log.info("=" * 80)
    log.info("Total entries: %d", results["total_entries"])
    log.info("Total decision runs: %d", results["total_runs"])
    log.info("Pipeline successes: %d", results["pipeline_success"])
    log.info("Eval successes: %d", results["eval_success"])
    log.info("Failed runs: %d", len(results["failed_runs"]))

    if results["failed_runs"]:
        log.warning("Failed run IDs: %s", ", ".join(results["failed_runs"]))

    # Display and save scores
    if score_data:
        display_score_summary(score_data)
        scores_path = output_base / "decision_scores.json"
        save_json(scores_path, score_data)
        log.info("Decision scores saved to: %s", scores_path)

    # Save summary
    summary_path = output_base / "batch_summary.json"
    save_json(summary_path, results)
    log.info("Batch summary saved to: %s", summary_path)

    # Log batch-level summary metrics to wandb
    if score_data:
        all_scores = [s["avg_score"] for s in score_data]
        wandb_log(batch_run, {
            "batch/total_entries": results["total_entries"],
            "batch/total_runs": results["total_runs"],
            "batch/pipeline_success_rate": results["pipeline_success"] / max(results["total_runs"], 1),
            "batch/eval_success_rate": results["eval_success"] / max(results["total_runs"], 1),
            "batch/failed_runs_count": len(results["failed_runs"]),
            "batch/avg_score_mean": float(np.mean(all_scores)),
            "batch/avg_score_std": float(np.std(all_scores)),
            "batch/avg_score_min": float(np.min(all_scores)),
            "batch/avg_score_max": float(np.max(all_scores)),
        })

    # Finalize wandb run
    wandb_finish(batch_run)

    # Exit with error if failures
    if results["failed_runs"]:
        sys.exit(1)


if __name__ == "__main__":
    main()