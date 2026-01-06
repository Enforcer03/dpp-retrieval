#!/usr/bin/env python3
"""
Batch processing script for running pipeline and evaluation on multiple PDFs.
Reads metadata.json and processes each entry (PDF + decisions + requirements).
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import logging
import sys
from datetime import datetime
from pathlib import Path

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
    parser.add_argument("--no_wandb", action="store_true", help="Disable Weights & Biases logging")

    args = parser.parse_args()

    # Setup logging
    output_base = Path(args.output)
    output_base.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(output_base / "batch_run.log", mode="w"),
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
        metadata = metadata[: args.limit]
        log.info("Limited to first %d entries (out of %d total)", args.limit, total_entries)

    log.info("Processing %d entries", len(metadata))

    # Generate batch timestamp
    batch_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log.info("Batch timestamp: %s", batch_timestamp)

    # Setup paths
    data_dir = Path(args.data_dir)

    # Initialize wandb for batch tracking
    cfg = load_config(args.config)

    # Override wandb enabled via CLI
    if args.no_wandb:
        try:
            cfg.wandb.enabled = False
        except Exception:
            # In case cfg.wandb is not a simple object; fail-safe
            pass
        log.info("W&B disabled via --no_wandb")

    def _path_to_str(obj):
        """Recursively convert Path objects to strings for wandb compatibility."""
        if isinstance(obj, Path):
            return str(obj)
        if isinstance(obj, dict):
            return {k: _path_to_str(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_path_to_str(item) for item in obj]
        return obj

    wandb_config = _path_to_str(asdict(cfg))
    wandb_config.update(
        {
            "total_entries": len(metadata),
            "eval_mode": args.eval_mode,
            "eval_model": args.eval_model,
            "config_path": args.config,
            "no_wandb": bool(args.no_wandb),
        }
    )

    batch_run = wandb_init(
        enabled=cfg.wandb.enabled,
        project=cfg.wandb.project,
        entity=cfg.wandb.entity,
        tags=["batch", f"batch_{batch_timestamp}"] + cfg.wandb.tags,
        name=f"batch_{batch_timestamp}",
        config=wandb_config,
    )

    results = {
        "batch_timestamp": batch_timestamp,
        "total_entries": len(metadata),
        "total_runs": 0,
        "pipeline_success": 0,
        "eval_success": 0,
        "failed_runs": [],
    }

    score_data = []

    # Backward-compatible naming + new baselines
    METHOD_EVAL_BUNDLES = {
        "greedy": "eval_bundle.json",
        "topk": "eval_bundle_topk.json",
        "greedy_cov": "eval_bundle_greedy_cov.json",
        "cost_norm": "eval_bundle_cost_norm.json",
        "dpp": "eval_bundle_dpp.json",
    }

    METHOD_CONTEXT_FILES = {
        "greedy": "context.json",
        "topk": "context_topk.json",
        "greedy_cov": "context_greedy_cov.json",
        "cost_norm": "context_cost_norm.json",
        "dpp": "context_dpp.json",
    }

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

        for pipeline_ok, eval_ok, run_id in entry_results:
            results["total_runs"] += 1

            if pipeline_ok:
                results["pipeline_success"] += 1
            if eval_ok:
                results["eval_success"] += 1
            if not (pipeline_ok and eval_ok):
                results["failed_runs"].append(run_id)

            run_dir = output_base / run_id

            # Always log diagnostics prominently
            diag_path = run_dir / "diagnostics.png"
            if diag_path.exists():
                wandb_log_artifact(batch_run, str(diag_path), f"{run_id}_diagnostics", "plot")

            # Log all available contexts (one per method)
            for m, ctx_name in METHOD_CONTEXT_FILES.items():
                p = run_dir / ctx_name
                if p.exists():
                    wandb_log_artifact(batch_run, str(p), f"{run_id}_context_{m}", "config")

            # Scores + eval artifacts (loop methods)
            if eval_ok:
                requirements_path = run_dir / "requirements.json"
                requirements_data = load_json(requirements_path) if requirements_path.exists() else None
                parent_path = entry.get("parent", {}).get("path", "unknown.pdf")

                for method, bundle_name in METHOD_EVAL_BUNDLES.items():
                    bpath = run_dir / bundle_name
                    if not bpath.exists():
                        continue

                    try:
                        eval_bundle = load_json(bpath)
                        score_info = calculate_decision_score(eval_bundle, requirements_data)
                        if not score_info:
                            continue

                        score_info["pdf_name"] = parent_path
                        score_info["run_id"] = run_id
                        score_info["method"] = method
                        score_data.append(score_info)

                        wandb_log_artifact(batch_run, str(bpath), f"{run_id}_eval_{method}", "evaluation")

                        wandb_log(
                            batch_run,
                            {
                                f"{run_id}/{method}/avg_score": score_info["avg_score"],
                                f"{run_id}/{method}/num_requirements": score_info["num_requirements"],
                                f"{run_id}/{method}/covered": score_info["status_breakdown"]["covered"],
                                f"{run_id}/{method}/partial": score_info["status_breakdown"]["partial"],
                                f"{run_id}/{method}/missing": score_info["status_breakdown"]["missing"],
                            },
                        )
                    except Exception as e:
                        log.warning("Failed to process score for %s (%s): %s", run_id, method, e)

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

    if score_data:
        display_score_summary(score_data)
        scores_path = output_base / "decision_scores.json"
        save_json(scores_path, score_data)
        log.info("Decision scores saved to: %s", scores_path)

    summary_path = output_base / "batch_summary.json"
    save_json(summary_path, results)
    log.info("Batch summary saved to: %s", summary_path)

    # Batch-level aggregate metrics
    if score_data:
        all_scores = [s["avg_score"] for s in score_data]
        wandb_log(
            batch_run,
            {
                "batch/total_entries": results["total_entries"],
                "batch/total_runs": results["total_runs"],
                "batch/pipeline_success_rate": results["pipeline_success"] / max(results["total_runs"], 1),
                "batch/eval_success_rate": results["eval_success"] / max(results["total_runs"], 1),
                "batch/failed_runs_count": len(results["failed_runs"]),
                "batch/avg_score_mean": float(np.mean(all_scores)),
                "batch/avg_score_std": float(np.std(all_scores)),
                "batch/avg_score_min": float(np.min(all_scores)),
                "batch/avg_score_max": float(np.max(all_scores)),
            },
        )

    wandb_finish(batch_run)

    if results["failed_runs"]:
        sys.exit(1)


if __name__ == "__main__":
    main()
