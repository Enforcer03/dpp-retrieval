#!/usr/bin/env python3
"""
Batch processing script for running pipeline and evaluation on multiple PDFs.
Reads metadata.json and processes each entry (PDF + decisions + requirements).
"""
from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)


def load_json(path: Path) -> Any:
    """Load JSON file."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: Path, data: Any) -> None:
    """Save data as JSON file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def score_requirement_status(status: str) -> int:
    """
    Map requirement status to numeric score.

    Returns:
        2 = covered/satisfied
        1 = partial
        0 = not satisfied/missing/unknown
    """
    status_lower = str(status).lower().strip()
    if status_lower in ("covered", "satisfied", "full", "complete"):
        return 2
    elif status_lower in ("partial", "weak"):
        return 1
    else:
        return 0


def calculate_decision_score(eval_bundle: dict, requirements_data: dict | None = None) -> dict | None:
    """
    Calculate average requirement score for a decision.

    Args:
        eval_bundle: The evaluation bundle JSON
        requirements_data: Optional requirements.json data to get requirement descriptions

    Returns:
        {
            "decision": str,
            "avg_score": float,
            "num_requirements": int,
            "status_breakdown": {"covered": int, "partial": int, "missing": int},
            "requirement_details": [{"id": str, "description": str, "status": str, "score": int}]
        }
    """
    try:
        decisions = (
            eval_bundle.get("evaluation", {})
            .get("retrieval_eval", {})
            .get("requirements_eval", {})
            .get("decisions", [])
        )

        if not decisions:
            return None

        # Take first decision (each run has one decision)
        decision_data = decisions[0]
        decision_text = decision_data.get("decision", "Unknown")
        requirements = decision_data.get("requirements", [])

        # Build requirement description map from requirements_data
        req_desc_map = {}
        if requirements_data:
            for req in requirements_data.get("requirements", []):
                if isinstance(req, dict):
                    req_id = req.get("id")
                    req_desc = req.get("description")
                    if req_id and req_desc:
                        req_desc_map[req_id] = req_desc

        if not requirements:
            return {
                "decision": decision_text,
                "avg_score": 0.0,
                "num_requirements": 0,
                "status_breakdown": {"covered": 0, "partial": 0, "missing": 0},
                "requirement_details": [],
            }

        # Score each requirement
        scores = []
        breakdown = {"covered": 0, "partial": 0, "missing": 0}
        requirement_details = []

        for req in requirements:
            req_id = req.get("id", "")
            status = req.get("status", "")
            score = score_requirement_status(status)
            scores.append(score)

            if score == 2:
                breakdown["covered"] += 1
            elif score == 1:
                breakdown["partial"] += 1
            else:
                breakdown["missing"] += 1

            # Add detailed info
            requirement_details.append({
                "id": req_id,
                "description": req_desc_map.get(req_id, "Description not found"),
                "status": status,
                "score": score,
            })

        avg_score = sum(scores) / len(scores) if scores else 0.0

        return {
            "decision": decision_text,
            "avg_score": avg_score,
            "num_requirements": len(requirements),
            "status_breakdown": breakdown,
            "requirement_details": requirement_details,
        }
    except Exception as e:
        log.warning("Failed to calculate decision score: %s", e)
        return None


def run_command(cmd: list[str], description: str) -> bool:
    """Run a command and return success status."""
    log.info("Running: %s", description)
    log.debug("Command: %s", " ".join(cmd))

    try:
        # Let subprocess output stream directly to terminal instead of capturing
        subprocess.run(
            cmd,
            check=True,
            text=True,
        )
        return True
    except subprocess.CalledProcessError as e:
        log.error("Command failed: %s", description)
        log.error("Return code: %s", e.returncode)
        return False


def process_entry(
    entry: dict,
    entry_index: int,
    config_path: str,
    data_dir: Path,
    output_base: Path,
    eval_mode: str,
    eval_model: str,
    disable_wandb: bool,
    recompute: bool = False,
) -> list[tuple[bool, bool]]:
    """
    Process a single metadata entry.
    Each decision within the entry is processed separately.

    Args:
        recompute: If True, force recompute PDF extraction (bypass cache)

    Returns:
        List of (pipeline_success, eval_success) tuples, one per decision
    """
    entry_id = entry.get("id")
    if not entry_id:
        log.warning("Entry missing 'id' field, skipping: %s", entry)
        return []

    # Extract metadata
    parent_path = entry.get("parent", {}).get("path")
    if not parent_path:
        log.error("Entry %s missing parent.path, skipping", entry_id)
        return []

    decisions = entry.get("task", {}).get("decisions", [])
    if not decisions:
        log.warning("Entry %s has no decisions, using empty query", entry_id)
        decisions = [""]

    requirements = entry.get("requirements", [])

    # Construct paths
    pdf_path = data_dir / parent_path
    if not pdf_path.exists():
        log.error("PDF not found: %s (entry_id=%s)", pdf_path, entry_id)
        return []

    # Process each decision separately
    results = []

    for decision_idx, decision in enumerate(decisions):
        # Create unique output directory for this entry + decision combination
        # Format: {entry_id}_{entry_index}_d{decision_index}
        output_dir_name = f"{entry_id}_{entry_index}_d{decision_idx}"
        entry_output = output_base / output_dir_name
        entry_output.mkdir(parents=True, exist_ok=True)

        pipeline_output_path = entry_output / "pipeline_output.json"
        requirements_path = entry_output / "requirements.json"
        eval_output_path = entry_output / "eval_bundle.json"

        log.info("=" * 80)
        log.info("Processing: %s (decision %d/%d)", output_dir_name, decision_idx + 1, len(decisions))
        log.info("Entry ID: %s (index %d)", entry_id, entry_index)
        log.info("PDF: %s", pdf_path)
        log.info("Decision: %s", decision[:200] + "..." if len(decision) > 200 else decision)
        log.info("Requirements: %d", len(requirements))
        log.info("Output dir: %s", entry_output)

        # Step 1: Run main pipeline
        pipeline_cmd = [
            sys.executable,
            "main.py",
            "--config", config_path,
            "--pdf", str(pdf_path),
            "--query", decision,
            "--run_output", str(pipeline_output_path),
        ]

        if recompute:
            pipeline_cmd.append("--recompute")

        pipeline_success = run_command(pipeline_cmd, f"Pipeline for {output_dir_name}")

        if not pipeline_success:
            log.error("Pipeline failed for %s, skipping eval", output_dir_name)
            results.append((False, False))
            continue

        # Verify pipeline output was created
        if not pipeline_output_path.exists():
            log.error("Pipeline output not found: %s", pipeline_output_path)
            results.append((False, False))
            continue

        # Step 2: Create requirements file (single decision for this run)
        requirements_data = {
            "id": f"{entry_id}_d{decision_idx}",
            "parent": entry.get("parent", {}),
            "task": {"decisions": [decision]},  # Single decision
            "requirements": requirements,
            "notes": entry.get("notes", {}),
        }
        save_json(requirements_path, requirements_data)
        log.info("Saved requirements to: %s", requirements_path)

        # Step 3: Run eval engine
        eval_cmd = [
            sys.executable,
            "-m", "eval_engine.main",
            "--input", str(pipeline_output_path),
            "--requirements", str(requirements_path),
            "--output", str(eval_output_path),
            "--mode", eval_mode,
            "--model", eval_model,
        ]

        if disable_wandb:
            eval_cmd.append("--disable_wandb")

        eval_success = run_command(eval_cmd, f"Evaluation for {output_dir_name}")

        if eval_success:
            log.info("Successfully processed: %s", output_dir_name)
        else:
            log.error("Evaluation failed for: %s", output_dir_name)

        results.append((pipeline_success, eval_success))

    return results


def display_score_summary(score_data: list[dict]) -> None:
    """Display a crisp summary table of decision scores."""
    if not score_data:
        log.warning("No score data to display")
        return

    # Group scores by PDF
    pdf_groups: dict[str, list[dict]] = {}
    for item in score_data:
        pdf_name = item.get("pdf_name", "unknown")
        if pdf_name not in pdf_groups:
            pdf_groups[pdf_name] = []
        pdf_groups[pdf_name].append(item)

    # Print header
    print("\n" + "=" * 120)
    print("DECISION SCORES SUMMARY (0=missing, 1=partial, 2=covered)")
    print("=" * 120)
    print(f"{'PDF':<25} {'Decision':<60} {'Score':>8} {'Reqs':>6} {'C/P/M':>10}")
    print("-" * 120)

    # Track overall stats
    all_scores = []
    total_covered = 0
    total_partial = 0
    total_missing = 0

    # Print each PDF group
    for pdf_name in sorted(pdf_groups.keys()):
        items = pdf_groups[pdf_name]

        for idx, item in enumerate(items):
            decision = item.get("decision", "Unknown")
            avg_score = item.get("avg_score", 0.0)
            num_reqs = item.get("num_requirements", 0)
            breakdown = item.get("status_breakdown", {})

            covered = breakdown.get("covered", 0)
            partial = breakdown.get("partial", 0)
            missing = breakdown.get("missing", 0)

            all_scores.append(avg_score)
            total_covered += covered
            total_partial += partial
            total_missing += missing

            # Truncate decision for display
            decision_display = decision[:57] + "..." if len(decision) > 60 else decision

            # Display PDF name only on first row of each group
            pdf_display = pdf_name if idx == 0 else ""

            print(
                f"{pdf_display:<25} {decision_display:<60} {avg_score:>8.2f} {num_reqs:>6} "
                f"{covered:>2}/{partial:>2}/{missing:>2}"
            )

        # Separator between PDFs
        if pdf_name != sorted(pdf_groups.keys())[-1]:
            print("-" * 120)

    # Overall summary
    print("=" * 120)
    overall_avg = sum(all_scores) / len(all_scores) if all_scores else 0.0
    total_reqs = total_covered + total_partial + total_missing

    print(f"{'OVERALL AVERAGE':<25} {'':<60} {overall_avg:>8.2f} {total_reqs:>6} "
          f"{total_covered:>2}/{total_partial:>2}/{total_missing:>2}")
    print("=" * 120)

    # Score distribution
    if all_scores:
        excellent = sum(1 for s in all_scores if s >= 1.5)
        good = sum(1 for s in all_scores if 1.0 <= s < 1.5)
        weak = sum(1 for s in all_scores if s < 1.0)

        print(f"\nScore Distribution: {excellent} excellent (≥1.5), {good} good (1.0-1.5), {weak} weak (<1.0)")
        print(f"Total Decisions Analyzed: {len(all_scores)}")
    print()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Batch process multiple PDFs with metadata"
    )
    parser.add_argument(
        "--metadata",
        type=str,
        default="data/metadata.json",
        help="Path to metadata JSON file (default: data/metadata.json)",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config/default_config.yaml",
        help="Path to config file (default: config/default_config.yaml)",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="data",
        help="Directory containing PDF files (default: data)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="output",
        help="Base output directory (default: output)",
    )
    parser.add_argument(
        "--eval_mode",
        type=str,
        default="all",
        choices=["requirements", "retrieval", "all"],
        help="Evaluation mode (default: all)",
    )
    parser.add_argument(
        "--eval_model",
        type=str,
        default="gpt-5.1",
        help="Model to use for evaluation (default: gpt-5.1)",
    )
    parser.add_argument(
        "--disable_wandb",
        action="store_true",
        help="Disable W&B logging",
    )
    parser.add_argument(
        "--log_level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level (default: INFO)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit processing to first N entries (for testing)",
    )
    parser.add_argument(
        "--recompute",
        action="store_true",
        help="Force recompute PDF extraction (bypass cache)",
    )

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
        log.info("Recompute enabled: All PDF extractions will bypass cache")

    # Load metadata
    metadata_path = Path(args.metadata)
    if not metadata_path.exists():
        log.error("Metadata file not found: %s", metadata_path)
        sys.exit(1)

    log.info("Loading metadata from: %s", metadata_path)
    metadata = load_json(metadata_path)

    if not isinstance(metadata, list):
        log.error("Metadata must be a JSON array, got: %s", type(metadata))
        sys.exit(1)

    total_entries = len(metadata)
    if args.limit:
        metadata = metadata[:args.limit]
        log.info("Limited to first %d entries (out of %d total)", args.limit, total_entries)

    log.info("Processing %d entries", len(metadata))

    # Setup paths
    data_dir = Path(args.data_dir)
    output_base = Path(args.output)
    output_base.mkdir(parents=True, exist_ok=True)

    # Process each entry
    results = {
        "total_entries": len(metadata),
        "total_runs": 0,  # Total number of decision runs
        "pipeline_success": 0,
        "eval_success": 0,
        "failed_runs": [],
    }

    score_data = []  # Collect scores for summary

    for idx, entry in enumerate(metadata):
        log.info("")
        log.info("=" * 80)
        log.info("Entry %d/%d", idx + 1, len(metadata))

        entry_results = process_entry(
            entry=entry,
            entry_index=idx,
            config_path=args.config,
            data_dir=data_dir,
            output_base=output_base,
            eval_mode=args.eval_mode,
            eval_model=args.eval_model,
            disable_wandb=args.disable_wandb,
            recompute=args.recompute,
        )

        # Count successes and failures for each decision run
        for decision_idx, (pipeline_ok, eval_ok) in enumerate(entry_results):
            results["total_runs"] += 1
            entry_id = entry.get("id", f"entry_{idx}")
            run_name = f"{entry_id}_{idx}_d{decision_idx}"

            if pipeline_ok:
                results["pipeline_success"] += 1
            if eval_ok:
                results["eval_success"] += 1
            if not (pipeline_ok and eval_ok):
                results["failed_runs"].append(run_name)

            # Calculate and collect scores for successful evals
            if eval_ok:
                eval_bundle_path = output_base / run_name / "eval_bundle.json"
                requirements_path = output_base / run_name / "requirements.json"
                if eval_bundle_path.exists():
                    try:
                        eval_bundle = load_json(eval_bundle_path)
                        # Load requirements data to get descriptions
                        requirements_data = None
                        if requirements_path.exists():
                            requirements_data = load_json(requirements_path)

                        score_info = calculate_decision_score(eval_bundle, requirements_data)
                        if score_info:
                            parent_path = entry.get("parent", {}).get("path", "unknown.pdf")
                            score_info["pdf_name"] = parent_path
                            score_info["run_name"] = run_name
                            score_data.append(score_info)
                    except Exception as e:
                        log.warning("Failed to process scores for %s: %s", run_name, e)

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

    # Display score summary
    if score_data:
        display_score_summary(score_data)
        # Save detailed scores
        scores_path = output_base / "decision_scores.json"
        save_json(scores_path, score_data)
        log.info("Decision scores saved to: %s", scores_path)

    # Save summary
    summary_path = output_base / "batch_summary.json"
    save_json(summary_path, results)
    log.info("Batch summary saved to: %s", summary_path)

    # Exit with error code if any failures
    if results["failed_runs"]:
        sys.exit(1)


if __name__ == "__main__":
    main()
