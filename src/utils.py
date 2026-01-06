from __future__ import annotations

import hashlib
import json
import logging
import os
import re
from dataclasses import asdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from .schema import BBox

from src.config import load_config
from src.wandb_logger import wandb_finish, wandb_init, wandb_log, wandb_log_artifact
import datetime
from datetime import datetime
import subprocess
import sys
log = logging.getLogger(__name__)

def ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p


def setup_logging(level: str, log_file: Path) -> None:
    ensure_dir(log_file.parent)
    root = logging.getLogger()
    root.handlers.clear()
    root.setLevel(level.upper())

    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")
    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    root.addHandler(sh)

    fh = logging.FileHandler(log_file, encoding="utf-8")
    fh.setFormatter(fmt)
    root.addHandler(fh)


def sha1(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8", errors="ignore")).hexdigest()


def stable_id(*parts: object) -> str:
    return sha1("|".join(map(str, parts)))[:16]


_word_re = re.compile(r"[A-Za-z0-9]+(?:[-_][A-Za-z0-9]+)*")


def tokenize(text: str) -> list[str]:
    return [t.lower() for t in _word_re.findall(text or "")]


def is_caption(text: str) -> bool:
    t = (text or "").strip().lower()
    return t.startswith(("fig", "figure", "table", "chart", "exhibit"))


def bbox_union(a: BBox, b: BBox) -> BBox:
    return (min(a[0], b[0]), min(a[1], b[1]), max(a[2], b[2]), max(a[3], b[3]))


def bbox_iou(a: BBox, b: BBox) -> float:
    x0 = max(a[0], b[0])
    y0 = max(a[1], b[1])
    x1 = min(a[2], b[2])
    y1 = min(a[3], b[3])
    iw = max(0.0, x1 - x0)
    ih = max(0.0, y1 - y0)
    inter = iw * ih
    ua = (a[2] - a[0]) * (a[3] - a[1])
    ub = (b[2] - b[0]) * (b[3] - b[1])
    den = ua + ub - inter
    return 0.0 if den <= 0 else inter / den


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    na = float(np.linalg.norm(a) + 1e-12)
    nb = float(np.linalg.norm(b) + 1e-12)
    return float(np.dot(a, b) / (na * nb))


def normalize_rows(x: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(x, axis=1, keepdims=True) + 1e-12
    return x / n


def rrf_fusion(rank_lists: list[list[str]], k: int) -> dict[str, float]:
    out: dict[str, float] = {}
    for lst in rank_lists:
        for i, key in enumerate(lst, start=1):
            out[key] = out.get(key, 0.0) + 1.0 / (k + i)
    return out


def write_json(path: Path, obj: object) -> None:
    ensure_dir(path.parent)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def read_json(path: Path) -> dict | list | None:
    """Read JSON file, return None if not found."""
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def dataclass_list_to_dicts(xs: Iterable[object]) -> list[dict]:
    return [asdict(x) for x in xs]


def env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except Exception:
        return default




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
    """Map requirement status to numeric score: 2=covered, 1=partial, 0=missing."""
    status_lower = str(status).lower().strip()
    if status_lower in ("covered", "satisfied", "full", "complete"):
        return 2
    elif status_lower in ("partial", "weak"):
        return 1
    return 0


def calculate_decision_score(eval_bundle: dict, requirements_data: dict | None = None) -> dict | None:
    """Calculate average requirement score for a decision."""
    try:
        decisions = (
            eval_bundle.get("evaluation", {})
            .get("retrieval_eval", {})
            .get("requirements_eval", {})
            .get("decisions", [])
        )

        if not decisions:
            return None

        decision_data = decisions[0]
        decision_text = decision_data.get("decision", "Unknown")
        requirements = decision_data.get("requirements", [])

        # Build requirement description map
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
    try:
        subprocess.run(cmd, check=True, text=True)
        return True
    except subprocess.CalledProcessError as e:
        log.error("Command failed: %s (return code: %s)", description, e.returncode)
        return False


def process_entry(
    entry: dict,
    entry_index: int,
    batch_timestamp: str,
    config_path: str,
    data_dir: Path,
    output_base: Path,
    eval_mode: str,
    eval_model: str,
    recompute: bool = False,
) -> list[tuple[bool, bool, str]]:
    """
    Process a single metadata entry.
    Each decision within the entry is processed separately.

    Args:
        batch_timestamp: Batch timestamp in YYYYMMDD_HHMMSS format

    Returns:
        List of (pipeline_success, eval_success, run_id) tuples, one per decision
    """
    entry_id = entry.get("id")
    if not entry_id:
        log.warning("Entry missing 'id' field, skipping")
        return []

    parent_path = entry.get("parent", {}).get("path")
    if not parent_path:
        log.error("Entry %s missing parent.path, skipping", entry_id)
        return []

    decisions = entry.get("task", {}).get("decisions", [])
    if not decisions:
        log.warning("Entry %s has no decisions, using empty query", entry_id)
        decisions = [""]

    requirements = entry.get("requirements", [])

    # Construct PDF path
    pdf_path = data_dir / parent_path
    if not pdf_path.exists():
        log.error("PDF not found: %s (entry_id=%s)", pdf_path, entry_id)
        return []

    results = []

    for decision_idx, decision in enumerate(decisions):
        # Generate run timestamp for this specific decision run
        run_timestamp = datetime.now().strftime("%H%M%S")

        # Create unique output directory with timestamps: {batch_timestamp}_{entry_id}_d{decision_idx}_{run_timestamp}
        output_dir_name = f"{batch_timestamp}_{entry_id}_d{decision_idx}_{run_timestamp}"
        entry_output = output_base / output_dir_name
        entry_output.mkdir(parents=True, exist_ok=True)

        log.info("")
        log.info("=" * 80)
        log.info("Processing: %s (decision %d/%d)", output_dir_name, decision_idx + 1, len(decisions))
        log.info("Entry ID: %s (index %d)", entry_id, entry_index)
        log.info("Batch timestamp: %s, Run timestamp: %s", batch_timestamp, run_timestamp)
        log.info("PDF: %s", pdf_path)
        log.info("Decision: %s", decision[:200])
        log.info("Requirements: %d", len(requirements))
        log.info("Output dir: %s", entry_output)

        # Step 1: Run pipeline
        pipeline_cmd = [
            sys.executable,
            "main.py",
            "--config", config_path,
            "--pdf", str(pdf_path),
            "--query", decision,
            "--output_dir", str(entry_output),
        ]

        if recompute:
            pipeline_cmd.append("--recompute")

        pipeline_success = run_command(pipeline_cmd, f"Pipeline for {output_dir_name}")

        if not pipeline_success:
            log.error("Pipeline failed for %s, skipping eval", output_dir_name)
            results.append((False, False, output_dir_name))
            continue

        # Verify context.json was created
        context_path = entry_output / "context.json"
        if not context_path.exists():
            log.error("Pipeline output not found: %s", context_path)
            results.append((False, False, output_dir_name))
            continue

        # Step 2: Create requirements file
        requirements_data = {
            "id": f"{entry_id}_d{decision_idx}",
            "parent": entry.get("parent", {}),
            "task": {"decisions": [decision]},
            "requirements": requirements,
            "notes": entry.get("notes", {}),
        }
        requirements_path = entry_output / "requirements.json"
        save_json(requirements_path, requirements_data)
        log.info("Saved requirements to: %s", requirements_path)

        # Step 3: Run evaluation
        eval_output_path = entry_output / "eval_bundle.json"
        eval_cmd = [
            sys.executable,
            "-m", "eval_engine.main",
            "--input", str(context_path),
            "--requirements", str(requirements_path),
            "--output", str(eval_output_path),
            "--mode", eval_mode,
            "--model", eval_model,
        ]

        eval_success = run_command(eval_cmd, f"Evaluation for {output_dir_name}")

        # Run evaluation for top-k selection if context_topk.json exists
        context_topk_path = entry_output / "context_topk.json"
        eval_topk_success = False
        if context_topk_path.exists():
            eval_topk_output = entry_output / "eval_bundle_topk.json"
            eval_topk_cmd = [
                sys.executable,
                "-m", "eval_engine.main",
                "--input", str(context_topk_path),
                "--requirements", str(requirements_path),
                "--output", str(eval_topk_output),
                "--mode", eval_mode,
                "--model", eval_model,
            ]
            eval_topk_success = run_command(eval_topk_cmd, f"Evaluation (topk) for {output_dir_name}")

        # Run evaluation for greedy+coverage selection if context_greedy_cov.json exists
        context_greedy_cov_path = entry_output / "context_greedy_cov.json"
        eval_greedy_cov_success = False
        if context_greedy_cov_path.exists():
            eval_greedy_cov_output = entry_output / "eval_bundle_greedy_cov.json"
            eval_greedy_cov_cmd = [
                sys.executable,
                "-m", "eval_engine.main",
                "--input", str(context_greedy_cov_path),
                "--requirements", str(requirements_path),
                "--output", str(eval_greedy_cov_output),
                "--mode", eval_mode,
                "--model", eval_model,
            ]
            eval_greedy_cov_success = run_command(eval_greedy_cov_cmd, f"Evaluation (greedy_cov) for {output_dir_name}")

        if eval_success:
            log.info("Successfully processed: %s", output_dir_name)
        else:
            log.error("Evaluation failed for: %s", output_dir_name)

        results.append((pipeline_success, eval_success, output_dir_name))

    return results


def display_score_summary(score_data: list[dict]) -> None:
    """Display a crisp summary table of decision scores with methods as columns."""
    if not score_data:
        log.warning("No score data to display")
        return

    # Group by (pdf_name, decision) to combine methods
    decision_groups: dict[tuple[str, str], dict[str, dict]] = {}
    for item in score_data:
        pdf = item.get("pdf_name", "unknown")
        decision = item.get("decision", "Unknown")
        method = item.get("method", "unknown")
        key = (pdf, decision)

        if key not in decision_groups:
            decision_groups[key] = {}
        decision_groups[key][method] = item

    # Print header
    sep_line = "=" * 152
    print(f"\n{sep_line}")
    print("DECISION SCORES SUMMARY (0=missing, 1=partial, 2=covered)")
    print(sep_line)
    print(f"{'PDF':<18} {'Decision':<35} {'Greedy':>8} {'G-C/P/M':>10} {'Gr+Cov':>8} {'GC-C/P/M':>10} {'TopK':>8} {'T-C/P/M':>10} {'Reqs':>6}")
    print("-" * 152)

    # Track overall stats per method
    greedy_scores = []
    greedy_cov_scores = []
    topk_scores = []
    greedy_breakdown = {"covered": 0, "partial": 0, "missing": 0}
    greedy_cov_breakdown = {"covered": 0, "partial": 0, "missing": 0}
    topk_breakdown = {"covered": 0, "partial": 0, "missing": 0}
    total_reqs = 0

    # Print each decision group
    current_pdf = None
    for (pdf, decision) in sorted(decision_groups.keys()):
        methods = decision_groups[(pdf, decision)]

        # Get data for each method
        greedy = methods.get("greedy", {})
        greedy_cov = methods.get("greedy_cov", {})
        topk = methods.get("topk", {})

        # Scores
        greedy_score = greedy.get("avg_score", 0.0)
        greedy_cov_score = greedy_cov.get("avg_score", 0.0)
        topk_score = topk.get("avg_score", 0.0)

        # Breakdowns
        g_bd = greedy.get("status_breakdown", {})
        gc_bd = greedy_cov.get("status_breakdown", {})
        t_bd = topk.get("status_breakdown", {})

        g_c, g_p, g_m = g_bd.get("covered", 0), g_bd.get("partial", 0), g_bd.get("missing", 0)
        gc_c, gc_p, gc_m = gc_bd.get("covered", 0), gc_bd.get("partial", 0), gc_bd.get("missing", 0)
        t_c, t_p, t_m = t_bd.get("covered", 0), t_bd.get("partial", 0), t_bd.get("missing", 0)

        # Number of requirements (same across methods)
        num_reqs = greedy.get("num_requirements", greedy_cov.get("num_requirements", topk.get("num_requirements", 0)))

        # Accumulate stats
        if greedy_score > 0:
            greedy_scores.append(greedy_score)
            greedy_breakdown["covered"] += g_c
            greedy_breakdown["partial"] += g_p
            greedy_breakdown["missing"] += g_m
        if greedy_cov_score > 0:
            greedy_cov_scores.append(greedy_cov_score)
            greedy_cov_breakdown["covered"] += gc_c
            greedy_cov_breakdown["partial"] += gc_p
            greedy_cov_breakdown["missing"] += gc_m
        if topk_score > 0:
            topk_scores.append(topk_score)
            topk_breakdown["covered"] += t_c
            topk_breakdown["partial"] += t_p
            topk_breakdown["missing"] += t_m
        if num_reqs > 0:
            total_reqs += num_reqs

        # Format decision text
        decision_display = decision[:32] + "..." if len(decision) > 35 else decision
        pdf_display = pdf if pdf != current_pdf else ""
        current_pdf = pdf

        # Print row
        print(
            f"{pdf_display:<18} {decision_display:<35} "
            f"{greedy_score:>8.2f} {g_c:>2}/{g_p:>2}/{g_m:>2}    "
            f"{greedy_cov_score:>8.2f} {gc_c:>2}/{gc_p:>2}/{gc_m:>2}    "
            f"{topk_score:>8.2f} {t_c:>2}/{t_p:>2}/{t_m:>2}    "
            f"{num_reqs:>6}"
        )

    # Overall summary
    print(sep_line)
    greedy_avg = sum(greedy_scores) / len(greedy_scores) if greedy_scores else 0.0
    greedy_cov_avg = sum(greedy_cov_scores) / len(greedy_cov_scores) if greedy_cov_scores else 0.0
    topk_avg = sum(topk_scores) / len(topk_scores) if topk_scores else 0.0

    g_c = greedy_breakdown["covered"]
    g_p = greedy_breakdown["partial"]
    g_m = greedy_breakdown["missing"]
    gc_c = greedy_cov_breakdown["covered"]
    gc_p = greedy_cov_breakdown["partial"]
    gc_m = greedy_cov_breakdown["missing"]
    t_c = topk_breakdown["covered"]
    t_p = topk_breakdown["partial"]
    t_m = topk_breakdown["missing"]

    print(
        f"{'OVERALL AVERAGE':<18} {'':<35} "
        f"{greedy_avg:>8.2f} {g_c:>2}/{g_p:>2}/{g_m:>2}    "
        f"{greedy_cov_avg:>8.2f} {gc_c:>2}/{gc_p:>2}/{gc_m:>2}    "
        f"{topk_avg:>8.2f} {t_c:>2}/{t_p:>2}/{t_m:>2}    "
        f"{total_reqs:>6}"
    )
    print(sep_line)

    # Score distribution
    total_decisions = len(decision_groups)
    print(f"\nTotal Decisions Analyzed: {total_decisions}")
    print(f"Methods Evaluated: {len([m for m in [greedy_scores, greedy_cov_scores, topk_scores] if m])}\n")

