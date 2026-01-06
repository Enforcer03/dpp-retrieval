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

    Returns:
        List of (pipeline_success, eval_success, run_id) tuples, one per decision.
        eval_success=True if at least one method eval succeeded.
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

    pdf_path = data_dir / parent_path
    if not pdf_path.exists():
        log.error("PDF not found: %s (entry_id=%s)", pdf_path, entry_id)
        return []

    # Context/eval bundle naming (backward-compatible + new baselines)
    METHODS = ["greedy", "topk", "greedy_cov", "cost_norm", "dpp"]
    CONTEXT_FILES = {
        "greedy": "context.json",
        "topk": "context_topk.json",
        "greedy_cov": "context_greedy_cov.json",
        "cost_norm": "context_cost_norm.json",
        "dpp": "context_dpp.json",
    }
    EVAL_BUNDLES = {
        "greedy": "eval_bundle.json",
        "topk": "eval_bundle_topk.json",
        "greedy_cov": "eval_bundle_greedy_cov.json",
        "cost_norm": "eval_bundle_cost_norm.json",
        "dpp": "eval_bundle_dpp.json",
    }

    results: list[tuple[bool, bool, str]] = []

    for decision_idx, decision in enumerate(decisions):
        run_timestamp = datetime.now().strftime("%H%M%S")
        output_dir_name = f"{batch_timestamp}_{entry_id}_d{decision_idx}_{run_timestamp}"
        entry_output = output_base / output_dir_name
        entry_output.mkdir(parents=True, exist_ok=True)

        log.info("")
        log.info("=" * 80)
        log.info("Processing: %s (decision %d/%d)", output_dir_name, decision_idx + 1, len(decisions))
        log.info("Entry ID: %s (index %d)", entry_id, entry_index)
        log.info("Batch timestamp: %s, Run timestamp: %s", batch_timestamp, run_timestamp)
        log.info("PDF: %s", pdf_path)
        log.info("Decision: %s", (decision or "")[:200])
        log.info("Requirements: %d", len(requirements))
        log.info("Output dir: %s", entry_output)

        # ----------------------------------------------------------------------
        # Step 1: Run pipeline (once)
        # ----------------------------------------------------------------------
        pipeline_cmd = [
            sys.executable,
            "main.py",
            "--config",
            config_path,
            "--pdf",
            str(pdf_path),
            "--query",
            decision,
            "--output_dir",
            str(entry_output),
        ]
        if recompute:
            pipeline_cmd.append("--recompute")

        pipeline_success = run_command(pipeline_cmd, f"Pipeline for {output_dir_name}")

        # Optional: write a simple marker log (run_command may already log; this is safe)
        try:
            (entry_output / "pipeline_cmd.txt").write_text(" ".join(pipeline_cmd), encoding="utf-8")
        except Exception:
            pass

        if not pipeline_success:
            log.error("Pipeline failed for %s, skipping eval", output_dir_name)
            results.append((False, False, output_dir_name))
            continue

        # Verify at least greedy context exists
        context_path = entry_output / CONTEXT_FILES["greedy"]
        if not context_path.exists():
            log.error("Pipeline output not found: %s", context_path)
            results.append((True, False, output_dir_name))
            continue

        # ----------------------------------------------------------------------
        # Step 2: Create requirements file (once)
        # ----------------------------------------------------------------------
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

        # ----------------------------------------------------------------------
        # Step 3: Run evaluation for all available contexts (loop)
        # ----------------------------------------------------------------------
        eval_success_any = False

        for method in METHODS:
            ctx_name = CONTEXT_FILES[method]
            ctx_path = entry_output / ctx_name
            if not ctx_path.exists():
                log.info("Skipping eval (%s): missing %s", method, ctx_name)
                continue

            out_name = EVAL_BUNDLES[method]
            eval_output_path = entry_output / out_name

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

            # Save the command for debugging
            try:
                (entry_output / f"eval_{method}_cmd.txt").write_text(" ".join(eval_cmd), encoding="utf-8")
            except Exception:
                pass

            ok = run_command(eval_cmd, f"Evaluation ({method}) for {output_dir_name}")
            if ok:
                eval_success_any = True
            else:
                log.error("Evaluation failed for %s (method=%s)", output_dir_name, method)

        if eval_success_any:
            log.info("Successfully processed: %s (eval_success_any=True)", output_dir_name)
        else:
            log.error("All evaluations failed for: %s", output_dir_name)

        results.append((True, eval_success_any, output_dir_name))

    return results


def display_score_summary(score_data: list[dict]) -> None:
    """Display a crisp summary table of decision scores with methods as columns."""
    if not score_data:
        log.warning("No score data to display")
        return

    # Group by (pdf_name, decision) -> method -> score_info
    decision_groups: dict[tuple[str, str], dict[str, dict]] = {}
    methods_seen: set[str] = set()

    for item in score_data:
        pdf = item.get("pdf_name", "unknown")
        decision = item.get("decision", "Unknown")
        method = item.get("method", "unknown")
        key = (pdf, decision)

        methods_seen.add(method)
        decision_groups.setdefault(key, {})[method] = item

    # Prefer a stable order (your baselines first, then anything else)
    preferred = ["greedy", "greedy_cov", "topk", "cost_norm", "dpp"]
    methods = [m for m in preferred if m in methods_seen] + sorted([m for m in methods_seen if m not in preferred])

    # Formatting helpers
    def fmt_bd(bd: dict) -> str:
        c = bd.get("covered", 0)
        p = bd.get("partial", 0)
        m = bd.get("missing", 0)
        return f"{c:>2}/{p:>2}/{m:>2}"

    sep_line = "=" * 180
    print(f"\n{sep_line}")
    print("DECISION SCORES SUMMARY (0=missing, 1=partial, 2=covered)")
    print(sep_line)

    # Header
    header = f"{'PDF':<18} {'Decision':<40} "
    for m in methods:
        header += f"{m:>10} {'C/P/M':>9} "
    header += f"{'Reqs':>6}"
    print(header)
    print("-" * 180)

    # Stats accumulators
    score_lists: dict[str, list[float]] = {m: [] for m in methods}
    breakdown_sums: dict[str, dict[str, int]] = {m: {"covered": 0, "partial": 0, "missing": 0} for m in methods}
    total_reqs = 0

    current_pdf = None
    for (pdf, decision) in sorted(decision_groups.keys()):
        row_methods = decision_groups[(pdf, decision)]

        decision_display = decision[:37] + "..." if len(decision) > 40 else decision
        pdf_display = pdf if pdf != current_pdf else ""
        current_pdf = pdf

        # Num requirements (pull from any method that has it)
        num_reqs = 0
        for m in methods:
            if m in row_methods:
                num_reqs = row_methods[m].get("num_requirements", 0) or 0
                if num_reqs:
                    break
        if num_reqs:
            total_reqs += num_reqs

        row = f"{pdf_display:<18} {decision_display:<40} "
        for m in methods:
            info = row_methods.get(m, {})
            score = float(info.get("avg_score", 0.0) or 0.0)
            bd = info.get("status_breakdown", {}) or {}

            row += f"{score:>10.2f} {fmt_bd(bd):>9} "

            if score > 0:
                score_lists[m].append(score)
                breakdown_sums[m]["covered"] += int(bd.get("covered", 0) or 0)
                breakdown_sums[m]["partial"] += int(bd.get("partial", 0) or 0)
                breakdown_sums[m]["missing"] += int(bd.get("missing", 0) or 0)

        row += f"{num_reqs:>6}"
        print(row)

    # Overall summary
    print(sep_line)
    overall = f"{'OVERALL AVERAGE':<18} {'':<40} "
    for m in methods:
        scores = score_lists[m]
        avg = (sum(scores) / len(scores)) if scores else 0.0
        bd = breakdown_sums[m]
        overall += f"{avg:>10.2f} {fmt_bd(bd):>9} "
    overall += f"{total_reqs:>6}"
    print(overall)
    print(sep_line)

    print(f"\nTotal Decisions Analyzed: {len(decision_groups)}")
    print(f"Methods Evaluated: {', '.join(methods)}\n")
