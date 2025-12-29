# eval_engine/main.py
from __future__ import annotations

import argparse
import logging
import sys

from .evaluators import run_evaluation
from .io import load_json, load_schema, save_json, validate_against_schema

log = logging.getLogger("eval_engine")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="pipeline_output.json OR dataset entry json")
    ap.add_argument("--requirements", default=None, help="optional dataset entry json containing task.decisions + requirements")
    ap.add_argument("--output", required=True)
    ap.add_argument("--mode", default="all", choices=["requirements", "retrieval", "all"])
    ap.add_argument("--model", default="gpt-5.1")
    ap.add_argument("--disable_wandb", action="store_true")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")

    try:
        inp = load_json(args.input)

        # If requirements are provided, merge via the adapter's hybrid support:
        # pipeline_output supplies chunks/budgets; dataset_entry supplies decisions/requirements.
        if args.requirements:
            req = load_json(args.requirements)
            inp = {"pipeline_output": inp, "dataset_entry": req}

        schema = load_schema()
        bundle = run_evaluation(inp, schema=schema, mode=args.mode, model=args.model, disable_wandb=args.disable_wandb)
        validate_against_schema(bundle, schema)
        save_json(args.output, bundle)
        log.info("wrote %s", args.output)
    except Exception as e:
        log.error(str(e))
        sys.exit(2)


if __name__ == "__main__":
    main()
