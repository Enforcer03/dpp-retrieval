from __future__ import annotations

import argparse
import logging
import sys

from .evaluators import run_evaluation
from .io import load_json, load_schema, save_json

log = logging.getLogger("eval_engine")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--mode", default="all", choices=["retrieval", "all"])
    ap.add_argument("--model", default="gpt-4.1-mini")
    ap.add_argument("--disable_wandb", action="store_true")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")

    try:
        inp = load_json(args.input)
        schema = load_schema()
        bundle = run_evaluation(inp, schema=schema, mode=args.mode, model=args.model, disable_wandb=args.disable_wandb)
        save_json(args.output, bundle)
        log.info("wrote %s", args.output)
    except Exception as e:
        log.error(str(e))
        sys.exit(2)


if __name__ == "__main__":
    main()
