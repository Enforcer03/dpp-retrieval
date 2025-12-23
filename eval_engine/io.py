from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from .utils import ensure_dir


def load_json(path: str | Path) -> Any:
    p = Path(path)
    return json.loads(p.read_text(encoding="utf-8"))


def save_json(path: str | Path, obj: Any) -> None:
    p = Path(path)
    ensure_dir(p.parent)
    p.write_text(json.dumps(obj, ensure_ascii=False, separators=(",", ":"), sort_keys=True), encoding="utf-8")


def load_schema(schema_path: str | Path | None = None) -> dict:
    path = schema_path or os.getenv("EVAL_SCHEMA_PATH") or os.getenv("ICBSUM_EVAL_SCHEMA_PATH")
    if path:
        return load_json(path)
    for name in ("eval_schema.json", "schema.json"):
        p = Path(name)
        if p.exists():
            return load_json(p)
    raise FileNotFoundError("Eval schema not found. Set EVAL_SCHEMA_PATH or place eval_schema.json in cwd.")


def validate_against_schema(obj: Any, schema: dict) -> None:
    try:
        import jsonschema  # type: ignore
    except Exception as e:
        raise RuntimeError(f"jsonschema not available: {e}")
    jsonschema.validate(instance=obj, schema=schema)
