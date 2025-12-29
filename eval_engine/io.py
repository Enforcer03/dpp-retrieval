# eval_engine/io.py
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any


def load_json(path: str) -> Any:
    p = Path(path)
    raw = p.read_text(encoding="utf-8", errors="ignore").strip()
    if not raw:
        return {}

    try:
        return json.loads(raw)
    except Exception:
        items: list[Any] = []
        for line in raw.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                items.append(json.loads(line))
            except Exception:
                continue
        if len(items) == 1:
            return items[0]
        if items:
            return {"entries": items}
        return {}


def save_json(path: str, obj: Any) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def load_schema() -> dict:
    envp = os.getenv("EVAL_SCHEMA_PATH")
    candidates = [envp] if envp else []
    candidates.append(str(Path.cwd() / "eval_schema.json"))
    for c in candidates:
        if not c:
            continue
        p = Path(c)
        if p.exists() and p.is_file():
            return json.loads(p.read_text(encoding="utf-8", errors="ignore"))
    raise FileNotFoundError("Eval schema not found. Set EVAL_SCHEMA_PATH or place eval_schema.json in cwd.")


def validate_against_schema(bundle: dict, schema: dict) -> None:
    try:
        import jsonschema  # type: ignore
    except Exception as e:
        raise RuntimeError("jsonschema is required for validation") from e
    jsonschema.validate(instance=bundle, schema=schema)
