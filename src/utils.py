from __future__ import annotations

import hashlib
import json
import logging
import os
import re
from dataclasses import asdict
from pathlib import Path
from typing import Iterable

import numpy as np

from .schema import BBox


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


def dataclass_list_to_dicts(xs: Iterable[object]) -> list[dict]:
    return [asdict(x) for x in xs]


def env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except Exception:
        return default
