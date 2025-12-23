from __future__ import annotations

import logging
import math
import re
from dataclasses import dataclass
from pathlib import Path

from PIL import Image

log = logging.getLogger(__name__)


_wordish = re.compile(r"\w+|[^\w\s]")


class TokenCounter:
    def __init__(self):
        self._enc = None
        try:
            import tiktoken  # type: ignore

            self._enc = tiktoken.get_encoding("o200k_base")
        except Exception:
            self._enc = None

    def count(self, text: str) -> int:
        t = text or ""
        if self._enc is not None:
            return len(self._enc.encode(t))
        return len(_wordish.findall(t))


@dataclass(frozen=True)
class Cost:
    text_tokens: int
    visual_tokens: int
    total: int


class CostProfiler:
    def __init__(self, token_counter: TokenCounter, alpha_visual: float, patch: int = 14):
        self.tc = token_counter
        self.alpha_visual = alpha_visual
        self.patch = patch

    def estimate_visual_tokens(self, image_path: Path) -> int:
        try:
            with Image.open(image_path) as im:
                w, h = im.size
        except Exception:
            return 0
        gw = int(math.ceil(w / self.patch))
        gh = int(math.ceil(h / self.patch))
        return gw * gh

    def cost(self, context_text: str, image_paths: list[Path]) -> Cost:
        tt = self.tc.count(context_text or "")
        vt = 0
        for p in image_paths:
            vt += self.estimate_visual_tokens(p)
        total = int(tt + self.alpha_visual * vt)
        return Cost(text_tokens=tt, visual_tokens=vt, total=total)
