from __future__ import annotations

import logging
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Dict

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


class SimpleCostProfiler:
    """
    Simple cost profiler based on token counting and visual tokens.
    """
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


class EmbeddingBasedCostProfiler:
    """
    Embedding-based cost profiler using reasoning complexity model.
    
    FIX: Complex units get LOWER cost (not higher) because they have higher
    decision utility per token. Simple fragments require more integration effort.
    """
    def __init__(
        self,
        token_counter: TokenCounter,
        alpha_visual: float,
        model_name: str = "davanstrien/ModernBERT-based-Reasoning-Required",
        device: str = "cpu",
        patch: int = 14,
        max_length: int = 512,
        complexity_scale: float = 50.0,  # Max tokens to discount for complexity
        type_weights: Optional[Dict[str, float]] = None,
    ):
        self.tc = token_counter
        self.alpha_visual = alpha_visual
        self.patch = patch
        self.device = device
        self.max_length = max_length
        self.complexity_scale = complexity_scale

        self.type_weights = type_weights or {
            "text": 1.0,
            "table_text": 1.15,
            "figure": 1.05,
            "page_image": 0.9,
        }

        # Lazy load
        self._tokenizer = None
        self._model = None
        self._model_name = model_name

    def _ensure_model_loaded(self):
        if self._tokenizer is None or self._model is None:
            try:
                import torch
                from transformers import AutoModelForSequenceClassification, AutoTokenizer

                log.info(f"Loading reasoning complexity model: {self._model_name}")
                self._tokenizer = AutoTokenizer.from_pretrained(self._model_name)
                self._model = AutoModelForSequenceClassification.from_pretrained(
                    self._model_name
                ).to(self.device).eval()
                log.info("Model loaded successfully")
            except Exception as e:
                log.error(f"Failed to load HuggingFace model: {e}")
                raise

    def estimate_visual_tokens(self, image_path: Path) -> int:
        try:
            with Image.open(image_path) as im:
                w, h = im.size
        except Exception:
            return 0
        gw = int(math.ceil(w / self.patch))
        gh = int(math.ceil(h / self.patch))
        return gw * gh

    def _complexity_score(self, text: str) -> float:
        """Get reasoning complexity score (0=simple, 4=complex)."""
        self._ensure_model_loaded()
        import torch

        inputs = self._tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_length,
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            out = self._model(**inputs)
        
        return float(out.logits.squeeze().item())

    def cost(self, context_text: str, image_paths: list[Path], unit_type: str = "text") -> Cost:
        """
        Compute cost: base_tokens - complexity_discount + visual_tokens
        
        Higher complexity → LOWER cost → preferred by selection.
        """
        # Base token count
        tokens = self.tc.count(context_text or "")
        type_w = float(self.type_weights.get(unit_type, 1.0))
        base_cost = max(1, int(round(tokens * type_w)))

        # Complexity discount: higher complexity = bigger discount
        complexity = self._complexity_score(context_text or "")
        complexity_norm = max(0.0, min(1.0, complexity / 4.0))  # Normalize to 0-1
        discount = int(complexity_norm * self.complexity_scale)
        
        # Apply discount (SUBTRACT, not add)
        text_cost = max(1, base_cost - discount)

        # Visual tokens
        vt = 0
        for p in image_paths:
            vt += self.estimate_visual_tokens(p)

        # Total
        total = int(text_cost + self.alpha_visual * vt)

        return Cost(text_tokens=text_cost, visual_tokens=vt, total=total)


# Backward compatibility alias
CostProfiler = SimpleCostProfiler