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
    Embedding-based cost profiler using HuggingFace reasoning complexity model.
    Estimates cognitive load based on text complexity using a transformer model.
    """
    def __init__(
        self,
        token_counter: TokenCounter,
        alpha_visual: float,
        model_name: str = "davanstrien/ModernBERT-based-Reasoning-Required",
        device: str = "cpu",
        patch: int = 14,
        max_length: int = 512,
        hf_max_token_penalty: int = 140,
        hf_gamma: float = 1.6,
        type_weights: Optional[Dict[str, float]] = None,
    ):
        self.tc = token_counter
        self.alpha_visual = alpha_visual
        self.patch = patch
        self.device = device
        self.max_length = max_length
        self.hf_max_token_penalty = hf_max_token_penalty
        self.hf_gamma = hf_gamma

        self.type_weights = type_weights or {
            "text": 1.0,
            "table_text": 1.15,
            "figure": 1.05,
            "page_image": 0.9,
        }

        # Lazy load the model
        self._tokenizer = None
        self._model = None
        self._model_name = model_name

    def _ensure_model_loaded(self):
        """Lazy load the HuggingFace model on first use."""
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
        """Estimate visual tokens from image dimensions."""
        try:
            with Image.open(image_path) as im:
                w, h = im.size
        except Exception:
            return 0
        gw = int(math.ceil(w / self.patch))
        gh = int(math.ceil(h / self.patch))
        return gw * gh

    def _hf_score(self, text: str) -> float:
        """Get reasoning complexity score from HuggingFace model (0..4-ish range)."""
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

    def _hf_to_penalty(self, hf_score_0_4: float) -> int:
        """
        Convert model output (roughly 0..4) to an integer token-equivalent penalty.
        Normalize to 0..1, apply gamma stretch, and scale by max penalty.
        """
        s01 = max(0.0, min(1.0, hf_score_0_4 / 4.0))
        s01 = s01 ** float(self.hf_gamma)
        return int(round(s01 * self.hf_max_token_penalty))

    def cost(self, context_text: str, image_paths: list[Path], unit_type: str = "text") -> Cost:
        """
        Compute cost including reasoning complexity penalty.

        Args:
            context_text: Text content to analyze
            image_paths: List of image paths for visual token estimation
            unit_type: Type of unit (text, table_text, figure, page_image)

        Returns:
            Cost object with text_tokens, visual_tokens, and total
        """
        # Base token count
        tokens = self.tc.count(context_text or "")

        # Apply type weight
        type_w = float(self.type_weights.get(unit_type, 1.0))
        base_cost = max(1, int(round(tokens * type_w)))

        # Get reasoning complexity score and convert to penalty
        hf_score = self._hf_score(context_text or "")
        extra_penalty = self._hf_to_penalty(hf_score)

        # Text cost includes base + reasoning penalty
        text_cost = max(1, base_cost + extra_penalty)

        # Visual tokens
        vt = 0
        for p in image_paths:
            vt += self.estimate_visual_tokens(p)

        # Total cost
        total = int(text_cost + self.alpha_visual * vt)

        return Cost(text_tokens=text_cost, visual_tokens=vt, total=total)


# Backward compatibility alias
CostProfiler = SimpleCostProfiler
