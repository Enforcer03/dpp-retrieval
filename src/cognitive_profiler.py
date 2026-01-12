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



def clamp01(x: float) -> float:
    return max(0.0, min(1.0, x))


class EmbeddingBasedCostProfiler:
    """
    Text-only embedding-based cost profiler.

    - Base cost: token count * type_weight
    - Complexity discount: reduces cost by up to max_discount_frac of base cost
      (bounded, scales with length => avoids collapsing short units to cost=1)
    - If images exist: apply a small multiplier (<1) to make them slightly cheaper
      (heuristic: images are useful for decisioning !!)
    """

    def __init__(
        self,
        token_counter,
        model_name: str = "davanstrien/ModernBERT-based-Reasoning-Required",
        device: str = "cpu",
        max_length: int = 512,
        max_discount_frac: float = 0.50,   # up to 50% off at max complexity
        image_cost_multiplier: float = 0.85,  # images => 15% cheaper (tune 0.8-0.95)
        type_weights: Optional[Dict[str, float]] = None,
    ):
        self.tc = token_counter
        self.device = device
        self.max_length = int(max_length)

        self.max_discount_frac = float(max_discount_frac)
        self.image_cost_multiplier = float(image_cost_multiplier)

        self.type_weights = type_weights or {
            "text": 1.0,
            "table_text": 1.15,
            "equation": 1.10,
            "figure": 1.0,
            "heading": 0.8,
            "reference": 3.90,
        }

        self._tokenizer = None
        self._model = None
        self._model_name = model_name

        # tiny cache for speed
        self._cache: dict[str, float] = {}

    def _ensure_model_loaded(self):
        if self._tokenizer is None or self._model is None:
            from transformers import AutoModelForSequenceClassification, AutoTokenizer

            log.info("Loading reasoning complexity model: %s", self._model_name)
            self._tokenizer = AutoTokenizer.from_pretrained(self._model_name)
            self._model = AutoModelForSequenceClassification.from_pretrained(
                self._model_name
            ).to(self.device).eval()
            log.info("Model loaded")

    def _complexity_norm01(self, text: str) -> float:
        """
        Map model output -> [0,1] without assuming logits are 0..4.
        Works for both regression (1 logit) and classification (C logits).
        """
        t = (text or "").strip()
        if not t:
            return 0.0

        key = t[:512]
        if key in self._cache:
            return self._cache[key]

        self._ensure_model_loaded()
        import torch
        import math

        inputs = self._tokenizer(
            t,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_length,
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            out = self._model(**inputs)
            logits = out.logits.squeeze()

        if logits.numel() == 1:
            # regression-like head: sigmoid to [0,1]
            val = float(logits.item())
            c01 = 1.0 / (1.0 + math.exp(-val))
        else:
            # classification head: expected label / (C-1)
            probs = torch.softmax(logits.float(), dim=0)
            C = int(probs.numel())
            expected = float((probs * torch.arange(C, device=probs.device)).sum().item())
            c01 = expected / max(1.0, float(C - 1))

        c01 = clamp01(float(c01))
        self._cache[key] = c01
        return c01

    def cost(self, context_text: str, image_paths: list, unit_type: str = "text"):
        # base
        tokens = max(1, int(self.tc.count(context_text or "")))
        type_w = float(self.type_weights.get(unit_type, 1.0))
        base_cost = max(1, int(round(tokens * type_w)))

        # complexity discount (bounded)
        c01 = self._complexity_norm01(context_text or "")
        discount = int(round(base_cost * self.max_discount_frac * c01))
        text_cost = max(1, base_cost - discount)

        # images are "slightly cheaper" (heuristic)
        if image_paths:
            text_cost = max(1, int(round(text_cost * self.image_cost_multiplier)))

        # Keep return shape compatible with your pipeline
        return Cost(text_tokens=text_cost, visual_tokens=0, total=int(text_cost))

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


# class EmbeddingBasedCostProfiler:
#     """
#     Embedding-based cost profiler using reasoning complexity model.
    
#     FIX: Complex units get LOWER cost (not higher) because they have higher
#     decision utility per token. Simple fragments require more integration effort.
#     """
#     def __init__(
#         self,
#         token_counter: TokenCounter,
#         alpha_visual: float,
#         model_name: str = "davanstrien/ModernBERT-based-Reasoning-Required",
#         device: str = "cpu",
#         patch: int = 14,
#         max_length: int = 512,
#         complexity_scale: float = 50.0,  # Max tokens to discount for complexity
#         type_weights: Optional[Dict[str, float]] = None,
#     ):
#         self.tc = token_counter
#         self.alpha_visual = alpha_visual
#         self.patch = patch
#         self.device = device
#         self.max_length = max_length
#         self.complexity_scale = complexity_scale

#         self.type_weights = type_weights or {
#             "text": 1.0,
#             "table_text": 1.15,
#             "figure": 1.05,
#             "page_image": 0.9,
#         }

#         # Lazy load
#         self._tokenizer = None
#         self._model = None
#         self._model_name = model_name

#     def _ensure_model_loaded(self):
#         if self._tokenizer is None or self._model is None:
#             try:
#                 import torch
#                 from transformers import AutoModelForSequenceClassification, AutoTokenizer

#                 log.info(f"Loading reasoning complexity model: {self._model_name}")
#                 self._tokenizer = AutoTokenizer.from_pretrained(self._model_name)
#                 self._model = AutoModelForSequenceClassification.from_pretrained(
#                     self._model_name
#                 ).to(self.device).eval()
#                 log.info("Model loaded successfully")
#             except Exception as e:
#                 log.error(f"Failed to load HuggingFace model: {e}")
#                 raise

#     def estimate_visual_tokens(self, image_path: Path) -> int:
#         try:
#             with Image.open(image_path) as im:
#                 w, h = im.size
#         except Exception:
#             return 0
#         gw = int(math.ceil(w / self.patch))
#         gh = int(math.ceil(h / self.patch))
#         return gw * gh

#     def _complexity_score(self, text: str) -> float:
#         """Get reasoning complexity score (0=simple, 4=complex)."""
#         self._ensure_model_loaded()
#         import torch

#         inputs = self._tokenizer(
#             text,
#             return_tensors="pt",
#             truncation=True,
#             max_length=self.max_length,
#         )
#         inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
#         with torch.no_grad():
#             out = self._model(**inputs)
        
#         return float(out.logits.squeeze().item())

#     def cost(self, context_text: str, image_paths: list[Path], unit_type: str = "text") -> Cost:
#         """
#         Compute cost: base_tokens - complexity_discount + visual_tokens
        
#         Higher complexity → LOWER cost → preferred by selection.
#         """
#         # Base token count
#         tokens = self.tc.count(context_text or "")
#         type_w = float(self.type_weights.get(unit_type, 1.0))
#         base_cost = max(1, int(round(tokens * type_w)))

#         # Complexity discount: higher complexity = bigger discount
#         complexity = self._complexity_score(context_text or "")
#         complexity_norm = max(0.0, min(1.0, complexity / 4.0))  # Normalize to 0-1
#         discount = int(complexity_norm * self.complexity_scale)
        
#         # Apply discount (SUBTRACT, not add)
#         text_cost = max(1, base_cost - discount)

#         # Visual tokens
#         vt = 0
#         for p in image_paths:
#             vt += self.estimate_visual_tokens(p)

#         # Total
#         total = int(text_cost + self.alpha_visual * vt)

#         return Cost(text_tokens=text_cost, visual_tokens=vt, total=total)


# Backward compatibility alias
CostProfiler = SimpleCostProfiler