from __future__ import annotations

import base64
import io
from dataclasses import dataclass
from typing import Optional

from .openai_client import OpenAIChatClient


@dataclass
class OCRResult:
    text: str
    model: str


class OCREngine:
    """
    Kept for compatibility with the rest of the pipeline.

    Uses OpenAI Vision to OCR a page image and return layout-aware unit-tagged text.
    Prefer using PdfLayoutExtractor (layout_aware_extractor.py) for full-PDF ingestion.
    """

    def __init__(self, client: OpenAIChatClient, model: str, temperature: float = 0.0):
        self.client = client
        self.model = model
        self.temperature = temperature

    def ocr_image_pil(self, pil_img, prompt: str, max_side: int = 1400, quality: int = 85) -> OCRResult:
        # Resize and encode to b64jpeg
        w, h = pil_img.size
        scale = min(1.0, float(max_side) / max(w, h))
        if scale < 1.0:
            pil_img = pil_img.resize((int(w * scale), int(h * scale)))

        buff = io.BytesIO()
        pil_img.save(buff, format="JPEG", quality=quality, optimize=True)
        b64jpeg = base64.b64encode(buff.getvalue()).decode("utf-8")

        res = self.client.chat_vision_b64jpeg(
            model=self.model,
            prompt=prompt,
            b64jpeg=b64jpeg,
            temperature=self.temperature,
            detail="high",
            max_tokens=22000,
        )
        return OCRResult(text=(res.text or "").strip(), model=res.model)
