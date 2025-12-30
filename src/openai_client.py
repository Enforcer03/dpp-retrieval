from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Dict, Optional

from dotenv import load_dotenv
from openai import OpenAI


load_dotenv()  # ensures .env is respected across the whole pipeline


@dataclass
class ChatResult:
    text: str
    model: str
    usage: Dict[str, Any]


class OpenAIChatClient:
    def __init__(self, api_key: Optional[str] = None, timeout_s: int = 60):
        api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY not found (env/.env).")
        self.client = OpenAI(api_key=api_key, timeout=timeout_s)

    def chat(self, model: str, system: str, user: str, temperature: float = 0.0, max_tokens: int = 16000) -> ChatResult:
        """
        Standard chat completion.

        Note: Default max_tokens=16000 is safe for gpt-4o (max output: 16,384).
        Adjust if using different models with lower limits.
        """
        resp = self.client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system or ""},
                {"role": "user", "content": user or ""},
            ],
            temperature=temperature,
            max_completion_tokens=max_tokens,
        )
        msg = resp.choices[0].message.content or ""
        usage = {}
        try:
            usage = resp.usage.model_dump() if resp.usage else {}
        except Exception:
            usage = {}
        return ChatResult(text=msg, model=model, usage=usage)

    def chat_vision_b64jpeg(
        self,
        model: str,
        prompt: str,
        b64jpeg: str,
        temperature: float = 0.0,
        detail: str = "high",
        max_tokens: int = 16000,
    ) -> ChatResult:
        """
        Single-image vision completion.

        Note: Default max_tokens=16000 is safe for gpt-4o (max output: 16,384).
        Caller typically overrides with more conservative values (e.g., 3500).
        """
        resp = self.client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt or ""},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{b64jpeg}", "detail": detail},
                        },
                    ],
                }
            ],
            temperature=temperature,
            max_completion_tokens=max_tokens,
        )
        msg = resp.choices[0].message.content or ""
        usage = {}
        try:
            usage = resp.usage.model_dump() if resp.usage else {}
        except Exception:
            usage = {}
        return ChatResult(text=msg, model=model, usage=usage)

    def chat_vision_multi_images(
        self,
        model: str,
        prompt: str,
        b64jpegs: list[str],
        temperature: float = 0.0,
        detail: str = "high",
        max_tokens: int = 16000,
    ) -> ChatResult:
        """
        Send multiple images in a single Vision API call.

        Args:
            model: Vision model (e.g., gpt-4o)
            prompt: System prompt for extraction
            b64jpegs: List of base64-encoded JPEG images
            temperature: Sampling temperature
            detail: Vision detail level ("high" or "low")
            max_tokens: Max output tokens (default 16000 for gpt-4o safety)

        Returns:
            ChatResult with concatenated text from all pages

        Note: gpt-4o has max_output_tokens=16,384. Caller should pass appropriate
        max_tokens based on batch size (e.g., 3500 * num_pages, capped at 16000).
        """
        if not b64jpegs:
            return ChatResult(text="", model=model, usage={})

        # Build multi-image content array
        content = [{"type": "text", "text": prompt}]
        for b64jpeg in b64jpegs:
            content.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{b64jpeg}",
                    "detail": detail
                },
            })

        resp = self.client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": content}],
            temperature=temperature,
            max_completion_tokens=max_tokens,
        )

        msg = resp.choices[0].message.content or ""
        usage = {}
        try:
            usage = resp.usage.model_dump() if resp.usage else {}
        except Exception:
            usage = {}

        return ChatResult(text=msg, model=model, usage=usage)
