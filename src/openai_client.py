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

    def chat(self, model: str, system: str, user: str, temperature: float = 0.0, max_tokens: int = 1800) -> ChatResult:
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
        max_tokens: int = 2200,
    ) -> ChatResult:
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
