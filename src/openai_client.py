from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Any

from dotenv import load_dotenv

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class ChatResult:
    text: str
    model: str


class OpenAIChatClient:
    def __init__(self, timeout_s: int = 60):
        load_dotenv(override=False)
        self.timeout_s = timeout_s
        self._client = None
        self._style = "new"
        self._init_client()

    def _init_client(self) -> None:
        try:
            from openai import OpenAI  # type: ignore

            self._client = OpenAI(timeout=self.timeout_s)
            self._style = "new"
            return
        except Exception:
            pass
        try:
            import openai  # type: ignore

            key = os.getenv("OPENAI_API_KEY")
            if not key:
                raise RuntimeError("OPENAI_API_KEY not set")
            openai.api_key = key
            self._client = openai
            self._style = "old"
        except Exception as e:
            raise RuntimeError(f"OpenAI client init failed: {e}")

    def chat(self, model: str, system: str, user: str, temperature: float = 0.1) -> ChatResult:
        msgs = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        if self._style == "new":
            return ChatResult(text=self._chat_new(model, msgs, temperature), model=model)
        return ChatResult(text=self._chat_old(model, msgs, temperature), model=model)

    def _chat_new(self, model: str, msgs: list[dict[str, str]], temperature: float) -> str:
        assert self._client is not None
        resp = self._client.chat.completions.create(model=model, messages=msgs, temperature=temperature, top_p=1)
        return (resp.choices[0].message.content or "").strip()

    def _chat_old(self, model: str, msgs: list[dict[str, str]], temperature: float) -> str:
        assert self._client is not None
        resp = self._client.ChatCompletion.create(model=model, messages=msgs, temperature=temperature, top_p=1)
        return (resp["choices"][0]["message"]["content"] or "").strip()
