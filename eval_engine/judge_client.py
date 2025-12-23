from __future__ import annotations

import json
import logging
import os
import re
import time
from dataclasses import dataclass
from typing import Any

from dotenv import load_dotenv

log = logging.getLogger(__name__)

_FENCE = re.compile(r"```(?:json)?|```", re.I)


@dataclass(frozen=True)
class JudgeRun:
    name: str
    model: str
    ok: bool
    retry_used: bool
    latency_ms: int
    error: str | None
    output: dict | None


class JudgeClient:
    def __init__(self, model: str, system_prompt: str, timeout_s: int = 60):
        load_dotenv(override=False)
        self.model = model
        self.system_prompt = system_prompt
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

    def run(self, name: str, user_prompt: str) -> JudgeRun:
        t0 = time.time()
        raw, parsed, err = self._call(user_prompt)
        retry_used = False
        if parsed is None:
            retry_used = True
            raw, parsed, err = self._call(self._repair_prompt(user_prompt, raw))
        ms = int((time.time() - t0) * 1000)
        return JudgeRun(name=name, model=self.model, ok=parsed is not None, retry_used=retry_used, latency_ms=ms, error=err, output=parsed)

    def _call(self, user_prompt: str) -> tuple[str, dict | None, str | None]:
        messages = [{"role": "system", "content": self.system_prompt}, {"role": "user", "content": user_prompt}]
        try:
            raw = self._call_new(messages) if self._style == "new" else self._call_old(messages)
            return raw, self._parse_json(raw), None
        except Exception as e:
            return "", None, str(e)

    def _call_new(self, messages: list[dict[str, str]]) -> str:
        assert self._client is not None
        try:
            resp = self._client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=0,
                top_p=1,
                response_format={"type": "json_object"},
            )
        except TypeError:
            resp = self._client.chat.completions.create(model=self.model, messages=messages, temperature=0, top_p=1)
        return (resp.choices[0].message.content or "").strip()

    def _call_old(self, messages: list[dict[str, str]]) -> str:
        assert self._client is not None
        resp = self._client.ChatCompletion.create(model=self.model, messages=messages, temperature=0, top_p=1)
        return (resp["choices"][0]["message"]["content"] or "").strip()

    def _repair_prompt(self, original_user_prompt: str, raw: str) -> str:
        bad = (raw or "")[:6000]
        return (
            "Return valid JSON only (no markdown). Keep the same output keys and intent.\n\n"
            f"Task:\n{original_user_prompt}\n\n"
            f"Invalid response:\n{bad}"
        )

    def _parse_json(self, text: str) -> dict:
        t = (text or "").strip()
        t = _FENCE.sub("", t).strip()
        i = t.find("{")
        j = t.rfind("}")
        if i < 0 or j <= i:
            raise ValueError("No JSON object found")
        s = t[i : j + 1]
        obj = json.loads(s)
        if not isinstance(obj, dict):
            raise ValueError("JSON root must be object")
        return obj
