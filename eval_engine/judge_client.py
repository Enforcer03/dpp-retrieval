# eval_engine/judge_client.py
from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass
from typing import Any

import httpx
from dotenv import load_dotenv

_CODE_FENCE = re.compile(r"^```(?:json)?\s*|\s*```$", re.M)
_JSON_OBJ = re.compile(r"\{.*\}", re.S)


@dataclass
class JudgeRun:
    name: str
    model: str
    ok: bool
    repaired: bool
    latency_s: float
    usage: dict
    error: str | None = None


def _strip_code_fences(s: str) -> str:
    return _CODE_FENCE.sub("", s).strip()


def _extract_json_obj(s: str) -> str | None:
    s = _strip_code_fences(s)
    m = _JSON_OBJ.search(s)
    if not m:
        return None
    return m.group(0)


class JudgeClient:
    def __init__(self, timeout_s: float = 60.0, base_url: str = "https://api.openai.com/v1"):
        load_dotenv()
        self.api_key = os.getenv("OPENAI_API_KEY") or ""
        if not self.api_key:
            raise RuntimeError("OPENAI_API_KEY not set")
        self.timeout_s = timeout_s
        self.base_url = base_url.rstrip("/")

    def _call(self, model: str, system_prompt: str, user_prompt: str, temperature: float) -> tuple[str, dict]:
        url = f"{self.base_url}/chat/completions"
        headers = {"Authorization": f"Bearer {self.api_key}"}
        payload = {
            "model": model,
            "temperature": float(temperature),
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        }
        with httpx.Client(timeout=self.timeout_s) as client:
            r = client.post(url, headers=headers, json=payload)
            r.raise_for_status()
            data = r.json()
        text = (
            ((data.get("choices") or [{}])[0].get("message") or {}).get("content")
            or ((data.get("choices") or [{}])[0].get("text") or "")
            or ""
        )
        usage = data.get("usage") if isinstance(data.get("usage"), dict) else {}
        return text, usage

    def judge_json(
        self,
        *,
        name: str,
        model: str,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.1,
    ) -> tuple[dict, JudgeRun]:
        t0 = time.time()
        repaired = False
        usage: dict[str, Any] = {}

        try:
            raw, usage = self._call(model=model, system_prompt=system_prompt, user_prompt=user_prompt, temperature=temperature)
            obj_s = _extract_json_obj(raw)
            if obj_s is None:
                raise ValueError("no_json_object_found")
            try:
                out = json.loads(obj_s)
                return out, JudgeRun(name=name, model=model, ok=True, repaired=False, latency_s=time.time() - t0, usage=usage)
            except Exception:
                repaired = True

            repair_prompt = (
                "You returned invalid JSON. Return ONLY a valid JSON object that matches the requested schema. "
                "Do not include code fences, markdown, or explanations.\n\n"
                "Previous answer:\n"
                f"{raw}"
            )
            raw2, usage2 = self._call(model=model, system_prompt=system_prompt, user_prompt=repair_prompt, temperature=temperature)
            usage = usage2 or usage
            obj_s2 = _extract_json_obj(raw2)
            if obj_s2 is None:
                raise ValueError("repair_no_json_object_found")
            out2 = json.loads(obj_s2)
            return out2, JudgeRun(name=name, model=model, ok=True, repaired=True, latency_s=time.time() - t0, usage=usage)

        except Exception as e:
            return (
                {},
                JudgeRun(
                    name=name,
                    model=model,
                    ok=False,
                    repaired=repaired,
                    latency_s=time.time() - t0,
                    usage=usage,
                    error=str(e),
                ),
            )
