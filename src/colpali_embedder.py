from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

from .utils import normalize_rows, tokenize

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class EmbedResult:
    ids: list[str]
    vecs: np.ndarray


class MultiModalEmbedder:
    def __init__(self, backend: str, model_name: str, device: str, batch_size: int, dim: int):
        self.backend = backend
        self.model_name = model_name
        self.device = device
        self.batch_size = batch_size
        self.dim = dim

        self._hf = None
        if backend.startswith("hf_"):
            self._hf = self._init_hf()

    def _init_hf(self):
        try:
            import torch
            from transformers import AutoProcessor, AutoModel  # type: ignore

            proc = AutoProcessor.from_pretrained(self.model_name)
            model = AutoModel.from_pretrained(self.model_name)
            model.to(self.device)
            model.eval()
            return torch, proc, model
        except Exception as e:
            log.warning("embedder.hf_init_failed backend=%s model=%s err=%s", self.backend, self.model_name, e)
            return None

    def embed_text(self, ids: list[str], texts: list[str]) -> EmbedResult:
        if self._hf is None:
            vecs = self._hash_text(texts, self.dim)
            return EmbedResult(ids=ids, vecs=normalize_rows(vecs))

        torch, proc, model = self._hf
        all_vecs: list[np.ndarray] = []
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i : i + self.batch_size]
            inputs = proc(text=batch, return_tensors="pt", padding=True, truncation=True)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            with torch.no_grad():
                if hasattr(model, "get_text_features"):
                    x = model.get_text_features(**inputs)
                else:
                    out = model(**inputs)
                    x = out.last_hidden_state[:, 0, :]
                x = x / (x.norm(dim=-1, keepdim=True) + 1e-12)
            all_vecs.append(x.detach().cpu().numpy().astype(np.float32))
        vecs = np.vstack(all_vecs) if all_vecs else np.zeros((0, self.dim), dtype=np.float32)
        return EmbedResult(ids=ids, vecs=vecs)

    def embed_images(self, ids: list[str], image_paths: list[Path]) -> EmbedResult:
        if self._hf is None:
            vecs = self._hash_images(image_paths, self.dim)
            return EmbedResult(ids=ids, vecs=normalize_rows(vecs))

        torch, proc, model = self._hf
        all_vecs: list[np.ndarray] = []
        for i in range(0, len(image_paths), self.batch_size):
            batch_paths = image_paths[i : i + self.batch_size]
            imgs = [Image.open(p).convert("RGB") for p in batch_paths]
            inputs = proc(images=imgs, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            with torch.no_grad():
                if hasattr(model, "get_image_features"):
                    x = model.get_image_features(**inputs)
                else:
                    out = model(**inputs)
                    x = out.last_hidden_state[:, 0, :]
                x = x / (x.norm(dim=-1, keepdim=True) + 1e-12)
            for im in imgs:
                im.close()
            all_vecs.append(x.detach().cpu().numpy().astype(np.float32))
        vecs = np.vstack(all_vecs) if all_vecs else np.zeros((0, self.dim), dtype=np.float32)
        return EmbedResult(ids=ids, vecs=vecs)

    def _hash_text(self, texts: list[str], dim: int) -> np.ndarray:
        out = np.zeros((len(texts), dim), dtype=np.float32)
        for i, t in enumerate(texts):
            toks = tokenize(t)
            if not toks:
                continue
            for tok in toks:
                h = abs(hash(tok))
                j = h % dim
                s = -1.0 if (h >> 1) & 1 else 1.0
                out[i, j] += s
        return out

    def _hash_images(self, paths: list[Path], dim: int) -> np.ndarray:
        rng = np.random.default_rng(12345)
        proj = rng.standard_normal((32 * 32, dim), dtype=np.float32)
        out = np.zeros((len(paths), dim), dtype=np.float32)
        for i, p in enumerate(paths):
            try:
                with Image.open(p) as im:
                    g = im.convert("L").resize((32, 32))
                    v = np.asarray(g, dtype=np.float32).reshape(-1) / 255.0
                out[i] = v @ proj
            except Exception:
                out[i] = 0.0
        return out
