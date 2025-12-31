from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Tuple

import numpy as np
from PIL import Image

from .utils import normalize_rows, tokenize

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class EmbedResult:
    ids: list[str]
    vecs: np.ndarray


class MultiModalEmbedder:
    """
    backend:
      - "hf_clip"   : force CLIPModel/CLIPProcessor
      - "hf_siglip" : force AutoModel/AutoProcessor (works for SigLIP)
      - "hf_auto" / "hf" / "hf_*": auto-detect by config.model_type where possible
      - anything else: hash fallback (debug mode)

    dim:
      - if <= 0 => infer automatically from the loaded model
    """

    def __init__(self, backend: str, model_name: str, device: str, batch_size: int, dim: int):
        self.backend = backend
        self.model_name = model_name
        self.device = self._resolve_device(device)
        self.batch_size = batch_size
        self.dim = int(dim) if dim is not None else 0
        if self.dim < 0:
            self.dim = 0

        self._hf: Optional[Tuple[Any, Any, Any, Any]] = None  # (torch, processor, model, tokenizer)
        self._st: Optional[Tuple[Any, Any]] = None  # (sentence_transformers module, model)
        self._use_dual_encoder: bool = False

        if backend.startswith("hf"):
            self._hf = self._init_hf()
        elif backend == "sentence_transformer":
            self._st = self._init_sentence_transformer()

        # If both HF and ST init failed, ensure we have a usable dim for hashing fallback
        if self._hf is None and self._st is None and self.dim == 0:
            self.dim = 256

    def _resolve_device(self, device: str) -> str:
        """Auto-detect available device if cuda is requested but not available."""
        if device == "cuda":
            try:
                import torch
                if torch.cuda.is_available():
                    log.info("embedder.device using device=cuda (CUDA available)")
                    return "cuda"
                else:
                    log.warning("embedder.device CUDA requested but not available, falling back to cpu")
                    return "cpu"
            except ImportError:
                log.warning("embedder.device torch not available, falling back to cpu")
                return "cpu"
        return device

    def _init_hf(self):
        try:
            import torch
            from transformers import AutoConfig  # type: ignore

            cfg = AutoConfig.from_pretrained(self.model_name)
            model_type = (getattr(cfg, "model_type", "") or "").lower()

            force_clip = self.backend in {"hf_clip"}
            force_siglip = self.backend in {"hf_siglip"}

            if force_clip or model_type == "clip" or "clip" in self.model_name.lower():
                from transformers import CLIPModel, CLIPProcessor, CLIPTokenizer  # type: ignore

                proc = self._from_pretrained_fast(CLIPProcessor, self.model_name)
                tokenizer = CLIPTokenizer.from_pretrained(self.model_name)
                model = CLIPModel.from_pretrained(self.model_name)
            else:
                # SigLIP and most other multimodal transformers work with Auto*
                from transformers import AutoProcessor, AutoModel, AutoTokenizer  # type: ignore

                proc = self._from_pretrained_fast(AutoProcessor, self.model_name)
                tokenizer = AutoTokenizer.from_pretrained(self.model_name)
                model = AutoModel.from_pretrained(self.model_name)

            model.to(self.device)
            model.eval()

            # Infer embedding dimension if dim unset or mismatched
            inferred_dim = self._infer_dim(torch, proc, model, tokenizer)
            if inferred_dim is not None:
                if self.dim in (0, None) or self.dim != inferred_dim:
                    log.info("embedder.dim set dim=%s (was=%s) model=%s backend=%s",
                             inferred_dim, self.dim, self.model_name, self.backend)
                    self.dim = int(inferred_dim)

            return torch, proc, model, tokenizer

        except Exception as e:
            log.warning("embedder.hf_init_failed backend=%s model=%s err=%s", self.backend, self.model_name, e)
            return None

    def _init_sentence_transformer(self):
        """Initialize SentenceTransformer model with dual encoder auto-detection."""
        try:
            import sentence_transformers as st_lib

            model = st_lib.SentenceTransformer(self.model_name, device=self.device)

            # Auto-detect dual encoder support
            self._use_dual_encoder = (
                hasattr(model, 'encode_query') and
                hasattr(model, 'encode_document')
            )

            log.info(
                "embedder.st_init backend=%s model=%s dual_encoder=%s",
                self.backend, self.model_name, self._use_dual_encoder
            )

            # Infer embedding dimension
            test_emb = model.encode(["test"], convert_to_numpy=True, show_progress_bar=False)
            inferred_dim = test_emb.shape[1]

            if self.dim in (0, None) or self.dim != inferred_dim:
                log.info(
                    "embedder.dim set dim=%s (was=%s) model=%s backend=%s",
                    inferred_dim, self.dim, self.model_name, self.backend
                )
                self.dim = int(inferred_dim)

            return st_lib, model

        except Exception as e:
            log.warning(
                "embedder.st_init_failed backend=%s model=%s err=%s",
                self.backend, self.model_name, e
            )
            return None

    @staticmethod
    def _from_pretrained_fast(cls, model_name: str):
        # Many processors now support use_fast; some don't. Try fast first.
        try:
            return cls.from_pretrained(model_name, use_fast=True)
        except TypeError:
            return cls.from_pretrained(model_name)

    def _infer_dim(self, torch, proc, model, tokenizer) -> Optional[int]:
        """
        Run a tiny forward pass to determine output dimension robustly.
        """
        # Try text first using tokenizer directly
        try:
            inputs = tokenizer(["test"], return_tensors="pt", padding=True, truncation=True)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            with torch.no_grad():
                if hasattr(model, "get_text_features"):
                    x = model.get_text_features(**inputs)
                else:
                    out = model(**inputs)
                    x = out.last_hidden_state[:, 0, :]
            return int(x.shape[-1])
        except Exception:
            pass

        # Fallback: try image
        try:
            dummy = Image.new("RGB", (224, 224), color=(0, 0, 0))
            inputs = proc(images=[dummy], return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            with torch.no_grad():
                if hasattr(model, "get_image_features"):
                    x = model.get_image_features(**inputs)
                else:
                    out = model(**inputs)
                    x = out.last_hidden_state[:, 0, :]
            return int(x.shape[-1])
        except Exception:
            return None

    def embed_text(self, ids: list[str], texts: list[str]) -> EmbedResult:
        if self._st is not None:
            st_lib, model = self._st
            all_vecs = []

            for i in range(0, len(texts), self.batch_size):
                batch = texts[i : i + self.batch_size]

                # Use encode_document for corpus text (dual encoder models)
                if hasattr(model, 'encode_document'):
                    vecs = model.encode_document(
                        batch,
                        convert_to_numpy=True,
                        show_progress_bar=False,
                        normalize_embeddings=True
                    )
                else:
                    vecs = model.encode(
                        batch,
                        convert_to_numpy=True,
                        show_progress_bar=False,
                        normalize_embeddings=True
                    )
                all_vecs.append(vecs.astype(np.float32))

            vecs = np.vstack(all_vecs) if all_vecs else np.zeros((0, self.dim), dtype=np.float32)
            return EmbedResult(ids=ids, vecs=vecs)

        if self._hf is None:
            vecs = self._hash_text(texts, self.dim)
            return EmbedResult(ids=ids, vecs=normalize_rows(vecs))

        torch, proc, model, tokenizer = self._hf
        all_vecs: list[np.ndarray] = []
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i : i + self.batch_size]
            # Use tokenizer directly for text to avoid processor kwargs issues
            inputs = tokenizer(batch, return_tensors="pt", padding=True, truncation=True)
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
        if self._st is not None:
            st_lib, model = self._st
            all_vecs = []

            for i in range(0, len(image_paths), self.batch_size):
                batch_paths = image_paths[i : i + self.batch_size]
                imgs = [Image.open(p).convert("RGB") for p in batch_paths]

                vecs = model.encode(
                    imgs,
                    convert_to_numpy=True,
                    show_progress_bar=False,
                    normalize_embeddings=True
                )

                for im in imgs:
                    im.close()

                all_vecs.append(vecs.astype(np.float32))

            vecs = np.vstack(all_vecs) if all_vecs else np.zeros((0, self.dim), dtype=np.float32)
            return EmbedResult(ids=ids, vecs=vecs)

        if self._hf is None:
            vecs = self._hash_images(image_paths, self.dim)
            return EmbedResult(ids=ids, vecs=normalize_rows(vecs))

        torch, proc, model, tokenizer = self._hf
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

    def embed_query(self, query_text: str) -> np.ndarray:
        """Embed a query using dual encoder if available."""
        if self._st is not None:
            st_lib, model = self._st

            # Use encode_query for dual encoder models
            if self._use_dual_encoder and hasattr(model, 'encode_query'):
                vec = model.encode_query(
                    [query_text],
                    convert_to_numpy=True,
                    show_progress_bar=False,
                    normalize_embeddings=True
                )
            else:
                vec = model.encode(
                    [query_text],
                    convert_to_numpy=True,
                    show_progress_bar=False,
                    normalize_embeddings=True
                )
            return vec[0].astype(np.float32)

        # Fall back to embed_text for HF models
        result = self.embed_text(["q"], [query_text])
        return result.vecs[0]
