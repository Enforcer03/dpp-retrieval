"""
Layout-aware PDF extractor using OpenAI Vision to produce <unit type="..."> blocks per page,
plus page image artifacts so visuals (tables/figures) remain retrievable.

Key features:
- Renders PDF -> page images (pdf2image)
- Calls OpenAI Vision on each page image to extract ALL content with <unit> tags
- Uses bounded concurrency + retries with exponential backoff & jitter
- Saves extraction JSON and page images under <out_dir>/extractions/
- Emits:
  - PageArtifact(page, width, height, page_image_path)
  - Element(type="page_image", image_path=..., bbox=(0,0,w,h))
  - Element(type mapped from unit tags, text=unit body)

NOTE:
- Vision transcription does not provide bboxes for individual units. Unit elements get a full-page bbox.
"""

from __future__ import annotations

import base64
import io
import json
import os
import random
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any, List, Optional, Tuple

from pdf2image import convert_from_path
from pdf2image.exceptions import PDFInfoNotInstalledError

from .openai_client import OpenAIChatClient
from .schema import DocumentArtifact, Element, PageArtifact
from .utils import ensure_dir, stable_id, write_json


_UNIT_RE = re.compile(
    r"<unit\s+type\s*=\s*['\"](?P<type>[^'\"]+)['\"]\s*>(?P<body>.*?)</unit>",
    re.I | re.S,
)

LAYOUT_AWARE_PROMPT = """Extract ALL content from this page with SEMANTIC UNIT TAGGING.

CRITICAL INSTRUCTIONS:
1. Wrap each semantic unit with XML tags: <unit type="TYPE">CONTENT</unit>
2. Unit types:
   - "heading": Section/subsection headers (e.g., "# Introduction")
   - "paragraph": Regular text paragraphs
   - "equation": Mathematical equations (preserve $$ notation)
   - "table": Tables (use Markdown table format)
   - "figure": Figure captions and descriptions (include caption text)
   - "list": Bulleted or numbered lists
   - "code": Code blocks or algorithms
   - "abstract": Abstract section
   - "reference": Citations/references section

3. Preserve ALL mathematical notation using LaTeX:
   - Display equations: $$equation$$
   - Inline math: $expression$

4. DO NOT include page numbers, headers, or footers
5. Maintain document structure and hierarchy
6. If content is unclear, use [UNCLEAR: approximate text]

Output MUST be only <unit ...> blocks, no extra commentary.

Now extract the content from the provided page:"""


def _cfg_get(cfg: Any, dotted_key: str, default: Any) -> Any:
    """
    Robust config getter:
    - If cfg has .get("a.b.c") -> uses it
    - Else tries attribute traversal cfg.a.b.c
    """
    if cfg is None:
        return default

    if hasattr(cfg, "get") and callable(getattr(cfg, "get")):
        try:
            return cfg.get(dotted_key, default)
        except Exception:
            pass

    cur = cfg
    for part in dotted_key.split("."):
        if hasattr(cur, part):
            cur = getattr(cur, part)
        else:
            return default
    return cur


def _pil_to_b64jpeg(pil_img, max_side: int = 1400, quality: int = 85) -> str:
    w, h = pil_img.size
    scale = min(1.0, float(max_side) / max(w, h))
    if scale < 1.0:
        pil_img = pil_img.resize((int(w * scale), int(h * scale)))

    buff = io.BytesIO()
    pil_img.save(buff, format="JPEG", quality=quality, optimize=True)
    return base64.b64encode(buff.getvalue()).decode("utf-8")


def _parse_units(page_text: str) -> List[Tuple[str, str]]:
    out: List[Tuple[str, str]] = []
    if not page_text:
        return out
    for m in _UNIT_RE.finditer(page_text):
        t = (m.group("type") or "").strip().lower()
        b = (m.group("body") or "").strip()
        if t and b:
            out.append((t, b))
    return out


def _unit_type_to_element_type(unit_type: str) -> str:
    ut = (unit_type or "").strip().lower()
    if ut == "table":
        return "table_text"
    if ut == "figure":
        return "figure"
    if ut == "equation":
        return "equation"
    if ut == "heading":
        return "heading"
    if ut == "list":
        return "list"
    if ut == "code":
        return "code"
    if ut == "abstract":
        return "abstract"
    if ut == "reference":
        return "reference"
    # paragraph + unknown fall back to "text"
    return "text"


class PdfLayoutExtractor:
    """
    New extractor signature:
      PdfLayoutExtractor(cfg)

    Compatible extract signatures:
      extract(pdf_path, out_dir=..., doc_id=optional)

    The rest of your pipeline expects DocumentArtifact with:
      - pages: list[PageArtifact]
      - elements: list[Element]
      - doc_id: str
    """

    def __init__(self, cfg: Any):
        self.cfg = cfg

        # OpenAI client (uses .env OPENAI_API_KEY inside OpenAIChatClient)
        self.client = OpenAIChatClient(timeout_s=int(_cfg_get(cfg, "openai.timeout_s", 60)))

        # Use a vision-capable model if you set extract.vision_model; else fall back.
        self.model = str(_cfg_get(cfg, "extract.vision_model", _cfg_get(cfg, "openai.model", "gpt-4o")))
        self.temperature = float(_cfg_get(cfg, "openai.temperature", 0.0))

        # Rendering/extraction settings
        self.dpi = int(_cfg_get(cfg, "extract.dpi", 200))
        self.keep_page_renders = bool(_cfg_get(cfg, "extract.keep_page_renders", True))
        self.use_layout_aware = bool(_cfg_get(cfg, "extract.use_layout_aware", True))

        # Concurrency/retry
        self.max_workers = int(_cfg_get(cfg, "extract.max_workers", 4))
        self.max_retries = int(_cfg_get(cfg, "extract.max_retries", 6))
        self.backoff_base_s = float(_cfg_get(cfg, "extract.backoff_base_s", 1.0))
        self.backoff_max_s = float(_cfg_get(cfg, "extract.backoff_max_s", 20.0))

        # Image encoding
        self.max_side = int(_cfg_get(cfg, "extract.vision_max_side", 1400))
        self.jpeg_quality = int(_cfg_get(cfg, "extract.vision_jpeg_quality", 85))

        self.prompt = LAYOUT_AWARE_PROMPT

    # -------------------------------------------------------------------------
    # Public API
    # -------------------------------------------------------------------------
    def extract(self, pdf_path: str | Path, out_dir: str | Path, doc_id: Optional[str] = None) -> DocumentArtifact:
        pdf_path = Path(pdf_path).resolve()
        out_dir = Path(out_dir)
        ensure_dir(out_dir)

        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF not found: {pdf_path}")

        # Where we save images + extraction JSON
        extraction_dir = ensure_dir(out_dir / "extractions")

        pdf_basename = pdf_path.stem
        mod_time = pdf_path.stat().st_mtime
        timestamp = datetime.fromtimestamp(mod_time).strftime("%Y%m%d_%H%M%S")

        extraction_file = extraction_dir / f"{pdf_basename}_extraction_{timestamp}.json"
        images_dir = extraction_dir / f"{pdf_basename}_images_{timestamp}"
        ensure_dir(images_dir)

        # Cache controls (optional)
        use_cache = bool(_cfg_get(self.cfg, "cache.use_cached_extraction", True))
        force_recompute = bool(_cfg_get(self.cfg, "cache.force_recompute", False))
        if force_recompute:
            use_cache = False

        # Determine doc_id for stable element ids
        doc_id_final = doc_id or stable_id(str(pdf_path), str(int(pdf_path.stat().st_size)))

        if use_cache and extraction_file.exists():
            data = json.loads(extraction_file.read_text(encoding="utf-8"))
            page_texts = data.get("page_texts", [])
            page_image_paths = [Path(p) for p in data.get("image_paths", [])]
            # Best-effort widths/heights from stored metadata
            page_sizes = data.get("page_sizes", None)  # list of [w,h]
            return self._build_document_artifact(
                doc_id=doc_id_final,
                pdf_path=pdf_path,
                page_texts=page_texts,
                page_image_paths=page_image_paths,
                page_sizes=page_sizes,
            )

        # Render PDF pages
        try:
            images = convert_from_path(str(pdf_path), dpi=self.dpi)
        except PDFInfoNotInstalledError as e:
            raise RuntimeError(
                "Poppler not installed. Install poppler-utils (Linux) / poppler (macOS/Windows)."
            ) from e

        # Save page images + prepare b64jpeg
        page_image_paths: List[Path] = []
        page_sizes: List[Tuple[int, int]] = []
        page_b64: List[Tuple[int, str]] = []

        for i, img in enumerate(images):
            pno = i + 1
            w, h = img.size
            page_sizes.append((int(w), int(h)))

            # Always save PNG page images if keep_page_renders
            img_path = images_dir / f"page_{pno:03d}.png"
            if self.keep_page_renders:
                img.save(str(img_path), "PNG")
                page_image_paths.append(img_path)
            else:
                # still append a path placeholder to keep alignment; but we prefer saving renders
                img.save(str(img_path), "PNG")
                page_image_paths.append(img_path)

            b64 = _pil_to_b64jpeg(img, max_side=self.max_side, quality=self.jpeg_quality)
            page_b64.append((pno, b64))

        # Vision extraction: concurrent + retries
        page_texts = self._extract_pages_concurrent(page_b64)

        # Save extraction JSON for reuse/debugging
        data = {
            "pdf_path": str(pdf_path),
            "pdf_basename": pdf_basename,
            "extraction_timestamp": timestamp,
            "pdf_modified_time": mod_time,
            "num_pages": len(images),
            "dpi": self.dpi,
            "page_texts": page_texts,
            "full_text": "\n\n".join(page_texts),
            "unit_count": sum(t.count("<unit") for t in page_texts),
            "char_count": sum(len(t) for t in page_texts),
            "image_directory": str(images_dir),
            "image_paths": [str(p) for p in page_image_paths],
            "page_sizes": [[w, h] for (w, h) in page_sizes],
            "model": self.model,
        }
        write_json(extraction_file, data)

        return self._build_document_artifact(
            doc_id=doc_id_final,
            pdf_path=pdf_path,
            page_texts=page_texts,
            page_image_paths=page_image_paths,
            page_sizes=[[w, h] for (w, h) in page_sizes],
        )

    # -------------------------------------------------------------------------
    # Concurrency + retries
    # -------------------------------------------------------------------------
    def _is_retryable_error(self, e: Exception) -> bool:
        msg = (str(e) or "").lower()

        if "rate limit" in msg or "too many requests" in msg or "429" in msg:
            return True
        if "timeout" in msg or "timed out" in msg:
            return True
        if "temporarily unavailable" in msg or "service unavailable" in msg or "503" in msg:
            return True
        if "internal server error" in msg or "server error" in msg or "500" in msg or "502" in msg:
            return True
        if "connection" in msg or "connection reset" in msg or "connection aborted" in msg:
            return True
        if "gateway" in msg or "bad gateway" in msg or "504" in msg:
            return True

        status = getattr(e, "status_code", None)
        if isinstance(status, int) and status in (408, 409, 429, 500, 502, 503, 504):
            return True

        code = getattr(e, "code", None)
        if isinstance(code, str) and code.lower() in ("rate_limit_exceeded", "timeout", "server_error"):
            return True

        return False

    def _call_vision_with_retry(self, page_num: int, b64jpeg: str) -> str:
        prompt = self.prompt if self.use_layout_aware else (
            "Transcribe this page into Markdown. Preserve headers, tables, and math equations ($$). "
            "Do not include page numbers or footers."
        )

        last_err: Exception | None = None
        for attempt in range(self.max_retries + 1):
            try:
                res = self.client.chat_vision_b64jpeg(
                    model=self.model,
                    prompt=prompt,
                    b64jpeg=b64jpeg,
                    temperature=self.temperature,
                    detail="high",
                    max_tokens=2200,
                )
                txt = (res.text or "").strip()
                if "<unit" not in txt:
                    txt = f"<unit type=\"paragraph\">{txt}</unit>"
                return txt
            except Exception as e:
                last_err = e
                if attempt >= self.max_retries or not self._is_retryable_error(e):
                    break

                # exponential backoff with jitter
                sleep_s = min(self.backoff_max_s, self.backoff_base_s * (2 ** attempt))
                sleep_s = sleep_s * (0.75 + 0.5 * random.random())  # jitter [0.75x, 1.25x]
                time.sleep(sleep_s)

        return f"<unit type='error'>[ERROR: Page {page_num} extraction failed: {last_err}]</unit>"

    def _extract_pages_concurrent(self, page_b64: List[Tuple[int, str]]) -> List[str]:
        """
        Concurrent extraction preserving input order.
        page_b64: [(page_num, b64jpeg), ...]
        returns:  [page_texts...] aligned to input order
        """
        if not page_b64:
            return []

        max_workers = max(1, int(self.max_workers))
        results: List[Optional[str]] = [None] * len(page_b64)

        def _job(idx: int, pn: int, b64: str) -> Tuple[int, str]:
            return idx, self._call_vision_with_retry(pn, b64)

        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            futs = {ex.submit(_job, idx, pn, b64): idx for idx, (pn, b64) in enumerate(page_b64)}
            for fut in as_completed(futs):
                idx, text = fut.result()
                results[idx] = text

        return [r if r is not None else "<unit type='error'>[ERROR: missing result]</unit>" for r in results]

    # -------------------------------------------------------------------------
    # Build schema artifacts
    # -------------------------------------------------------------------------
    def _build_document_artifact(
        self,
        doc_id: str,
        pdf_path: Path,
        page_texts: List[str],
        page_image_paths: List[Path],
        page_sizes: Any = None,
    ) -> DocumentArtifact:
        pages: List[PageArtifact] = []
        elements: List[Element] = []

        # page_sizes may come as list[list[int,int]] or list[tuple[int,int]] or None
        sizes: List[Tuple[int, int]] = []
        if isinstance(page_sizes, list) and page_sizes:
            for s in page_sizes:
                try:
                    w = int(s[0])
                    h = int(s[1])
                    sizes.append((w, h))
                except Exception:
                    sizes.append((0, 0))
        else:
            # best-effort: infer from images on disk
            for p in page_image_paths:
                try:
                    from PIL import Image
                    with Image.open(p) as im:
                        sizes.append((int(im.size[0]), int(im.size[1])))
                except Exception:
                    sizes.append((0, 0))

        for i, img_path in enumerate(page_image_paths):
            pno = i + 1
            w, h = sizes[i] if i < len(sizes) else (0, 0)

            # FIX: PageArtifact requires width & height (and possibly others)
            pages.append(
                PageArtifact(
                    page=pno,
                    width=int(w),
                    height=int(h),
                    page_image_path=str(img_path),
                )
            )

            # Always emit page_image element
            bbox = (0.0, 0.0, float(w) if w else 1.0, float(h) if h else 1.0)
            elements.append(
                Element(
                    id=stable_id(doc_id, pno, "page_image"),
                    page=pno,
                    type="page_image",
                    text="",
                    bbox=bbox,
                    image_path=str(img_path),
                )
            )

            # Emit unit-derived text elements
            units = _parse_units(page_texts[i] if i < len(page_texts) else "")
            for j, (utype, body) in enumerate(units):
                etype = _unit_type_to_element_type(utype)
                elements.append(
                    Element(
                        id=stable_id(doc_id, pno, etype, j),
                        page=pno,
                        type=etype,
                        text=body,
                        bbox=bbox,        # full-page bbox (no per-unit bboxes available)
                        image_path=None,  # unit text has no specific crop image
                    )
                )

        return DocumentArtifact(
            doc_id=doc_id,
            pdf_path=str(pdf_path),
            pages=pages,
            elements=elements,
        )
