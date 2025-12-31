"""
Simplified layout-aware PDF extractor with clear visibility into extraction.
"""

from __future__ import annotations

import base64
import io
import json
import logging
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any, List, Optional, Tuple

from pdf2image import convert_from_path
from PIL import Image

from .openai_client import OpenAIChatClient
from .schema import DocumentArtifact, Element, PageArtifact
from .utils import ensure_dir, stable_id, write_json

log = logging.getLogger(__name__)

UNIT_TAG_PATTERN = re.compile(
    r"<unit\s+type\s*=\s*['\"](?P<type>[^'\"]+)['\"]\s*>(?P<body>.*?)</unit>",
    re.I | re.S,
)

VISION_PROMPT = """Extract ALL content from this page with SEMANTIC UNIT TAGGING.

CRITICAL INSTRUCTIONS:
1. Wrap each semantic unit with XML tags: <unit type="TYPE">CONTENT</unit>
2. Unit types:
   - "heading": Section/subsection headers
   - "paragraph": Regular text paragraphs
   - "equation": Mathematical equations (preserve $$ notation)
   - "table": Tables (use Markdown table format)
   - "figure": Figure captions and descriptions
   - "list": Bulleted or numbered lists
   - "code": Code blocks or algorithms
   - "abstract": Abstract section
   - "reference": Citations/references section

3. Preserve ALL mathematical notation using LaTeX
4. DO NOT include page numbers, headers, or footers
5. If content is unclear, use [UNCLEAR: approximate text]

Output MUST be only <unit ...> blocks, no extra commentary."""

TYPE_MAPPING = {
    "table": "table_text",
    "figure": "figure",
    "equation": "equation",
    "heading": "heading",
    "list": "list",
    "code": "code",
    "abstract": "abstract",
    "reference": "reference",
}


class PdfLayoutExtractor:
    def __init__(self, cfg: Any, force_recompute: bool = False):
        self.cfg = cfg
        self.client = OpenAIChatClient(timeout_s=cfg.extract.timeout_s)
        self.model = cfg.extract.vision_model
        self.temperature = cfg.openai.temperature
        self.dpi = cfg.extract.dpi
        self.max_workers = cfg.extract.max_workers
        self.max_retries = cfg.extract.max_retries
        self.retry_wait_s = cfg.extract.retry_wait_s
        self.batch_size = cfg.extract.batch_size
        self.force_recompute = force_recompute

    def extract(self, pdf_path: str | Path, out_dir: str | Path, doc_id: Optional[str] = None) -> DocumentArtifact:
        pdf_path = Path(pdf_path).resolve()
        out_dir = Path(out_dir)
        ensure_dir(out_dir)

        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF not found: {pdf_path}")

        extraction_dir = ensure_dir(out_dir / "extractions")
        images_dir = ensure_dir(extraction_dir / f"{pdf_path.stem}_images")

        # Check cache
        cache_file = extraction_dir / f"{pdf_path.stem}_extraction.json"
        use_cache = not self.force_recompute
        
        doc_id = doc_id or stable_id(str(pdf_path), str(pdf_path.stat().st_size))

        if use_cache and cache_file.exists():
            log.info("extract.cache_hit path=%s", cache_file)
            return self._load_from_cache(cache_file, doc_id, pdf_path)

        # Fresh extraction
        log.info("extract.start pdf=%s pages=?", pdf_path.name)
        images = convert_from_path(str(pdf_path), dpi=self.dpi)
        
        # Save page images + get dimensions
        page_images = []
        page_sizes = []
        for i, img in enumerate(images):
            page_num = i + 1
            img_path = images_dir / f"page_{page_num:03d}.png"
            img.save(str(img_path), "PNG")
            page_images.append(img_path)
            page_sizes.append((img.width, img.height))
            
        log.info("extract.rendered pages=%d dpi=%d", len(images), self.dpi)

        # Vision extraction
        page_texts = self._extract_with_vision(images)
        
        # Parse units and log stats
        total_units = 0
        type_counts = {}
        for i, text in enumerate(page_texts):
            units = self._parse_units(text)
            total_units += len(units)
            for utype, _ in units:
                type_counts[utype] = type_counts.get(utype, 0) + 1
            log.info("extract.page=%d units=%d types=%s", 
                     i + 1, len(units), [ut for ut, _ in units[:5]])
        
        log.info("extract.done total_units=%d type_counts=%s", total_units, type_counts)

        # Save extraction data
        extraction_data = {
            "pdf_path": str(pdf_path),
            "num_pages": len(images),
            "dpi": self.dpi,
            "page_texts": page_texts,
            "page_sizes": page_sizes,
            "image_paths": [str(p) for p in page_images],
            "unit_count": total_units,
            "type_counts": type_counts,
            "extracted_at": datetime.now().isoformat(),
            "model": self.model,
        }
        write_json(cache_file, extraction_data)
        
        # Save markdown
        markdown_path = extraction_dir / f"{pdf_path.stem}_extracted.md"
        self._save_markdown(page_texts, markdown_path)
        log.info("extract.saved json=%s markdown=%s", cache_file, markdown_path)

        return self._build_artifact(doc_id, pdf_path, page_texts, page_images, page_sizes)

    def _extract_with_vision(self, images: List[Any]) -> List[str]:
        """Extract text from images using vision API with batching."""
        batches = [images[i:i + self.batch_size] for i in range(0, len(images), self.batch_size)]
        log.info("vision.extract batches=%d batch_size=%d workers=%d",
                 len(batches), self.batch_size, self.max_workers)

        all_results = [None] * len(images)

        def process_batch(batch_idx: int, batch_images: List[Any]) -> Tuple[int, List[str]]:
            results = []
            for img in batch_images:
                # Convert to base64 JPEG
                buffer = io.BytesIO()
                img.save(buffer, format="JPEG", quality=85)
                b64 = base64.b64encode(buffer.getvalue()).decode()
                
                # Call vision API with retry
                for attempt in range(self.max_retries + 1):
                    try:
                        response = self.client.chat_vision_b64jpeg(
                            model=self.model,
                            prompt=VISION_PROMPT,
                            b64jpeg=b64,
                            temperature=self.temperature,
                            detail="high",
                            max_tokens=3500,
                        )
                        text = response.text.strip()
                        if "<unit" not in text:
                            text = f"<unit type='paragraph'>{text}</unit>"
                        results.append(text)
                        break
                    except Exception as e:
                        if attempt >= self.max_retries:
                            log.warning("vision.failed attempt=%d err=%s", attempt, e)
                            results.append(f"<unit type='error'>[ERROR: {e}]</unit>")
                            break
                        time.sleep(self.retry_wait_s)
            
            return batch_idx, results

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {executor.submit(process_batch, i, batch): i 
                      for i, batch in enumerate(batches)}
            
            for future in as_completed(futures):
                batch_idx, texts = future.result()
                start_idx = batch_idx * self.batch_size
                for offset, text in enumerate(texts):
                    if start_idx + offset < len(all_results):
                        all_results[start_idx + offset] = text

        return [r or "<unit type='error'>[ERROR: missing]</unit>" for r in all_results]

    def _parse_units(self, page_text: str) -> List[Tuple[str, str]]:
        """Parse <unit type="...">...</unit> tags."""
        units = []
        for match in UNIT_TAG_PATTERN.finditer(page_text):
            unit_type = match.group("type").strip().lower()
            body = match.group("body").strip()
            if unit_type and body:
                units.append((unit_type, body))
        return units

    def _save_markdown(self, page_texts: List[str], output_path: Path) -> None:
        """Save extracted content as markdown for easy inspection."""
        lines = ["# Extracted Document Content\n\n"]
        
        for i, text in enumerate(page_texts):
            page_num = i + 1
            units = self._parse_units(text)
            
            lines.append(f"## Page {page_num}\n\n")
            lines.append(f"**Units extracted:** {len(units)}\n\n")
            
            for unit_type, body in units:
                lines.append(f"### `{unit_type}`\n\n")
                preview = body[:300] + "..." if len(body) > 300 else body
                lines.append(f"{preview}\n\n")
            
            lines.append("---\n\n")
        
        output_path.write_text("".join(lines), encoding="utf-8")

    def _load_from_cache(self, cache_file: Path, doc_id: str, pdf_path: Path) -> DocumentArtifact:
        """Load extraction from cached JSON."""
        data = json.loads(cache_file.read_text(encoding="utf-8"))
        page_texts = data.get("page_texts", [])
        page_images = [Path(p) for p in data.get("image_paths", [])]
        page_sizes = data.get("page_sizes", None)
        
        return self._build_artifact(doc_id, pdf_path, page_texts, page_images, page_sizes)

    def _build_artifact(
        self,
        doc_id: str,
        pdf_path: Path,
        page_texts: List[str],
        page_images: List[Path],
        page_sizes: Optional[List[Tuple[int, int]]],
    ) -> DocumentArtifact:
        """Build DocumentArtifact with proper bboxes."""
        pages = []
        elements = []

        # Get page dimensions (from cache or read from images)
        sizes = page_sizes or []
        if not sizes or len(sizes) != len(page_images):
            sizes = []
            for img_path in page_images:
                try:
                    with Image.open(img_path) as img:
                        sizes.append((img.width, img.height))
                except Exception:
                    sizes.append((0, 0))
                    log.warning("build.missing_dimensions path=%s", img_path)

        for i, img_path in enumerate(page_images):
            page_num = i + 1
            width, height = sizes[i] if i < len(sizes) else (0, 0)

            # Create page artifact
            pages.append(PageArtifact(
                page=page_num,
                width=width,
                height=height,
                page_image_path=str(img_path),
            ))

            # Full-page bbox
            bbox = (0.0, 0.0, float(width), float(height))

            # Page image element
            elements.append(Element(
                id=stable_id(doc_id, page_num, "page_image"),
                page=page_num,
                type="page_image",
                text="",
                bbox=bbox,
                image_path=str(img_path),
            ))

            # Text elements from units
            units = self._parse_units(page_texts[i] if i < len(page_texts) else "")
            for j, (unit_type, body) in enumerate(units):
                element_type = TYPE_MAPPING.get(unit_type, "text")
                elements.append(Element(
                    id=stable_id(doc_id, page_num, element_type, j),
                    page=page_num,
                    type=element_type,
                    text=body,
                    bbox=bbox,
                    image_path=None,
                ))

        return DocumentArtifact(
            doc_id=doc_id,
            pdf_path=str(pdf_path),
            pages=pages,
            elements=elements,
        )