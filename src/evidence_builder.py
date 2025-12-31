"""
Simplified evidence builder with transparent unit creation.
"""
from __future__ import annotations

import logging
from pathlib import Path

from .schema import DocumentArtifact, EvidenceUnit
from .utils import stable_id

log = logging.getLogger(__name__)


class EvidenceBuilder:
    def __init__(self, max_text_chunk_tokens: int, caption_search_px: int, stopwords: set[str]):
        self.max_text_chunk_tokens = max_text_chunk_tokens
        self.caption_search_px = caption_search_px
        self.stopwords = stopwords

    def build(self, doc: DocumentArtifact, token_counter) -> list[EvidenceUnit]:
        """Convert document elements into evidence units."""
        units = []
        
        # Group elements by page
        by_page = {}
        for el in doc.elements:
            page = el.page
            if page not in by_page:
                by_page[page] = []
            by_page[page].append(el)
        
        # Process each page
        for page_num in sorted(by_page.keys()):
            elements = sorted(by_page[page_num], key=lambda e: (e.bbox[1], e.bbox[0]))
            
            # Separate by type
            page_images = [e for e in elements if e.type == "page_image"]
            tables = [e for e in elements if e.type == "table_text"]
            figures = [e for e in elements if e.type == "figure"]
            text_elements = [e for e in elements if e.type not in {"page_image", "table_text", "figure"}]
            
            log.debug("page=%d elements: page_img=%d tables=%d figures=%d text=%d", 
                     page_num, len(page_images), len(tables), len(figures), len(text_elements))
            
            # Create units for visuals (tables/figures) - one unit per element
            for el in tables:
                units.append(EvidenceUnit(
                    id=stable_id(doc.doc_id, page_num, "table_text", el.id),
                    page=page_num,
                    bbox=el.bbox,
                    type="table_text",
                    retrieval_text=el.text or "",
                    context_text=el.text or "",
                    image_paths=[Path(el.image_path)] if el.image_path else [],
                    source_element_ids=[el.id],
                ))
            
            for el in figures:
                units.append(EvidenceUnit(
                    id=stable_id(doc.doc_id, page_num, "figure", el.id),
                    page=page_num,
                    bbox=el.bbox,
                    type="figure",
                    retrieval_text=el.text or "",
                    context_text=el.text or "",
                    image_paths=[Path(el.image_path)] if el.image_path else [],
                    source_element_ids=[el.id],
                ))
            
            # Create units for text elements - one unit per element (no chunking merge)
            for el in text_elements:
                if not el.text or not el.text.strip():
                    continue
                
                # Each element becomes its own unit
                units.append(EvidenceUnit(
                    id=stable_id(doc.doc_id, page_num, el.type, el.id),
                    page=page_num,
                    bbox=el.bbox,
                    type=el.type,
                    retrieval_text=el.text,
                    context_text=el.text,
                    image_paths=[],
                    source_element_ids=[el.id],
                ))
            
            # Add page image unit
            for el in page_images:
                units.append(EvidenceUnit(
                    id=stable_id(doc.doc_id, page_num, "page_image", el.id),
                    page=page_num,
                    bbox=el.bbox,
                    type="page_image",
                    retrieval_text="",
                    context_text="",
                    image_paths=[Path(el.image_path)] if el.image_path else [],
                    source_element_ids=[el.id],
                ))
        
        log.info("evidence.units doc_id=%s total=%d", doc.doc_id, len(units))
        
        # Count by type
        type_counts = {}
        for u in units:
            type_counts[u.type] = type_counts.get(u.type, 0) + 1
        log.info("evidence.type_counts=%s", type_counts)
        
        return units