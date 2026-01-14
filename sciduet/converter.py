"""
SciDuet to DocumentArtifact conversion utilities.

This module handles:
- ID normalization (gem_id, slide_id, paper_id)
- Requirements lookup from metadata
- Converting SciDuet samples to DocumentArtifact format
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Tuple

from src.schema import DocumentArtifact, Element, PageArtifact
from src.utils import stable_id


# ============================================================================
# ID Normalization & Requirements Lookup
# ============================================================================

def _str(x: Any) -> str:
    """Safe string conversion."""
    return str(x) if x is not None else ""


def normalize_gem_id(gem_id: str) -> str:
    """
    Fix SciDuet gem_id bug where ID becomes duplicated with '#slide-' prefix.
    
    Some SciDuet variants store gem_id as:
        "GEM-...#paper-954#slide-" + "GEM-...#paper-954#slide-0"
    We keep only the suffix full GEM id.
    """
    gem_id = (gem_id or "").strip()
    if "#slide-" in gem_id:
        suffix = gem_id.split("#slide-", 1)[1]
        if suffix.startswith("GEM-"):
            return suffix
    return gem_id


def extract_sample_ids(sample: Dict[str, Any]) -> Tuple[str, str, str, str]:
    """
    Extract and normalize IDs from a SciDuet sample.
    
    Returns:
        Tuple of (paper_id, gem_id, slide_id, slide_num)
    """
    paper_id = _str(
        sample.get("paper_id") or 
        sample.get("paper") or 
        sample.get("id") or 
        "unknown"
    )
    gem_id = normalize_gem_id(_str(sample.get("gem_id")))
    slide_id = _str(sample.get("slide_id"))

    # Fall back to slide_id if gem_id is empty
    if not gem_id and slide_id:
        gem_id = slide_id

    # Extract slide number
    id_for_slide = slide_id or gem_id
    slide_num = id_for_slide.split("#slide-")[-1] if "#slide-" in id_for_slide else "0"
    
    return paper_id, gem_id, slide_id, slide_num


def lookup_requirements(
    requirements_map: Dict[str, List[Dict[str, Any]]],
    paper_id: str,
    gem_id: str,
    slide_id: str,
) -> Tuple[str, List[Dict[str, Any]]]:
    """
    Lookup requirements with fallback key candidates.
    
    Tries multiple key formats since metadata format varies across pipelines.
    
    Returns:
        Tuple of (matched_key, requirements_list)
    """
    candidates = [
        f"{paper_id}_{slide_id}" if slide_id else "",
        f"{paper_id}_{gem_id}" if gem_id else "",
        slide_id,
        gem_id,
    ]
    for key in candidates:
        if key and key in requirements_map:
            return key, requirements_map[key]
    
    # No match - return empty list with constructed key
    return f"{paper_id}_{slide_id or gem_id}".strip("_"), []


# ============================================================================
# SciDuet to DocumentArtifact Converter
# ============================================================================

class SciDuetConverter:
    """
    Convert SciDuet paper data to DocumentArtifact format.
    
    Strategy:
    - Each section header defines a "page"
    - Content under each section becomes text elements
    - No actual PDF or images (page_image_path=None)
    - Synthetic bounding boxes for layout
    """

    def __init__(self, max_section_length: int = 4000, target_chunk_size: int = 1200):
        """
        Args:
            max_section_length: Max characters per text element before splitting
            target_chunk_size: Target size when merging small chunks
        """
        self.max_section_length = max_section_length
        self.target_chunk_size = target_chunk_size

    def convert(self, sample: Dict[str, Any]) -> DocumentArtifact:
        """
        Convert SciDuet sample to DocumentArtifact.
        
        Args:
            sample: SciDuet dataset sample with paper_content, paper_headers, etc.
            
        Returns:
            DocumentArtifact with synthetic pages and elements
        """
        paper_id = str(sample.get("paper_id", "unknown"))

        content_chunks = self._extract_list(sample.get("paper_content", {}), "paper_content_text")
        headers = self._extract_list(sample.get("paper_headers", {}), "paper_header_content")

        sections = self._create_sections(content_chunks, headers)
        pages, elements = self._build_artifacts(paper_id, sections)

        return DocumentArtifact(
            doc_id=paper_id,
            pdf_path=Path(f"{paper_id}.txt"),
            pages=pages,
            elements=elements,
            meta={
                "source": "sciduet",
                "paper_title": sample.get("paper_title", ""),
                "paper_abstract": sample.get("paper_abstract", ""),
            },
        )

    @staticmethod
    def _extract_list(data: Any, key: str) -> List[str]:
        """Extract list of strings from nested dict structure."""
        if isinstance(data, dict):
            return [str(c) for c in data.get(key, []) if c]
        return []

    def _create_sections(
        self, content_chunks: List[str], headers: List[str]
    ) -> List[Tuple[str, List[str]]]:
        """
        Merge chunks and split by headers into sections.
        
        Returns:
            List of (header, [chunks]) tuples
        """
        if not content_chunks:
            return [("", [])]

        merged = self._merge_chunks(content_chunks)

        if not headers:
            # No headers - group by fixed size
            return [
                (f"Section {i // 6 + 1}", merged[i : i + 6])
                for i in range(0, len(merged), 6)
            ]

        return self._split_by_headers(merged, headers)

    def _merge_chunks(self, chunks: List[str]) -> List[str]:
        """Merge small consecutive chunks into larger blocks."""
        merged, current = [], ""
        for chunk in chunks:
            if len(current) + len(chunk) < self.target_chunk_size:
                current = f"{current} {chunk}" if current else chunk
            else:
                if current:
                    merged.append(current)
                current = chunk
        if current:
            merged.append(current)
        return merged

    def _split_by_headers(
        self, merged: List[str], headers: List[str]
    ) -> List[Tuple[str, List[str]]]:
        """Split merged chunks into sections based on headers."""
        sections: List[Tuple[str, List[str]]] = []
        current_header, current_chunks = "Introduction", []
        header_idx = 0

        for chunk in merged:
            if header_idx < len(headers):
                next_header = headers[header_idx]
                if self._chunk_contains_header(chunk, next_header):
                    if current_chunks:
                        sections.append((current_header, current_chunks))
                    current_header, current_chunks = next_header, [chunk]
                    header_idx += 1
                    continue
            current_chunks.append(chunk)

        if current_chunks:
            sections.append((current_header, current_chunks))
        return sections or [("", merged)]

    @staticmethod
    def _chunk_contains_header(chunk: str, header: str) -> bool:
        """Check if chunk contains/starts with header text."""
        chunk_lower = chunk.lower()
        header_lower = header.lower()
        return (
            header_lower in chunk_lower[:100]
            or chunk_lower.strip().startswith(header_lower[:20])
        )

    def _build_artifacts(
        self, paper_id: str, sections: List[Tuple[str, List[str]]]
    ) -> Tuple[List[PageArtifact], List[Element]]:
        """Build PageArtifact and Element lists from sections."""
        pages, elements = [], []

        for page_num, (header, section_chunks) in enumerate(sections, start=1):
            pages.append(PageArtifact(
                page=page_num, 
                width=1000.0, 
                height=1000.0, 
                page_image_path=None
            ))

            if header:
                elements.append(Element(
                    id=stable_id(paper_id, page_num, "header", header),
                    page=page_num,
                    bbox=(0.0, 0.0, 1000.0, 50.0),
                    type="header",
                    text=header,
                    image_path=None,
                ))

            y_pos = 100.0
            for chunk_idx, chunk in enumerate(section_chunks):
                if not chunk.strip():
                    continue
                for sub_idx, sub_chunk in enumerate(self._split_long_text(chunk)):
                    if not sub_chunk.strip():
                        continue
                    elements.append(Element(
                        id=stable_id(paper_id, page_num, "text", f"{chunk_idx}_{sub_idx}"),
                        page=page_num,
                        bbox=(0.0, y_pos, 1000.0, y_pos + 200.0),
                        type="text",
                        text=sub_chunk,
                        image_path=None,
                    ))
                    y_pos += 220.0

        return pages, elements

    def _split_long_text(self, text: str) -> List[str]:
        """Split text exceeding max_section_length into sentence-aware chunks."""
        if len(text) <= self.max_section_length:
            return [text]

        chunks, current, current_len = [], [], 0
        for sent in text.split(". "):
            sent_len = len(sent) + 2
            if current_len + sent_len > self.max_section_length and current:
                chunks.append(". ".join(current) + ".")
                current, current_len = [sent], sent_len
            else:
                current.append(sent)
                current_len += sent_len
        if current:
            chunks.append(". ".join(current) + ".")
        return chunks