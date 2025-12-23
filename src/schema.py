from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

BBox = tuple[float, float, float, float]


@dataclass(frozen=True)
class PageArtifact:
    page: int
    width: float
    height: float
    page_image_path: Path | None = None


@dataclass(frozen=True)
class Element:
    id: str
    page: int
    bbox: BBox
    type: Literal["text", "caption", "header", "footer", "figure", "page_image"]
    text: str | None = None
    image_path: Path | None = None


@dataclass
class DocumentArtifact:
    doc_id: str
    pdf_path: Path
    pages: list[PageArtifact]
    elements: list[Element]
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass
class EvidenceUnit:
    id: str
    page: int
    bbox: BBox
    type: Literal["text", "table_text", "figure", "page_image"]
    retrieval_text: str
    context_text: str
    image_paths: list[Path]
    source_element_ids: list[str]
