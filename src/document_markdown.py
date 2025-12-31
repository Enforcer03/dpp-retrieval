"""Generate markdown document with embedded PDF page images."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .schema import Document, EvidenceUnit

log = logging.getLogger(__name__)


def build_document_markdown(
    doc: Document,
    selected_units: list[EvidenceUnit] | None = None,
    output_dir: Path | None = None,
    include_selected_highlights: bool = True,
) -> str:
    """
    Build a markdown document with all PDF page images.

    Args:
        doc: Document with page artifacts and metadata
        selected_units: Optional list of selected evidence units to highlight
        output_dir: Output directory (for calculating relative image paths)
        include_selected_highlights: Whether to show which chunks were selected

    Returns:
        Markdown formatted string with embedded page images
    """
    lines = []

    # Header
    pdf_name = Path(doc.pdf_path).name if hasattr(doc, 'pdf_path') else "Document"
    lines.append(f"# {pdf_name}\n")
    lines.append(f"**Total Pages:** {len(doc.pages)}\n")

    if selected_units and include_selected_highlights:
        lines.append(f"**Selected Chunks:** {len(selected_units)}\n")

    lines.append("---\n")

    # Build page-to-selected-chunks mapping
    page_to_chunks: dict[int, list[EvidenceUnit]] = {}
    if selected_units:
        for unit in selected_units:
            page = int(unit.page)
            if page not in page_to_chunks:
                page_to_chunks[page] = []
            page_to_chunks[page].append(unit)

    # Generate markdown for each page
    for page_artifact in sorted(doc.pages, key=lambda p: p.page):
        page_num = int(page_artifact.page)

        # Page header
        lines.append(f"\n## Page {page_num}\n")

        # Add page image if available
        if page_artifact.page_image_path:
            img_path = Path(page_artifact.page_image_path)

            # Calculate relative path from output_dir if provided
            if output_dir:
                try:
                    rel_path = img_path.relative_to(output_dir.parent)
                    img_ref = f"../{rel_path}"
                except ValueError:
                    # If not relative, use absolute path
                    img_ref = str(img_path)
            else:
                img_ref = str(img_path)

            lines.append(f"![Page {page_num}]({img_ref})\n")
        else:
            lines.append(f"*Page image not available*\n")

        # Show selected chunks for this page if any
        if page_num in page_to_chunks and include_selected_highlights:
            chunks = page_to_chunks[page_num]
            lines.append(f"\n### Selected Content ({len(chunks)} chunk{'s' if len(chunks) > 1 else ''})\n")

            for i, unit in enumerate(chunks, 1):
                lines.append(f"\n**Chunk {i}** ({unit.type}):\n")
                lines.append(f"```\n{unit.context_text[:500]}{'...' if len(unit.context_text) > 500 else ''}\n```\n")

        lines.append("\n---\n")

    return "".join(lines)


def save_document_markdown(
    doc: Document,
    output_path: Path,
    selected_units: list[EvidenceUnit] | None = None,
    include_selected_highlights: bool = True,
) -> None:
    """
    Generate and save markdown document with PDF page images.

    Args:
        doc: Document with page artifacts
        output_path: Path where markdown file should be saved
        selected_units: Optional list of selected evidence units
        include_selected_highlights: Whether to show selected chunks
    """
    output_dir = output_path.parent

    markdown = build_document_markdown(
        doc=doc,
        selected_units=selected_units,
        output_dir=output_dir,
        include_selected_highlights=include_selected_highlights,
    )

    output_path.write_text(markdown, encoding="utf-8")
    log.info("document_markdown.saved path=%s pages=%d", output_path, len(doc.pages))
