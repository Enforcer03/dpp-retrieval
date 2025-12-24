from __future__ import annotations

import logging
from pathlib import Path

import fitz  # PyMuPDF

log = logging.getLogger(__name__)


def highlight_pdf(pdf_path: Path, selected_units, out_pdf: Path) -> None:
    """
    Highlights selected units in a PDF.

    IMPORTANT: pipeline page numbers are 1-indexed, PyMuPDF is 0-indexed.
    This function safely converts and clamps page indices.
    """
    pdf_path = Path(pdf_path)
    out_pdf = Path(out_pdf)

    doc = fitz.open(str(pdf_path))
    n_pages = doc.page_count

    for u in selected_units:
        # Units from our pipeline are 1-indexed pages
        pno_1 = getattr(u, "page", None)
        if not isinstance(pno_1, int):
            continue

        # Convert to 0-index for PyMuPDF
        pno = pno_1 - 1

        # Clamp / skip invalid pages
        if pno < 0 or pno >= n_pages:
            log.warning(
                "highlight.skip invalid_page unit_id=%s unit_page=%s doc_pages=%s",
                getattr(u, "id", None),
                pno_1,
                n_pages,
            )
            continue

        page = doc.load_page(pno)

        # bbox is expected in (x0, y0, x1, y1)
        bbox = getattr(u, "bbox", None)
        if not bbox or len(bbox) != 4:
            continue

        try:
            rect = fitz.Rect(float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3]))
            # Add highlight annotation (or rectangle)
            annot = page.add_rect_annot(rect)
            annot.set_colors(stroke=(1, 0, 0))  # red outline
            annot.set_border(width=1)
            annot.update()
        except Exception as e:
            log.warning("highlight.failed unit_id=%s err=%s", getattr(u, "id", None), e)

    doc.save(str(out_pdf))
    doc.close()
