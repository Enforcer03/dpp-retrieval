from __future__ import annotations

import logging
from pathlib import Path

from .schema import EvidenceUnit
from .utils import ensure_dir

try:
    import fitz
except Exception as e:  # pragma: no cover
    fitz = None
    _fitz_err = e

log = logging.getLogger(__name__)


def highlight_pdf(pdf_path: Path, units: list[EvidenceUnit], out_path: Path) -> None:
    if fitz is None:  # pragma: no cover
        raise RuntimeError(f"PyMuPDF not available: {_fitz_err}")

    ensure_dir(out_path.parent)
    doc = fitz.open(pdf_path)
    by_page: dict[int, list[EvidenceUnit]] = {}
    for u in units:
        by_page.setdefault(u.page, []).append(u)

    for pno, us in by_page.items():
        page = doc.load_page(pno)
        for u in us:
            x0, y0, x1, y1 = u.bbox
            r = fitz.Rect(x0, y0, x1, y1)
            annot = page.add_rect_annot(r)
            annot.set_colors(stroke=(1, 0, 0))
            annot.set_opacity(0.25)
            annot.update()

    doc.save(str(out_path))
    log.info("highlight.saved path=%s", out_path)
