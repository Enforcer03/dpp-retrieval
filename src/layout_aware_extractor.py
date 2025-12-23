from __future__ import annotations

import logging
from pathlib import Path

from PIL import Image

from .ocr_engine import OCREngine
from .schema import DocumentArtifact, Element, PageArtifact
from .utils import ensure_dir, is_caption, stable_id

try:
    import fitz
except Exception as e:  # pragma: no cover
    fitz = None
    _fitz_err = e


log = logging.getLogger(__name__)


class PdfLayoutExtractor:
    def __init__(
        self,
        dpi: int,
        work_dir: Path,
        keep_page_renders: bool,
        keep_crops: bool,
        mode: str = "auto",
        ocr: OCREngine | None = None,
        ocr_on_pages: bool = True,
        ocr_on_figures: bool = False,
    ):
        self.dpi = dpi
        self.work_dir = work_dir
        self.keep_page_renders = keep_page_renders
        self.keep_crops = keep_crops
        self.mode = (mode or "auto").lower()
        self.ocr = ocr
        self.ocr_on_pages = ocr_on_pages
        self.ocr_on_figures = ocr_on_figures

    def extract(self, pdf_path: Path, doc_id: str) -> DocumentArtifact:
        if fitz is None:  # pragma: no cover
            raise RuntimeError(f"PyMuPDF not available: {_fitz_err}")

        pdf_path = pdf_path.resolve()
        out_dir = ensure_dir(self.work_dir / doc_id)
        pages_dir = ensure_dir(out_dir / "pages")
        crops_dir = ensure_dir(out_dir / "crops")

        doc = fitz.open(pdf_path)
        pages: list[PageArtifact] = []
        elements: list[Element] = []

        log.info("extract.start doc_id=%s pages=%d mode=%s ocr=%s", doc_id, doc.page_count, self.mode, bool(self.ocr and self.ocr.available))

        for pno in range(doc.page_count):
            page = doc.load_page(pno)
            w, h = float(page.rect.width), float(page.rect.height)

            page_img_path = None
            pix_w = pix_h = None
            if self.keep_page_renders:
                page_img_path = pages_dir / f"page_{pno:04d}.png"
                pix_w, pix_h = self._render_page(page, page_img_path)

            pages.append(PageArtifact(page=pno, width=w, height=h, page_image_path=page_img_path))

            native_has_text = False
            if self.mode in ("native", "auto"):
                native_has_text = self._extract_native_text(doc_id, pno, page, w, h, elements)

            run_ocr = (
                self.ocr is not None
                and self.ocr.available
                and self.ocr_on_pages
                and (self.mode == "ocr" or (self.mode == "auto" and not native_has_text))
            )

            if run_ocr:
                if page_img_path is None:
                    page_img_path = pages_dir / f"page_{pno:04d}.png"
                    pix_w, pix_h = self._render_page(page, page_img_path)
                if pix_w is None or pix_h is None:
                    with Image.open(page_img_path) as im:
                        pix_w, pix_h = im.size
                self._extract_ocr_text(doc_id, pno, w, h, int(pix_w), int(pix_h), page_img_path, elements)

            self._extract_figures(doc_id, pno, page, crops_dir, elements)

            if not any(e.page == pno and e.type in ("text", "caption", "header", "footer") for e in elements):
                if page_img_path is None:
                    page_img_path = pages_dir / f"page_{pno:04d}.png"
                    self._render_page(page, page_img_path)
                    pages[-1] = PageArtifact(page=pno, width=w, height=h, page_image_path=page_img_path)
                eid = stable_id(doc_id, pno, "page_image")
                elements.append(
                    Element(
                        id=eid,
                        page=pno,
                        bbox=(0.0, 0.0, w, h),
                        type="page_image",
                        text=None,
                        image_path=page_img_path,
                    )
                )

        log.info("extract.done doc_id=%s elements=%d", doc_id, len(elements))
        return DocumentArtifact(doc_id=doc_id, pdf_path=pdf_path, pages=pages, elements=elements)

    def _extract_native_text(self, doc_id: str, pno: int, page, w: float, h: float, elements: list[Element]) -> bool:
        text_blocks = page.get_text("blocks") or []
        added = 0
        for b in text_blocks:
            x0, y0, x1, y1, text = float(b[0]), float(b[1]), float(b[2]), float(b[3]), str(b[4] or "")
            t = text.strip()
            if not t:
                continue
            etype = "caption" if is_caption(t) else "text"
            if y0 <= 0.08 * h and len(t) <= 120:
                etype = "header"
            if y1 >= 0.92 * h and len(t) <= 120:
                etype = "footer"
            eid = stable_id(doc_id, pno, x0, y0, x1, y1, etype, t[:32])
            elements.append(Element(id=eid, page=pno, bbox=(x0, y0, x1, y1), type=etype, text=t, image_path=None))
            added += 1
        return added > 0

    def _extract_ocr_text(
        self,
        doc_id: str,
        pno: int,
        w_pdf: float,
        h_pdf: float,
        w_px: int,
        h_px: int,
        img_path: Path,
        elements: list[Element],
    ) -> None:
        assert self.ocr is not None
        lines = self.ocr.extract_lines(img_path)
        if not lines:
            log.info("ocr.empty page=%d", pno)
            return
        sx = w_pdf / max(1, w_px)
        sy = h_pdf / max(1, h_px)

        added = 0
        for ln in lines:
            x0, y0, x1, y1 = ln.bbox_px
            bx0, by0, bx1, by1 = float(x0) * sx, float(y0) * sy, float(x1) * sx, float(y1) * sy
            t = (ln.text or "").strip()
            if not t:
                continue
            etype = "caption" if is_caption(t) else "text"
            if by0 <= 0.08 * h_pdf and len(t) <= 120:
                etype = "header"
            if by1 >= 0.92 * h_pdf and len(t) <= 120:
                etype = "footer"
            eid = stable_id(doc_id, pno, bx0, by0, bx1, by1, etype, t[:32])
            elements.append(Element(id=eid, page=pno, bbox=(bx0, by0, bx1, by1), type=etype, text=t, image_path=None))
            added += 1

        log.info("ocr.page page=%d lines=%d", pno, added)

    def _extract_figures(self, doc_id: str, pno: int, page, crops_dir: Path, elements: list[Element]) -> None:
        img_list = page.get_images(full=True) or []
        for img in img_list:
            xref = img[0]
            rects = page.get_image_rects(xref) or []
            for r in rects:
                rect = fitz.Rect(r)
                ipath = None
                if self.keep_crops:
                    ipath = crops_dir / f"img_{pno:04d}_{xref}_{int(rect.x0)}_{int(rect.y0)}.png"
                    self._render_crop(page, rect, ipath)

                ftxt = None
                if self.ocr is not None and self.ocr.available and self.ocr_on_figures and ipath is not None:
                    ftxt = self.ocr.extract_text(ipath) or None

                eid = stable_id(doc_id, pno, rect.x0, rect.y0, rect.x1, rect.y1, "figure", xref)
                elements.append(
                    Element(
                        id=eid,
                        page=pno,
                        bbox=(float(rect.x0), float(rect.y0), float(rect.x1), float(rect.y1)),
                        type="figure",
                        text=ftxt,
                        image_path=ipath,
                    )
                )

    def _render_page(self, page, out_path: Path) -> tuple[int, int]:
        ensure_dir(out_path.parent)
        pix = page.get_pixmap(dpi=self.dpi, alpha=False)
        pix.save(str(out_path))
        return int(pix.width), int(pix.height)

    def _render_crop(self, page, rect, out_path: Path) -> None:
        ensure_dir(out_path.parent)
        pix = page.get_pixmap(clip=rect, dpi=self.dpi, alpha=False)
        pix.save(str(out_path))
