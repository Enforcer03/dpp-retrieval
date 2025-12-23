from __future__ import annotations

import logging
import re
from collections import defaultdict
from pathlib import Path

from .schema import DocumentArtifact, EvidenceUnit, Element
from .utils import bbox_iou, bbox_union, stable_id

log = logging.getLogger(__name__)


_num_re = re.compile(r"\d")


def _looks_like_table(text: str) -> bool:
    t = text.strip()
    if t.count("\n") >= 3 and t.count("  ") >= 6:
        return True
    digs = len(_num_re.findall(t))
    return digs >= 10 and ("\n" in t) and (digs / max(1, len(t))) > 0.08


def _h_overlap(a, b) -> float:
    x0 = max(a[0], b[0])
    x1 = min(a[2], b[2])
    return max(0.0, x1 - x0)


def _near_caption(fig: Element, caps: list[Element], px: float) -> Element | None:
    fx0, fy0, fx1, fy1 = fig.bbox
    best = None
    best_d = 1e18
    for c in caps:
        cx0, cy0, cx1, cy1 = c.bbox
        if _h_overlap((fx0, 0, fx1, 0), (cx0, 0, cx1, 0)) <= 0:
            continue
        d = min(abs(cy0 - fy1), abs(fy0 - cy1))
        if d <= px and d < best_d:
            best = c
            best_d = d
    return best


class EvidenceBuilder:
    def __init__(self, max_text_chunk_tokens: int, caption_search_px: int, stopwords: set[str]):
        self.max_text_chunk_tokens = max_text_chunk_tokens
        self.caption_search_px = caption_search_px
        self.stopwords = stopwords

    def build(self, doc: DocumentArtifact, token_counter) -> list[EvidenceUnit]:
        by_page: dict[int, list[Element]] = defaultdict(list)
        for e in doc.elements:
            by_page[e.page].append(e)

        units: list[EvidenceUnit] = []
        for pno, els in sorted(by_page.items()):
            els = sorted(els, key=lambda x: (x.bbox[1], x.bbox[0]))
            caps = [e for e in els if e.type == "caption" and e.text]
            figs = [e for e in els if e.type == "figure" and e.image_path]
            texts = [e for e in els if e.type in ("text", "header", "footer") and e.text]

            used_caps: set[str] = set()
            for fig in figs:
                cap = _near_caption(fig, caps, self.caption_search_px)
                cap_text = ""
                cap_ids: list[str] = []
                bbox = fig.bbox
                if cap is not None:
                    cap_text = cap.text or ""
                    used_caps.add(cap.id)
                    cap_ids.append(cap.id)
                    bbox = bbox_union(bbox, cap.bbox)

                fig_text = (fig.text or "").strip()
                rt = "\n".join([t for t in [cap_text.strip(), fig_text] if t]).strip()

                uid = stable_id(doc.doc_id, pno, "figure", fig.id, rt[:32])
                units.append(
                    EvidenceUnit(
                        id=uid,
                        page=pno,
                        bbox=bbox,
                        type="figure",
                        retrieval_text=rt,
                        context_text=rt,
                        image_paths=[Path(fig.image_path)],
                        source_element_ids=[fig.id] + cap_ids,
                    )
                )

            usable_texts = [e for e in texts if e.id not in used_caps]
            buf: list[Element] = []
            buf_texts: list[str] = []
            buf_bbox = None
            buf_tokens = 0

            def flush() -> None:
                nonlocal buf, buf_texts, buf_bbox, buf_tokens
                if not buf:
                    return
                text = "\n".join(buf_texts).strip()
                if not text:
                    buf, buf_texts, buf_bbox, buf_tokens = [], [], None, 0
                    return
                ttype = "table_text" if _looks_like_table(text) else "text"
                uid = stable_id(doc.doc_id, pno, ttype, buf[0].id, buf[-1].id)
                units.append(
                    EvidenceUnit(
                        id=uid,
                        page=pno,
                        bbox=buf_bbox or buf[0].bbox,
                        type=ttype,
                        retrieval_text=text,
                        context_text=text,
                        image_paths=[],
                        source_element_ids=[e.id for e in buf],
                    )
                )
                buf, buf_texts, buf_bbox, buf_tokens = [], [], None, 0

            for e in usable_texts:
                t = (e.text or "").strip()
                if not t:
                    continue
                t_tokens = token_counter.count(t)
                if buf and (buf_tokens + t_tokens) > self.max_text_chunk_tokens:
                    flush()
                buf.append(e)
                buf_texts.append(t)
                buf_bbox = e.bbox if buf_bbox is None else bbox_union(buf_bbox, e.bbox)
                buf_tokens += t_tokens

            flush()

            page_images = [e for e in els if e.type == "page_image" and e.image_path]
            for pi in page_images:
                uid = stable_id(doc.doc_id, pno, "page_image", pi.id)
                units.append(
                    EvidenceUnit(
                        id=uid,
                        page=pno,
                        bbox=pi.bbox,
                        type="page_image",
                        retrieval_text="",
                        context_text="",
                        image_paths=[Path(pi.image_path)],
                        source_element_ids=[pi.id],
                    )
                )

        log.info("evidence.units doc_id=%s units=%d", doc.doc_id, len(units))
        return self._dedupe(units)

    def _dedupe(self, units: list[EvidenceUnit]) -> list[EvidenceUnit]:
        out: list[EvidenceUnit] = []
        for u in sorted(units, key=lambda x: (x.page, x.bbox[1], x.bbox[0], x.type)):
            keep = True
            for v in out[-6:]:
                if u.page == v.page and u.type == v.type and bbox_iou(u.bbox, v.bbox) > 0.92:
                    keep = False
                    break
            if keep:
                out.append(u)
        return out
