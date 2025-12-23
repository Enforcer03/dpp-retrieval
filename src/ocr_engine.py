from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

from PIL import Image

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class OcrLine:
    text: str
    bbox_px: tuple[int, int, int, int]
    conf: float


class OCREngine:
    def __init__(self, engine: str, lang: str, min_conf: int, psm: int):
        self.engine = (engine or "").lower()
        self.lang = lang
        self.min_conf = int(min_conf)
        self.psm = int(psm)
        self._easy_reader = None
        self._available = self._probe()

    @property
    def available(self) -> bool:
        return self._available

    def extract_lines(self, image_path: Path) -> list[OcrLine]:
        if not self._available:
            return []
        if self.engine == "easyocr":
            return self._easyocr_lines(image_path)
        return self._tesseract_lines(image_path)

    def extract_text(self, image_path: Path) -> str:
        lines = self.extract_lines(image_path)
        return "\n".join([l.text for l in lines if l.text]).strip()

    def _probe(self) -> bool:
        if self.engine == "easyocr":
            try:
                import easyocr  # type: ignore

                self._easy_reader = easyocr.Reader([self._map_easy_lang(self.lang)], gpu=False)
                return True
            except Exception as e:
                log.warning("ocr.unavailable engine=easyocr err=%s", e)
                return False
        try:
            import pytesseract  # type: ignore

            _ = pytesseract.get_tesseract_version()
            return True
        except Exception as e:
            log.warning("ocr.unavailable engine=tesseract err=%s", e)
            return False

    def _tesseract_lines(self, image_path: Path) -> list[OcrLine]:
        import pytesseract  # type: ignore
        from pytesseract import Output  # type: ignore

        cfg = f"--psm {self.psm}"
        with Image.open(image_path) as im:
            data = pytesseract.image_to_data(im, lang=self.lang, config=cfg, output_type=Output.DICT)

        n = len(data.get("text", []))
        groups: dict[tuple[int, int, int], dict] = {}

        for i in range(n):
            txt = (data["text"][i] or "").strip()
            if not txt:
                continue
            try:
                conf = float(data["conf"][i])
            except Exception:
                conf = -1.0
            if conf < self.min_conf:
                continue

            key = (int(data["block_num"][i]), int(data["par_num"][i]), int(data["line_num"][i]))
            x, y, w, h = int(data["left"][i]), int(data["top"][i]), int(data["width"][i]), int(data["height"][i])
            g = groups.get(key)
            if g is None:
                groups[key] = {"t": [txt], "x0": x, "y0": y, "x1": x + w, "y1": y + h, "c": [conf]}
            else:
                g["t"].append(txt)
                g["x0"] = min(g["x0"], x)
                g["y0"] = min(g["y0"], y)
                g["x1"] = max(g["x1"], x + w)
                g["y1"] = max(g["y1"], y + h)
                g["c"].append(conf)

        out: list[OcrLine] = []
        for g in groups.values():
            text = " ".join(g["t"]).strip()
            if not text:
                continue
            conf = float(sum(g["c"]) / max(1, len(g["c"])))
            out.append(OcrLine(text=text, bbox_px=(int(g["x0"]), int(g["y0"]), int(g["x1"]), int(g["y1"])), conf=conf))
        out.sort(key=lambda x: (x.bbox_px[1], x.bbox_px[0]))
        return out

    def _easyocr_lines(self, image_path: Path) -> list[OcrLine]:
        import numpy as np

        if self._easy_reader is None:
            return []
        res = self._easy_reader.readtext(str(image_path))
        out: list[OcrLine] = []
        for bbox, text, conf in res:
            t = (text or "").strip()
            if not t:
                continue
            c = float(conf) * 100.0
            if c < self.min_conf:
                continue
            pts = np.array(bbox, dtype=np.float32)
            x0, y0 = float(pts[:, 0].min()), float(pts[:, 1].min())
            x1, y1 = float(pts[:, 0].max()), float(pts[:, 1].max())
            out.append(OcrLine(text=t, bbox_px=(int(x0), int(y0), int(x1), int(y1)), conf=c))
        out.sort(key=lambda x: (x.bbox_px[1], x.bbox_px[0]))
        return out

    def _map_easy_lang(self, lang: str) -> str:
        l = (lang or "").lower()
        if l in ("eng", "en"):
            return "en"
        return l
