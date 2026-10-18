"""OCR engines (PaddleOCR primary, Tesseract fallback).

Output is always an :class:`OcrPage`. If no engine can read the page the result carries an
``error`` and an empty text: this module never invents text.
"""

from __future__ import annotations

import logging
import statistics
import time
from typing import Any, Protocol

import numpy as np
from PIL import Image

from config import OcrSettings
from pipeline.schemas import OcrLine, OcrPage

log = logging.getLogger(__name__)


class OcrUnavailableError(RuntimeError):
    """The OCR engine cannot run (not installed, model missing, ...). Message includes how to fix it."""


class OcrEngine(Protocol):
    name: str

    def read(self, image: Image.Image) -> list[OcrLine]: ...


# ----------------------------------------------------------------------
def _poly_to_bbox(poly: Any) -> list[float]:
    pts = np.asarray(poly, dtype=float).reshape(-1, 2)
    return [float(pts[:, 0].min()), float(pts[:, 1].min()), float(pts[:, 0].max()), float(pts[:, 1].max())]


class PaddleOcrEngine:
    """PaddleOCR wrapper that understands both the 2.x (``.ocr``) and 3.x (``.predict``) result formats."""

    name = "paddleocr"

    def __init__(self, lang: str = "en") -> None:
        self.lang = lang
        self._engine: Any = None

    def _load(self) -> Any:
        if self._engine is not None:
            return self._engine
        try:
            import paddleocr  # type: ignore
        except ImportError as exc:
            raise OcrUnavailableError(
                "PaddleOCR is not installed. Run:  pip install paddleocr paddlepaddle   "
                "(or set GST_OCR_ENGINE=tesseract to use Tesseract instead)."
            ) from exc
        try:
            try:  # PaddleOCR 3.x
                self._engine = paddleocr.PaddleOCR(lang=self.lang, use_textline_orientation=True)
            except (TypeError, ValueError):  # PaddleOCR 2.x
                self._engine = paddleocr.PaddleOCR(lang=self.lang, use_angle_cls=True, show_log=False)
        except Exception as exc:
            raise OcrUnavailableError(
                f"PaddleOCR could not start ({type(exc).__name__}). Its models are downloaded on first use, "
                "so the first run needs internet access. Check your install, or switch to Tesseract."
            ) from exc
        return self._engine

    def read(self, image: Image.Image) -> list[OcrLine]:
        engine = self._load()
        array = np.asarray(image.convert("RGB"))[:, :, ::-1].copy()  # PaddleOCR expects BGR
        lines: list[OcrLine] = []
        if hasattr(engine, "predict"):
            for res in engine.predict(array):
                data = res if isinstance(res, dict) else getattr(res, "json", {}).get("res", {})
                texts = data.get("rec_texts", [])
                scores = data.get("rec_scores", [])
                polys = data.get("rec_polys", data.get("dt_polys", []))
                for i, text in enumerate(texts):
                    if str(text).strip():
                        lines.append(OcrLine(
                            text=str(text).strip(),
                            confidence=float(scores[i]) if i < len(scores) else None,
                            bbox=_poly_to_bbox(polys[i]) if i < len(polys) else [],
                        ))
        else:
            result = engine.ocr(array, cls=True)
            for page in result or []:
                for item in page or []:
                    poly, (text, score) = item[0], item[1]
                    if str(text).strip():
                        lines.append(OcrLine(text=str(text).strip(), confidence=float(score), bbox=_poly_to_bbox(poly)))
        return lines


class TesseractEngine:
    name = "tesseract"

    def __init__(self, lang: str = "en", cmd: str | None = None, psm: int = 4) -> None:
        self.lang = {"en": "eng"}.get(lang, lang)
        self.cmd = cmd
        self.psm = psm

    def read(self, image: Image.Image) -> list[OcrLine]:
        try:
            import pytesseract
        except ImportError as exc:
            raise OcrUnavailableError("pytesseract is not installed. Run:  pip install pytesseract") from exc
        if self.cmd:
            pytesseract.pytesseract.tesseract_cmd = self.cmd
        try:
            data = pytesseract.image_to_data(image, lang=self.lang, config=f"--psm {self.psm}", output_type=pytesseract.Output.DICT)
        except pytesseract.TesseractNotFoundError as exc:
            raise OcrUnavailableError(
                "The Tesseract program was not found. Install it (Windows: UB Mannheim installer; "
                "macOS: brew install tesseract; Linux: apt install tesseract-ocr) and, if needed, "
                "set GST_TESSERACT_CMD to its full path."
            ) from exc
        except pytesseract.TesseractError as exc:
            raise OcrUnavailableError(f"Tesseract failed: {str(exc)[:200]}") from exc

        # Group words into lines using Tesseract's own block/paragraph/line numbering.
        groups: dict[tuple[int, int, int], list[int]] = {}
        for i, word in enumerate(data["text"]):
            if str(word).strip() and float(data["conf"][i]) >= 0:
                key = (data["block_num"][i], data["par_num"][i], data["line_num"][i])
                groups.setdefault(key, []).append(i)
        lines: list[OcrLine] = []
        for idxs in groups.values():
            idxs.sort(key=lambda i: data["left"][i])
            text = " ".join(str(data["text"][i]).strip() for i in idxs)
            conf = statistics.fmean(float(data["conf"][i]) for i in idxs) / 100.0
            x0 = min(data["left"][i] for i in idxs)
            y0 = min(data["top"][i] for i in idxs)
            x1 = max(data["left"][i] + data["width"][i] for i in idxs)
            y1 = max(data["top"][i] + data["height"][i] for i in idxs)
            lines.append(OcrLine(text=text, confidence=conf, bbox=[float(x0), float(y0), float(x1), float(y1)]))
        return lines


# ----------------------------------------------------------------------
def build_engine(name: str, cfg: OcrSettings) -> OcrEngine:
    if name == "paddleocr":
        return PaddleOcrEngine(cfg.lang)
    if name == "tesseract":
        return TesseractEngine(cfg.lang, cfg.tesseract_cmd, cfg.tesseract_psm)
    raise OcrUnavailableError(f"Unknown OCR engine '{name}'.")


def lines_to_text(lines: list[OcrLine]) -> str:
    """Rebuild reading order: cluster boxes into rows by vertical centre, then sort left to right."""
    boxed = [ln for ln in lines if len(ln.bbox) == 4]
    unboxed = [ln for ln in lines if len(ln.bbox) != 4]
    if not boxed:
        return "\n".join(ln.text for ln in unboxed)
    heights = [ln.bbox[3] - ln.bbox[1] for ln in boxed]
    tolerance = max(4.0, statistics.median(heights) * 0.6)
    rows: list[list[OcrLine]] = []
    for ln in sorted(boxed, key=lambda x: (x.bbox[1] + x.bbox[3]) / 2):
        centre = (ln.bbox[1] + ln.bbox[3]) / 2
        if rows:
            last = rows[-1]
            last_centre = statistics.fmean((b.bbox[1] + b.bbox[3]) / 2 for b in last)
            if abs(centre - last_centre) <= tolerance:
                last.append(ln)
                continue
        rows.append([ln])
    out = ["   ".join(b.text for b in sorted(row, key=lambda x: x.bbox[0])) for row in rows]
    out.extend(ln.text for ln in unboxed)
    return "\n".join(out)


def run_ocr(image: Image.Image, cfg: OcrSettings, engines: list[OcrEngine] | None = None) -> OcrPage:
    """Try the primary engine, then the fallback. Never raises; failures are returned in ``error``."""
    if engines is None:
        names = [cfg.engine] + ([cfg.fallback_engine] if cfg.fallback_engine and cfg.fallback_engine != cfg.engine else [])
        engines = [build_engine(n, cfg) for n in names if n]
    problems: list[str] = []
    for engine in engines:
        start = time.perf_counter()
        try:
            lines = engine.read(image)
        except OcrUnavailableError as exc:
            problems.append(f"{engine.name}: {exc}")
            log.warning("OCR engine %s unavailable", engine.name)
            continue
        except Exception as exc:  # unexpected engine crash on this page
            problems.append(f"{engine.name}: failed on this page ({type(exc).__name__})")
            log.warning("OCR engine %s crashed: %s", engine.name, type(exc).__name__)
            continue
        elapsed = int((time.perf_counter() - start) * 1000)
        if not lines:
            problems.append(f"{engine.name}: no text was detected on this page")
            continue
        confs = [ln.confidence for ln in lines if ln.confidence is not None]
        return OcrPage(
            text=lines_to_text(lines),
            lines=lines,
            mean_confidence=round(statistics.fmean(confs), 4) if confs else None,
            engine=engine.name,
            elapsed_ms=elapsed,
        )
    message = "OCR could not read this page. " + " | ".join(problems) if problems else "OCR could not run."
    return OcrPage(engine="none", error=message)
