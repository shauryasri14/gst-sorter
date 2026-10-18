"""Shared UI helpers."""

from __future__ import annotations

import importlib.util
import shutil
import time
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path

import streamlit as st
from PIL import Image, ImageDraw

from config import Settings
from pipeline.extract import OllamaStatus, check_ollama
from pipeline.runner import JobManager, Services
from pipeline.schemas import InvoiceStatus, OcrLine, Severity

STATUS_ICON = {
    InvoiceStatus.PENDING: "⏳",
    InvoiceStatus.PROCESSING: "⚙️",
    InvoiceStatus.EXTRACTED: "🔹",
    InvoiceStatus.NEEDS_REVIEW: "🟠",
    InvoiceStatus.REVIEWED: "✅",
    InvoiceStatus.FAILED: "❌",
}
SEVERITY_ICON = {Severity.ERROR: "🔴", Severity.WARNING: "🟠", Severity.INFO: "🔵"}

CSS = """
<style>
.block-container {padding-top: 1.6rem; max-width: 1500px;}
[data-testid="stMetricValue"] {font-size: 1.6rem;}
.gst-card {border: 1px solid #dfe3ea; border-radius: 10px; padding: 0.8rem 1rem; background: #fff; margin-bottom: .6rem;}
.gst-note {color:#5b6675; font-size:.85rem;}
</style>
"""


def services_and_jobs() -> tuple[Services, JobManager]:
    return st.session_state["services"], st.session_state["jobs"]


def money(value: Decimal | None) -> str:
    return "—" if value is None else f"₹{value:,.2f}"


def status_label(status: InvoiceStatus) -> str:
    return f"{STATUS_ICON[status]} {status.value}"


def ollama_status(settings: Settings, force: bool = False) -> OllamaStatus:
    """Cached for a few seconds so the sidebar does not hit Ollama on every click."""
    cached = st.session_state.get("_ollama")
    if not force and cached and time.time() - cached[0] < 8 and cached[2] == settings.ollama.model:
        return cached[1]
    status = check_ollama(settings.ollama)
    st.session_state["_ollama"] = (time.time(), status, settings.ollama.model)
    return status


@dataclass
class OcrAvailability:
    paddle: bool
    tesseract: bool

    @property
    def any(self) -> bool:
        return self.paddle or self.tesseract


def ocr_availability(settings: Settings) -> OcrAvailability:
    paddle = importlib.util.find_spec("paddleocr") is not None
    cmd = settings.ocr.tesseract_cmd
    tesseract = importlib.util.find_spec("pytesseract") is not None and bool(shutil.which(cmd or "tesseract"))
    return OcrAvailability(paddle, tesseract)


def ocr_problem(settings: Settings) -> str | None:
    """Human-readable setup problem for the configured OCR engines, or None when something usable exists."""
    avail = ocr_availability(settings)
    wanted = [settings.ocr.engine, settings.ocr.fallback_engine]
    if (settings.ocr.engine == "paddleocr" and avail.paddle) or (settings.ocr.engine == "tesseract" and avail.tesseract):
        return None
    if settings.ocr.fallback_engine == "tesseract" and avail.tesseract:
        return None
    if settings.ocr.fallback_engine == "paddleocr" and avail.paddle:
        return None
    return (
        f"No configured OCR engine is installed ({', '.join(str(w) for w in wanted if w)}). "
        "Install PaddleOCR with `pip install paddleocr paddlepaddle`, or install Tesseract "
        "(see README, 'Installing OCR')."
    )


def render_overlay(image_path: str | Path, lines: list[OcrLine], threshold: float, max_width: int = 1100) -> Image.Image:
    """Draw OCR boxes on the page: red = below the confidence threshold, green = fine."""
    img = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(img)
    thickness = max(2, img.width // 500)
    for ln in lines:
        if len(ln.bbox) != 4:
            continue
        low = ln.confidence is not None and ln.confidence < threshold
        draw.rectangle(ln.bbox, outline=(214, 40, 40) if low else (46, 160, 67), width=thickness)
    if img.width > max_width:
        img = img.resize((max_width, int(img.height * max_width / img.width)), Image.LANCZOS)
    return img


def short_name(name: str, width: int = 38) -> str:
    return name if len(name) <= width else name[: width - 1] + "…"
