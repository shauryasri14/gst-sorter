"""Shared fixtures. Nothing here touches the network, a real Ollama server, or real invoices."""

from __future__ import annotations

import io
import json
import random
from datetime import date
from decimal import Decimal
from pathlib import Path
from typing import Any

import pytest
from PIL import Image

from config import Settings, load_settings
from eval.synthetic import InvoiceSpec, make_spec, render_invoice
from pipeline.ocr import OcrUnavailableError
from pipeline.runner import Services
from pipeline.schemas import ExtractedInvoice, OcrLine
from pipeline.storage import Storage

TODAY = date(2026, 10, 17)


# ----------------------------------------------------------------------
# Test doubles
# ----------------------------------------------------------------------
class FakeOcr:
    """OCR engine returning fixed text. ``text`` may be a callable(image) -> str."""

    name = "fake-ocr"

    def __init__(self, text: Any = "", confidence: float = 0.97, error: Exception | None = None) -> None:
        self.text, self.confidence, self.error, self.calls = text, confidence, error, 0

    def read(self, image: Image.Image) -> list[OcrLine]:
        self.calls += 1
        if self.error:
            raise self.error
        text = self.text(image) if callable(self.text) else self.text
        lines = []
        for i, raw in enumerate(str(text).splitlines()):
            if raw.strip():
                lines.append(OcrLine(text=raw, confidence=self.confidence, bbox=[10.0, 20.0 + 30 * i, 400.0, 45.0 + 30 * i]))
        return lines


class FakeChat:
    """Chat client returning queued replies (the last one repeats). Records every call."""

    def __init__(self, *replies: Any) -> None:
        self.replies = list(replies)
        self.calls: list[list[dict[str, str]]] = []

    def chat(self, messages: list[dict[str, str]], schema: dict[str, Any] | None = None) -> str:
        self.calls.append(messages)
        reply = self.replies[min(len(self.calls) - 1, len(self.replies) - 1)]
        if isinstance(reply, Exception):
            raise reply
        return reply if isinstance(reply, str) else json.dumps(reply)


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------
def ocr_text_for(spec: InvoiceSpec) -> str:
    """OCR text that matches a synthetic invoice (what a perfect OCR would return)."""
    lines = [spec.vendor_name.upper(), f"GSTIN: {spec.vendor_gstin}", "TAX INVOICE",
             f"Invoice No: {spec.invoice_number}", f"Date: {spec.invoice_date.strftime('%d/%m/%Y')}",
             "Bill To:", spec.buyer_name, f"GSTIN: {spec.buyer_gstin}", f"Taxable Value {spec.taxable_amount:,.2f}"]
    if spec.igst is not None:
        lines.append(f"IGST @ {spec.rate}% {spec.igst:,.2f}")
    else:
        lines += [f"CGST @ {spec.rate / 2:g}% {spec.cgst:,.2f}", f"SGST @ {spec.rate / 2:g}% {spec.sgst:,.2f}"]
    lines.append(f"Grand Total {spec.total_amount:,.2f}")
    return "\n".join(lines)


def llm_json_for(spec: InvoiceSpec, **overrides: Any) -> dict[str, Any]:
    """What a correct model would answer for ``spec`` (date copied as printed)."""
    data: dict[str, Any] = {
        "vendor_name": spec.vendor_name, "vendor_gstin": spec.vendor_gstin, "buyer_name": spec.buyer_name,
        "buyer_gstin": spec.buyer_gstin, "invoice_number": spec.invoice_number,
        "invoice_date": spec.invoice_date.strftime("%d/%m/%Y"),
        "taxable_amount": float(spec.taxable_amount),
        "cgst": None if spec.cgst is None else float(spec.cgst),
        "sgst": None if spec.sgst is None else float(spec.sgst),
        "igst": None if spec.igst is None else float(spec.igst),
        "total_amount": float(spec.total_amount), "currency": "INR", "extraction_notes": [],
    }
    data.update(overrides)
    return data


def png_bytes(spec: InvoiceSpec | None = None, size: tuple[int, int] | None = None) -> bytes:
    img = render_invoice(spec) if spec else Image.new("RGB", size or (300, 200), "white")
    buf = io.BytesIO()
    img.save(buf, "PNG")
    return buf.getvalue()


def pdf_bytes(specs: list[InvoiceSpec]) -> bytes:
    pages = [render_invoice(s).resize((620, 877)) for s in specs]
    buf = io.BytesIO()
    pages[0].save(buf, "PDF", save_all=True, append_images=pages[1:], resolution=72.0)
    return buf.getvalue()


def specs(n: int, seed: int = 1) -> list[InvoiceSpec]:
    rng = random.Random(seed)
    return [make_spec(rng) for _ in range(n)]


# ----------------------------------------------------------------------
# Fixtures
# ----------------------------------------------------------------------
@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    return load_settings(
        env={}, dotenv_file=tmp_path / "missing.env",
        overrides={"data_dir": str(tmp_path / "data"), "ocr": {"engine": "tesseract", "fallback_engine": None},
                   "preprocess": {"enabled": False}, "limits": {"pdf_dpi": 72}},
    )


@pytest.fixture
def storage(settings: Settings) -> Storage:
    return Storage(settings)


@pytest.fixture
def make_services(settings: Settings, storage: Storage):
    def _make(ocr: Any = None, chat: Any = None) -> Services:
        return Services(settings, storage, ocr_engines=[ocr or FakeOcr("")], chat_client=chat or FakeChat("{}"))

    return _make


@pytest.fixture
def one_spec() -> InvoiceSpec:
    return specs(1, seed=3)[0]


def invoice(**kw: Any) -> ExtractedInvoice:
    """A clean, intra-state invoice (both GSTINs from Maharashtra) unless overridden."""
    base: dict[str, Any] = {
        "vendor_name": "Demo Sharma Electronics", "vendor_gstin": "27AAPFU0939F1ZV",
        "buyer_name": "Alpha Test Services LLP", "buyer_gstin": "27AAPFU0939F1ZV",
        "invoice_number": "INV-2041", "invoice_date": "2026-09-14",
        "taxable_amount": Decimal("10000.00"), "cgst": Decimal("900.00"), "sgst": Decimal("900.00"),
        "igst": None, "total_amount": Decimal("11800.00"), "currency": "INR",
    }
    base.update(kw)
    return ExtractedInvoice(**base)


OCR_CLEAN = (
    "Demo Sharma Electronics\nGSTIN 27AAPFU0939F1ZV\nInvoice No INV-2041\nDate 14/09/2026\n"
    "Alpha Test Services LLP\nGSTIN 27AAPFU0939F1ZV\nTaxable 10,000.00\nCGST 900.00\nSGST 900.00\nTotal 11,800.00"
)
