"""Data models shared by every pipeline stage."""

from __future__ import annotations

import math
import re
from decimal import Decimal, InvalidOperation
from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator


class Severity(str, Enum):
    ERROR = "error"
    WARNING = "warning"
    INFO = "info"


class InvoiceStatus(str, Enum):
    PENDING = "Pending"
    PROCESSING = "Processing"
    # Machine-extracted, no validation errors or warnings, NOT yet checked by a human.
    EXTRACTED = "Extracted"
    NEEDS_REVIEW = "Needs Review"
    REVIEWED = "Reviewed"
    FAILED = "Failed"


class ValidationIssue(BaseModel):
    """One deterministic check result."""

    code: str
    severity: Severity
    message: str
    fields: list[str] = Field(default_factory=list)

    def short(self) -> str:
        return f"[{self.code}] {self.message}"


# ----------------------------------------------------------------------
# Extraction schema (what the local LLM is asked to return)
# ----------------------------------------------------------------------
_NULL_WORDS = {"", "null", "none", "n/a", "na", "nil", "unknown", "not available", "not found", "-", "--"}
_CURRENCY_NOISE = re.compile(r"(?i)(₹|rs\.?|inr|rupees?|/-|\s)")


def _clean_text(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, bool):
        raise ValueError("expected text, got a boolean")
    if isinstance(value, (int, float, Decimal)):
        value = str(value)
    if not isinstance(value, str):
        raise ValueError("expected text")
    value = value.strip()
    if value.lower() in _NULL_WORDS:
        return None
    return value


MAX_AMOUNT_DIGITS = 15      # amounts up to 10^15 rupees (1,000 trillion) are accepted
MAX_AMOUNT_DECIMALS = 20


def parse_amount(value: Any) -> Decimal | None:
    """Convert LLM/user input to an exact Decimal, or None when the value is missing.

    The digits are never rounded. Unknown values stay ``None`` (never 0).
    """
    if value is None:
        return None
    if isinstance(value, bool):
        raise ValueError("expected an amount, got a boolean")
    if isinstance(value, Decimal):
        number = value
    elif isinstance(value, int):
        number = Decimal(value)
    elif isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("amount is not finite")
        number = Decimal(repr(value))
    elif isinstance(value, str):
        text = value.strip()
        if text.lower() in _NULL_WORDS:
            return None
        negative = text.startswith("(") and text.endswith(")")
        text = _CURRENCY_NOISE.sub("", text.strip("()")).replace(",", "")
        if text.endswith("-") and not text.startswith("-"):  # "1500-"
            text = "-" + text[:-1]
        try:
            number = Decimal(text)
        except InvalidOperation as exc:
            raise ValueError(f"'{value}' is not a valid amount") from exc
        if negative:
            number = -abs(number)
    else:
        raise ValueError("expected an amount")
    if not number.is_finite():
        raise ValueError("amount is not finite")
    # Guard against absurd exponents such as 1e999999999 (formatting them would exhaust memory).
    exponent = number.as_tuple().exponent
    if number != 0 and (number.adjusted() > MAX_AMOUNT_DIGITS or (isinstance(exponent, int) and exponent < -MAX_AMOUNT_DECIMALS)):
        raise ValueError("amount is outside the supported range")
    return number


class ExtractedInvoice(BaseModel):
    """Structured invoice fields. ``None`` means missing/unreadable, never zero."""

    model_config = ConfigDict(extra="forbid")

    vendor_name: str | None = None
    vendor_gstin: str | None = None
    buyer_name: str | None = None
    buyer_gstin: str | None = None
    invoice_number: str | None = None
    invoice_date: str | None = None
    taxable_amount: Decimal | None = None
    cgst: Decimal | None = None
    sgst: Decimal | None = None
    igst: Decimal | None = None
    total_amount: Decimal | None = None
    currency: str | None = "INR"
    extraction_notes: list[str] = Field(default_factory=list)

    @field_validator(
        "vendor_name", "vendor_gstin", "buyer_name", "buyer_gstin", "invoice_number", "invoice_date",
        mode="before",
    )
    @classmethod
    def _text(cls, v: Any) -> Any:
        return _clean_text(v)

    @field_validator("currency", mode="before")
    @classmethod
    def _currency(cls, v: Any) -> Any:
        v = _clean_text(v)
        if v is None:
            return None
        v = v.upper()
        return "INR" if v in {"₹", "RS", "RS."} else v

    @field_validator("taxable_amount", "cgst", "sgst", "igst", "total_amount", mode="before")
    @classmethod
    def _amount(cls, v: Any) -> Any:
        return parse_amount(v)

    @field_validator("extraction_notes", mode="before")
    @classmethod
    def _notes(cls, v: Any) -> Any:
        if v is None:
            return []
        if isinstance(v, str):
            return [v] if v.strip() else []
        return [str(x) for x in v]


TEXT_FIELDS = ["vendor_name", "vendor_gstin", "buyer_name", "buyer_gstin", "invoice_number", "invoice_date", "currency"]
AMOUNT_FIELDS = ["taxable_amount", "cgst", "sgst", "igst", "total_amount"]
EXTRACTION_FIELDS = [
    "vendor_name", "vendor_gstin", "buyer_name", "buyer_gstin", "invoice_number", "invoice_date",
    "taxable_amount", "cgst", "sgst", "igst", "total_amount", "currency",
]
FIELD_LABELS = {
    "vendor_name": "Vendor name",
    "vendor_gstin": "Vendor GSTIN",
    "buyer_name": "Buyer name",
    "buyer_gstin": "Buyer GSTIN",
    "invoice_number": "Invoice number",
    "invoice_date": "Invoice date",
    "taxable_amount": "Taxable amount",
    "cgst": "CGST",
    "sgst": "SGST",
    "igst": "IGST",
    "total_amount": "Total amount",
    "currency": "Currency",
}


# ----------------------------------------------------------------------
# OCR output (kept separate from the document and from the extracted data)
# ----------------------------------------------------------------------
class OcrLine(BaseModel):
    text: str
    confidence: float | None = None  # 0..1
    # Axis-aligned box in the pixel space of the image that was OCR'd: [x0, y0, x1, y1]
    bbox: list[float] = Field(default_factory=list)


class OcrPage(BaseModel):
    text: str = ""
    lines: list[OcrLine] = Field(default_factory=list)
    mean_confidence: float | None = None
    engine: str = ""
    error: str | None = None
    elapsed_ms: int = 0
    used_original_image: bool = False

    @property
    def ok(self) -> bool:
        return self.error is None


def to_jsonable(value: Any) -> Any:
    """Decimal-safe conversion used when storing invoices as JSON (Decimals become strings)."""
    if isinstance(value, Decimal):
        return format(value, "f")
    if isinstance(value, dict):
        return {k: to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(v) for v in value]
    return value
