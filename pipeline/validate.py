"""Deterministic GST validation. No LLM is involved anywhere in this module.

Principles
* A check that cannot be decided from the available data produces a *warning for review*, never a
  silent pass or a definitive legal conclusion.
* Extracted amounts are never changed to make the numbers add up.
* A GSTIN that passes the format and checksum tests is only *well-formed*. This module cannot tell
  whether it is registered or active (that needs the GST portal).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date, datetime
from decimal import Decimal
from typing import Any, Iterable

from config import ValidationSettings, load_rate_periods, load_state_codes
from pipeline.schemas import (
    AMOUNT_FIELDS,
    EXTRACTION_FIELDS,
    FIELD_LABELS,
    ExtractedInvoice,
    OcrLine,
    Severity,
    ValidationIssue,
)

ZERO = Decimal("0")
GST_START = date(2017, 7, 1)

# ======================================================================
# GSTIN
# ======================================================================
GSTIN_CHARS = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"
# 2-digit state, 10-char PAN (5 letters, 4 digits, 1 letter), entity no., 'Z', check character.
# Covers regular taxpayers. Special-purpose ids (TDS, UN bodies, ...) use other patterns.
GSTIN_PATTERN = re.compile(r"^[0-9]{2}[A-Z]{5}[0-9]{4}[A-Z][1-9A-Z]Z[0-9A-Z]$")
_STRIP = re.compile(r"[\s\-​ ]+")


def normalize_gstin(value: str | None) -> str | None:
    """Upper-case and remove whitespace/hyphens only. Never swaps look-alike characters (O/0, I/1)."""
    if value is None:
        return None
    cleaned = _STRIP.sub("", value).upper()
    return cleaned or None


def gstin_checksum_char(first14: str) -> str | None:
    """Check character for the first 14 characters (mod-36 weighted sum), or None for invalid input."""
    if len(first14) != 14 or any(c not in GSTIN_CHARS for c in first14):
        return None
    total = 0
    for i, ch in enumerate(first14):
        product = GSTIN_CHARS.index(ch) * (1 if i % 2 == 0 else 2)
        total += product // 36 + product % 36
    return GSTIN_CHARS[(36 - total % 36) % 36]


def is_valid_gstin_format(gstin: str | None) -> bool:
    return bool(gstin) and bool(GSTIN_PATTERN.match(gstin))  # type: ignore[arg-type]


def is_valid_gstin_checksum(gstin: str | None) -> bool:
    return bool(gstin) and len(gstin) == 15 and gstin_checksum_char(gstin[:14]) == gstin[14]  # type: ignore[index]


@dataclass(frozen=True)
class GstinCheck:
    raw: str | None
    normalized: str | None
    format_ok: bool
    checksum_ok: bool
    state_code: str | None
    state_name: str | None
    state_known: bool

    @property
    def fully_valid(self) -> bool:
        return self.format_ok and self.checksum_ok


def check_gstin(value: str | None, state_codes: dict[str, Any] | None = None) -> GstinCheck:
    state_codes = state_codes if state_codes is not None else load_state_codes()
    norm = normalize_gstin(value)
    fmt = is_valid_gstin_format(norm)
    chk = is_valid_gstin_checksum(norm)
    code = norm[:2] if norm and len(norm) >= 2 and norm[:2].isdigit() else None
    info = state_codes.get(code) if code else None
    return GstinCheck(value, norm, fmt, chk, code, info["name"] if info else None, info is not None)


# ======================================================================
# Dates
# ======================================================================
_MONTHS = {
    m: i for i, names in enumerate(
        [("jan", "january"), ("feb", "february"), ("mar", "march"), ("apr", "april"), ("may",), ("jun", "june"),
         ("jul", "july"), ("aug", "august"), ("sep", "sept", "september"), ("oct", "october"),
         ("nov", "november"), ("dec", "december")], start=1)
    for m in names
}
_ISO = re.compile(r"^(\d{4})[-/.](\d{1,2})[-/.](\d{1,2})$")
_NUMERIC = re.compile(r"^(\d{1,2})[-/.](\d{1,2})[-/.](\d{2}|\d{4})$")
_DAY_MON_YEAR = re.compile(r"^(\d{1,2})(?:st|nd|rd|th)?[\s\-/.,]*([A-Za-z]{3,9})\.?[\s\-/.,]*(\d{2}|\d{4})$")
_MON_DAY_YEAR = re.compile(r"^([A-Za-z]{3,9})\.?[\s\-/.]*(\d{1,2})(?:st|nd|rd|th)?[\s,\-/.]*(\d{2}|\d{4})$")


@dataclass(frozen=True)
class DateParse:
    iso: str | None
    ambiguous: bool = False
    two_digit_year: bool = False
    problem: str | None = None  # 'invalid' | 'unrecognised'
    detail: str = ""


def _build(y: int, m: int, d: int, **flags: Any) -> DateParse:
    try:
        return DateParse(date(y, m, d).isoformat(), **flags)
    except ValueError:
        return DateParse(None, problem="invalid", detail=f"{d:02d}/{m:02d}/{y} is not a real calendar date")


def _year(text: str) -> tuple[int, bool]:
    return (2000 + int(text), True) if len(text) == 2 else (int(text), False)


def parse_invoice_date(text: str | None, day_first: bool = True) -> DateParse:
    """Parse the date formats seen on Indian invoices. Returns ISO text, or why it could not be trusted."""
    if text is None or not text.strip():
        return DateParse(None, problem="unrecognised", detail="empty")
    s = re.sub(r"\s+", " ", text.strip())
    if m := _ISO.match(s):
        return _build(int(m[1]), int(m[2]), int(m[3]))
    if m := _NUMERIC.match(s):
        a, b = int(m[1]), int(m[2])
        year, short = _year(m[3])
        if a > 12 and b <= 12:
            day, month, ambiguous = a, b, False
        elif b > 12 and a <= 12:
            day, month, ambiguous = b, a, False
        elif a <= 12 and b <= 12:
            day, month = (a, b) if day_first else (b, a)
            ambiguous = a != b
        else:
            return DateParse(None, problem="invalid", detail=f"'{s}' has no valid day/month combination")
        return _build(year, month, day, ambiguous=ambiguous, two_digit_year=short)
    if m := _DAY_MON_YEAR.match(s):
        month = _MONTHS.get(m[2].lower())
        if month:
            year, short = _year(m[3])
            return _build(year, month, int(m[1]), two_digit_year=short)
    if m := _MON_DAY_YEAR.match(s):
        month = _MONTHS.get(m[1].lower())
        if month:
            year, short = _year(m[3])
            return _build(year, month, int(m[2]), two_digit_year=short)
    return DateParse(None, problem="unrecognised", detail=f"'{s}' is not a date format this tool understands")


_DATE_CANDIDATES = re.compile(
    r"\d{4}[-/.]\d{1,2}[-/.]\d{1,2}|\d{1,2}[-/.]\d{1,2}[-/.](?:\d{4}|\d{2})\b|"
    r"\d{1,2}(?:st|nd|rd|th)?[ \-/.,]*[A-Za-z]{3,9}\.?[ \-/.,]*\d{2,4}\b"
)


# ======================================================================
# Helpers
# ======================================================================
def _issue(code: str, severity: Severity, message: str, *fields: str) -> ValidationIssue:
    return ValidationIssue(code=code, severity=severity, message=message, fields=list(fields))


def _money(value: Decimal | None) -> str:
    return "missing" if value is None else f"₹{value:,.2f}"


def _present(*values: Decimal | None) -> list[Decimal]:
    return [v for v in values if v is not None]


_NUMBER = re.compile(r"\d[\d,]*(?:\.\d+)?")


def numbers_in(text: str) -> list[Decimal]:
    out: list[Decimal] = []
    for token in _NUMBER.findall(text):
        try:
            out.append(Decimal(token.replace(",", "").rstrip(".")))
        except Exception:
            continue
    return out


def _squash(text: str) -> str:
    return re.sub(r"[^A-Z0-9]", "", text.upper())


_SPECIAL_PATTERNS = {
    "reverse_charge": re.compile(r"reverse\s+charge[^\n]{0,30}?(?:applicable|:|-)?\s*(?:yes|y\b|applicable)", re.I),
    "exempt": re.compile(r"\b(?:nil[\s-]?rated|exempt(?:ed)?|bill\s+of\s+supply)\b", re.I),
    "zero_rated": re.compile(r"\b(?:zero[\s-]?rated|export\s+invoice|supply\s+to\s+sez|under\s+lut|letter\s+of\s+undertaking)\b", re.I),
    "composition": re.compile(r"composition\s+(?:taxable\s+person|dealer|scheme)", re.I),
}
_EXTRA_CHARGE = re.compile(r"\b(?:cess|round(?:ing)?[\s-]*off|tcs|discount|freight|shipping|packing|insurance|other\s+charges)\b", re.I)


def detect_special_cases(text: str, notes: Iterable[str] = ()) -> set[str]:
    haystack = text + "\n" + "\n".join(notes)
    return {name for name, pattern in _SPECIAL_PATTERNS.items() if pattern.search(haystack)}


# ======================================================================
# Normalization (deterministic clean-up that is reported, never silent)
# ======================================================================
def normalize_invoice(invoice: ExtractedInvoice, settings: ValidationSettings) -> ExtractedInvoice:
    """Normalise GSTIN case/spacing and unambiguous dates to ISO. Amounts are never touched."""
    updates: dict[str, Any] = {}
    notes = list(invoice.extraction_notes)
    for name in ("vendor_gstin", "buyer_gstin"):
        original = getattr(invoice, name)
        normalized = normalize_gstin(original)
        if original is not None and normalized != original:
            updates[name] = normalized
            notes.append(f"{FIELD_LABELS[name]} normalised from '{original}' to '{normalized}' (case/spacing only)")
    if invoice.invoice_date:
        parsed = parse_invoice_date(invoice.invoice_date, settings.date_day_first)
        if parsed.iso and not parsed.ambiguous and not parsed.two_digit_year and parsed.iso != invoice.invoice_date:
            notes.append(f"Invoice date normalised from '{invoice.invoice_date}' to {parsed.iso}")
            updates["invoice_date"] = parsed.iso
    if updates:
        updates["extraction_notes"] = notes
        return invoice.model_copy(update=updates)
    return invoice


def normalized_date_or_raw(invoice_date: str | None, day_first: bool = True) -> str | None:
    if not invoice_date:
        return None
    return parse_invoice_date(invoice_date, day_first).iso or invoice_date


# ======================================================================
# Individual checks
# ======================================================================
def check_required_fields(inv: ExtractedInvoice, required: Iterable[str]) -> list[ValidationIssue]:
    issues = []
    for name in required:
        if getattr(inv, name, None) in (None, ""):
            extra = ""
            if name == "vendor_gstin":
                extra = " (receipts from unregistered sellers legitimately have none)"
            issues.append(_issue("MISSING_FIELD", Severity.WARNING, f"{FIELD_LABELS.get(name, name)} is missing{extra}.", name))
    return issues


def check_gstin_fields(inv: ExtractedInvoice, state_codes: dict[str, Any]) -> list[ValidationIssue]:
    issues: list[ValidationIssue] = []
    for field_name in ("vendor_gstin", "buyer_gstin"):
        value = getattr(inv, field_name)
        if not value:
            continue
        label = FIELD_LABELS[field_name]
        res = check_gstin(value, state_codes)
        if not res.format_ok:
            issues.append(_issue(
                "GSTIN_FORMAT_INVALID", Severity.ERROR,
                f"{label} '{res.normalized}' does not match the standard 15-character GSTIN pattern "
                "(state code, PAN, entity number, 'Z', check character). Compare it with the document; "
                "special-purpose ids (e.g. TDS, UN bodies) use other patterns.", field_name))
        elif not res.checksum_ok:
            issues.append(_issue(
                "GSTIN_CHECKSUM_INVALID", Severity.ERROR,
                f"{label} '{res.normalized}' has the right shape but its check character is wrong. "
                "This usually means a misread or mistyped character.", field_name))
        if res.format_ok:
            if res.state_code is not None and not res.state_known:
                issues.append(_issue("UNKNOWN_STATE_CODE", Severity.WARNING,
                                     f"{label} starts with state code {res.state_code}, which is not in the state-code table.", field_name))
            elif res.state_code and state_codes.get(res.state_code, {}).get("legacy"):
                issues.append(_issue("LEGACY_STATE_CODE", Severity.INFO,
                                     f"{label} uses state code {res.state_code} ({res.state_name}), which is a legacy code.", field_name))
    return issues


def check_date(inv: ExtractedInvoice, settings: ValidationSettings, today: date) -> list[ValidationIssue]:
    if not inv.invoice_date:
        return []
    parsed = parse_invoice_date(inv.invoice_date, settings.date_day_first)
    if parsed.problem == "invalid":
        return [_issue("INVALID_DATE", Severity.ERROR, f"Invoice date '{inv.invoice_date}': {parsed.detail}.", "invoice_date")]
    if parsed.problem == "unrecognised":
        return [_issue("UNRECOGNISED_DATE", Severity.WARNING, f"Invoice date {parsed.detail}. Enter it as YYYY-MM-DD.", "invoice_date")]
    issues = []
    order = "day/month" if settings.date_day_first else "month/day"
    if parsed.ambiguous and settings.flag_ambiguous_dates:
        issues.append(_issue("AMBIGUOUS_DATE", Severity.WARNING,
                             f"Invoice date '{inv.invoice_date}' could be read two ways; it was read as {order} = {parsed.iso}. Please confirm.",
                             "invoice_date"))
    if parsed.two_digit_year:
        issues.append(_issue("TWO_DIGIT_YEAR", Severity.WARNING,
                             f"Invoice date '{inv.invoice_date}' has a 2-digit year; it was read as {parsed.iso}. Please confirm.",
                             "invoice_date"))
    d = date.fromisoformat(parsed.iso)  # type: ignore[arg-type]
    if d > today and settings.warn_on_future_date:
        issues.append(_issue("FUTURE_DATE", Severity.WARNING,
                             f"Invoice date {d.isoformat()} is in the future. This is a warning, not proof of an error "
                             "(post-dated invoices exist, but misread digits are more common).", "invoice_date"))
    if d < GST_START:
        issues.append(_issue("DATE_BEFORE_GST", Severity.WARNING,
                             f"Invoice date {d.isoformat()} is before GST began (1 July 2017). Check the year.", "invoice_date"))
    return issues


def check_amounts(inv: ExtractedInvoice, settings: ValidationSettings, special: set[str], extra_charge_hint: bool) -> list[ValidationIssue]:
    issues: list[ValidationIssue] = []
    tol = settings.amount_tolerance
    taxes = {"cgst": inv.cgst, "sgst": inv.sgst, "igst": inv.igst}

    for name in AMOUNT_FIELDS:
        value = getattr(inv, name)
        if value is not None and value < 0:
            issues.append(_issue("NEGATIVE_AMOUNT", Severity.WARNING,
                                 f"{FIELD_LABELS[name]} is negative ({_money(value)}). This is normal for a credit note and unusual otherwise.", name))

    if inv.taxable_amount is None:
        issues.append(_issue("MISSING_AMOUNT", Severity.WARNING, "Taxable amount is missing, so the total cannot be reconciled.", "taxable_amount"))
    if inv.total_amount is None:
        issues.append(_issue("MISSING_AMOUNT", Severity.WARNING, "Total amount is missing, so it cannot be reconciled.", "total_amount"))

    no_tax_extracted = all(v is None for v in taxes.values())
    special_no_tax = bool(special & {"exempt", "zero_rated", "composition"})
    if no_tax_extracted:
        if special_no_tax:
            issues.append(_issue("SPECIAL_CASE_NO_TAX", Severity.INFO,
                                 "No tax amounts were found and the document looks exempt / zero-rated / bill of supply "
                                 f"({', '.join(sorted(special))}), so none are expected.", "cgst", "sgst", "igst"))
        else:
            detail = ""
            if inv.taxable_amount is not None and inv.total_amount is not None and abs(inv.total_amount - inv.taxable_amount) > tol:
                detail = (f" The total ({_money(inv.total_amount)}) is {_money(inv.total_amount - inv.taxable_amount)} above the "
                          "taxable amount, which may be tax that was not extracted.")
            issues.append(_issue("TAX_AMOUNTS_MISSING", Severity.WARNING,
                                 "No CGST, SGST or IGST amount was extracted." + detail, "cgst", "sgst", "igst"))
    elif inv.taxable_amount is not None and inv.total_amount is not None:
        expected = inv.taxable_amount + sum(_present(*taxes.values()), ZERO)
        diff = inv.total_amount - expected
        if abs(diff) > tol:
            soft = bool(special) or extra_charge_hint
            hint = " The document mentions cess, round-off, discount or other charges, which are not part of this check." if extra_charge_hint else ""
            if special:
                hint += f" Special case detected ({', '.join(sorted(special))})."
            missing_parts = [n.upper() for n, v in taxes.items() if v is None]
            if missing_parts and not soft:
                hint += f" {', '.join(missing_parts)} was not extracted and is assumed to be absent."
            issues.append(_issue(
                "TOTAL_MISMATCH", Severity.WARNING if soft else Severity.ERROR,
                f"Taxable {_money(inv.taxable_amount)} + taxes {_money(expected - inv.taxable_amount)} = {_money(expected)}, "
                f"but the total is {_money(inv.total_amount)} (difference {_money(diff)}, tolerance ₹{tol}).{hint}",
                "taxable_amount", "cgst", "sgst", "igst", "total_amount"))
        elif all(v == ZERO for v in _present(*taxes.values())) and not special_no_tax and len(_present(*taxes.values())) == 3:
            issues.append(_issue("ZERO_TAX", Severity.WARNING,
                                 "All three taxes are 0. Confirm the supply is exempt, nil-rated, zero-rated or from a composition dealer.",
                                 "cgst", "sgst", "igst"))
    return issues


def check_tax_consistency(inv: ExtractedInvoice, settings: ValidationSettings) -> list[ValidationIssue]:
    issues: list[ValidationIssue] = []
    tol = settings.amount_tolerance
    c, s, i = inv.cgst, inv.sgst, inv.igst
    if (c is None) != (s is None) and (c or s):
        missing = "SGST" if s is None else "CGST"
        issues.append(_issue("CGST_SGST_INCOMPLETE", Severity.WARNING,
                             f"Only one of CGST/SGST was extracted ({missing} is missing). They are normally charged together in equal amounts.",
                             "cgst", "sgst"))
    if c is not None and s is not None and abs(c - s) > tol:
        issues.append(_issue("CGST_SGST_UNEQUAL", Severity.WARNING,
                             f"CGST ({_money(c)}) and SGST ({_money(s)}) differ by more than ₹{tol}. They are normally equal "
                             "(different rates per line item or a misread digit can cause this).", "cgst", "sgst"))
    if i is not None and i > ZERO and ((c is not None and c > ZERO) or (s is not None and s > ZERO)):
        issues.append(_issue("INTRA_INTER_CONFLICT", Severity.WARNING,
                             "IGST (inter-state) and CGST/SGST (intra-state) are both charged. A single supply is normally one or the other; "
                             "this can happen with mixed supplies or a misread value.", "cgst", "sgst", "igst"))
    return issues


def check_state_consistency(inv: ExtractedInvoice, state_codes: dict[str, Any]) -> list[ValidationIssue]:
    v = check_gstin(inv.vendor_gstin, state_codes)
    b = check_gstin(inv.buyer_gstin, state_codes)
    if not (v.fully_valid and b.fully_valid and v.state_known and b.state_known):
        return []
    c, s, i = inv.cgst, inv.sgst, inv.igst
    intra = ((c or ZERO) > ZERO or (s or ZERO) > ZERO) and not (i or ZERO) > ZERO
    inter = (i or ZERO) > ZERO and not ((c or ZERO) > ZERO or (s or ZERO) > ZERO)
    same_state = v.state_code == b.state_code
    caveat = ("The place of supply is not always the buyer's registered state (e.g. ship-to address, SEZ, "
              "services), so this is a prompt to check, not a conclusion.")
    if same_state and inter:
        return [_issue("STATE_TAX_MISMATCH", Severity.WARNING,
                       f"Seller and buyer GSTINs are both from {v.state_name}, but IGST was charged instead of CGST+SGST. {caveat}",
                       "vendor_gstin", "buyer_gstin", "igst")]
    if not same_state and intra:
        return [_issue("STATE_TAX_MISMATCH", Severity.WARNING,
                       f"Seller GSTIN is from {v.state_name} and buyer GSTIN is from {b.state_name}, but CGST+SGST was charged instead of IGST. {caveat}",
                       "vendor_gstin", "buyer_gstin", "cgst", "sgst")]
    return []


def check_rate(inv: ExtractedInvoice, settings: ValidationSettings, today: date, periods: list[dict[str, Any]]) -> list[ValidationIssue]:
    """Optional, warning-only. The implied overall rate is compared with the configured reference rates."""
    if not settings.rate_check_enabled or not periods:
        return []
    if inv.taxable_amount is None or inv.taxable_amount <= ZERO:
        return []
    taxes = _present(inv.cgst, inv.sgst, inv.igst)
    if not taxes:
        return []
    when = today
    parsed = parse_invoice_date(inv.invoice_date, settings.date_day_first) if inv.invoice_date else None
    if parsed and parsed.iso:
        when = date.fromisoformat(parsed.iso)
    period = next((p for p in periods
                   if date.fromisoformat(p["from"]) <= when and (p.get("to") is None or when <= date.fromisoformat(p["to"]))), None)
    if period is None:
        return []
    rates = [Decimal(r) for r in period["rates_pct"]]
    implied = sum(taxes, ZERO) / inv.taxable_amount * 100
    if any(abs(implied - r) <= settings.rate_tolerance_pct for r in rates):
        return []
    return [_issue("UNUSUAL_IMPLIED_RATE", Severity.WARNING,
                   f"Total tax / taxable amount = {implied:.2f}%, which matches none of the reference rates for {when.isoformat()} "
                   f"({', '.join(str(r) for r in rates)}%). A mixed-rate invoice shows a blended rate, so this is only a prompt to check. "
                   "The reference rates are editable data, not a legal determination.",
                   "taxable_amount", "cgst", "sgst", "igst")]


# ======================================================================
# Grounding and OCR confidence (is the extracted value really in the OCR text?)
# ======================================================================
def _name_tokens(text: str) -> set[str]:
    return {t for t in re.findall(r"[a-z0-9]+", text.lower()) if len(t) > 1}


def check_grounding(inv: ExtractedInvoice, ocr_text: str, settings: ValidationSettings) -> list[ValidationIssue]:
    """Flag values the model returned that cannot be found in the OCR text (possible hallucination)."""
    issues: list[ValidationIssue] = []
    squashed = _squash(ocr_text)
    nums = numbers_in(ocr_text)

    def missing(name: str, what: str) -> None:
        issues.append(_issue("NOT_FOUND_IN_OCR", Severity.WARNING,
                             f"{FIELD_LABELS[name]} {what} was not found in the OCR text. The model may have guessed it; check the document.", name))

    for name in ("vendor_gstin", "buyer_gstin", "invoice_number"):
        value = getattr(inv, name)
        if value and _squash(value) not in squashed:
            missing(name, f"'{value}'")
    for name in ("vendor_name", "buyer_name"):
        value = getattr(inv, name)
        if value:
            tokens = _name_tokens(value)
            if tokens and len(tokens & _name_tokens(ocr_text)) / len(tokens) < 0.6:
                missing(name, f"'{value}'")
    for name in AMOUNT_FIELDS:
        value = getattr(inv, name)
        if value is not None and not any(abs(n - abs(value)) == 0 for n in nums) and value != ZERO:
            missing(name, _money(value))
    if inv.invoice_date:
        wanted = parse_invoice_date(inv.invoice_date, settings.date_day_first).iso
        if wanted:
            found = False
            for candidate in _DATE_CANDIDATES.findall(ocr_text):
                got = parse_invoice_date(candidate, settings.date_day_first)
                if got.iso == wanted:
                    found = True
                    break
            if not found:
                missing("invoice_date", f"'{inv.invoice_date}'")
    return issues


def _line_matches(line: str, name: str, value: Any) -> bool:
    if name in AMOUNT_FIELDS:
        return any(n == abs(value) for n in numbers_in(line))
    if name == "invoice_date":
        parsed = parse_invoice_date(str(value))
        return bool(parsed.iso) and any(parse_invoice_date(c).iso == parsed.iso for c in _DATE_CANDIDATES.findall(line))
    if name in ("vendor_name", "buyer_name"):
        tokens = _name_tokens(str(value))
        return bool(tokens) and len(tokens & _name_tokens(line)) / len(tokens) >= 0.5
    return _squash(str(value)) in _squash(line) and bool(_squash(str(value)))


def field_ocr_confidence(inv: ExtractedInvoice, lines: list[OcrLine]) -> dict[str, float]:
    """Lowest OCR confidence among the lines that contain each extracted value."""
    out: dict[str, float] = {}
    scored = [ln for ln in lines if ln.confidence is not None]
    for name in EXTRACTION_FIELDS:
        value = getattr(inv, name)
        if value in (None, "") or name == "currency":
            continue
        confs = [ln.confidence for ln in scored if _line_matches(ln.text, name, value)]
        if confs:
            out[name] = min(confs)  # type: ignore[type-var]
    return out


def check_low_confidence(inv: ExtractedInvoice, lines: list[OcrLine], threshold: float) -> list[ValidationIssue]:
    issues = []
    for name, conf in field_ocr_confidence(inv, lines).items():
        if conf < threshold:
            issues.append(_issue("LOW_OCR_CONFIDENCE", Severity.WARNING,
                                 f"{FIELD_LABELS[name]} was read with low OCR confidence ({conf:.0%}, threshold {threshold:.0%}). Check it against the document.",
                                 name))
    return issues


# ======================================================================
# Orchestration
# ======================================================================
def validate_invoice(
    invoice: ExtractedInvoice,
    ocr_text: str | None = None,
    settings: ValidationSettings | None = None,
    *,
    ocr_lines: list[OcrLine] | None = None,
    low_confidence_threshold: float = 0.8,
    state_codes: dict[str, Any] | None = None,
    rate_periods: list[dict[str, Any]] | None = None,
    today: date | None = None,
) -> list[ValidationIssue]:
    """Run every deterministic check and return the issues (errors first)."""
    settings = settings or ValidationSettings()
    state_codes = state_codes if state_codes is not None else load_state_codes()
    rate_periods = rate_periods if rate_periods is not None else load_rate_periods()
    today = today or datetime.now().date()
    text = ocr_text or ""
    special = detect_special_cases(text, invoice.extraction_notes)
    extra_charge_hint = bool(_EXTRA_CHARGE.search(text))

    issues: list[ValidationIssue] = []
    issues += check_required_fields(invoice, settings.required_fields)
    issues += check_gstin_fields(invoice, state_codes)
    issues += check_date(invoice, settings, today)
    issues += check_amounts(invoice, settings, special, extra_charge_hint)
    issues += check_tax_consistency(invoice, settings)
    issues += check_state_consistency(invoice, state_codes)
    issues += check_rate(invoice, settings, today, rate_periods)
    if ocr_text:
        issues += check_grounding(invoice, ocr_text, settings)
    if ocr_lines:
        issues += check_low_confidence(invoice, ocr_lines, low_confidence_threshold)

    order = {Severity.ERROR: 0, Severity.WARNING: 1, Severity.INFO: 2}
    return sorted(issues, key=lambda x: order[x.severity])


def count_by_severity(issues: Iterable[ValidationIssue]) -> dict[str, int]:
    counts = {s.value: 0 for s in Severity}
    for issue in issues:
        counts[issue.severity.value] += 1
    return counts


def field_review_status(
    invoice: ExtractedInvoice,
    issues: list[ValidationIssue],
    ocr_lines: list[OcrLine] | None = None,
    edited_fields: Iterable[str] = (),
    low_confidence_threshold: float = 0.8,
) -> dict[str, dict[str, Any]]:
    """Per-field status for the review screen and export: ok | missing | low_ocr_confidence | flagged | edited."""
    confidences = field_ocr_confidence(invoice, ocr_lines or [])
    edited = set(edited_fields)
    result: dict[str, dict[str, Any]] = {}
    for name in EXTRACTION_FIELDS:
        value = getattr(invoice, name)
        reasons = [i.short() for i in issues if name in i.fields and i.severity != Severity.INFO]
        if value in (None, ""):
            status = "missing"
        elif any(i.severity == Severity.ERROR and name in i.fields for i in issues):
            status = "flagged"
        elif name in confidences and confidences[name] < low_confidence_threshold:
            status = "low_ocr_confidence"
        elif reasons:
            status = "flagged"
        else:
            status = "ok"
        if name in edited:
            reasons = ["Edited by a person"] + reasons
        result[name] = {
            "status": status,
            "ocr_confidence": round(confidences[name], 4) if name in confidences else None,
            "reasons": reasons,
            "edited": name in edited,
        }
    return result
