"""Likely-duplicate detection across invoices.

Duplicates are only *flagged* for a person to review. Nothing is merged or deleted automatically.
(Byte-identical files are already skipped at import time by their content hash.)
"""

from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass
from decimal import Decimal
from itertools import combinations
from typing import Iterable

_STRENGTH_RANK = {"high": 3, "medium": 2, "low": 1}


@dataclass(frozen=True)
class DupInput:
    invoice_id: str
    vendor_gstin: str | None = None
    vendor_name: str | None = None
    invoice_number: str | None = None
    invoice_date: str | None = None   # ISO if parseable, otherwise raw text
    total_amount: Decimal | None = None
    source_file: str = ""


@dataclass(frozen=True)
class DuplicateCandidate:
    other_id: str
    reason: str
    strength: str  # 'high' | 'medium' | 'low'


def _alnum(text: str | None) -> str | None:
    if not text:
        return None
    out = re.sub(r"[^a-z0-9]", "", text.lower())
    return out or None


def normalize_invoice_number(text: str | None) -> str | None:
    """'INV-0041' and 'inv 41'... only separators/case are ignored; leading zeros are kept on purpose."""
    return _alnum(text)


def find_duplicates(items: Iterable[DupInput]) -> dict[str, list[DuplicateCandidate]]:
    """Return {invoice_id: [candidates]} for every invoice that looks like a duplicate of another."""
    items = list(items)
    found: dict[tuple[str, str], DuplicateCandidate] = {}
    by_id = {i.invoice_id: i for i in items}

    def record(a: str, b: str, reason: str, strength: str) -> None:
        for x, y in ((a, b), (b, a)):
            current = found.get((x, y))
            if current is None or _STRENGTH_RANK[strength] > _STRENGTH_RANK[current.strength]:
                found[(x, y)] = DuplicateCandidate(y, reason, strength)

    def buckets(key_fn) -> Iterable[list[DupInput]]:
        groups: dict[tuple, list[DupInput]] = defaultdict(list)
        for item in items:
            key = key_fn(item)
            if key is not None:
                groups[key].append(item)
        return [g for g in groups.values() if len(g) > 1]

    gstin = lambda i: (i.vendor_gstin or "").replace(" ", "").upper() or None  # noqa: E731
    number = lambda i: normalize_invoice_number(i.invoice_number)  # noqa: E731
    name = lambda i: _alnum(i.vendor_name)  # noqa: E731

    # 1. Same vendor GSTIN + same invoice number (strongest signal)
    for group in buckets(lambda i: (gstin(i), number(i)) if gstin(i) and number(i) else None):
        for a, b in combinations(group, 2):
            if a.invoice_date and b.invoice_date and a.invoice_date != b.invoice_date:
                record(a.invoice_id, b.invoice_id,
                       "Same vendor GSTIN and invoice number but different dates (could be a different financial year)", "medium")
            else:
                record(a.invoice_id, b.invoice_id, "Same vendor GSTIN, invoice number and date", "high")

    # 2. Vendor GSTIN missing on one side: same vendor name + same invoice number
    for group in buckets(lambda i: (name(i), number(i)) if name(i) and number(i) else None):
        for a, b in combinations(group, 2):
            if (a.invoice_id, b.invoice_id) not in {(x, y) for (x, y) in found}:
                record(a.invoice_id, b.invoice_id, "Same vendor name and invoice number (GSTIN missing or different)", "medium")

    # 3. No invoice number to compare: same vendor + date + total
    def weak_key(i: DupInput):
        vendor = gstin(i) or name(i)
        if vendor and i.invoice_date and i.total_amount is not None and not number(i):
            return (vendor, i.invoice_date, i.total_amount)
        return None

    for group in buckets(weak_key):
        for a, b in combinations(group, 2):
            record(a.invoice_id, b.invoice_id, "Same vendor, date and total, and no invoice number to compare", "low")

    # 4. Same invoice number, date and total, vendor unknown on both
    def anon_key(i: DupInput):
        if not gstin(i) and not name(i) and number(i) and i.invoice_date and i.total_amount is not None:
            return (number(i), i.invoice_date, i.total_amount)
        return None

    for group in buckets(anon_key):
        for a, b in combinations(group, 2):
            record(a.invoice_id, b.invoice_id, "Same invoice number, date and total (vendor unknown)", "low")

    result: dict[str, list[DuplicateCandidate]] = defaultdict(list)
    for (owner, _), cand in found.items():
        if cand.other_id in by_id:
            result[owner].append(cand)
    for owner in result:
        result[owner].sort(key=lambda c: -_STRENGTH_RANK[c.strength])
    return dict(result)
