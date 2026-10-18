"""Duplicate-candidate detection. Candidates are only flagged, never merged or removed."""

from __future__ import annotations

from decimal import Decimal

from pipeline.duplicates import DupInput, find_duplicates, normalize_invoice_number

G1, G2 = "27AAPFU0939F1ZV", "07AAPFU0939F1ZX"


def D(i, **kw):
    return DupInput(i, **kw)


def test_same_gstin_number_and_date_is_a_high_strength_duplicate_both_ways():
    out = find_duplicates([D("a", vendor_gstin=G1, invoice_number="INV-1", invoice_date="2026-09-14"),
                           D("b", vendor_gstin=G1, invoice_number="inv 1", invoice_date="2026-09-14")])
    assert out["a"][0].other_id == "b" and out["a"][0].strength == "high"
    assert out["b"][0].other_id == "a"


def test_same_gstin_and_number_but_different_date_is_only_medium():
    out = find_duplicates([D("a", vendor_gstin=G1, invoice_number="42", invoice_date="2026-09-14"),
                           D("b", vendor_gstin=G1, invoice_number="42", invoice_date="2025-09-14")])
    assert out["a"][0].strength == "medium" and "different dates" in out["a"][0].reason


def test_different_vendors_with_same_invoice_number_are_not_duplicates():
    assert find_duplicates([D("a", vendor_gstin=G1, invoice_number="42"), D("b", vendor_gstin=G2, invoice_number="42")]) == {}


def test_fallback_to_vendor_name_when_gstin_is_missing():
    out = find_duplicates([D("a", vendor_name="Demo Traders", invoice_number="42"),
                           D("b", vendor_name="DEMO  TRADERS", invoice_number="42")])
    assert out["a"][0].strength == "medium"


def test_fallback_without_invoice_number_uses_vendor_date_and_total_and_is_low():
    kw = dict(vendor_gstin=G1, invoice_date="2026-09-14", total_amount=Decimal("500.00"))
    out = find_duplicates([D("a", **kw), D("b", **kw)])
    assert out["a"][0].strength == "low"
    assert find_duplicates([D("a", **kw), D("b", **{**kw, "total_amount": Decimal("501.00")})]) == {}


def test_anonymous_invoices_match_on_number_date_and_total():
    kw = dict(invoice_number="77", invoice_date="2026-09-14", total_amount=Decimal("90"))
    assert find_duplicates([D("a", **kw), D("b", **kw)])["a"][0].strength == "low"


def test_nothing_comparable_means_no_candidates():
    assert find_duplicates([D("a"), D("b"), D("c", vendor_name="x")]) == {}


def test_three_way_duplicates_list_each_other_strongest_first():
    kw = dict(vendor_gstin=G1, invoice_number="9", invoice_date="2026-01-01")
    out = find_duplicates([D("a", **kw), D("b", **kw), D("c", **{**kw, "invoice_date": "2025-01-01"})])
    assert {c.other_id for c in out["a"]} == {"b", "c"}
    assert out["a"][0].strength == "high"


def test_leading_zeros_are_kept_because_001_and_1_are_different_invoices():
    assert normalize_invoice_number("INV-001") != normalize_invoice_number("INV-1")
    assert normalize_invoice_number("inv 001") == normalize_invoice_number("INV-001")
    assert find_duplicates([D("a", vendor_gstin=G1, invoice_number="001"), D("b", vendor_gstin=G1, invoice_number="1")]) == {}


def test_input_is_not_modified():
    items = [D("a", vendor_gstin=G1, invoice_number="1"), D("b", vendor_gstin=G1, invoice_number="1")]
    before = list(items)
    find_duplicates(items)
    assert items == before
