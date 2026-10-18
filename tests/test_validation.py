"""Deterministic GST validation: amounts, taxes, states, dates, grounding, special cases."""

from __future__ import annotations

from datetime import date
from decimal import Decimal

import pytest

from config import ValidationSettings
from pipeline.schemas import ExtractedInvoice, OcrLine, Severity
from pipeline.validate import (
    count_by_severity,
    detect_special_cases,
    field_review_status,
    normalize_invoice,
    parse_invoice_date,
    validate_invoice,
)
from tests.conftest import OCR_CLEAN, TODAY, invoice

D = Decimal
DELHI = "07AAPFU0939F1Z"  # first 14 chars; a valid check character is appended below


def gstin_for(state: str) -> str:
    from pipeline.validate import gstin_checksum_char

    first14 = f"{state}AAPFU0939F1Z"
    return first14 + gstin_checksum_char(first14)  # type: ignore[operator]


def run(inv: ExtractedInvoice, text: str | None = OCR_CLEAN, **kw):
    return validate_invoice(inv, text, kw.pop("settings", None), today=TODAY, **kw)


def codes(issues, severity: Severity | None = None) -> list[str]:
    return [i.code for i in issues if severity is None or i.severity == severity]


# ----------------------------------------------------------------------
def test_clean_invoice_has_no_errors_or_warnings():
    issues = run(invoice())
    assert [i for i in issues if i.severity != Severity.INFO] == []


def test_every_issue_carries_code_severity_message_and_fields():
    issues = run(invoice(total_amount=D("12000.00")))
    assert issues
    for i in issues:
        assert i.code and i.message and isinstance(i.severity, Severity) and isinstance(i.fields, list)


# ---- monetary reconciliation -------------------------------------------------------------
def test_total_mismatch_is_an_error_and_names_the_fields():
    issues = run(invoice(total_amount=D("11850.00")))
    mismatch = [i for i in issues if i.code == "TOTAL_MISMATCH"]
    assert len(mismatch) == 1 and mismatch[0].severity == Severity.ERROR
    assert {"taxable_amount", "total_amount"} <= set(mismatch[0].fields)
    assert "50.00" in mismatch[0].message


@pytest.mark.parametrize("total, expect_flag", [("11800.00", False), ("11800.99", False), ("11801.00", False),
                                                  ("11801.01", True), ("11798.99", True)])
def test_rounding_tolerance_is_one_rupee_by_default(total, expect_flag):
    assert ("TOTAL_MISMATCH" in codes(run(invoice(total_amount=D(total))))) is expect_flag


def test_tolerance_is_configurable():
    strict = ValidationSettings(amount_tolerance=D("0.01"))
    assert "TOTAL_MISMATCH" in codes(run(invoice(total_amount=D("11800.50")), settings=strict))
    assert "TOTAL_MISMATCH" not in codes(run(invoice(total_amount=D("11800.50"))))


def test_missing_amounts_are_reported_separately_from_mismatches():
    issues = run(invoice(total_amount=None))
    assert "MISSING_AMOUNT" in codes(issues) and "TOTAL_MISMATCH" not in codes(issues)
    issues = run(invoice(taxable_amount=None))
    assert "MISSING_AMOUNT" in codes(issues) and "TOTAL_MISMATCH" not in codes(issues)


def test_missing_taxes_are_a_warning_not_treated_as_zero():
    issues = run(invoice(cgst=None, sgst=None, total_amount=D("11800.00")))
    assert "TAX_AMOUNTS_MISSING" in codes(issues, Severity.WARNING)
    assert "TOTAL_MISMATCH" not in codes(issues)


def test_validation_never_changes_the_extracted_amounts():
    inv = invoice(total_amount=D("99999.00"))
    before = inv.model_dump()
    run(inv)
    assert inv.model_dump() == before


def test_negative_amount_is_a_warning_for_review():
    assert "NEGATIVE_AMOUNT" in codes(run(invoice(total_amount=D("-11800.00"), taxable_amount=D("-10000.00"),
                                                    cgst=D("-900.00"), sgst=D("-900.00"))), Severity.WARNING)


# ---- tax consistency ---------------------------------------------------------------------
def test_unequal_cgst_and_sgst_is_flagged():
    issues = run(invoice(cgst=D("900.00"), sgst=D("850.00"), total_amount=D("11750.00")))
    assert "CGST_SGST_UNEQUAL" in codes(issues, Severity.WARNING)


def test_cgst_sgst_difference_within_tolerance_is_fine():
    assert "CGST_SGST_UNEQUAL" not in codes(run(invoice(cgst=D("900.00"), sgst=D("900.50"), total_amount=D("11800.50"))))


def test_only_one_of_cgst_sgst_is_flagged():
    assert "CGST_SGST_INCOMPLETE" in codes(run(invoice(sgst=None, total_amount=D("10900.00"))))


def test_igst_together_with_cgst_sgst_is_flagged():
    issues = run(invoice(igst=D("1800.00"), total_amount=D("13600.00")))
    assert "INTRA_INTER_CONFLICT" in codes(issues, Severity.WARNING)


def test_igst_only_invoice_is_accepted():
    inv = invoice(cgst=None, sgst=None, igst=D("1800.00"), buyer_gstin=gstin_for("07"))
    assert [i for i in run(inv, OCR_CLEAN.replace("CGST 900.00\nSGST 900.00", "IGST 1,800.00").replace("11,800", "11,800")
                           .replace("27AAPFU0939F1ZV\nTaxable", gstin_for("07") + "\nTaxable"))
            if i.severity != Severity.INFO] == []


def test_all_zero_taxes_ask_for_confirmation():
    inv = invoice(cgst=D("0"), sgst=D("0"), igst=D("0"), total_amount=D("10000.00"))
    assert "ZERO_TAX" in codes(run(inv), Severity.WARNING)


# ---- state codes -------------------------------------------------------------------------
def test_same_state_with_igst_is_a_warning_with_a_caveat():
    inv = invoice(cgst=None, sgst=None, igst=D("1800.00"))
    issue = next(i for i in run(inv) if i.code == "STATE_TAX_MISMATCH")
    assert issue.severity == Severity.WARNING            # never an error: place of supply is not just the GSTIN state
    assert "not always" in issue.message and "not a conclusion" in issue.message


def test_different_states_with_cgst_sgst_is_a_warning():
    inv = invoice(buyer_gstin=gstin_for("07"))
    issue = next(i for i in run(inv) if i.code == "STATE_TAX_MISMATCH")
    assert issue.severity == Severity.WARNING
    assert "Maharashtra" in issue.message and "Delhi" in issue.message


def test_states_are_only_compared_when_both_gstins_are_valid():
    bad_buyer = invoice(buyer_gstin="07AAPFU0939F1ZY")   # right shape, wrong check character (correct one is X)
    assert "STATE_TAX_MISMATCH" not in codes(run(bad_buyer))
    assert "GSTIN_CHECKSUM_INVALID" in codes(run(bad_buyer), Severity.ERROR)
    assert "STATE_TAX_MISMATCH" not in codes(run(invoice(buyer_gstin=None)))


def test_state_mapping_is_configurable():
    inv = invoice(buyer_gstin=gstin_for("07"))
    issues = validate_invoice(inv, OCR_CLEAN, today=TODAY, state_codes={"27": {"name": "Somewhere"}})
    assert "STATE_TAX_MISMATCH" not in codes(issues)      # buyer state unknown -> no comparison, no claim


def test_invalid_gstin_format_and_checksum_are_errors():
    assert "GSTIN_FORMAT_INVALID" in codes(run(invoice(vendor_gstin="27AAPFU0939F1")), Severity.ERROR)
    assert "GSTIN_CHECKSUM_INVALID" in codes(run(invoice(vendor_gstin="27AAPFU0939F1ZW")), Severity.ERROR)


def test_valid_gstin_is_not_described_as_registered():
    text = " ".join(i.message for i in run(invoice()))
    assert "registered" not in text.lower() and "active" not in text.lower()


# ---- dates -------------------------------------------------------------------------------
@pytest.mark.parametrize("raw, iso", [("14/09/2026", "2026-09-14"), ("2026-09-14", "2026-09-14"), ("14-09-2026", "2026-09-14"),
                                      ("14.09.2026", "2026-09-14"), ("14 Sep 2026", "2026-09-14"),
                                      ("14th September 2026", "2026-09-14"), ("Sep 14, 2026", "2026-09-14"), ("25/12/2025", "2025-12-25")])
def test_dates_are_normalised_to_iso(raw, iso):
    p = parse_invoice_date(raw)
    assert p.iso == iso and not p.ambiguous


def test_ambiguous_dates_are_flagged_and_follow_the_day_first_setting():
    p = parse_invoice_date("03/04/2026")
    assert p.iso == "2026-04-03" and p.ambiguous
    assert parse_invoice_date("03/04/2026", day_first=False).iso == "2026-03-04"
    assert not parse_invoice_date("13/04/2026").ambiguous
    assert not parse_invoice_date("05/05/2026").ambiguous        # same either way
    assert "AMBIGUOUS_DATE" in codes(run(invoice(invoice_date="03/04/2026"), "03/04/2026"), Severity.WARNING)


def test_ambiguous_date_flag_can_be_switched_off_but_the_date_is_still_read_day_first():
    s = ValidationSettings(flag_ambiguous_dates=False)
    assert "AMBIGUOUS_DATE" not in codes(run(invoice(invoice_date="03/04/2026"), "03/04/2026", settings=s))
    assert "AMBIGUOUS_DATE" in codes(run(invoice(invoice_date="03/04/2026"), "03/04/2026"))      # default stays on


def test_invalid_calendar_date_is_an_error():
    assert "INVALID_DATE" in codes(run(invoice(invoice_date="31/02/2026"), "31/02/2026"), Severity.ERROR)
    assert parse_invoice_date("2026-13-40").problem == "invalid"


def test_unrecognised_date_text_is_a_warning():
    assert "UNRECOGNISED_DATE" in codes(run(invoice(invoice_date="yesterday")), Severity.WARNING)


def test_two_digit_year_is_flagged():
    assert "TWO_DIGIT_YEAR" in codes(run(invoice(invoice_date="14/09/26"), "14/09/26"), Severity.WARNING)


def test_future_date_is_a_warning_not_an_error():
    issues = run(invoice(invoice_date="2026-12-31"), OCR_CLEAN.replace("14/09/2026", "31/12/2026"))
    future = [i for i in issues if i.code == "FUTURE_DATE"]
    assert len(future) == 1 and future[0].severity == Severity.WARNING
    assert "not proof" in future[0].message


def test_future_date_warning_can_be_switched_off():
    s = ValidationSettings(warn_on_future_date=False)
    assert "FUTURE_DATE" not in codes(run(invoice(invoice_date="2026-12-31"), settings=s))


def test_date_before_gst_is_flagged():
    assert "DATE_BEFORE_GST" in codes(run(invoice(invoice_date="2015-01-01"), "01/01/2015"))


def test_today_is_not_in_the_future():
    assert "FUTURE_DATE" not in codes(run(invoice(invoice_date=TODAY.isoformat()), TODAY.strftime("%d/%m/%Y")))


# ---- missing fields ----------------------------------------------------------------------
def test_missing_required_fields_are_warnings():
    issues = run(invoice(vendor_name=None, invoice_number=None, vendor_gstin=None), None)
    missing = {f for i in issues if i.code == "MISSING_FIELD" for f in i.fields}
    assert {"vendor_name", "invoice_number", "vendor_gstin"} <= missing
    assert all(i.severity == Severity.WARNING for i in issues if i.code == "MISSING_FIELD")


def test_missing_gstin_message_mentions_unregistered_sellers():
    msg = next(i.message for i in run(invoice(vendor_gstin=None), None) if i.code == "MISSING_FIELD")
    assert "unregistered" in msg


# ---- special invoice types ---------------------------------------------------------------
@pytest.mark.parametrize("text, case", [("BILL OF SUPPLY", "exempt"), ("Nil rated supply", "exempt"),
                                         ("Export invoice under LUT", "zero_rated"), ("Supply to SEZ", "zero_rated"),
                                         ("Composition taxable person, not eligible to collect tax", "composition"),
                                         ("Reverse charge applicable: Yes", "reverse_charge")])
def test_special_cases_are_detected(text, case):
    assert case in detect_special_cases(text)


def test_bill_of_supply_without_tax_is_not_treated_as_missing_tax():
    inv = invoice(cgst=None, sgst=None, igst=None, total_amount=D("10000.00"))
    issues = run(inv, OCR_CLEAN + "\nBILL OF SUPPLY")
    assert "TAX_AMOUNTS_MISSING" not in codes(issues) and "TOTAL_MISMATCH" not in codes(issues)
    assert "SPECIAL_CASE_NO_TAX" in codes(issues, Severity.INFO)


def test_zero_rated_export_with_zero_igst_is_not_flagged_as_zero_tax():
    inv = invoice(buyer_gstin=None, cgst=None, sgst=None, igst=D("0"), total_amount=D("10000.00"))
    assert "ZERO_TAX" not in codes(run(inv, OCR_CLEAN + "\nExport invoice under LUT"))


def test_reverse_charge_mismatch_is_softened_to_a_warning():
    inv = invoice(total_amount=D("10000.00"))                      # total excludes tax payable by the buyer
    issues = run(inv, OCR_CLEAN + "\nReverse charge applicable: Yes")
    mismatch = next(i for i in issues if i.code == "TOTAL_MISMATCH")
    assert mismatch.severity == Severity.WARNING and "reverse_charge" in mismatch.message


def test_cess_or_round_off_softens_a_mismatch_to_a_warning():
    inv = invoice(total_amount=D("11805.00"))
    mismatch = next(i for i in run(inv, OCR_CLEAN + "\nCess 5.00") if i.code == "TOTAL_MISMATCH")
    assert mismatch.severity == Severity.WARNING


# ---- optional rate check -----------------------------------------------------------------
def test_rate_check_is_off_by_default():
    assert "UNUSUAL_IMPLIED_RATE" not in codes(run(invoice(cgst=D("625"), sgst=D("625"), total_amount=D("11250"))))


def test_rate_check_warns_only_when_enabled_and_never_errors():
    s = ValidationSettings(rate_check_enabled=True)
    odd = invoice(cgst=D("625.00"), sgst=D("625.00"), total_amount=D("11250.00"))      # blended 12.5%
    issue = next(i for i in run(odd, settings=s) if i.code == "UNUSUAL_IMPLIED_RATE")
    assert issue.severity == Severity.WARNING and "not a legal determination" in issue.message
    assert "UNUSUAL_IMPLIED_RATE" not in codes(run(invoice(), settings=s))            # 18% is a listed rate


def test_rate_check_uses_the_rate_table_for_the_invoice_date():
    periods = [{"from": "2017-07-01", "to": None, "rates_pct": ["5"]}]
    s = ValidationSettings(rate_check_enabled=True)
    issues = validate_invoice(invoice(), OCR_CLEAN, s, today=TODAY, rate_periods=periods)
    assert "UNUSUAL_IMPLIED_RATE" in codes(issues)


# ---- grounding and OCR confidence --------------------------------------------------------
def test_values_missing_from_the_ocr_text_are_flagged():
    issues = run(invoice(invoice_number="INV-9999"))
    flagged = [i for i in issues if i.code == "NOT_FOUND_IN_OCR"]
    assert [i.fields for i in flagged] == [["invoice_number"]]


def test_amount_not_in_ocr_text_is_flagged():
    inv = invoice(taxable_amount=D("10001.00"), total_amount=D("11801.00"))
    assert any(i.code == "NOT_FOUND_IN_OCR" and "taxable_amount" in i.fields for i in run(inv))


def test_low_ocr_confidence_is_flagged_per_field():
    lines = [OcrLine(text="Total 11,800.00", confidence=0.42), OcrLine(text="Invoice No INV-2041", confidence=0.99)]
    issues = run(invoice(), ocr_lines=lines)
    low = [i for i in issues if i.code == "LOW_OCR_CONFIDENCE"]
    assert [i.fields for i in low] == [["total_amount"]]
    assert "42%" in low[0].message


# ---- normalisation and field status -------------------------------------------------------
def test_normalise_cleans_gstin_and_date_but_never_amounts():
    inv = invoice(vendor_gstin=" 27aapfu0939f1zv ", invoice_date="14/09/2026", total_amount=D("11800.00"))
    out = normalize_invoice(inv, ValidationSettings())
    assert out.vendor_gstin == "27AAPFU0939F1ZV" and out.invoice_date == "2026-09-14"
    assert out.total_amount == D("11800.00") and out.cgst == D("900.00")
    assert any("normalised" in n for n in out.extraction_notes)


def test_ambiguous_date_is_not_silently_rewritten():
    out = normalize_invoice(invoice(invoice_date="03/04/2026"), ValidationSettings())
    assert out.invoice_date == "03/04/2026"


def test_field_review_status_marks_missing_flagged_and_edited():
    inv = invoice(buyer_name=None, total_amount=D("12000.00"))
    issues = run(inv)
    status = field_review_status(inv, issues, edited_fields=["vendor_name"])
    assert status["buyer_name"]["status"] == "missing"
    assert status["total_amount"]["status"] == "flagged" and status["total_amount"]["reasons"]
    assert status["vendor_name"]["edited"] and status["vendor_name"]["reasons"][0] == "Edited by a person"
    assert status["invoice_number"]["status"] == "ok"


def test_count_by_severity():
    counts = count_by_severity(run(invoice(total_amount=D("12000.00"))))
    assert counts["error"] >= 1 and set(counts) == {"error", "warning", "info"}
