"""The evaluation script itself: scoring logic and its three modes (with test doubles, never real measurements)."""

from __future__ import annotations

import csv
import io
import random
import shutil
from decimal import Decimal
from pathlib import Path

import pytest

from eval import evaluate as ev
from eval.synthetic import GROUND_TRUTH_COLUMNS, make_batch, make_spec
from pipeline.export import INVOICE_COLUMNS
from tests.conftest import FakeChat, FakeOcr, llm_json_for, ocr_text_for

D = Decimal


def truth_row(**kw):
    base = {"file": "a.png", "vendor_name": "Demo Traders", "vendor_gstin": "27AAPFU0939F1ZV", "buyer_name": "Buyer Co", "buyer_gstin": "",
            "invoice_number": "INV-1", "invoice_date": "2026-09-14", "taxable_amount": "100.00", "cgst": "9.00", "sgst": "9.00",
            "igst": "", "total_amount": "118.00"}
    base.update(kw)
    return base


def pred_row(**kw):
    base = {k: v for k, v in truth_row().items() if k != "file"}
    base.update({"_status": "Succeeded", "_flagged": False})
    base.update(kw)
    return base


# ---- field comparison ----------------------------------------------------------------------------
def test_field_equal_rules():
    assert ev.field_equal("vendor_name", "Demo  Traders", "demo traders")           # case and spacing are ignored for names
    assert not ev.field_equal("vendor_name", "Demo Traders", "Demo Trader")
    assert ev.field_equal("vendor_gstin", "27aapfu0939f1zv", " 27AAPFU0939F1ZV")
    assert not ev.field_equal("vendor_gstin", "27AAPFU0939F1ZV", "27AAPFUO939F1ZV")  # O vs 0 is a real error
    assert not ev.field_equal("invoice_number", "INV-1", "INV1")                     # hyphens are not ignored
    assert ev.field_equal("invoice_date", "2026-09-14", "14/09/2026")                # same date, different notation
    assert not ev.field_equal("invoice_date", "2026-09-14", "2026-09-15")
    assert ev.field_equal("igst", "", None) and not ev.field_equal("igst", "", "0")  # empty == empty, but 0 is not "missing"
    assert ev.field_equal("total_amount", "1,180.00", "1180") and not ev.field_equal("total_amount", "118", "118.01")


def test_numeric_tolerance_is_separate_from_exact_match():
    assert not ev.field_equal("total_amount", "118.00", "118.50")
    assert ev.field_equal("total_amount", "118.00", "118.50", D("1"))
    assert not ev.field_equal("total_amount", "118.00", "119.50", D("1"))
    assert not ev.field_equal("total_amount", "118.00", "", D("1"))


# ---- compare ---------------------------------------------------------------------------------------
def test_perfect_predictions_score_100_percent():
    rep = ev.compare({"a.png": truth_row()}, {"a.png": pred_row()})
    assert rep.invoice_success_rate == 1.0 and rep.gstin_exact == 1.0 and all(v == 1.0 for v in rep.field_exact.values())
    assert rep.wrong_invoices == 0 and rep.flagged_rate == 0.0 and rep.accuracy_of_unflagged == 1.0


def test_wrong_fields_are_counted_per_field_and_per_invoice():
    truth = {"a.png": truth_row(), "b.png": truth_row(file="b.png")}
    preds = {"a.png": pred_row(vendor_gstin="27AAPFUO939F1ZV", total_amount="118.50"), "b.png": pred_row()}
    rep = ev.compare(truth, preds, D("1"))
    assert rep.field_exact["vendor_gstin"] == 0.5 and rep.field_exact["total_amount"] == 0.5 and rep.field_exact["vendor_name"] == 1.0
    assert rep.field_numeric["total_amount"] == 1.0                                   # 0.50 is inside a ₹1 tolerance
    assert rep.gstin_exact == 0.75                                                    # 3 of 4 GSTIN cells (vendor+buyer, 2 invoices)
    assert rep.invoice_success_rate == 0.5 and rep.outcomes[0].wrong_fields == ["vendor_gstin"]


def test_validation_detection_buckets():
    truth = {n: truth_row(file=n) for n in "abcd"}
    preds = {"a": pred_row(),                                           # right, not flagged
             "b": pred_row(total_amount="1"), "c": pred_row(total_amount="1", _flagged=True),   # wrong: silent / caught
             "d": pred_row(_flagged=True)}                              # right but flagged
    rep = ev.compare(truth, preds)
    assert (rep.wrong_invoices, rep.wrong_and_flagged, rep.wrong_not_flagged, rep.correct_but_flagged) == (2, 1, 1, 1)
    assert rep.flagged_rate == 0.5 and rep.accuracy_of_unflagged == 0.5


def test_failed_and_missing_files_count_as_wrong_and_flagged():
    truth = {"a.png": truth_row(), "b.png": truth_row(file="b.png")}
    rep = ev.compare(truth, {"a.png": pred_row(_status="Failed")})
    assert rep.failed_invoices == 2 and rep.missing_predictions == ["b.png"] and rep.invoice_success_rate == 0.0
    assert rep.wrong_and_flagged == 2 and rep.accuracy_of_unflagged is None


def test_report_text_contains_every_required_metric():
    rep = ev.compare({"a.png": truth_row()}, {"a.png": pred_row(total_amount="9")})
    rep.timings = {"per_page_ms": {"mean": 1.0, "median": 1.0, "p95": 1.0, "n": 1}, "per_invoice_ms": {"mean": 2.0, "median": 2.0, "p95": 2.0, "n": 1}}
    text = ev.format_report(rep)
    for needle in ("Exact match", "Within tolerance", "GSTIN exact match", "Invoice-level success", "Flagged for manual review",
                   "slipped through", "per_page_ms", "per_invoice_ms", "identical before and after"):
        assert needle in text, needle


# ---- loading ----------------------------------------------------------------------------------------------
def test_example_ground_truth_file_is_valid():
    rows = ev.load_truth(Path(__file__).resolve().parent.parent / "eval" / "ground_truth.example.csv")
    assert len(rows) == 3
    first = next(iter(rows.values()))
    assert set(GROUND_TRUTH_COLUMNS) <= set(first)


def test_predictions_are_read_from_an_app_csv_export(tmp_path):
    header = INVOICE_COLUMNS
    row = {c: "" for c in header}
    row.update({"Source File": "a.png", "Vendor Name": "Demo Traders", "Total Amount": "118.00", "Extraction Status": "Succeeded",
                "Review Status": "Needs review", "Validation Warnings": "[X] something"})
    path = tmp_path / "export.csv"
    with path.open("w", newline="", encoding="utf-8-sig") as fh:
        w = csv.DictWriter(fh, fieldnames=header)
        w.writeheader()
        w.writerow(row)
    preds = ev.load_predictions(path)
    assert preds["a.png"]["vendor_name"] == "Demo Traders" and preds["a.png"]["_flagged"] is True and preds["a.png"]["total_amount"] == "118.00"


def test_wrong_csv_shapes_are_rejected_with_a_helpful_message(tmp_path):
    bad = tmp_path / "bad.csv"
    bad.write_text("x,y\n1,2\n")
    with pytest.raises(SystemExit, match="'file' column"):
        ev.load_truth(bad)
    with pytest.raises(SystemExit, match="Source File"):
        ev.load_predictions(bad)


# ---- synthetic generator -------------------------------------------------------------------------------------
def test_synthetic_batch_is_reproducible_and_labelled_fictitious(tmp_path):
    a = make_batch(tmp_path / "a", count=6, seed=5, pdfs=2)
    b = make_batch(tmp_path / "b", count=6, seed=5, pdfs=2)
    assert a == b
    files = sorted(p.suffix for p in (tmp_path / "a").iterdir() if p.name != "ground_truth.csv")
    assert files.count(".pdf") == 2 and len(files) == 6
    assert {r["variant"] for r in a} >= {"clean", "rotated", "blurry"}


def test_synthetic_invoices_are_internally_consistent():
    rng = random.Random(0)
    for _ in range(100):
        s = make_spec(rng)
        assert s.taxable_amount + (s.cgst or 0) + (s.sgst or 0) + (s.igst or 0) == s.total_amount
        assert (s.igst is None) != (s.cgst is None)                                 # intra-state XOR inter-state


# ---- the two pipeline modes, with test doubles --------------------------------------------------------------
def test_run_mode_scores_the_pipeline_and_reports_timings(tmp_path):
    folder = tmp_path / "batch"
    rows = make_batch(folder, count=3, seed=11, pdfs=0, variants=["clean"])
    specs_by_file = {}
    rng = random.Random(11)
    for r in rows:
        specs_by_file[r["file"]] = make_spec(rng)
    order = iter(rows)
    current = {}

    def ocr_text(img):
        current["row"] = next(order)
        return ocr_text_for(specs_by_file[current["row"]["file"]])

    class Chat(FakeChat):
        def chat(self, messages, schema=None):
            return __import__("json").dumps(llm_json_for(specs_by_file[current["row"]["file"]]))

    truth = ev.load_truth(folder / "ground_truth.csv")
    preds, timings = ev.run_pipeline(folder, set(truth), {"preprocess": {"enabled": False}, "validation": {"flag_ambiguous_dates": False}},
                                     ocr_engines=[FakeOcr(ocr_text)], chat_client=Chat(), log=lambda *_: None)
    rep = ev.compare(truth, preds)
    rep.timings = timings
    assert rep.invoice_success_rate == 1.0 and rep.count == 3 and rep.flagged_rate == 0.0
    assert timings["per_invoice_ms"]["n"] == 3 and timings["per_page_ms"]["mean"] > 0


def test_run_mode_stops_with_a_clear_message_when_ollama_is_down(tmp_path):
    from pipeline.extract import OllamaUnavailableError

    folder = tmp_path / "batch"
    make_batch(folder, count=1, seed=1, pdfs=0, variants=["clean"])
    with pytest.raises(SystemExit, match="--ocr-only"):
        ev.run_pipeline(folder, None, {"preprocess": {"enabled": False}}, ocr_engines=[FakeOcr("some text")],
                        chat_client=FakeChat(OllamaUnavailableError("Ollama is not running")), log=lambda *_: None)


def test_ocr_only_mode_measures_recall_with_a_fake_engine(tmp_path):
    folder = tmp_path / "batch"
    rows = make_batch(folder, count=2, seed=13, pdfs=0, variants=["clean"])
    rng = random.Random(13)
    texts = {r["file"]: ocr_text_for(make_spec(rng)) for r in rows}
    queue = iter(texts.values())
    truth = ev.load_truth(folder / "ground_truth.csv")
    recall, timings = ev.ocr_recall(folder, truth, {"preprocess": {"enabled": False}}, ocr_engines=[FakeOcr(lambda img: next(queue))], log=lambda *_: None)
    assert recall["vendor_gstin"] == 1.0 and recall["total_amount"] == 1.0 and recall["invoice_date"] == 1.0 and timings["files"] == 2


def test_ocr_only_mode_with_garbled_text_reports_low_recall(tmp_path):
    folder = tmp_path / "batch"
    make_batch(folder, count=2, seed=13, pdfs=0, variants=["clean"])
    truth = ev.load_truth(folder / "ground_truth.csv")
    recall, _ = ev.ocr_recall(folder, truth, {"preprocess": {"enabled": False}}, ocr_engines=[FakeOcr("nothing useful here")], log=lambda *_: None)
    assert recall["vendor_gstin"] == 0.0 and recall["total_amount"] == 0.0


def test_command_line_scores_an_export(tmp_path, capsys):
    folder = tmp_path / "t"
    rows = make_batch(folder, count=2, seed=2, pdfs=0, variants=["clean"])
    export = tmp_path / "export.csv"
    with export.open("w", newline="", encoding="utf-8-sig") as fh:
        w = csv.DictWriter(fh, fieldnames=INVOICE_COLUMNS)
        w.writeheader()
        for r in rows:
            rec = {c: "" for c in INVOICE_COLUMNS}
            rec.update({"Source File": r["file"], "Extraction Status": "Succeeded", "Review Status": "Not yet reviewed (no flags)",
                        "Vendor Name": r["vendor_name"], "Vendor GSTIN": r["vendor_gstin"], "Total Amount": r["total_amount"]})
            w.writerow(rec)
    assert ev.main(["--truth", str(folder / "ground_truth.csv"), "--predictions", str(export), "--json", str(tmp_path / "r.json")]) == 0
    out = capsys.readouterr().out
    assert "Evaluation of export.csv" in out and "Invoice-level success" in out and (tmp_path / "r.json").exists()
