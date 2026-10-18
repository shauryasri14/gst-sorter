"""Evaluation: compare extraction results with a ground-truth CSV. Reports ONLY measured numbers.

Three ways to use it
--------------------
1. Score an export you already made with the app (works anywhere):
       python eval/evaluate.py --truth ground_truth.csv --predictions export.csv

2. Run the real pipeline on a folder and score it (needs OCR installed AND a running Ollama with the model):
       python eval/evaluate.py --truth data/synthetic_batch/ground_truth.csv --run data/synthetic_batch --ocr tesseract

3. OCR only, no LLM needed: "can the value be found in the OCR text at all?" (the ceiling for any extractor):
       python eval/evaluate.py --truth data/synthetic_batch/ground_truth.csv --ocr-only data/synthetic_batch

Ground truth columns: file, vendor_name, vendor_gstin, buyer_name, buyer_gstin, invoice_number, invoice_date (ISO),
taxable_amount, cgst, sgst, igst, total_amount   (empty cell = the field is not on the invoice)

What "before / after deterministic validation" means here
---------------------------------------------------------
Validation never changes extracted values, so field accuracy is the same before and after. What it adds is
*detection*: it flags invoices for a human. The report therefore shows how many wrong invoices were caught
(flagged), how many slipped through silently, and how many correct invoices were flagged needlessly.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import statistics
import sys
import tempfile
import time
from dataclasses import asdict, dataclass, field
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pipeline.schemas import AMOUNT_FIELDS  # noqa: E402
from pipeline.validate import _DATE_CANDIDATES, numbers_in, parse_invoice_date  # noqa: E402

TEXT_FIELDS = ["vendor_name", "vendor_gstin", "buyer_name", "buyer_gstin", "invoice_number"]
FIELDS = TEXT_FIELDS + ["invoice_date"] + AMOUNT_FIELDS
GSTIN_FIELDS = ["vendor_gstin", "buyer_gstin"]

# App export column -> field name
EXPORT_COLUMNS = {
    "Vendor Name": "vendor_name", "Vendor GSTIN": "vendor_gstin", "Buyer Name": "buyer_name", "Buyer GSTIN": "buyer_gstin",
    "Invoice Number": "invoice_number", "Invoice Date": "invoice_date", "Taxable Amount": "taxable_amount", "CGST": "cgst",
    "SGST": "sgst", "IGST": "igst", "Total Amount": "total_amount",
}


# ======================================================================
# Comparison
# ======================================================================
def _blank(value: Any) -> bool:
    return value is None or str(value).strip() == ""


def _decimal(value: Any) -> Decimal | None:
    if _blank(value):
        return None
    try:
        return Decimal(str(value).replace(",", "").strip().lstrip("'"))
    except InvalidOperation:
        return None


def _text_key(value: Any) -> str | None:
    if _blank(value):
        return None
    return re.sub(r"\s+", " ", str(value).strip().lstrip("'")).casefold()


def _gstin_key(value: Any) -> str | None:
    return None if _blank(value) else re.sub(r"[\s-]+", "", str(value).strip().lstrip("'")).upper()


def _date_key(value: Any) -> str | None:
    if _blank(value):
        return None
    text = str(value).strip().lstrip("'")
    return parse_invoice_date(text).iso or text


def field_equal(name: str, truth: Any, pred: Any, tolerance: Decimal | None = None) -> bool:
    """Exact match (``tolerance=None``) or, for amounts, match within ``tolerance`` rupees. Empty equals empty."""
    if name in AMOUNT_FIELDS:
        a, b = _decimal(truth), _decimal(pred)
        if a is None or b is None:
            return a is None and b is None
        return a == b if tolerance is None else abs(a - b) <= tolerance
    if name in GSTIN_FIELDS:
        return _gstin_key(truth) == _gstin_key(pred)
    if name == "invoice_date":
        return _date_key(truth) == _date_key(pred)
    return _text_key(truth) == _text_key(pred)              # case- and spacing-insensitive, otherwise literal


@dataclass
class InvoiceOutcome:
    file: str
    failed: bool
    flagged: bool
    wrong_fields: list[str]
    correct: bool                       # every field right (amounts within tolerance) and not failed


@dataclass
class Report:
    count: int = 0
    tolerance: str = "0.01"
    field_exact: dict[str, float] = field(default_factory=dict)
    field_numeric: dict[str, float] = field(default_factory=dict)
    gstin_exact: float | None = None
    invoice_success_rate: float = 0.0
    failed_invoices: int = 0
    missing_predictions: list[str] = field(default_factory=list)
    flagged_rate: float = 0.0
    wrong_invoices: int = 0
    wrong_and_flagged: int = 0
    wrong_not_flagged: int = 0
    correct_but_flagged: int = 0
    accuracy_of_unflagged: float | None = None
    timings: dict[str, Any] = field(default_factory=dict)
    ocr_recall: dict[str, float] = field(default_factory=dict)
    outcomes: list[InvoiceOutcome] = field(default_factory=list)


def _rate(hits: int, total: int) -> float:
    return hits / total if total else 0.0


def compare(truth: dict[str, dict[str, Any]], preds: dict[str, dict[str, Any]], tolerance: Decimal = Decimal("0.01")) -> Report:
    """Score predictions against the ground truth. Truth files with no prediction count as failed."""
    rep = Report(count=len(truth), tolerance=str(tolerance))
    exact_hits = {f: 0 for f in FIELDS}
    numeric_hits = {f: 0 for f in AMOUNT_FIELDS}
    gstin_hits = gstin_total = 0

    for name, t in truth.items():
        p = preds.get(name)
        if p is None:
            rep.missing_predictions.append(name)
        failed = p is None or str(p.get("_status", "")).lower() == "failed"
        flagged = failed or bool(p.get("_flagged"))
        p = p or {}
        wrong: list[str] = []
        for f in FIELDS:
            if failed:
                wrong.append(f)
                if f in GSTIN_FIELDS:
                    gstin_total += 1
                continue
            exact = field_equal(f, t.get(f), p.get(f))
            exact_hits[f] += exact
            if f in AMOUNT_FIELDS:
                within = field_equal(f, t.get(f), p.get(f), tolerance)
                numeric_hits[f] += within
                if not within:
                    wrong.append(f)
            elif not exact:
                wrong.append(f)
            if f in GSTIN_FIELDS:
                gstin_total += 1
                gstin_hits += exact
        correct = not failed and not wrong
        rep.outcomes.append(InvoiceOutcome(name, failed, flagged, wrong, correct))

    n = len(truth)
    rep.field_exact = {f: _rate(exact_hits[f], n) for f in FIELDS}
    rep.field_numeric = {f: _rate(numeric_hits[f], n) for f in AMOUNT_FIELDS}
    rep.gstin_exact = _rate(gstin_hits, gstin_total) if gstin_total else None
    rep.failed_invoices = sum(o.failed for o in rep.outcomes)
    rep.invoice_success_rate = _rate(sum(o.correct for o in rep.outcomes), n)
    rep.flagged_rate = _rate(sum(o.flagged for o in rep.outcomes), n)
    rep.wrong_invoices = sum(not o.correct for o in rep.outcomes)
    rep.wrong_and_flagged = sum((not o.correct) and o.flagged for o in rep.outcomes)
    rep.wrong_not_flagged = sum((not o.correct) and not o.flagged for o in rep.outcomes)
    rep.correct_but_flagged = sum(o.correct and o.flagged for o in rep.outcomes)
    unflagged = [o for o in rep.outcomes if not o.flagged]
    rep.accuracy_of_unflagged = _rate(sum(o.correct for o in unflagged), len(unflagged)) if unflagged else None
    return rep


# ======================================================================
# Loading
# ======================================================================
def load_truth(path: Path) -> dict[str, dict[str, Any]]:
    with Path(path).open(newline="", encoding="utf-8-sig") as fh:
        rows = list(csv.DictReader(fh))
    if not rows or "file" not in rows[0]:
        raise SystemExit(f"{path}: expected a CSV with a 'file' column (see eval/ground_truth.example.csv)")
    return {r["file"]: r for r in rows}


def load_predictions(path: Path) -> dict[str, dict[str, Any]]:
    """Read an export made by the app (CSV)."""
    with Path(path).open(newline="", encoding="utf-8-sig") as fh:
        rows = list(csv.DictReader(fh))
    if rows and "Source File" not in rows[0]:
        raise SystemExit(f"{path}: expected the app's CSV export (a 'Source File' column)")
    out: dict[str, dict[str, Any]] = {}
    for r in rows:
        rec: dict[str, Any] = {EXPORT_COLUMNS[c]: r.get(c) for c in EXPORT_COLUMNS}
        rec["_status"] = r.get("Extraction Status", "")
        rec["_flagged"] = r.get("Review Status", "") in ("Needs review", "Failed") or bool((r.get("Validation Errors") or "").strip()) \
            or bool((r.get("Validation Warnings") or "").strip())
        out[r["Source File"]] = rec
    return out


def records_to_predictions(records: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    out = {}
    for r in records:
        rec: dict[str, Any] = {EXPORT_COLUMNS[c]: r.get(c) for c in EXPORT_COLUMNS}
        rec["_status"] = r.get("Extraction Status", "")
        rec["_flagged"] = r.get("Review Status") in ("Needs review", "Failed") or bool(r.get("Validation Errors")) or bool(r.get("Validation Warnings"))
        out[r["Source File"]] = rec
    return out


# ======================================================================
# Running the real pipeline
# ======================================================================
def _stats(values: list[float]) -> dict[str, float]:
    if not values:
        return {}
    ordered = sorted(values)
    p95 = ordered[min(len(ordered) - 1, int(round(0.95 * (len(ordered) - 1))))]
    return {"mean": round(statistics.fmean(values), 1), "median": round(statistics.median(values), 1), "p95": round(p95, 1), "n": len(values)}


def run_pipeline(folder: Path, truth_files: set[str] | None, overrides: dict[str, Any], limit: int | None = None,
                 chat_client: Any = None, ocr_engines: Any = None, log=print) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    """Process a folder with the real pipeline in a throw-away data directory. Returns (predictions, timings)."""
    from config import load_settings
    from pipeline.export import gather, invoice_records
    from pipeline.runner import ServiceUnavailable, Services, import_folder, process_invoice
    from pipeline.storage import Storage

    with tempfile.TemporaryDirectory(prefix="gst_eval_") as tmp:
        settings = load_settings(env={}, dotenv_file=Path(tmp) / "none.env", overrides={"data_dir": tmp, **overrides})
        storage = Storage(settings)
        services = Services(settings, storage, ocr_engines=ocr_engines, chat_client=chat_client)
        report = import_folder(folder, storage, settings)
        log(f"Imported: {report.summary()}")
        ids = storage.pending_ids()
        if truth_files is not None:
            wanted = {r.invoice_id for r in storage.list_invoices() if r.filename in truth_files}
            ids = [i for i in ids if i in wanted]
        if limit:
            ids = ids[:limit]
        wall: list[float] = []
        per_page: list[float] = []
        for n, iid in enumerate(ids, 1):
            started = time.perf_counter()
            try:
                status = process_invoice(iid, services)
            except ServiceUnavailable as exc:
                raise SystemExit(f"\nCannot run the full pipeline: {exc}\nUse --ocr-only to measure OCR without a model.") from exc
            elapsed = (time.perf_counter() - started) * 1000
            row = storage.get_invoice(iid)
            wall.append(elapsed)
            per_page.append(elapsed / max(1, row.page_end - row.page_start + 1))
            log(f"  [{n}/{len(ids)}] {row.filename}: {status.value} ({elapsed / 1000:.1f}s)")
        rows, dups, conf = gather(storage, settings.validation.date_day_first)
        preds = records_to_predictions(invoice_records(rows, dups, conf, settings.validation.date_day_first))
        ocr_ms = [p.ocr.elapsed_ms for r in rows for p in storage.get_pages(r.doc_id, r.page_start, r.page_end) if p.ocr.elapsed_ms]
        llm_ms = [r.llm_ms for r in rows if r.llm_ms]
        timings = {
            "per_invoice_ms": _stats(wall), "per_page_ms": _stats(per_page),
            "ocr_ms_per_page": _stats([float(x) for x in ocr_ms]), "llm_ms_per_invoice": _stats([float(x) for x in llm_ms]),
            "model": settings.ollama.model, "ocr_engine": settings.ocr.engine,
        }
    return preds, timings


def ocr_recall(folder: Path, truth: dict[str, dict[str, Any]], overrides: dict[str, Any], limit: int | None = None,
               ocr_engines: Any = None, log=print) -> tuple[dict[str, float], dict[str, Any]]:
    """How often the true value appears in the OCR text at all. No LLM involved."""
    from config import load_settings
    from pipeline import ingest
    from pipeline.ocr import run_ocr
    from pipeline.preprocess import preprocess_image

    hits = {f: 0 for f in FIELDS if f != "currency"}
    seen = {f: 0 for f in hits}
    ms: list[float] = []
    names = [n for n in truth if (folder / n).exists()]
    if limit:
        names = names[:limit]
    if not names:
        raise SystemExit(f"None of the files listed in the ground truth were found in {folder}")
    with tempfile.TemporaryDirectory(prefix="gst_eval_") as tmp:
        settings = load_settings(env={}, dotenv_file=Path(tmp) / "none.env", overrides={"data_dir": tmp, **overrides})
        for i, name in enumerate(names, 1):
            f = ingest.ingest_path(folder / name, settings)
            texts, started = [], time.perf_counter()
            for page_no in range(1, f.page_count + 1):
                img = ingest.load_page_image(f.stored_path, f.kind, page_no, settings)
                page = run_ocr(preprocess_image(img, settings.preprocess).image, settings.ocr, ocr_engines)
                if page.error:
                    log(f"  {name}: OCR error: {page.error}")
                texts.append(page.text)
            ms.append((time.perf_counter() - started) * 1000 / f.page_count)
            text = "\n".join(texts)
            squashed = re.sub(r"[^A-Z0-9]", "", text.upper())
            numbers = set(numbers_in(text))
            iso_dates = {parse_invoice_date(c).iso for c in _DATE_CANDIDATES.findall(text)}
            t = truth[name]
            for fld in hits:
                value = t.get(fld)
                if _blank(value):
                    continue
                seen[fld] += 1
                if fld in AMOUNT_FIELDS:
                    ok = _decimal(value) in numbers
                elif fld == "invoice_date":
                    ok = _date_key(value) in iso_dates
                elif fld == "vendor_name" or fld == "buyer_name":
                    ok = re.sub(r"[^A-Z0-9]", "", str(value).upper()) in squashed
                else:
                    ok = re.sub(r"[^A-Z0-9]", "", str(value).upper()) in squashed
                hits[fld] += ok
            log(f"  [{i}/{len(names)}] {name}")
    recall = {f: _rate(hits[f], seen[f]) for f in hits if seen[f]}
    return recall, {"per_page_ms": _stats(ms), "ocr_engine": settings.ocr.engine, "files": len(names)}


# ======================================================================
# Output
# ======================================================================
def _pct(x: float | None) -> str:
    return "n/a" if x is None else f"{x * 100:5.1f}%"


def format_report(rep: Report, title: str = "Evaluation") -> str:
    out = [f"# {title}", "", f"Invoices in ground truth: {rep.count}   Numeric tolerance: ₹{rep.tolerance}", ""]
    if rep.missing_predictions:
        out += [f"No prediction for {len(rep.missing_predictions)} file(s): {', '.join(rep.missing_predictions[:5])}"
                + (" ..." if len(rep.missing_predictions) > 5 else ""), ""]
    if rep.field_exact:
        out += ["## Field accuracy", "", "| Field | Exact match | Within tolerance |", "|---|---|---|"]
        for f in FIELDS:
            out.append(f"| {f} | {_pct(rep.field_exact[f])} | {_pct(rep.field_numeric.get(f)) if f in AMOUNT_FIELDS else '-'} |")
        out += ["", f"GSTIN exact match (vendor + buyer): {_pct(rep.gstin_exact)}",
                f"Invoice-level success (every field right): {_pct(rep.invoice_success_rate)}   Failed files: {rep.failed_invoices}", ""]
        out += ["## Validation: what it caught", "",
                f"Flagged for manual review: {_pct(rep.flagged_rate)} of invoices",
                f"Invoices with at least one wrong field: {rep.wrong_invoices}",
                f"  caught (flagged):               {rep.wrong_and_flagged}",
                f"  slipped through (not flagged):  {rep.wrong_not_flagged}",
                f"Correct invoices flagged needlessly: {rep.correct_but_flagged}",
                f"Accuracy of invoices WITHOUT flags: {_pct(rep.accuracy_of_unflagged)}", "",
                "(Validation never edits values, so field accuracy is identical before and after; it adds the flags above.)", ""]
        silent = [o for o in rep.outcomes if not o.correct and not o.flagged]
        if silent:
            out += ["Wrong but not flagged:"] + [f"  - {o.file}: {', '.join(o.wrong_fields)}" for o in silent[:10]] + [""]
    if rep.ocr_recall:
        out += ["## OCR recall (is the true value present in the OCR text?)", "", "| Field | Found in OCR text |", "|---|---|"]
        out += [f"| {f} | {_pct(v)} |" for f, v in rep.ocr_recall.items()] + [""]
    if rep.timings:
        out += ["## Timing (milliseconds, measured on this machine)", ""]
        for k, v in rep.timings.items():
            if isinstance(v, dict) and v:
                out.append(f"- {k}: mean {v['mean']}, median {v['median']}, p95 {v['p95']} (n={v['n']})")
            elif not isinstance(v, dict):
                out.append(f"- {k}: {v}")
        out.append("")
    return "\n".join(out)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--truth", type=Path, required=True, help="ground-truth CSV")
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--predictions", type=Path, help="CSV exported from the app")
    mode.add_argument("--run", type=Path, help="folder of invoices: run the real pipeline (needs OCR + Ollama)")
    mode.add_argument("--ocr-only", type=Path, help="folder of invoices: measure OCR only (no Ollama needed)")
    ap.add_argument("--tolerance", default="0.01", help="rupee tolerance for 'within tolerance' amount accuracy (default 0.01)")
    ap.add_argument("--ocr", choices=["paddleocr", "tesseract"], help="OCR engine for --run / --ocr-only (default: from settings)")
    ap.add_argument("--model", help="Ollama model for --run (default: from settings)")
    ap.add_argument("--limit", type=int, help="only process the first N files")
    ap.add_argument("--json", type=Path, help="also write the full report as JSON")
    args = ap.parse_args(argv)

    truth = load_truth(args.truth)
    tolerance = Decimal(args.tolerance)
    overrides: dict[str, Any] = {}
    if args.ocr:
        overrides["ocr"] = {"engine": args.ocr, "fallback_engine": None}
    if args.model:
        overrides["ollama"] = {"model": args.model}

    if args.predictions:
        rep = compare(truth, load_predictions(args.predictions), tolerance)
        title = f"Evaluation of {args.predictions.name}"
    elif args.run:
        preds, timings = run_pipeline(args.run, set(truth), overrides, args.limit)
        subset = {k: v for k, v in truth.items() if k in preds} if args.limit else truth
        rep = compare(subset, preds, tolerance)
        rep.timings = timings
        title = f"End-to-end evaluation ({timings['ocr_engine']} + {timings['model']})"
    else:
        recall, timings = ocr_recall(args.ocr_only, truth, overrides, args.limit)
        rep = Report(count=timings["files"], tolerance=str(tolerance), ocr_recall=recall, timings=timings)
        title = f"OCR-only evaluation ({timings['ocr_engine']})"

    print(format_report(rep, title))
    if args.json:
        args.json.write_text(json.dumps(asdict(rep), indent=2, default=str), encoding="utf-8")
        print(f"JSON written to {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
