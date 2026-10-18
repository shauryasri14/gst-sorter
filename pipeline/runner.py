"""Pipeline orchestration. Independent of Streamlit.

import -> render pages -> preprocess -> OCR -> local LLM extraction -> deterministic validation -> store

* Every stage reports progress through an optional callback.
* A failure in one file never stops the batch, except "Ollama is not running / model missing", which would
  fail every file: that pauses the job and puts the current file back in the queue.
* Logs never contain OCR text, GSTINs or amounts. Only short ids, stages, durations and exception types.
"""

from __future__ import annotations

import logging
import logging.handlers
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from config import Settings, load_rate_periods, load_state_codes
from pipeline import ingest
from pipeline.duplicates import DupInput, DuplicateCandidate, find_duplicates
from pipeline.extract import (
    ChatClient,
    OllamaError,
    OllamaModelMissingError,
    OllamaUnavailableError,
    extract_invoice,
)
from pipeline.ocr import OcrEngine, run_ocr
from pipeline.preprocess import preprocess_image
from pipeline.schemas import ExtractedInvoice, InvoiceStatus, OcrLine, OcrPage, Severity, ValidationIssue
from pipeline.storage import InvoiceRow, PageRecord, Storage
from pipeline.validate import (
    field_review_status,
    normalize_invoice,
    normalized_date_or_raw,
    validate_invoice,
)

log = logging.getLogger("gst_sorter")

ProgressFn = Callable[[str, str], None]  # (stage, message)


def setup_logging(settings: Settings) -> None:
    """Rotating local log file. Safe to call more than once."""
    settings.ensure_dirs()
    root = logging.getLogger("gst_sorter")
    if any(isinstance(h, logging.handlers.RotatingFileHandler) for h in root.handlers):
        return
    handler = logging.handlers.RotatingFileHandler(settings.log_file, maxBytes=1_000_000, backupCount=3, encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
    root.addHandler(handler)
    root.setLevel(getattr(logging, settings.logging.level.upper(), logging.INFO))
    root.propagate = False
    for name in ("pipeline",):
        logging.getLogger(name).addHandler(handler)
        logging.getLogger(name).setLevel(root.level)


class ServiceUnavailable(RuntimeError):
    """Ollama is down or the model is missing. The batch should stop and the user should fix it."""


@dataclass
class Services:
    settings: Settings
    storage: Storage
    ocr_engines: list[OcrEngine] | None = None  # None -> built from settings
    chat_client: ChatClient | None = None       # None -> real Ollama client
    state_codes: dict[str, Any] = field(default_factory=load_state_codes)
    rate_periods: list[dict[str, Any]] = field(default_factory=load_rate_periods)


# ======================================================================
# Import
# ======================================================================
@dataclass
class ImportReport:
    added: list[str] = field(default_factory=list)
    duplicates: list[str] = field(default_factory=list)
    rejected: list[tuple[str, str]] = field(default_factory=list)

    def summary(self) -> str:
        parts = [f"{len(self.added)} added"]
        if self.duplicates:
            parts.append(f"{len(self.duplicates)} skipped (already imported)")
        if self.rejected:
            parts.append(f"{len(self.rejected)} rejected")
        return ", ".join(parts)


def import_files(files: list[tuple[str, bytes]], storage: Storage, settings: Settings) -> ImportReport:
    """Validate and register uploaded files. Same content is never imported twice."""
    report = ImportReport()
    for name, data in files:
        display = ingest.sanitize_filename(name)
        try:
            f = ingest.ingest_bytes(name, data, settings)
        except ingest.IngestError as exc:
            report.rejected.append((display, str(exc)))
            storage.add_rejected_file(display, str(exc), f"{display}:{ingest.sha256_bytes(data)}")
            continue
        (report.added if storage.add_document(f) else report.duplicates).append(f.filename)
    return report


def import_folder(folder: str | Path, storage: Storage, settings: Settings, recursive: bool = False) -> ImportReport:
    """Import the supported files of one explicitly chosen folder."""
    report = ImportReport()
    for path in ingest.list_folder(folder, recursive):
        try:
            f = ingest.ingest_path(path, settings)
        except ingest.IngestError as exc:
            report.rejected.append((path.name, str(exc)))
            storage.add_rejected_file(path.name, str(exc), str(path.resolve()))
            continue
        (report.added if storage.add_document(f) else report.duplicates).append(f.filename)
    return report


# ======================================================================
# One invoice
# ======================================================================
def _notify(progress: ProgressFn | None, stage: str, message: str) -> None:
    if progress:
        progress(stage, message)


def _combine_text(pages: list[PageRecord]) -> str:
    if len(pages) == 1:
        return pages[0].ocr.text
    return "\n\n".join(f"--- page {p.page_no} ---\n{p.ocr.text}" for p in pages if p.ocr.text)


def ocr_page(row: InvoiceRow, page_no: int, services: Services) -> PageRecord:
    """Render, preprocess and OCR one page, then save the result. Never raises for OCR problems."""
    s = services.settings
    original, original_path = ingest.render_and_store_page(Path(row.stored_path or ""), row.kind or "", row.doc_id, page_no, s)
    pre = preprocess_image(original, s.preprocess)
    result = run_ocr(pre.image, s.ocr, services.ocr_engines)
    ocr_image, ocr_path = pre.image, original_path
    used_original = pre.used_original

    # If preprocessing did not help, also try the untouched page and keep whichever was read better.
    low = result.mean_confidence is None or result.mean_confidence < s.ocr.retry_with_original_below
    if result.error is None and low and not pre.used_original:
        alt = run_ocr(original, s.ocr, services.ocr_engines)
        if alt.error is None and (alt.mean_confidence or 0) > (result.mean_confidence or 0):
            result, ocr_image, used_original = alt, original, True
    result.used_original_image = used_original

    if not used_original and pre.steps:
        ocr_path = ingest.page_image_path(s, row.doc_id, page_no, "_ocr")
        if not ocr_path.exists():
            ocr_image.save(ocr_path, format="PNG")
    record = PageRecord(
        row.doc_id, page_no, str(original_path), str(ocr_path), result,
        {"steps": pre.steps, "rotation": pre.rotation_deg, "skew": round(pre.skew_deg, 2),
         "score_before": pre.score_before, "score_after": pre.score_after, "used_original": used_original},
    )
    services.storage.save_page(record)
    return record


def run_validation(
    effective: ExtractedInvoice, ocr_text: str, lines: list[OcrLine], edited: list[str], services: Services,
) -> tuple[list[ValidationIssue], dict[str, Any], InvoiceStatus]:
    s = services.settings
    issues = validate_invoice(
        effective, ocr_text, s.validation, ocr_lines=lines, low_confidence_threshold=s.ocr.low_confidence_threshold,
        state_codes=services.state_codes, rate_periods=services.rate_periods,
    )
    field_status = field_review_status(effective, issues, lines, edited, s.ocr.low_confidence_threshold)
    needs_review = any(i.severity in (Severity.ERROR, Severity.WARNING) for i in issues)
    return issues, field_status, (InvoiceStatus.NEEDS_REVIEW if needs_review else InvoiceStatus.EXTRACTED)


def process_invoice(
    invoice_id: str,
    services: Services,
    progress: ProgressFn | None = None,
    reuse_ocr: bool = False,
    keep_corrections: bool = True,
) -> InvoiceStatus:
    """Run the whole pipeline for one invoice and store the result.

    Raises :class:`ServiceUnavailable` (after putting the invoice back in the queue) when Ollama is
    not reachable or the model is not installed. Every other problem ends as ``Failed`` with a message.
    """
    storage, s = services.storage, services.settings
    row = storage.get_invoice(invoice_id)
    if row is None:
        raise KeyError(f"Unknown invoice {invoice_id}")
    short = invoice_id[:8]
    started = time.perf_counter()
    storage.set_status(invoice_id, InvoiceStatus.PROCESSING)
    try:
        if not row.stored_path or not Path(row.stored_path).exists():
            raise ingest.IngestError("The stored copy of this file is missing. Import the file again.")

        # ---- render + preprocess + OCR ----
        pages = storage.get_pages(row.doc_id, row.page_start, row.page_end)
        for page in pages:
            if reuse_ocr and page.ocr.error is None and page.ocr.text.strip():
                continue
            _notify(progress, "OCR", f"{row.filename}: page {page.page_no}/{row.page_end}")
            ocr_page(row, page.page_no, services)
        pages = storage.get_pages(row.doc_id, row.page_start, row.page_end)
        good = [p for p in pages if p.ocr.error is None and p.ocr.text.strip()]
        if not good:
            problem = next((p.ocr.error for p in pages if p.ocr.error), "No text could be read.")
            storage.set_status(invoice_id, InvoiceStatus.FAILED, f"{problem} Fix the OCR setup, then use 'Retry failed'.",
                               total_ms=int((time.perf_counter() - started) * 1000))
            log.warning("invoice %s: OCR produced no text", short)
            return InvoiceStatus.FAILED
        text = _combine_text(good)
        lines = [ln for p in good for ln in p.ocr.lines]

        # ---- local LLM extraction ----
        _notify(progress, "Extracting", f"{row.filename}: asking {s.ollama.model}")
        try:
            result = extract_invoice(text, s.ollama, services.chat_client)
        except (OllamaUnavailableError, OllamaModelMissingError) as exc:
            storage.set_status(invoice_id, InvoiceStatus.PENDING)
            raise ServiceUnavailable(str(exc)) from exc
        except OllamaError as exc:
            storage.set_status(invoice_id, InvoiceStatus.FAILED, str(exc), total_ms=int((time.perf_counter() - started) * 1000))
            return InvoiceStatus.FAILED
        if not result.ok or result.invoice is None:
            storage.set_status(invoice_id, InvoiceStatus.FAILED, result.error, llm_ms=result.elapsed_ms,
                               total_ms=int((time.perf_counter() - started) * 1000))
            log.warning("invoice %s: extraction failed (retried=%s)", short, result.retried)
            return InvoiceStatus.FAILED

        # ---- deterministic validation ----
        _notify(progress, "Validating", row.filename)
        extracted = normalize_invoice(result.invoice, s.validation)
        pages_failed = [p.page_no for p in pages if p not in good]
        if pages_failed:
            extracted = extracted.model_copy(update={"extraction_notes": [
                *extracted.extraction_notes, f"OCR failed on page(s) {', '.join(map(str, pages_failed))}; they were not used"]})
        corrected = row.corrected if keep_corrections else {}
        effective = ExtractedInvoice.model_validate({**extracted.model_dump(), **corrected})
        issues, field_status, status = run_validation(effective, text, lines, sorted(corrected), services)
        storage.save_extraction(
            invoice_id, extracted, issues, field_status, status, result.model, result.elapsed_ms,
            int((time.perf_counter() - started) * 1000), keep_corrections=keep_corrections,
        )
        log.info("invoice %s: %s in %d ms (issues=%d)", short, status.value, int((time.perf_counter() - started) * 1000), len(issues))
        return status
    except ServiceUnavailable:
        raise
    except ingest.IngestError as exc:
        storage.set_status(invoice_id, InvoiceStatus.FAILED, str(exc))
        log.warning("invoice %s: ingest problem", short)
        return InvoiceStatus.FAILED
    except Exception as exc:  # last resort: never lose the batch to one bad file
        log.exception("invoice %s: unexpected %s", short, type(exc).__name__)
        storage.set_status(invoice_id, InvoiceStatus.FAILED,
                           f"Unexpected problem ({type(exc).__name__}). Details are in the log file; try 'Retry failed'.")
        return InvoiceStatus.FAILED


def retry_page_ocr(invoice_id: str, page_no: int, services: Services) -> OcrPage:
    """Re-run OCR for one page only. The caller can then re-run extraction (``reuse_ocr=True``)."""
    row = services.storage.get_invoice(invoice_id)
    if row is None or not row.stored_path:
        raise KeyError("Unknown invoice")
    try:
        return ocr_page(row, page_no, services).ocr
    except ingest.IngestError as exc:
        return OcrPage(engine="none", error=str(exc))


def apply_corrections(
    invoice_id: str, corrections: dict[str, Any], services: Services, mark_reviewed: bool = False,
) -> tuple[InvoiceStatus, list[ValidationIssue]]:
    """Store human corrections and re-run the deterministic checks on the corrected values."""
    storage = services.storage
    row = storage.get_invoice(invoice_id)
    if row is None or row.extracted is None:
        raise ValueError("This invoice has no extracted data to correct yet.")
    effective = ExtractedInvoice.model_validate({**row.extracted, **corrections})  # raises on invalid amounts
    pages = [p for p in storage.get_pages(row.doc_id, row.page_start, row.page_end) if p.ocr.text.strip()]
    text = _combine_text(pages) if pages else ""
    lines = [ln for p in pages for ln in p.ocr.lines]
    issues, field_status, status = run_validation(effective, text, lines, sorted(corrections), services)
    if mark_reviewed:
        status = InvoiceStatus.REVIEWED
    storage.save_corrections(invoice_id, corrections, issues, field_status, status)
    return status, issues


def mark_reviewed(invoice_id: str, services: Services, reviewed: bool = True) -> InvoiceStatus:
    row = services.storage.get_invoice(invoice_id)
    if row is None or row.extracted is None:
        raise ValueError("Nothing to review yet.")
    return apply_corrections(invoice_id, row.corrected, services, mark_reviewed=reviewed)[0]


def compute_duplicates(rows: list[InvoiceRow], day_first: bool = True) -> dict[str, list[DuplicateCandidate]]:
    items = []
    for r in rows:
        inv = r.effective
        if inv is None or r.status == InvoiceStatus.FAILED:
            continue
        items.append(DupInput(
            r.invoice_id, inv.vendor_gstin, inv.vendor_name, inv.invoice_number,
            normalized_date_or_raw(inv.invoice_date, day_first), inv.total_amount, r.filename))
    return find_duplicates(items)


# ======================================================================
# Background batch job (used by the UI; also usable from scripts)
# ======================================================================
@dataclass
class JobProgress:
    state: str = "idle"  # idle | running | paused | cancelling | finished | cancelled | stopped
    total: int = 0
    done: int = 0
    failed: int = 0
    current_id: str | None = None
    current_name: str = ""
    stage: str = ""
    message: str = ""
    started_at: float | None = None
    finished_at: float | None = None
    events: list[str] = field(default_factory=list)

    @property
    def active(self) -> bool:
        return self.state in {"running", "paused", "cancelling"}

    @property
    def fraction(self) -> float:
        return self.done / self.total if self.total else 0.0


class JobManager:
    """Runs a batch on a background thread so the UI stays responsive."""

    def __init__(self, services: Services) -> None:
        self.services = services
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._pause = threading.Event()
        self._cancel = threading.Event()
        self._progress = JobProgress()
        self._events: deque[str] = deque(maxlen=60)

    # ---- control ------------------------------------------------------
    def start(self, invoice_ids: list[str] | None = None, limit: int | None = None,
              reuse_ocr: bool = False, keep_corrections: bool = True) -> int:
        with self._lock:
            if self._progress.active:
                raise RuntimeError("A batch is already running.")
            ids = list(invoice_ids) if invoice_ids is not None else self.services.storage.pending_ids()
            if limit:
                ids = ids[:limit]
            if not ids:
                return 0
            self._pause.clear()
            self._cancel.clear()
            self._events.clear()
            self._progress = JobProgress(state="running", total=len(ids), started_at=time.time())
            self._thread = threading.Thread(target=self._run, args=(ids, reuse_ocr, keep_corrections), daemon=True, name="gst-batch")
            self._thread.start()
            return len(ids)

    def pause(self) -> None:
        with self._lock:
            if self._progress.state == "running":
                self._pause.set()
                self._progress.state = "paused"

    def resume(self) -> None:
        with self._lock:
            if self._progress.state == "paused":
                self._pause.clear()
                self._progress.state = "running"

    def cancel(self) -> None:
        """Stop after the file that is being processed. Remaining files stay Pending."""
        with self._lock:
            if self._progress.active:
                self._cancel.set()
                self._pause.clear()
                self._progress.state = "cancelling"

    def snapshot(self) -> JobProgress:
        with self._lock:
            snap = JobProgress(**{**self._progress.__dict__, "events": list(self._events)})
        return snap

    def wait(self, timeout: float | None = None) -> None:
        t = self._thread
        if t:
            t.join(timeout)

    # ---- worker -------------------------------------------------------
    def _update(self, **kw: Any) -> None:
        with self._lock:
            for k, v in kw.items():
                setattr(self._progress, k, v)

    def _event(self, text: str) -> None:
        with self._lock:
            self._events.append(f"{time.strftime('%H:%M:%S')}  {text}")

    def _run(self, ids: list[str], reuse_ocr: bool, keep_corrections: bool) -> None:
        storage = self.services.storage
        final = "finished"
        message = ""
        try:
            for iid in ids:
                if self._cancel.is_set():
                    final = "cancelled"
                    break
                while self._pause.is_set() and not self._cancel.is_set():
                    time.sleep(0.2)
                if self._cancel.is_set():
                    final = "cancelled"
                    break
                row = storage.get_invoice(iid)
                name = row.filename if row else iid
                self._update(current_id=iid, current_name=name, stage="Starting")

                def on_progress(stage: str, msg: str, _self: JobManager = self) -> None:
                    _self._update(stage=stage, message=msg)

                try:
                    status = process_invoice(iid, self.services, on_progress, reuse_ocr, keep_corrections)
                except ServiceUnavailable as exc:
                    final, message = "stopped", str(exc)
                    self._event(f"Stopped: {exc}")
                    break
                except Exception as exc:  # pragma: no cover - process_invoice already guards
                    log.exception("batch: unexpected %s", type(exc).__name__)
                    status = InvoiceStatus.FAILED
                with self._lock:
                    self._progress.done += 1
                    if status == InvoiceStatus.FAILED:
                        self._progress.failed += 1
                self._event(f"{name}: {status.value}")
        finally:
            leftovers = storage.reset_stale_processing()
            if leftovers:
                log.info("batch ended with %d unfinished item(s) returned to the queue", leftovers)
            self._update(state=final, message=message, finished_at=time.time(), current_id=None, stage="")
