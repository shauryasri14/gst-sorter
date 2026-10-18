"""Local persistence (SQLite). Everything stays under ``settings.data_dir`` on this computer.

Three kinds of data are kept apart on purpose:
* the original file (``uploads/``) and rendered page images (``pages/``) - never modified;
* OCR output (``pages`` table);
* extracted values (``invoices.extracted_json``) vs. human corrections (``invoices.corrected_json``).
  A re-run of extraction replaces the first but never the second unless the user asks for that.
"""

from __future__ import annotations

import json
import logging
import shutil
import sqlite3
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

from config import Settings
from pipeline.ingest import IngestedFile
from pipeline.schemas import (
    EXTRACTION_FIELDS,
    ExtractedInvoice,
    InvoiceStatus,
    OcrLine,
    OcrPage,
    ValidationIssue,
    to_jsonable,
)

log = logging.getLogger(__name__)

SCHEMA = """
CREATE TABLE IF NOT EXISTS documents (
    doc_id TEXT PRIMARY KEY,
    filename TEXT NOT NULL,
    ext TEXT, kind TEXT, size_bytes INTEGER, page_count INTEGER DEFAULT 0,
    stored_path TEXT, added_at TEXT
);
CREATE TABLE IF NOT EXISTS pages (
    doc_id TEXT NOT NULL, page_no INTEGER NOT NULL,
    image_path TEXT, ocr_image_path TEXT,
    ocr_text TEXT DEFAULT '', ocr_lines_json TEXT DEFAULT '[]',
    ocr_mean_conf REAL, ocr_engine TEXT, ocr_error TEXT, ocr_ms INTEGER, preprocess_json TEXT DEFAULT '{}',
    PRIMARY KEY (doc_id, page_no)
);
CREATE TABLE IF NOT EXISTS invoices (
    invoice_id TEXT PRIMARY KEY,
    doc_id TEXT NOT NULL,
    page_start INTEGER DEFAULT 1, page_end INTEGER DEFAULT 1,
    status TEXT NOT NULL,
    extracted_json TEXT, corrected_json TEXT DEFAULT '{}',
    issues_json TEXT DEFAULT '[]', field_status_json TEXT DEFAULT '{}',
    error TEXT, model TEXT, llm_ms INTEGER, total_ms INTEGER,
    reviewed_at TEXT, updated_at TEXT, processed_at TEXT
);
CREATE INDEX IF NOT EXISTS idx_invoices_status ON invoices(status);
CREATE INDEX IF NOT EXISTS idx_invoices_doc ON invoices(doc_id);
"""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


@dataclass
class PageRecord:
    doc_id: str
    page_no: int
    image_path: str | None = None
    ocr_image_path: str | None = None
    ocr: OcrPage = field(default_factory=OcrPage)
    preprocess: dict[str, Any] = field(default_factory=dict)


@dataclass
class InvoiceRow:
    invoice_id: str
    doc_id: str
    filename: str
    ext: str | None
    stored_path: str | None
    kind: str | None
    page_count: int
    page_start: int
    page_end: int
    status: InvoiceStatus
    extracted: dict[str, Any] | None
    corrected: dict[str, Any]
    issues: list[ValidationIssue]
    field_status: dict[str, Any]
    error: str | None
    model: str | None
    llm_ms: int | None
    total_ms: int | None
    reviewed_at: str | None
    updated_at: str | None

    @property
    def pages(self) -> range:
        return range(self.page_start, self.page_end + 1)

    @property
    def edited_fields(self) -> list[str]:
        return sorted(self.corrected)

    @property
    def extracted_invoice(self) -> ExtractedInvoice | None:
        return ExtractedInvoice.model_validate(self.extracted) if self.extracted is not None else None

    @property
    def effective(self) -> ExtractedInvoice | None:
        """Extracted values with the human corrections applied on top."""
        if self.extracted is None:
            return None
        return ExtractedInvoice.model_validate({**self.extracted, **self.corrected})


def make_invoice_id(doc_id: str, page: int | None = None) -> str:
    return f"{doc_id[:32]}" if page is None else f"{doc_id[:32]}-p{page}"


class Storage:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        settings.ensure_dirs()
        with self._conn() as c:
            c.executescript(SCHEMA)

    # ---- connection ---------------------------------------------------
    @contextmanager
    def _conn(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(self.settings.db_path, timeout=30)
        conn.row_factory = sqlite3.Row
        try:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA foreign_keys=ON")
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    # ---- documents ----------------------------------------------------
    def document_exists(self, doc_id: str) -> bool:
        with self._conn() as c:
            return c.execute("SELECT 1 FROM documents WHERE doc_id=?", (doc_id,)).fetchone() is not None

    def add_document(self, f: IngestedFile, pdf_mode: str | None = None) -> bool:
        """Register a validated file and create Pending invoice rows. Returns False if already imported."""
        mode = pdf_mode or self.settings.processing.pdf_mode
        if self.document_exists(f.doc_id):
            return False
        now = _now()
        with self._conn() as c:
            c.execute(
                "INSERT INTO documents(doc_id, filename, ext, kind, size_bytes, page_count, stored_path, added_at) VALUES (?,?,?,?,?,?,?,?)",
                (f.doc_id, f.filename, f.ext, f.kind, f.size_bytes, f.page_count, str(f.stored_path), now),
            )
            for n in range(1, f.page_count + 1):
                c.execute("INSERT INTO pages(doc_id, page_no) VALUES (?,?)", (f.doc_id, n))
            if f.kind == "pdf" and mode == "per_page" and f.page_count > 1:
                spans = [(make_invoice_id(f.doc_id, n), n, n) for n in range(1, f.page_count + 1)]
            else:
                spans = [(make_invoice_id(f.doc_id), 1, f.page_count)]
            for invoice_id, start, end in spans:
                c.execute(
                    "INSERT INTO invoices(invoice_id, doc_id, page_start, page_end, status, updated_at) VALUES (?,?,?,?,?,?)",
                    (invoice_id, f.doc_id, start, end, InvoiceStatus.PENDING.value, now),
                )
        return True

    def add_rejected_file(self, filename: str, reason: str, token: str) -> str:
        """Keep a visible record of a file that could not be imported (nothing is stored for it)."""
        doc_id = f"rejected-{uuid.uuid5(uuid.NAMESPACE_URL, token).hex}"
        now = _now()
        with self._conn() as c:
            c.execute(
                "INSERT OR REPLACE INTO documents(doc_id, filename, page_count, added_at) VALUES (?,?,0,?)",
                (doc_id, filename, now),
            )
            c.execute(
                "INSERT OR REPLACE INTO invoices(invoice_id, doc_id, status, error, updated_at) VALUES (?,?,?,?,?)",
                (doc_id[:32], doc_id, InvoiceStatus.FAILED.value, reason, now),
            )
        return doc_id[:32]

    # ---- pages / OCR --------------------------------------------------
    def save_page(self, rec: PageRecord) -> None:
        with self._conn() as c:
            c.execute(
                """UPDATE pages SET image_path=?, ocr_image_path=?, ocr_text=?, ocr_lines_json=?, ocr_mean_conf=?,
                   ocr_engine=?, ocr_error=?, ocr_ms=?, preprocess_json=? WHERE doc_id=? AND page_no=?""",
                (rec.image_path, rec.ocr_image_path, rec.ocr.text,
                 json.dumps([ln.model_dump() for ln in rec.ocr.lines]), rec.ocr.mean_confidence,
                 rec.ocr.engine, rec.ocr.error, rec.ocr.elapsed_ms, json.dumps(rec.preprocess), rec.doc_id, rec.page_no),
            )

    @staticmethod
    def _page_from_row(r: sqlite3.Row) -> PageRecord:
        lines = [OcrLine.model_validate(x) for x in json.loads(r["ocr_lines_json"] or "[]")]
        ocr = OcrPage(text=r["ocr_text"] or "", lines=lines, mean_confidence=r["ocr_mean_conf"],
                      engine=r["ocr_engine"] or "", error=r["ocr_error"], elapsed_ms=r["ocr_ms"] or 0)
        return PageRecord(r["doc_id"], r["page_no"], r["image_path"], r["ocr_image_path"], ocr,
                          json.loads(r["preprocess_json"] or "{}"))

    def get_page(self, doc_id: str, page_no: int) -> PageRecord | None:
        with self._conn() as c:
            r = c.execute("SELECT * FROM pages WHERE doc_id=? AND page_no=?", (doc_id, page_no)).fetchone()
        return self._page_from_row(r) if r else None

    def get_pages(self, doc_id: str, first: int | None = None, last: int | None = None) -> list[PageRecord]:
        with self._conn() as c:
            rows = c.execute("SELECT * FROM pages WHERE doc_id=? ORDER BY page_no", (doc_id,)).fetchall()
        pages = [self._page_from_row(r) for r in rows]
        return [p for p in pages if (first is None or p.page_no >= first) and (last is None or p.page_no <= last)]

    # ---- invoices -----------------------------------------------------
    _SELECT = """SELECT i.*, d.filename, d.ext, d.stored_path, d.kind, d.page_count
                 FROM invoices i JOIN documents d ON d.doc_id = i.doc_id"""

    @staticmethod
    def _row_from(r: sqlite3.Row) -> InvoiceRow:
        return InvoiceRow(
            invoice_id=r["invoice_id"], doc_id=r["doc_id"], filename=r["filename"], ext=r["ext"],
            stored_path=r["stored_path"], kind=r["kind"], page_count=r["page_count"] or 0,
            page_start=r["page_start"] or 1, page_end=r["page_end"] or 1,
            status=InvoiceStatus(r["status"]),
            extracted=json.loads(r["extracted_json"]) if r["extracted_json"] else None,
            corrected=json.loads(r["corrected_json"] or "{}"),
            issues=[ValidationIssue.model_validate(x) for x in json.loads(r["issues_json"] or "[]")],
            field_status=json.loads(r["field_status_json"] or "{}"),
            error=r["error"], model=r["model"], llm_ms=r["llm_ms"], total_ms=r["total_ms"],
            reviewed_at=r["reviewed_at"], updated_at=r["updated_at"],
        )

    def get_invoice(self, invoice_id: str) -> InvoiceRow | None:
        with self._conn() as c:
            r = c.execute(self._SELECT + " WHERE i.invoice_id=?", (invoice_id,)).fetchone()
        return self._row_from(r) if r else None

    def list_invoices(self, status: InvoiceStatus | None = None) -> list[InvoiceRow]:
        sql = self._SELECT + (" WHERE i.status=?" if status else "") + " ORDER BY d.added_at, d.filename, i.page_start"
        with self._conn() as c:
            rows = c.execute(sql, (status.value,) if status else ()).fetchall()
        return [self._row_from(r) for r in rows]

    def pending_ids(self) -> list[str]:
        with self._conn() as c:
            rows = c.execute(
                "SELECT i.invoice_id FROM invoices i JOIN documents d ON d.doc_id=i.doc_id WHERE i.status=? "
                "ORDER BY d.added_at, d.filename, i.page_start", (InvoiceStatus.PENDING.value,)).fetchall()
        return [r["invoice_id"] for r in rows]

    def set_status(self, invoice_id: str, status: InvoiceStatus, error: str | None = None, **extra: Any) -> None:
        cols = {"status": status.value, "error": error, "updated_at": _now(), **extra}
        sets = ", ".join(f"{k}=?" for k in cols)
        with self._conn() as c:
            c.execute(f"UPDATE invoices SET {sets} WHERE invoice_id=?", (*cols.values(), invoice_id))

    def save_extraction(
        self, invoice_id: str, invoice: ExtractedInvoice, issues: list[ValidationIssue], field_status: dict[str, Any],
        status: InvoiceStatus, model: str, llm_ms: int, total_ms: int, keep_corrections: bool = True,
    ) -> None:
        with self._conn() as c:
            c.execute(
                """UPDATE invoices SET extracted_json=?, issues_json=?, field_status_json=?, status=?, error=NULL, model=?,
                   llm_ms=?, total_ms=?, reviewed_at=NULL, updated_at=?, processed_at=?""" +
                ("" if keep_corrections else ", corrected_json='{}'") + " WHERE invoice_id=?",
                (json.dumps(to_jsonable(invoice.model_dump())), json.dumps([i.model_dump(mode="json") for i in issues]),
                 json.dumps(field_status), status.value, model, llm_ms, total_ms, _now(), _now(), invoice_id),
            )

    def save_corrections(
        self, invoice_id: str, corrections: dict[str, Any], issues: list[ValidationIssue], field_status: dict[str, Any],
        status: InvoiceStatus,
    ) -> None:
        """Replace the stored corrections with ``corrections`` (field -> value) and store the re-validation."""
        unknown = set(corrections) - set(EXTRACTION_FIELDS) - {"extraction_notes"}
        if unknown:
            raise ValueError(f"Unknown field(s): {', '.join(sorted(unknown))}")
        reviewed_at = _now() if status == InvoiceStatus.REVIEWED else None
        with self._conn() as c:
            c.execute(
                "UPDATE invoices SET corrected_json=?, issues_json=?, field_status_json=?, status=?, reviewed_at=?, updated_at=? WHERE invoice_id=?",
                (json.dumps(to_jsonable(corrections)), json.dumps([i.model_dump(mode="json") for i in issues]),
                 json.dumps(field_status), status.value, reviewed_at, _now(), invoice_id),
            )

    def reset_to_pending(self, invoice_ids: list[str]) -> int:
        """Queue invoices for processing again (extracted data is kept until the new result replaces it)."""
        n = 0
        with self._conn() as c:
            for iid in invoice_ids:
                cur = c.execute(
                    "UPDATE invoices SET status=?, error=NULL, updated_at=? WHERE invoice_id=? AND doc_id NOT LIKE 'rejected-%'",
                    (InvoiceStatus.PENDING.value, _now(), iid))
                n += cur.rowcount
        return n

    def retry_failed(self) -> int:
        ids = [r.invoice_id for r in self.list_invoices(InvoiceStatus.FAILED)]
        return self.reset_to_pending(ids)

    def reset_stale_processing(self) -> int:
        """After a crash or restart nothing is really 'Processing'. Put those items back in the queue."""
        with self._conn() as c:
            cur = c.execute("UPDATE invoices SET status=?, updated_at=? WHERE status=?",
                            (InvoiceStatus.PENDING.value, _now(), InvoiceStatus.PROCESSING.value))
            return cur.rowcount

    # ---- housekeeping -------------------------------------------------
    def purge_all(self, include_logs: bool = False) -> dict[str, int]:
        """Delete every record and every generated file (uploads, page images, temp files)."""
        with self._conn() as c:
            counts = {
                "invoices": c.execute("SELECT COUNT(*) FROM invoices").fetchone()[0],
                "documents": c.execute("SELECT COUNT(*) FROM documents").fetchone()[0],
            }
            c.executescript("DELETE FROM invoices; DELETE FROM pages; DELETE FROM documents;")
        removed = 0
        targets = [self.settings.uploads_dir, self.settings.pages_dir, self.settings.tmp_dir]
        if include_logs:
            targets.append(self.settings.logs_dir)
        for directory in targets:
            if directory.exists():
                for child in directory.iterdir():
                    if child.is_dir():
                        shutil.rmtree(child, ignore_errors=True)
                    else:
                        child.unlink(missing_ok=True)
                    removed += 1
        with self._conn() as c:
            c.execute("VACUUM")
        counts["files_removed"] = removed
        return counts

    def stats(self) -> dict[str, int]:
        counts = {s.value: 0 for s in InvoiceStatus}
        with self._conn() as c:
            for r in c.execute("SELECT status, COUNT(*) n FROM invoices GROUP BY status"):
                counts[r["status"]] = r["n"]
        return counts
