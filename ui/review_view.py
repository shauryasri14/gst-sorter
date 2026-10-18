"""Invoice review: original page beside editable fields, with the exact reason for every flag."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import streamlit as st

from pipeline.runner import (
    ServiceUnavailable,
    apply_corrections,
    compute_duplicates,
    process_invoice,
    retry_page_ocr,
)
from pipeline.schemas import (
    AMOUNT_FIELDS,
    FIELD_LABELS,
    InvoiceStatus,
    Severity,
    parse_amount,
)
from pipeline.storage import InvoiceRow
from pipeline.validate import normalize_gstin
from ui.common import SEVERITY_ICON, render_overlay, services_and_jobs, short_name, status_label

FORM_LAYOUT = [
    ("vendor_name", "vendor_gstin"),
    ("buyer_name", "buyer_gstin"),
    ("invoice_number", "invoice_date"),
    ("taxable_amount", "total_amount"),
    ("cgst", "sgst"),
    ("igst", "currency"),
]
HELP = {
    "invoice_date": "Use YYYY-MM-DD if you can. Day-first dates such as 14/09/2026 are also understood.",
    "vendor_gstin": "15 characters. A valid check character only means the number is well-formed, not that it is registered.",
    "buyer_gstin": "Optional for small B2C receipts.",
    "taxable_amount": "Value before tax. Leave empty if not on the document. Empty means unknown, not zero.",
    "cgst": "Tax amount in rupees (not the percentage).",
    "sgst": "Tax amount in rupees (not the percentage).",
    "igst": "Tax amount in rupees (not the percentage).",
}


def render() -> None:
    services, _ = services_and_jobs()
    storage, settings = services.storage, services.settings
    st.header("Invoice review")

    all_rows = storage.list_invoices()
    dups = compute_duplicates(all_rows, settings.validation.date_day_first)
    rows = [r for r in all_rows if r.status not in (InvoiceStatus.PENDING, InvoiceStatus.PROCESSING)]
    if not rows:
        st.info("No processed invoices yet. Import documents and process them first.")
        return

    # ---- filters ----
    f1, f2, f3, f4 = st.columns([3, 3, 2, 2])
    statuses = f1.multiselect("Status", [s.value for s in InvoiceStatus if s not in (InvoiceStatus.PENDING, InvoiceStatus.PROCESSING)],
                              default=[], placeholder="All statuses")
    vendor_q = f2.text_input("Vendor contains", placeholder="name or GSTIN").strip().lower()
    severity = f3.selectbox("Validation", ["Any", "Has errors", "Has warnings", "No flags"])
    only_dups = f4.checkbox("Duplicate candidates only")

    def keep(r: InvoiceRow) -> bool:
        if statuses and r.status.value not in statuses:
            return False
        inv = r.effective
        if vendor_q and not (inv and vendor_q in f"{inv.vendor_name or ''} {inv.vendor_gstin or ''}".lower()):
            return False
        has_err = any(i.severity == Severity.ERROR for i in r.issues) or r.status == InvoiceStatus.FAILED
        has_warn = any(i.severity == Severity.WARNING for i in r.issues)
        if severity == "Has errors" and not has_err:
            return False
        if severity == "Has warnings" and not has_warn:
            return False
        if severity == "No flags" and (has_err or has_warn):
            return False
        return not (only_dups and r.invoice_id not in dups)

    filtered = [r for r in rows if keep(r)]
    st.caption(f"Showing {len(filtered)} of {len(rows)} processed invoices.")
    if not filtered:
        st.warning("No invoices match these filters.")
        return

    # ---- navigation ----
    ids = [r.invoice_id for r in filtered]
    labels = {r.invoice_id: f"{status_label(r.status)}  {short_name(r.filename)}" + (f" (p{r.page_start})" if r.page_end == r.page_start and r.page_count > 1 else "")
              for r in filtered}
    if st.session_state.get("review_select") not in ids:
        st.session_state["review_select"] = ids[0]
    pos = ids.index(st.session_state["review_select"])

    def go(delta: int) -> None:
        st.session_state["review_select"] = ids[(ids.index(st.session_state["review_select"]) + delta) % len(ids)]

    n1, n2, n3 = st.columns([1, 6, 1])
    n1.button("◀ Previous", on_click=go, args=(-1,), disabled=len(ids) < 2)
    n2.selectbox("Invoice", ids, format_func=lambda i: labels[i], key="review_select", label_visibility="collapsed")
    n3.button("Next ▶", on_click=go, args=(1,), disabled=len(ids) < 2)
    st.caption(f"Invoice {pos + 1} of {len(ids)}")

    row = storage.get_invoice(st.session_state["review_select"])
    if row is None:
        st.error("This invoice no longer exists.")
        return

    left, right = st.columns([5, 6], gap="large")
    with left:
        _document_panel(row, services)
    with right:
        _data_panel(row, services, dups.get(row.invoice_id, []), {r.invoice_id: r.filename for r in all_rows})


# ----------------------------------------------------------------------
def _document_panel(row: InvoiceRow, services) -> None:  # noqa: ANN001
    storage, settings = services.storage, services.settings
    st.subheader("Original document")
    page_no = row.page_start
    if len(row.pages) > 1:
        page_no = st.selectbox("Page", list(row.pages), key=f"page_{row.invoice_id}")
    page = storage.get_page(row.doc_id, page_no)

    o1, o2 = st.columns(2)
    show_boxes = o1.checkbox("Show OCR boxes (red = low confidence)", key=f"boxes_{row.invoice_id}")
    show_ocr_image = o2.checkbox("Show the image OCR read", key=f"ocrimg_{row.invoice_id}",
                                 help="The page after rotation / deskew / contrast steps. The original is never changed.")
    if page is None or not page.image_path or not Path(page.image_path).exists():
        st.warning("No page image is available for this invoice yet (processing may have failed before rendering).")
    else:
        use_ocr_image = (show_boxes or show_ocr_image) and page.ocr_image_path and Path(page.ocr_image_path).exists()
        path = page.ocr_image_path if use_ocr_image else page.image_path
        if show_boxes and use_ocr_image:
            st.image(render_overlay(path, page.ocr.lines, settings.ocr.low_confidence_threshold))  # type: ignore[arg-type]
        else:
            st.image(path)  # type: ignore[arg-type]
        steps = ", ".join(page.preprocess.get("steps", [])) or "none"
        conf = f"{page.ocr.mean_confidence:.0%}" if page.ocr.mean_confidence is not None else "n/a"
        st.caption(f"OCR engine: {page.ocr.engine or 'n/a'} · mean confidence {conf} · preprocessing: {steps}"
                   + (" (original used)" if page.preprocess.get("used_original") else ""))
        if page.ocr.error:
            st.error(page.ocr.error)
        with st.expander("OCR text for this page"):
            st.text(page.ocr.text or "(no text)")
        if st.button("Retry OCR for this page", key=f"retry_ocr_{row.invoice_id}_{page_no}"):
            with st.spinner("Reading the page again…"):
                result = retry_page_ocr(row.invoice_id, page_no, services)
            if result.error:
                st.error(result.error)
            else:
                st.session_state["flash_review"] = ("success", f"Page {page_no} was read again ({len(result.lines)} lines). Use 'Re-run extraction' to update the fields.")
                st.rerun()


def _open(invoice_id: str) -> None:
    st.session_state["review_select"] = invoice_id


def _note_for(row: InvoiceRow, name: str) -> None:
    info = row.field_status.get(name)
    if not info:
        return
    status = info.get("status")
    conf = info.get("ocr_confidence")
    reasons = info.get("reasons", [])
    if status == "missing":
        st.caption(":red[Missing: not found or not readable. Fill it in if it is on the document.]")
    elif status == "low_ocr_confidence":
        st.caption(f":orange[Low OCR confidence ({conf:.0%}). Check against the document.]")
    elif status == "flagged":
        st.caption(":orange[Flagged by a check: see the list above.]")
    elif conf is not None:
        st.caption(f":green[OCR confidence {conf:.0%}]")
    if info.get("edited"):
        st.caption(":blue[Edited by you]")
    del reasons


def _data_panel(row: InvoiceRow, services, dup_candidates, names: dict[str, str]) -> None:  # noqa: ANN001
    st.subheader("Extracted data")
    kind, text = st.session_state.pop("flash_review", (None, None))
    if kind:
        getattr(st, kind)(text)

    pages = f"pages {row.page_start}-{row.page_end}" if row.page_end > row.page_start else (f"page {row.page_start}" if row.page_count > 1 else "")
    st.markdown(f"**{status_label(row.status)}** · `{row.filename}` {pages}")
    if row.status == InvoiceStatus.REVIEWED:
        st.success(f"Reviewed by a person on {row.reviewed_at}. Editing a reviewed invoice asks for a new review.")
    elif row.status in (InvoiceStatus.EXTRACTED, InvoiceStatus.NEEDS_REVIEW):
        st.info("Not verified yet. Extraction only means the model produced values. A person must confirm them against the document.")
    if row.error:
        st.error(row.error)

    for cand in dup_candidates:
        c1, c2 = st.columns([6, 1])
        c1.warning(f"Possible duplicate ({cand.strength}): **{names.get(cand.other_id, cand.other_id)}** · {cand.reason}")
        c2.button("Open", key=f"open_{row.invoice_id}_{cand.other_id}", on_click=_open, args=(cand.other_id,),
                  help="Jump to that invoice (it must match the current filters).")

    if row.extracted is None:
        st.warning("There is no extracted data for this file. Fix the problem above, then retry from **Import documents**.")
        _rerun_controls(row, services)
        return

    # ---- validation findings ----
    errors = [i for i in row.issues if i.severity == Severity.ERROR]
    warns = [i for i in row.issues if i.severity == Severity.WARNING]
    infos = [i for i in row.issues if i.severity == Severity.INFO]
    with st.expander(f"Validation: {len(errors)} error(s), {len(warns)} warning(s), {len(infos)} note(s)", expanded=bool(errors or warns)):
        if not row.issues:
            st.success("All automatic checks passed. A person still has to confirm the values.")
        for i in row.issues:
            fields = ", ".join(FIELD_LABELS.get(f, f) for f in i.fields)
            st.markdown(f"{SEVERITY_ICON[i.severity]} **{i.code}** · {i.message}" + (f"  \n<span class='gst-note'>Fields: {fields}</span>" if fields else ""),
                        unsafe_allow_html=True)
        st.caption("GSTIN checks confirm format and check character only. They cannot tell whether a GSTIN is registered or active.")

    notes = row.extracted.get("extraction_notes") or []
    if notes:
        with st.expander(f"Notes from extraction ({len(notes)})"):
            for n in notes:
                st.write(f"• {n}")

    # ---- editable fields ----
    effective = row.effective
    assert effective is not None
    version = row.updated_at or "0"
    values: dict[str, str] = {}
    with st.form(f"form_{row.invoice_id}_{version}"):
        for pair in FORM_LAYOUT:
            cols = st.columns(2)
            for col, name in zip(cols, pair):
                with col:
                    current = getattr(effective, name)
                    shown = "" if current is None else (format(current, "f") if name in AMOUNT_FIELDS else str(current))
                    values[name] = st.text_input(FIELD_LABELS[name], value=shown, help=HELP.get(name),
                                                 key=f"{row.invoice_id}_{name}_{version}")
                    _note_for(row, name)
        b1, b2 = st.columns(2)
        save = b1.form_submit_button("Save corrections")
        save_reviewed = b2.form_submit_button("Save & mark as reviewed", type="primary")

    if save or save_reviewed:
        corrections, problems = _collect(row, values)
        if problems:
            for p in problems:
                st.error(p)
        else:
            try:
                apply_corrections(row.invoice_id, corrections, services, mark_reviewed=save_reviewed)
            except ValueError as exc:
                st.error(str(exc))
            else:
                st.session_state["flash_review"] = ("success", "Saved and re-checked." + (" Marked as reviewed." if save_reviewed else ""))
                st.rerun()

    _rerun_controls(row, services)


def _collect(row: InvoiceRow, values: dict[str, str]) -> tuple[dict[str, Any], list[str]]:
    """Turn form text into corrections: only fields that differ from the extracted value are kept."""
    assert row.extracted is not None
    corrections: dict[str, Any] = {}
    problems: list[str] = []
    for name, raw in values.items():
        raw = raw.strip()
        old = row.extracted.get(name)
        if name in AMOUNT_FIELDS:
            try:
                new = parse_amount(raw) if raw else None
                previous = parse_amount(old) if old not in (None, "") else None
            except ValueError:
                problems.append(f"{FIELD_LABELS[name]}: '{raw}' is not a valid amount. Use digits and an optional decimal point.")
                continue
            if new != previous:
                corrections[name] = None if new is None else format(new, "f")
        else:
            new_text: str | None = raw or None
            if name in ("vendor_gstin", "buyer_gstin"):
                new_text = normalize_gstin(new_text)
            if name == "currency" and new_text:
                new_text = new_text.upper()
            if new_text != (old or None):
                corrections[name] = new_text
    return corrections, problems


def _rerun_controls(row: InvoiceRow, services) -> None:  # noqa: ANN001
    st.divider()
    st.markdown("**Other actions**")
    overwrite = st.checkbox("Overwrite my corrections when re-running", value=False, key=f"ow_{row.invoice_id}",
                            help="Off (default): your edits are kept and the new extraction fills in everything else.")
    c1, c2, c3 = st.columns(3)
    if c1.button("Re-run extraction", key=f"rerun_{row.invoice_id}", help="Uses the saved OCR text; does not re-read the image."):
        _run(row, services, reuse_ocr=True, keep=not overwrite)
    if c2.button("Re-run OCR + extraction", key=f"rerun_all_{row.invoice_id}"):
        _run(row, services, reuse_ocr=False, keep=not overwrite)
    if row.corrected and c3.button("Discard my corrections", key=f"discard_{row.invoice_id}"):
        apply_corrections(row.invoice_id, {}, services)
        st.session_state["flash_review"] = ("info", "Your corrections were removed.")
        st.rerun()
    if row.status == InvoiceStatus.REVIEWED and st.button("Remove the 'reviewed' mark", key=f"unreview_{row.invoice_id}"):
        apply_corrections(row.invoice_id, row.corrected, services, mark_reviewed=False)
        st.rerun()


def _run(row: InvoiceRow, services, reuse_ocr: bool, keep: bool) -> None:  # noqa: ANN001
    with st.spinner("Working… the local model can take a while on a laptop."):
        try:
            status = process_invoice(row.invoice_id, services, reuse_ocr=reuse_ocr, keep_corrections=keep)
        except ServiceUnavailable as exc:
            st.error(str(exc))
            return
    kind = "error" if status == InvoiceStatus.FAILED else "success"
    st.session_state["flash_review"] = (kind, f"Finished: {status.value}.")
    st.rerun()
