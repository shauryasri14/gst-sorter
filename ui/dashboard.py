"""Dashboard: counts, provisional totals and system health."""

from __future__ import annotations

from collections import Counter
from decimal import Decimal

import pandas as pd
import streamlit as st

from pipeline.runner import compute_duplicates
from pipeline.schemas import InvoiceStatus, Severity
from ui.common import money, ocr_problem, ollama_status, services_and_jobs


def render() -> None:
    services, _ = services_and_jobs()
    storage, settings = services.storage, services.settings
    st.header("Dashboard")

    rows = storage.list_invoices()
    if not rows:
        st.info("Nothing imported yet. Open **Import documents** to add invoices.")
        _health(services)
        return

    by_status = Counter(r.status for r in rows)
    docs = len({r.doc_id for r in rows})
    processed = sum(by_status[s] for s in (InvoiceStatus.EXTRACTED, InvoiceStatus.NEEDS_REVIEW, InvoiceStatus.REVIEWED, InvoiceStatus.FAILED))
    successful = sum(by_status[s] for s in (InvoiceStatus.EXTRACTED, InvoiceStatus.NEEDS_REVIEW, InvoiceStatus.REVIEWED))
    dups = compute_duplicates(rows, settings.validation.date_day_first)

    cols = st.columns(6)
    cols[0].metric("Files imported", docs)
    cols[1].metric("Processed", processed, help="Invoices that finished the pipeline (successfully or not).")
    cols[2].metric("Successful extractions", successful, help="Extraction succeeded. This does NOT mean the values are correct.")
    cols[3].metric("Need review", by_status[InvoiceStatus.NEEDS_REVIEW])
    cols[4].metric("Failed", by_status[InvoiceStatus.FAILED])
    cols[5].metric("Duplicate candidates", len(dups), help="Invoices that look like a copy of another one. Nothing is removed automatically.")

    left, right = st.columns([3, 2])
    with left:
        st.subheader("Provisional totals")
        usable = [r for r in rows if r.effective is not None and r.status != InvoiceStatus.FAILED]
        inr = [r for r in usable if (r.effective.currency or "INR") == "INR"]  # type: ignore[union-attr]

        def total(attr: str) -> Decimal:
            return sum((getattr(r.effective, attr) for r in inr if getattr(r.effective, attr) is not None), Decimal("0"))

        reviewed = sum(1 for r in inr if r.status == InvoiceStatus.REVIEWED)
        t = st.columns(5)
        for col, (label, attr) in zip(t, [("Taxable", "taxable_amount"), ("CGST", "cgst"), ("SGST", "sgst"),
                                           ("IGST", "igst"), ("Total", "total_amount")]):
            col.metric(label, money(total(attr)))
        st.warning(
            f"**Provisional.** These sums use values straight from extraction (plus your corrections) for {len(inr)} INR invoice(s); "
            f"only {reviewed} of them {'is' if reviewed == 1 else 'are'} reviewed. Do not use them for filing until every row is reviewed."
        )
        if len(inr) != len(usable):
            st.caption(f"{len(usable) - len(inr)} invoice(s) in another currency are excluded.")

    with right:
        st.subheader("Status")
        frame = pd.DataFrame({"Status": [s.value for s in InvoiceStatus], "Invoices": [by_status[s] for s in InvoiceStatus]})
        st.bar_chart(frame, x="Status", y="Invoices")

    st.subheader("Most common validation findings")
    codes = Counter((i.code, i.severity) for r in rows for i in r.issues if i.severity != Severity.INFO)
    if codes:
        table = pd.DataFrame(
            [{"Check": c, "Severity": sev.value, "Invoices": n} for (c, sev), n in codes.most_common(10)]
        )
        st.dataframe(table, hide_index=True)
    else:
        st.caption("No warnings or errors so far.")
    _health(services)


def _health(services) -> None:  # noqa: ANN001
    st.subheader("System check")
    settings = services.settings
    c1, c2, c3 = st.columns(3)
    status = ollama_status(settings)
    with c1:
        (st.success if status.ready else st.error)(status.message)
    with c2:
        problem = ocr_problem(settings)
        if problem:
            st.error(problem)
        else:
            st.success(f"OCR engine ready ({settings.ocr.engine}, fallback: {settings.ocr.fallback_engine or 'none'}).")
    with c3:
        st.success(f"All data stays in `{settings.data_path}` on this computer.")
