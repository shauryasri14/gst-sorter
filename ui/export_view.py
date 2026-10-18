"""Export screen and the local-data deletion control."""

from __future__ import annotations

import streamlit as st

from pipeline import export
from pipeline.schemas import InvoiceStatus
from ui.common import services_and_jobs


def render() -> None:
    services, jobs = services_and_jobs()
    storage, settings = services.storage, services.settings
    st.header("Export")

    rows = storage.list_invoices()
    done = [r for r in rows if r.status not in (InvoiceStatus.PENDING, InvoiceStatus.PROCESSING)]
    unreviewed = [r for r in done if r.status != InvoiceStatus.REVIEWED and r.status != InvoiceStatus.FAILED]
    if not done:
        st.info("Nothing to export yet. Process some invoices first.")
    else:
        st.write(f"{len(done)} invoice(s) will be exported. Every row carries its source filename, review status and validation messages.")
        if unreviewed:
            st.warning(f"{len(unreviewed)} invoice(s) are **not reviewed yet**. They are exported with their review status so nobody mistakes them for verified data.")
        include_sheet = st.checkbox("Include a 'Validation' sheet with every error and warning", value=True)
        day_first = settings.validation.date_day_first

        c1, c2, c3 = st.columns(3)
        c1.download_button("⬇ Excel (.xlsx)", data=export.export_all(storage, "xlsx", day_first, include_sheet),
                           file_name="gst_invoices.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", type="primary")
        c2.download_button("⬇ CSV", data=export.export_all(storage, "csv", day_first), file_name="gst_invoices.csv", mime="text/csv")
        c3.download_button("⬇ Validation report (CSV)", data=export.export_all(storage, "validation_csv"),
                           file_name="gst_validation_report.csv", mime="text/csv")
        st.caption("Text that starts with =, +, - or @ is neutralised so a spreadsheet cannot run it as a formula. "
                   "Amounts are written exactly as extracted or corrected.")

    st.divider()
    st.subheader("Delete local data")
    st.write(f"Removes all records, copies of imported files, page images and temporary files from `{settings.data_path}`. "
             "Your original files elsewhere on this computer are not touched. This cannot be undone.")
    snap = jobs.snapshot()
    if snap.active:
        st.info("Wait for the running batch to finish or cancel it first.")
        return
    confirm = st.text_input("Type DELETE to confirm", key="purge_confirm")
    wipe_logs = st.checkbox("Also delete the log file", value=False)
    if st.button("Delete all local data", disabled=confirm.strip() != "DELETE", type="secondary"):
        counts = storage.purge_all(include_logs=wipe_logs)
        for key in list(st.session_state):
            if key.startswith(("review_", "boxes_", "page_", "ocrimg_", "purge_")):
                del st.session_state[key]
        st.success(f"Deleted {counts['invoices']} record(s) and {counts['files_removed']} stored file/folder(s).")
