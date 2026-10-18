"""Import documents: upload / folder, queue, start-pause-cancel, progress."""

from __future__ import annotations

import time

import pandas as pd
import streamlit as st

from pipeline import ingest
from pipeline.runner import import_files, import_folder
from pipeline.schemas import InvoiceStatus
from ui.common import ocr_problem, ollama_status, services_and_jobs, status_label


def render() -> None:
    services, jobs = services_and_jobs()
    storage, settings = services.storage, services.settings
    st.header("Import documents")
    st.caption("Everything is processed on this computer. Files are copied into the app's own data folder; your originals are not changed.")

    tab_upload, tab_folder = st.tabs(["Upload files", "Process a folder"])
    with tab_upload:
        key = f"uploader_{st.session_state.get('uploader_n', 0)}"
        uploaded = st.file_uploader(
            "PDF, JPG, JPEG or PNG (several files allowed)", type=["pdf", "jpg", "jpeg", "png"],
            accept_multiple_files=True, key=key)
        if st.button("Add to queue", disabled=not uploaded, key="add_uploads"):
            report = import_files([(f.name, f.getvalue()) for f in uploaded], storage, settings)
            _show_report(report)
            st.session_state["uploader_n"] = st.session_state.get("uploader_n", 0) + 1
            st.rerun()
    with tab_folder:
        folder = st.text_input("Folder path on this computer", placeholder=r"C:\Users\me\Documents\invoices   or   /home/me/invoices", key="folder_path")
        recursive = st.checkbox("Include sub-folders", value=False)
        st.caption("Only this folder is read. Shortcuts/symlinks are skipped, and nothing outside it is touched.")
        if st.button("Scan folder and add files", disabled=not folder.strip(), key="add_folder"):
            try:
                report = import_folder(folder.strip(), storage, settings, recursive)
                _show_report(report)
                if report.added or report.duplicates or report.rejected:
                    st.rerun()
                else:
                    st.info("No PDF, JPG or PNG files were found in that folder.")
            except ingest.IngestError as exc:
                st.error(str(exc))

    st.divider()
    _controls(services, jobs)
    _queue_table(services)


def _show_report(report) -> None:  # noqa: ANN001
    if report.added:
        st.session_state["flash"] = ("success", f"Added: {report.summary()}.")
    elif report.duplicates and not report.rejected:
        st.session_state["flash"] = ("info", f"Nothing new: {report.summary()}.")
    for name, reason in report.rejected:
        st.session_state.setdefault("flash_errors", []).append(reason)
    if report.rejected and not report.added:
        st.session_state["flash"] = ("warning", f"{report.summary()}.")


def _controls(services, jobs) -> None:  # noqa: ANN001
    storage, settings = services.storage, services.settings
    kind, text = st.session_state.pop("flash", (None, None))
    if kind:
        getattr(st, kind)(text)
    for msg in st.session_state.pop("flash_errors", []):
        st.error(msg)

    snap = jobs.snapshot()
    counts = storage.stats()
    pending = counts[InvoiceStatus.PENDING.value]
    failed = counts[InvoiceStatus.FAILED.value]

    st.subheader("Processing")
    ollama = ollama_status(settings)
    ocr_issue = ocr_problem(settings)
    if not ollama.ready:
        st.error(ollama.message)
        if st.button("Check again", key="recheck_ollama"):
            ollama_status(settings, force=True)
            st.rerun()
    if ocr_issue:
        st.error(ocr_issue)

    if snap.active:
        _progress_fragment()
        c1, c2, c3 = st.columns(3)
        if snap.state == "paused":
            if c1.button("▶ Resume", key="resume"):
                jobs.resume()
                st.rerun()
        elif snap.state == "running":
            if c1.button("⏸ Pause after this file", key="pause"):
                jobs.pause()
                st.rerun()
        if c2.button("⏹ Cancel (rest stays queued)", key="cancel", disabled=snap.state == "cancelling"):
            jobs.cancel()
            st.rerun()
        return

    if snap.state in {"finished", "cancelled", "stopped"} and snap.total:
        label = {"finished": "Finished", "cancelled": "Cancelled", "stopped": "Stopped"}[snap.state]
        text = f"{label}: {snap.done} of {snap.total} processed, {snap.failed} failed."
        (st.warning if snap.state != "finished" else st.success)(text + (f" {snap.message}" if snap.message else ""))

    ready = ollama.ready and not ocr_issue
    n = st.number_input("Files per batch", min_value=1, max_value=500, value=int(settings.processing.batch_size), step=1,
                        help="'Process next batch' handles this many queued files, then stops. Smaller batches let you review sooner.")
    c1, c2, c3 = st.columns(3)
    if c1.button(f"▶ Process next batch ({min(pending, int(n))})", disabled=not (ready and pending), key="start_batch"):
        jobs.start(limit=int(n))
        st.rerun()
    if c2.button(f"▶▶ Process all queued ({pending})", disabled=not (ready and pending), key="start_all"):
        jobs.start()
        st.rerun()
    if c3.button(f"↻ Retry failed ({failed})", disabled=not (ready and failed), key="retry_failed"):
        storage.retry_failed()
        jobs.start()
        st.rerun()
    if not pending and not failed:
        st.caption("The queue is empty. Add files above.")


def _progress_fragment() -> None:
    _, jobs = services_and_jobs()

    @st.fragment(run_every=1)
    def live() -> None:
        snap = jobs.snapshot()
        if not snap.active:
            st.rerun()  # full refresh so buttons and the table update
            return
        st.progress(min(1.0, snap.fraction), text=f"{snap.done} of {snap.total} done" + (f" · {snap.failed} failed" if snap.failed else ""))
        state = {"running": "Working on", "paused": "Paused before", "cancelling": "Finishing"}[snap.state]
        st.caption(f"{state}: **{snap.current_name or '…'}** · stage: {snap.stage or '…'} {snap.message}")
        if snap.events:
            with st.expander("Activity", expanded=False):
                st.code("\n".join(reversed(snap.events[-15:])), language="text")

    live()


def _queue_table(services) -> None:  # noqa: ANN001
    rows = services.storage.list_invoices()
    st.subheader(f"Queue ({len(rows)})")
    if not rows:
        return
    frame = pd.DataFrame([{
        "File": r.filename,
        "Pages": f"{r.page_start}-{r.page_end}" if r.page_end > r.page_start else (str(r.page_start) if r.page_count else ""),
        "Status": status_label(r.status),
        "Message": (r.error or "")[:200],
        "Time (s)": round(r.total_ms / 1000, 1) if r.total_ms else None,
    } for r in rows])
    st.dataframe(frame, hide_index=True)
    st.caption("Same file content is never imported twice. A page that cannot be read does not stop the rest of the batch.")
