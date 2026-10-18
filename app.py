"""GST Invoice & Receipt Sorter: Streamlit entry point.

Run with:  streamlit run app.py
"""

from __future__ import annotations

import streamlit as st

from config import ConfigError, load_settings
from pipeline.runner import JobManager, Services, setup_logging
from pipeline.storage import Storage
from ui import dashboard, export_view, import_view, review_view
from ui.common import CSS, ollama_status, services_and_jobs

st.set_page_config(page_title="GST Invoice & Receipt Sorter", page_icon="🧾", layout="wide")


def bootstrap() -> None:
    if "services" in st.session_state:
        return
    try:
        settings = load_settings()
    except ConfigError as exc:
        st.error(f"Configuration problem: {exc}")
        st.stop()
    setup_logging(settings)
    storage = Storage(settings)
    storage.reset_stale_processing()
    services = Services(settings, storage)
    st.session_state["services"] = services
    st.session_state["jobs"] = JobManager(services)


def sidebar() -> str:
    services, jobs = services_and_jobs()
    settings = services.settings
    with st.sidebar:
        st.title("🧾 GST Sorter")
        page = st.radio("Go to", ["Dashboard", "Import documents", "Invoice review", "Export"], label_visibility="collapsed")
        st.divider()
        status = ollama_status(settings)
        st.caption(("🟢 " if status.ready else "🔴 ") + ("Local model ready" if status.ready else "Local model not ready"))
        st.caption(f"Model: `{settings.ollama.model}`")
        st.caption(f"OCR: `{settings.ocr.engine}`")
        snap = jobs.snapshot()
        if snap.active:
            st.caption(f"⚙️ Processing {snap.done}/{snap.total}")
        st.divider()
        st.caption("🔒 Runs on this computer only. No cloud AI, no telemetry. Offline after setup.")
    return page


def main() -> None:
    st.markdown(CSS, unsafe_allow_html=True)
    bootstrap()
    page = sidebar()
    {"Dashboard": dashboard.render, "Import documents": import_view.render,
     "Invoice review": review_view.render, "Export": export_view.render}[page]()


main()
