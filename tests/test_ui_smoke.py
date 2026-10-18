"""Headless Streamlit smoke tests: every page renders without an exception, empty and with data."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest  # noqa: E402

from pipeline.runner import Services, apply_corrections, import_files, process_invoice  # noqa: E402
from pipeline.schemas import InvoiceStatus  # noqa: E402
from tests.conftest import FakeChat, FakeOcr, llm_json_for, ocr_text_for, png_bytes, specs  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
PAGES = ["Dashboard", "Import documents", "Invoice review", "Export"]


@pytest.fixture
def app(tmp_path, monkeypatch):
    monkeypatch.setenv("GST_DATA_DIR", str(tmp_path / "ui_data"))
    monkeypatch.setenv("GST_OCR_ENGINE", "tesseract")
    monkeypatch.setenv("GST_OCR_FALLBACK_ENGINE", "none")
    monkeypatch.setenv("GST_OLLAMA_HOST", "http://127.0.0.1:9")          # nothing listens here: Ollama "not running"
    at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=60)
    at.run()
    assert not at.exception, [e.value for e in at.exception]
    return at


def goto(at: AppTest, page: str) -> AppTest:
    at.sidebar.radio[0].set_value(page).run()
    assert not at.exception, (page, [e.value for e in at.exception])
    return at


def populate(at: AppTest, n: int = 3) -> list[str]:
    """Process a few invoices through the real pipeline (test doubles for OCR/LLM) inside the app's own storage."""
    services: Services = at.session_state["services"]
    sp = specs(n, seed=77)
    ids = []
    for i, s in enumerate(sp):
        import_files([(f"inv_{i}.png", png_bytes(s))], services.storage, services.settings)
        iid = services.storage.list_invoices()[-1].invoice_id
        bad = llm_json_for(s, total_amount=float(s.total_amount) + 100) if i == 1 else llm_json_for(s)
        process_invoice(iid, Services(services.settings, services.storage, ocr_engines=[FakeOcr(ocr_text_for(s))], chat_client=FakeChat(bad)))
        ids.append(iid)
    return ids


def test_app_starts_on_the_dashboard_and_binds_local_settings(app):
    assert app.sidebar.radio[0].value == "Dashboard"
    text = " ".join(m.value for m in app.sidebar.caption)
    assert "Local model not ready" in text and "No cloud AI" in text


@pytest.mark.parametrize("page", PAGES)
def test_every_page_renders_when_empty(app, page):
    goto(app, page)


@pytest.mark.parametrize("page", PAGES)
def test_every_page_renders_with_data(app, page):
    populate(app)
    goto(app, page)


def test_dashboard_labels_totals_as_provisional(app):
    populate(app)
    goto(app, "Dashboard")
    text = " ".join(str(x.value) for x in list(app.subheader) + list(app.warning))
    assert "Provisional totals" in text and "Do not use them for filing until every row is reviewed" in text
    labels = {m.label for m in app.metric}
    assert {"Files imported", "Successful extractions", "Need review", "Failed", "Duplicate candidates"} <= labels
    values = {m.label: m.value for m in app.metric}
    assert values["Files imported"] == "3" and values["Need review"] == "1" and values["Failed"] == "0"


def select_validation(app, option: str):
    app.selectbox(key=None) if False else None
    box = next(b for b in app.selectbox if b.label == "Validation")
    box.set_value(option).run()
    assert not app.exception
    return app


def test_review_screen_shows_the_exact_reason_for_a_flagged_invoice(app):
    populate(app)
    goto(app, "Invoice review")
    select_validation(app, "Has errors")
    body = " ".join(str(x.value) for x in app.markdown)
    assert "TOTAL_MISMATCH" in body and "difference" in body            # the exact explanation, not just a red mark
    assert any("Not verified yet" in str(x.value) for x in app.info)    # extraction success is never presented as verified


def test_review_filters_narrow_the_list(app):
    populate(app)
    goto(app, "Invoice review")
    assert "Showing 3 of 3" in " ".join(c.value for c in app.caption)
    select_validation(app, "No flags")
    assert "Showing 2 of 3" in " ".join(c.value for c in app.caption)


def test_correcting_a_field_in_the_form_saves_and_marks_reviewed(app):
    ids = populate(app)
    storage = app.session_state["services"].storage
    goto(app, "Invoice review")
    select_validation(app, "Has errors")
    good_total = storage.get_invoice(ids[1]).extracted["total_amount"]
    wrong = storage.get_invoice(ids[1]).effective.total_amount
    assert wrong != 0
    corrected = format(wrong - 100, "f")                                # the model added 100 to the true total
    next(t for t in app.text_input if t.label == "Total amount").set_value(corrected)
    next(b for b in app.button if "mark as reviewed" in b.label).click().run()
    assert not app.exception, [e.value for e in app.exception]
    row = storage.get_invoice(ids[1])
    assert row.status == InvoiceStatus.REVIEWED and row.corrected["total_amount"] == corrected and row.extracted["total_amount"] == good_total
    assert not [i for i in row.issues if i.code == "TOTAL_MISMATCH"]


def test_an_invalid_amount_is_rejected_by_the_form_and_nothing_is_saved(app):
    ids = populate(app)
    storage = app.session_state["services"].storage
    goto(app, "Invoice review")
    next(t for t in app.text_input if t.label == "Total amount").set_value("twelve")
    next(b for b in app.button if b.label == "Save corrections").click().run()
    assert any("not a valid amount" in str(e.value) for e in app.error)
    assert storage.get_invoice(ids[0]).corrected == {}


def test_export_page_offers_downloads_and_requires_typing_delete(app):
    populate(app)
    goto(app, "Export")
    buttons = [b.label for b in app.button]
    assert any("Delete all local data" in b for b in buttons)
    delete = next(b for b in app.button if "Delete all local data" in b.label)
    assert delete.disabled                                              # disabled until DELETE is typed
    app.text_input(key="purge_confirm").set_value("DELETE").run()
    delete = next(b for b in app.button if "Delete all local data" in b.label)
    assert not delete.disabled
    delete.click().run()
    assert not app.exception
    assert app.session_state["services"].storage.list_invoices() == []


def test_import_page_warns_when_ollama_is_not_running(app):
    goto(app, "Import documents")
    body = " ".join(str(x.value) for x in list(app.warning) + list(app.error) + list(app.info) + list(app.markdown))
    assert "Ollama" in body
