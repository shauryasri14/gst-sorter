"""End-to-end pipeline behaviour with test doubles for OCR and the LLM (no network, no Ollama)."""

from __future__ import annotations

import logging
from decimal import Decimal

import pytest

from pipeline import ingest
from pipeline.extract import OllamaModelMissingError, OllamaUnavailableError
from pipeline.ocr import OcrUnavailableError
from pipeline.runner import (
    JobManager,
    ServiceUnavailable,
    Services,
    apply_corrections,
    compute_duplicates,
    import_files,
    import_folder,
    mark_reviewed,
    process_invoice,
    retry_page_ocr,
    setup_logging,
)
from pipeline.schemas import InvoiceStatus
from pipeline.storage import Storage
from tests.conftest import FakeChat, FakeOcr, llm_json_for, ocr_text_for, pdf_bytes, png_bytes, specs

D = Decimal


def add_png(storage, settings, spec, name="inv.png"):
    report = import_files([(name, png_bytes(spec))], storage, settings)
    assert report.added == [name], report
    return storage.list_invoices()[-1].invoice_id


@pytest.fixture
def happy(one_spec, make_services, storage, settings):
    services = make_services(FakeOcr(ocr_text_for(one_spec)), FakeChat(llm_json_for(one_spec)))
    return services, add_png(storage, settings, one_spec), one_spec


# ---- ingestion ---------------------------------------------------------------------------------
def test_duplicate_content_is_imported_once_even_with_a_different_name(storage, settings, one_spec):
    data = png_bytes(one_spec)
    first = import_files([("a.png", data)], storage, settings)
    second = import_files([("copy of a.png", data)], storage, settings)
    assert first.added == ["a.png"] and second.added == [] and second.duplicates == ["copy of a.png"]
    assert len(storage.list_invoices()) == 1


def test_duplicate_inside_one_batch_is_skipped(storage, settings, one_spec):
    data = png_bytes(one_spec)
    report = import_files([("a.png", data), ("b.png", data)], storage, settings)
    assert report.added == ["a.png"] and report.duplicates == ["b.png"]


@pytest.mark.parametrize("name, data", [("malware.exe", b"MZ\x90\x00"), ("notes.txt", b"hello"), ("noext", b"hello"),
                                         ("script.png", b"#!/bin/sh\nrm -rf /\n"), ("fake.pdf", b"not a pdf at all"),
                                         ("empty.png", b""), ("wrong.jpg", None)])
def test_unsupported_or_mismatched_files_are_rejected_safely(storage, settings, one_spec, name, data):
    data = png_bytes(one_spec) if data is None else data         # a real PNG named .jpg: content must match extension
    report = import_files([(name, data)], storage, settings)
    assert report.added == [] and len(report.rejected) == 1
    assert list(settings.uploads_dir.iterdir()) == []              # nothing stored for a rejected file
    failed = [r for r in storage.list_invoices() if r.status == InvoiceStatus.FAILED]
    assert len(failed) == 1 and failed[0].error                     # but the rejection stays visible to the user


def test_hostile_file_names_never_influence_paths(storage, settings, one_spec):
    for name in ("../../../etc/passwd.png", "..\\..\\evil.png", "a/b/c.png", "x\x00y.png"):
        import_files([(name, png_bytes(specs(1, seed=hash(name) % 1000)[0]))], storage, settings)
    stored = {p.name for p in settings.uploads_dir.iterdir()}
    assert all(len(n) == 36 and n.endswith(".png") for n in stored)       # 32 hex chars of the hash + extension
    assert not (settings.data_path.parent / "evil.png").exists()
    assert {r.filename for r in storage.list_invoices()} >= {"passwd.png", "evil.png", "c.png"}


def test_sanitize_filename():
    assert ingest.sanitize_filename("../a/b\\c.pdf") == "c.pdf" and ingest.sanitize_filename("") == "file"
    assert len(ingest.sanitize_filename("x" * 500 + ".png")) <= 150


def test_file_size_limit(storage, settings, one_spec):
    small = settings.model_copy(update={"limits": settings.limits.model_copy(update={"max_file_mb": 0.0001})})
    report = import_files([("big.png", png_bytes(one_spec))], Storage(small), small)
    assert report.rejected and "limit" in report.rejected[0][1]


def test_image_dimension_limit(storage, settings):
    tight = settings.model_copy(update={"limits": settings.limits.model_copy(update={"max_image_side": 100})})
    report = import_files([("wide.png", png_bytes(size=(300, 200)))], Storage(tight), tight)
    assert report.rejected and "above the configured limit" in report.rejected[0][1]


def test_corrupted_png_is_rejected(storage, settings):
    report = import_files([("bad.png", b"\x89PNG\r\n\x1a\n" + b"garbage" * 20)], storage, settings)
    assert report.rejected and "could not be decoded" in report.rejected[0][1]


def test_corrupted_pdf_is_rejected_with_friendly_message(storage, settings):
    report = import_files([("bad.pdf", b"%PDF-1.4\n" + b"\x00garbage" * 50)], storage, settings)
    assert report.rejected and "Could not read this PDF" in report.rejected[0][1]


def test_page_limit(storage, settings):
    limited = settings.model_copy(update={"limits": settings.limits.model_copy(update={"max_pages": 2})})
    report = import_files([("three.pdf", pdf_bytes(specs(3)))], Storage(limited), limited)
    assert report.rejected and "3 pages" in report.rejected[0][1]


def test_folder_import_only_takes_supported_files_and_skips_symlinks(tmp_path, storage, settings, one_spec):
    folder = tmp_path / "scans"
    folder.mkdir()
    (folder / "a.png").write_bytes(png_bytes(one_spec))
    (folder / "readme.txt").write_text("hi")
    (folder / "link.png").symlink_to(folder / "a.png")
    (folder / "sub").mkdir()
    (folder / "sub" / "deep.png").write_bytes(png_bytes(specs(1, seed=8)[0]))
    report = import_folder(folder, storage, settings)
    assert report.added == ["a.png"] and not report.rejected
    assert import_folder(folder, Storage(settings), settings, recursive=True).added == ["deep.png"]    # a.png already known


def test_missing_folder_is_a_clean_error(storage, settings):
    with pytest.raises(ingest.IngestError, match="not a folder"):
        import_folder("/definitely/not/here", storage, settings)


# ---- multi-page PDFs ----------------------------------------------------------------------------
def test_multipage_pdf_is_one_invoice_spanning_all_pages_by_default(storage, settings):
    sp = specs(3)
    report = import_files([("three.pdf", pdf_bytes(sp))], storage, settings)
    assert report.added == ["three.pdf"]
    rows = storage.list_invoices()
    assert len(rows) == 1 and rows[0].page_count == 3 and (rows[0].page_start, rows[0].page_end) == (1, 3)
    assert len(storage.get_pages(rows[0].doc_id)) == 3


def test_multipage_pdf_can_be_one_invoice_per_page(storage, settings):
    per_page = settings.model_copy(update={"processing": settings.processing.model_copy(update={"pdf_mode": "per_page"})})
    st = Storage(per_page)
    import_files([("three.pdf", pdf_bytes(specs(3)))], st, per_page)
    rows = st.list_invoices()
    assert [(r.page_start, r.page_end) for r in rows] == [(1, 1), (2, 2), (3, 3)]
    assert len({r.invoice_id for r in rows}) == 3 and all(r.invoice_id.endswith(f"-p{r.page_start}") for r in rows)


def test_every_page_of_a_pdf_is_ocrd_and_combined_for_the_model(storage, settings, make_services):
    pages = iter(["PAGE ONE TEXT", "PAGE TWO TEXT", "PAGE THREE TEXT"])
    ocr = FakeOcr(lambda img: next(pages))
    chat = FakeChat(llm_json_for(specs(1)[0]))
    services = make_services(ocr, chat)
    import_files([("three.pdf", pdf_bytes(specs(3)))], storage, settings)
    iid = storage.list_invoices()[0].invoice_id
    process_invoice(iid, services)
    assert ocr.calls == 3
    prompt = chat.calls[0][0]["content"]
    assert "--- page 1 ---" in prompt and "PAGE TWO TEXT" in prompt and "--- page 3 ---" in prompt
    assert [p.ocr.text for p in storage.get_pages(storage.list_invoices()[0].doc_id)] == ["PAGE ONE TEXT", "PAGE TWO TEXT", "PAGE THREE TEXT"]


def test_page_images_are_stored_for_the_review_screen(storage, settings, make_services):
    import_files([("three.pdf", pdf_bytes(specs(2)))], storage, settings)
    row = storage.list_invoices()[0]
    process_invoice(row.invoice_id, make_services(FakeOcr("text"), FakeChat(llm_json_for(specs(1)[0]))))
    for page in storage.get_pages(row.doc_id):
        assert page.image_path and ingest.Path(page.image_path).exists()


# ---- the happy path -------------------------------------------------------------------------------
def test_good_extraction_is_stored_but_not_marked_reviewed(happy, storage):
    services, iid, spec = happy
    status = process_invoice(iid, services)
    row = storage.get_invoice(iid)
    assert status == InvoiceStatus.EXTRACTED and row.status != InvoiceStatus.REVIEWED
    assert row.reviewed_at is None and row.issues == []
    inv = row.effective
    assert inv.total_amount == spec.total_amount and inv.vendor_gstin == spec.vendor_gstin
    assert inv.invoice_date == spec.invoice_date.isoformat()            # printed dd/mm/yyyy normalised to ISO
    assert row.model == "qwen2.5:7b-instruct" and row.total_ms is not None


def test_wrong_extraction_is_flagged_for_review_with_reasons(one_spec, make_services, storage, settings):
    wrong = llm_json_for(one_spec, total_amount=float(one_spec.total_amount) + 500)
    iid = add_png(storage, settings, one_spec)
    status = process_invoice(iid, make_services(FakeOcr(ocr_text_for(one_spec)), FakeChat(wrong)))
    row = storage.get_invoice(iid)
    assert status == InvoiceStatus.NEEDS_REVIEW
    assert {"TOTAL_MISMATCH", "NOT_FOUND_IN_OCR"} <= {i.code for i in row.issues}
    assert row.field_status["total_amount"]["status"] == "flagged"


def test_a_hallucinated_gstin_is_caught_by_the_ocr_grounding_check(one_spec, make_services, storage, settings):
    from eval.synthetic import make_gstin
    import random

    fake = make_gstin(random.Random(1), "27")
    iid = add_png(storage, settings, one_spec)
    process_invoice(iid, make_services(FakeOcr(ocr_text_for(one_spec)), FakeChat(llm_json_for(one_spec, vendor_gstin=fake))))
    assert any(i.code == "NOT_FOUND_IN_OCR" and "vendor_gstin" in i.fields for i in storage.get_invoice(iid).issues)


def test_unprocessed_invoices_stay_pending_until_run(happy, storage):
    _, iid, _ = happy
    assert storage.get_invoice(iid).status == InvoiceStatus.PENDING and storage.pending_ids() == [iid]


# ---- failure handling -------------------------------------------------------------------------------
def test_failed_ocr_marks_the_file_failed_without_inventing_text(one_spec, make_services, storage, settings):
    iid = add_png(storage, settings, one_spec)
    chat = FakeChat(llm_json_for(one_spec))
    status = process_invoice(iid, make_services(FakeOcr(error=OcrUnavailableError("PaddleOCR is not installed. Run: pip install paddleocr")), chat))
    row = storage.get_invoice(iid)
    assert status == InvoiceStatus.FAILED and chat.calls == []          # the model is never asked to guess from nothing
    assert "PaddleOCR is not installed" in row.error and "Retry failed" in row.error
    assert row.extracted is None and storage.get_pages(row.doc_id)[0].ocr.text == ""


def test_ocr_crash_is_contained(one_spec, make_services, storage, settings):
    iid = add_png(storage, settings, one_spec)
    status = process_invoice(iid, make_services(FakeOcr(error=RuntimeError("boom")), FakeChat({})))
    assert status == InvoiceStatus.FAILED and "failed on this page" in storage.get_invoice(iid).error


def test_blank_page_without_text_fails_instead_of_guessing(one_spec, make_services, storage, settings):
    iid = add_png(storage, settings, one_spec)
    assert process_invoice(iid, make_services(FakeOcr(""), FakeChat({}))) == InvoiceStatus.FAILED
    assert "no text was detected" in storage.get_invoice(iid).error


def test_fallback_engine_is_used_when_the_primary_is_unavailable(one_spec, storage, settings):
    primary = FakeOcr(error=OcrUnavailableError("primary missing"))
    secondary = FakeOcr(ocr_text_for(one_spec))
    secondary.name = "fallback"
    services = Services(settings, storage, ocr_engines=[primary, secondary], chat_client=FakeChat(llm_json_for(one_spec)))
    iid = add_png(storage, settings, one_spec)
    assert process_invoice(iid, services) == InvoiceStatus.EXTRACTED
    assert storage.get_pages(storage.get_invoice(iid).doc_id)[0].ocr.engine == "fallback"


@pytest.mark.parametrize("error", [OllamaUnavailableError("Ollama is not running at http://127.0.0.1:11434."),
                                    OllamaModelMissingError("The model 'x' is not installed. ollama pull x")])
def test_missing_ollama_stops_the_run_and_keeps_the_file_queued(one_spec, make_services, storage, settings, error):
    iid = add_png(storage, settings, one_spec)
    services = make_services(FakeOcr(ocr_text_for(one_spec)), FakeChat(error))
    with pytest.raises(ServiceUnavailable) as exc:
        process_invoice(iid, services)
    assert str(error) in str(exc.value)
    assert storage.get_invoice(iid).status == InvoiceStatus.PENDING        # nothing is lost; run again after starting Ollama


def test_malformed_model_output_ends_as_failed_with_a_message_after_one_retry(one_spec, make_services, storage, settings):
    iid = add_png(storage, settings, one_spec)
    chat = FakeChat("garbage", "more garbage")
    status = process_invoice(iid, make_services(FakeOcr(ocr_text_for(one_spec)), chat))
    assert status == InvoiceStatus.FAILED and len(chat.calls) == 2
    assert "even after one retry" in storage.get_invoice(iid).error


def test_malformed_then_valid_output_recovers(one_spec, make_services, storage, settings):
    iid = add_png(storage, settings, one_spec)
    chat = FakeChat("```oops```", llm_json_for(one_spec))
    assert process_invoice(iid, make_services(FakeOcr(ocr_text_for(one_spec)), chat)) == InvoiceStatus.EXTRACTED


def test_missing_stored_file_fails_cleanly(happy, storage):
    services, iid, _ = happy
    ingest.Path(storage.get_invoice(iid).stored_path).unlink()
    assert process_invoice(iid, services) == InvoiceStatus.FAILED and "stored copy" in storage.get_invoice(iid).error


def test_retry_failed_requeues_only_failed_items(one_spec, make_services, storage, settings):
    iid = add_png(storage, settings, one_spec)
    process_invoice(iid, make_services(FakeOcr(""), FakeChat({})))
    import_files([("bad.exe", b"MZ")], storage, settings)                   # rejected file: can never be retried
    assert storage.retry_failed() == 1
    assert storage.get_invoice(iid).status == InvoiceStatus.PENDING


# ---- corrections persist ----------------------------------------------------------------------------
def test_manual_corrections_are_saved_revalidated_and_survive_a_restart(happy, storage, settings):
    services, iid, spec = happy
    process_invoice(iid, services)
    status, issues = apply_corrections(iid, {"vendor_name": "Corrected Name", "total_amount": "12345.67"}, services)
    assert status == InvoiceStatus.NEEDS_REVIEW and any(i.code == "TOTAL_MISMATCH" for i in issues)    # re-validated with the edit

    reopened = Storage(settings).get_invoice(iid)                          # a brand new Storage = app restart
    assert reopened.effective.vendor_name == "Corrected Name" and reopened.effective.total_amount == D("12345.67")
    assert reopened.extracted["vendor_name"] == spec.vendor_name           # the machine's answer is kept separately
    assert reopened.edited_fields == ["total_amount", "vendor_name"]
    assert reopened.field_status["vendor_name"]["edited"]


def test_correcting_back_to_the_right_value_clears_the_error(happy, storage):
    services, iid, spec = happy
    process_invoice(iid, services)
    apply_corrections(iid, {"total_amount": "1"}, services)
    status, issues = apply_corrections(iid, {"total_amount": str(spec.total_amount)}, services)
    assert status == InvoiceStatus.EXTRACTED and not [i for i in issues if i.code == "TOTAL_MISMATCH"]


def test_invalid_corrections_are_refused_and_nothing_is_saved(happy, storage):
    services, iid, _ = happy
    process_invoice(iid, services)
    with pytest.raises(Exception):
        apply_corrections(iid, {"total_amount": "twelve"}, services)
    with pytest.raises(ValueError):
        storage.save_corrections(iid, {"not_a_field": 1}, [], {}, InvoiceStatus.NEEDS_REVIEW)
    assert storage.get_invoice(iid).corrected == {}


def test_marking_reviewed_is_explicit_and_reversible(happy, storage):
    services, iid, _ = happy
    process_invoice(iid, services)
    assert mark_reviewed(iid, services) == InvoiceStatus.REVIEWED and storage.get_invoice(iid).reviewed_at
    assert mark_reviewed(iid, services, reviewed=False) == InvoiceStatus.EXTRACTED


def test_cannot_mark_an_unextracted_invoice_reviewed(one_spec, make_services, storage, settings):
    iid = add_png(storage, settings, one_spec)
    with pytest.raises(ValueError):
        mark_reviewed(iid, make_services())


def test_rerunning_extraction_keeps_human_corrections_unless_told_otherwise(happy, storage):
    services, iid, spec = happy
    process_invoice(iid, services)
    apply_corrections(iid, {"vendor_name": "Corrected Name"}, services, mark_reviewed=True)
    process_invoice(iid, services, reuse_ocr=True)                                    # default keeps corrections
    row = storage.get_invoice(iid)
    assert row.effective.vendor_name == "Corrected Name" and row.extracted["vendor_name"] == spec.vendor_name
    assert row.status != InvoiceStatus.REVIEWED and row.reviewed_at is None           # a new extraction must be re-reviewed
    process_invoice(iid, services, reuse_ocr=True, keep_corrections=False)             # explicit overwrite
    assert storage.get_invoice(iid).corrected == {} and storage.get_invoice(iid).effective.vendor_name == spec.vendor_name


def test_reuse_ocr_does_not_run_ocr_again(happy):
    services, iid, _ = happy
    process_invoice(iid, services)
    calls = services.ocr_engines[0].calls
    process_invoice(iid, services, reuse_ocr=True)
    assert services.ocr_engines[0].calls == calls


def test_retry_page_ocr_replaces_only_that_pages_text(storage, settings, make_services):
    import_files([("two.pdf", pdf_bytes(specs(2)))], storage, settings)
    iid = storage.list_invoices()[0].invoice_id
    ocr = FakeOcr("first text")
    services = make_services(ocr, FakeChat(llm_json_for(specs(1)[0])))
    process_invoice(iid, services)
    ocr.text = "second try"
    page = retry_page_ocr(iid, 2, services)
    texts = [p.ocr.text for p in storage.get_pages(storage.list_invoices()[0].doc_id)]
    assert page.text == "second try" and texts == ["first text", "second try"]


# ---- duplicates across the stored batch -------------------------------------------------------------------
def test_duplicate_invoices_in_a_batch_are_flagged_not_removed(storage, settings, make_services):
    s = specs(2, seed=21)[0]
    other = specs(1, seed=22)[0]
    ids = [add_png(storage, settings, sp, name=n) for sp, n in ((s, "a.png"), (other, "b.png"))]
    for iid, sp in zip(ids, (s, s)):                                    # b.png's model output is the same invoice as a.png
        process_invoice(iid, make_services(FakeOcr(ocr_text_for(sp)), FakeChat(llm_json_for(sp))))
    dups = compute_duplicates(storage.list_invoices())
    assert {c.other_id for c in dups[ids[0]]} == {ids[1]} and dups[ids[0]][0].strength == "high"
    assert len(storage.list_invoices()) == 2


# ---- housekeeping, logging, jobs ------------------------------------------------------------------------------
def test_purge_removes_records_and_generated_files(happy, storage, settings):
    services, iid, _ = happy
    process_invoice(iid, services)
    assert list(settings.uploads_dir.iterdir()) and list(settings.pages_dir.iterdir())
    counts = storage.purge_all()
    assert counts["invoices"] == 1 and counts["documents"] == 1
    assert storage.list_invoices() == [] and list(settings.uploads_dir.iterdir()) == [] and list(settings.pages_dir.iterdir()) == []
    assert settings.db_path.exists()                                    # empty database remains; the app keeps working


def test_stale_processing_items_return_to_the_queue(happy, storage):
    _, iid, _ = happy
    storage.set_status(iid, InvoiceStatus.PROCESSING)
    assert storage.reset_stale_processing() == 1 and storage.get_invoice(iid).status == InvoiceStatus.PENDING


def test_logs_contain_no_invoice_text_gstins_or_amounts(happy, settings, storage):
    services, iid, spec = happy
    logging.getLogger("gst_sorter").handlers.clear()
    setup_logging(settings)
    process_invoice(iid, services)
    apply_corrections(iid, {"vendor_name": "Secret Vendor Ltd"}, services)
    for handler in logging.getLogger("gst_sorter").handlers:
        handler.flush()
    log = settings.log_file.read_text(encoding="utf-8")
    assert log.strip(), "expected some log output"
    for secret in (spec.vendor_gstin, spec.buyer_gstin, spec.vendor_name, spec.invoice_number, f"{spec.total_amount:,.2f}",
                   str(spec.total_amount), "Secret Vendor Ltd"):
        assert secret not in log, secret


def test_batch_job_processes_every_pending_file_in_the_background(storage, settings, make_services):
    sp = specs(3, seed=31)
    for i, s in enumerate(sp):
        add_png(storage, settings, s, name=f"{i}.png")
    texts = {i: ocr_text_for(s) for i, s in enumerate(sp)}
    order = iter(range(3))
    services = make_services(FakeOcr(lambda img: texts[next(order)]), FakeChat(*[llm_json_for(s) for s in sp]))
    job = JobManager(services)
    assert job.start() == 3
    job.wait(30)
    snap = job.snapshot()
    assert snap.state == "finished" and snap.done == 3 and snap.failed == 0 and snap.fraction == 1.0
    assert storage.stats()["Pending"] == 0


def test_batch_job_stops_when_ollama_is_down_and_leaves_files_queued(storage, settings, make_services):
    sp = specs(3, seed=41)
    for i, s in enumerate(sp):
        add_png(storage, settings, s, name=f"{i}.png")
    services = make_services(FakeOcr(ocr_text_for(sp[0])), FakeChat(OllamaUnavailableError("Ollama is not running")))
    job = JobManager(services)
    job.start()
    job.wait(30)
    snap = job.snapshot()
    assert snap.state == "stopped" and "Ollama is not running" in snap.message and snap.done == 0
    assert storage.stats()["Pending"] == 3


def test_batch_job_can_be_cancelled_and_keeps_the_rest_pending(storage, settings, make_services):
    sp = specs(4, seed=51)
    for i, s in enumerate(sp):
        add_png(storage, settings, s, name=f"{i}.png")
    job_ref = {}

    class CancellingChat(FakeChat):
        def chat(self, messages, schema=None):
            job_ref["job"].cancel()                                    # cancel while the first file is in progress
            return super().chat(messages, schema)

    services = make_services(FakeOcr(ocr_text_for(sp[0])), CancellingChat(llm_json_for(sp[0])))
    job = job_ref["job"] = JobManager(services)
    job.start()
    job.wait(30)
    assert job.snapshot().state == "cancelled" and job.snapshot().done == 1
    assert storage.stats()["Pending"] == 3                              # untouched files stay queued


def test_only_one_batch_runs_at_a_time(storage, settings, make_services):
    s = specs(1, seed=61)[0]
    add_png(storage, settings, s)
    release = {}

    class SlowChat(FakeChat):
        def chat(self, messages, schema=None):
            import time
            time.sleep(0.4)
            return super().chat(messages, schema)

    job = JobManager(make_services(FakeOcr(ocr_text_for(s)), SlowChat(llm_json_for(s))))
    job.start()
    with pytest.raises(RuntimeError, match="already running"):
        job.start()
    job.wait(30)
