"""OCR wrappers and image preprocessing. Includes a real Tesseract run when the binary is installed."""

from __future__ import annotations

import hashlib
import random
import re
import shutil

import pytest
from PIL import Image

from config import OcrSettings, PreprocessSettings
from eval.synthetic import degrade, make_spec, render_invoice
from pipeline.ocr import OcrUnavailableError, PaddleOcrEngine, TesseractEngine, build_engine, lines_to_text, run_ocr
from pipeline.preprocess import estimate_skew, pil_to_gray, preprocess_image
from pipeline.schemas import OcrLine
from tests.conftest import FakeOcr

needs_tesseract = pytest.mark.skipif(shutil.which("tesseract") is None, reason="Tesseract binary not installed")


def squash(text: str) -> str:
    return re.sub(r"[^A-Z0-9]", "", text.upper())


# ---- text assembly --------------------------------------------------------------------------------
def test_lines_are_rebuilt_in_reading_order():
    lines = [OcrLine(text="Total", bbox=[10, 300, 80, 330]), OcrLine(text="Invoice", bbox=[10, 10, 90, 40]),
             OcrLine(text="11,800.00", bbox=[400, 302, 520, 332]), OcrLine(text="INV-1", bbox=[300, 12, 380, 38])]
    assert lines_to_text(lines).splitlines() == ["Invoice   INV-1", "Total   11,800.00"]


def test_lines_without_boxes_are_kept():
    assert lines_to_text([OcrLine(text="a"), OcrLine(text="b")]) == "a\nb"


# ---- engine selection and failure handling ------------------------------------------------------------
def test_unknown_engine_name_is_an_ocr_error_not_a_crash():
    with pytest.raises(OcrUnavailableError):
        build_engine("magic", OcrSettings())


def test_ocr_settings_reject_unknown_engines_and_bad_psm():
    with pytest.raises(ValueError):
        OcrSettings(engine="cloudvision")
    with pytest.raises(ValueError):
        OcrSettings(tesseract_psm=99)
    assert OcrSettings(fallback_engine="none").fallback_engine is None


def test_all_engines_failing_returns_an_error_and_empty_text():
    page = run_ocr(Image.new("RGB", (50, 50)), OcrSettings(), [FakeOcr(error=OcrUnavailableError("one")), FakeOcr(error=RuntimeError("two"))])
    assert page.error and "one" in page.error and "failed on this page" in page.error
    assert page.text == "" and page.lines == [] and not page.ok and page.engine == "none"


def test_a_page_with_no_detected_text_is_an_error_not_empty_success():
    page = run_ocr(Image.new("RGB", (50, 50)), OcrSettings(), [FakeOcr("")])
    assert not page.ok and "no text was detected" in page.error


def test_confidence_and_boxes_are_returned():
    page = run_ocr(Image.new("RGB", (50, 50)), OcrSettings(), [FakeOcr("hello\nworld", confidence=0.9)])
    assert page.ok and page.mean_confidence == 0.9 and len(page.lines) == 2 and all(len(l.bbox) == 4 for l in page.lines)
    assert page.engine == "fake-ocr"


@pytest.mark.skipif(PaddleOcrEngine("en").__class__ is None, reason="n/a")
def test_missing_paddleocr_gives_install_instructions_instead_of_fake_text():
    try:
        import paddleocr  # noqa: F401
        pytest.skip("PaddleOCR is installed here")
    except ImportError:
        pass
    page = run_ocr(Image.new("RGB", (50, 50)), OcrSettings(), [PaddleOcrEngine("en")])
    assert not page.ok and "pip install paddleocr" in page.error and page.text == ""


def test_missing_tesseract_binary_gives_instructions(monkeypatch):
    pytesseract = pytest.importorskip("pytesseract")
    monkeypatch.setattr(pytesseract.pytesseract, "tesseract_cmd", "/no/such/tesseract")
    with pytest.raises(OcrUnavailableError, match="Tesseract program was not found"):
        TesseractEngine("en", "/no/such/tesseract").read(Image.new("RGB", (50, 50), "white"))


# ---- real OCR (Tesseract) --------------------------------------------------------------------------------
@needs_tesseract
def test_real_tesseract_reads_a_clean_synthetic_invoice_including_the_large_heading():
    """Regression: Tesseract's '--psm 6' dropped the big vendor-name heading; the default mode must keep it."""
    spec = make_spec(random.Random(3))
    page = run_ocr(render_invoice(spec), OcrSettings(engine="tesseract", fallback_engine=None))
    assert page.ok and page.engine == "tesseract"
    text = squash(page.text)
    assert squash(spec.vendor_name) in text
    assert squash(spec.invoice_number) in text
    assert f"{spec.total_amount:,.2f}" in page.text
    assert page.mean_confidence and page.mean_confidence > 0.7


@needs_tesseract
def test_real_tesseract_psm_can_be_overridden():
    spec = make_spec(random.Random(3))
    lines = TesseractEngine("en", None, psm=6).read(render_invoice(spec))
    assert lines                                                    # works (the heading may be missing in this mode)


# ---- preprocessing --------------------------------------------------------------------------------------------
def digest(img: Image.Image) -> str:
    return hashlib.sha256(img.tobytes()).hexdigest()


def test_disabled_preprocessing_returns_the_original_image():
    img = render_invoice(make_spec(random.Random(1)))
    res = preprocess_image(img, PreprocessSettings(enabled=False))
    assert res.used_original and res.steps == [] and digest(res.image) == digest(img)


def test_the_original_image_is_never_modified():
    img = degrade(render_invoice(make_spec(random.Random(2))), "rotated", random.Random(2))
    before = digest(img)
    preprocess_image(img, PreprocessSettings(auto_orient=False))
    assert digest(img) == before


def test_each_step_can_be_switched_off():
    img = degrade(render_invoice(make_spec(random.Random(4))), "noisy", random.Random(4))
    res = preprocess_image(img, PreprocessSettings(auto_orient=False, deskew=False, denoise=False, contrast=False, grayscale=False))
    assert res.steps == [] and res.used_original
    res = preprocess_image(img, PreprocessSettings(auto_orient=False, deskew=False, adaptive_threshold=True, keep_original_if_worse=False))
    assert {"denoise", "contrast", "threshold"} <= set(res.steps)


def test_deskew_measures_and_corrects_a_tilted_scan():
    base = render_invoice(make_spec(random.Random(7)))
    for tilt in (-4.0, 2.5, 4.0):
        tilted = degrade(base, "rotated", type("R", (), {"choice": lambda self, seq: tilt, "randint": lambda *a: 0})())
        angle = estimate_skew(pil_to_gray(tilted))
        assert abs(angle + tilt) < 0.8, (tilt, angle)              # returns the CORRECTION: the opposite of the tilt
    assert estimate_skew(pil_to_gray(base)) == 0.0                  # a straight page is left alone


def test_preprocessing_deskews_and_reports_the_angle():
    base = render_invoice(make_spec(random.Random(7)))
    tilted = degrade(base, "rotated", type("R", (), {"choice": lambda self, seq: 4.0, "randint": lambda *a: 0})())
    res = preprocess_image(tilted, PreprocessSettings(auto_orient=False))
    assert not res.used_original and abs(res.skew_deg + 4.0) < 0.8 and any(s.startswith("deskew") for s in res.steps)


def test_blank_page_is_not_deskewed():
    assert estimate_skew(pil_to_gray(Image.new("RGB", (400, 400), "white"))) == 0.0


def test_small_images_are_upscaled():
    res = preprocess_image(Image.new("RGB", (300, 200), "white"), PreprocessSettings(auto_orient=False, deskew=False, keep_original_if_worse=False))
    assert "upscale" in res.steps and res.image.size == (600, 400)


def test_when_filters_make_the_page_worse_the_original_is_kept():
    img = render_invoice(make_spec(random.Random(5)))
    res = preprocess_image(img, PreprocessSettings(auto_orient=False, deskew=False, adaptive_threshold=True, keep_original_if_worse=True))
    assert res.used_original or res.score_after >= res.score_before * 0.9
