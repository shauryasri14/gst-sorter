"""Image preprocessing for OCR.

Every step is optional (see ``settings.preprocess``) and the original image is never modified.
If preprocessing makes the page measurably *less* readable, the original is used instead.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import cv2
import numpy as np
from PIL import Image

from config import PreprocessSettings

log = logging.getLogger(__name__)


@dataclass
class PreprocessResult:
    image: Image.Image                  # RGB image to hand to the OCR engine
    steps: list[str] = field(default_factory=list)
    used_original: bool = True
    rotation_deg: int = 0               # 90-degree rotation applied by orientation correction
    skew_deg: float = 0.0               # small deskew angle applied
    score_before: float = 0.0
    score_after: float = 0.0

    @property
    def geometry_changed(self) -> bool:
        return bool(self.rotation_deg or abs(self.skew_deg) > 0.0 or "upscale" in self.steps)


# ----------------------------------------------------------------------
def pil_to_gray(img: Image.Image) -> np.ndarray:
    return cv2.cvtColor(np.asarray(img.convert("RGB")), cv2.COLOR_RGB2GRAY)


def gray_to_pil(gray: np.ndarray) -> Image.Image:
    return Image.fromarray(gray).convert("RGB")


def readability_score(gray: np.ndarray) -> float:
    """Cheap 0..1 heuristic: contrast and edge sharpness. Only used to compare two versions of a page."""
    if gray.size == 0:
        return 0.0
    small = gray
    longest = max(gray.shape)
    if longest > 1600:
        scale = 1600 / longest
        small = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    contrast = min(1.0, float(small.std()) / 70.0)
    sharp = min(1.0, float(cv2.Laplacian(small, cv2.CV_64F).var()) / 1500.0)
    return round(0.5 * contrast + 0.5 * sharp, 4)


def detect_rotation_osd(gray: np.ndarray, min_confidence: float = 2.0) -> int:
    """Return 0/90/180/270 (degrees to rotate clockwise to make text upright) using Tesseract OSD.

    Returns 0 when OSD is unavailable or not confident: the OCR engine's own angle classifier
    still handles upside-down text lines.
    """
    try:
        import pytesseract

        small = gray
        if max(gray.shape) > 1600:
            scale = 1600 / max(gray.shape)
            small = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        osd = pytesseract.image_to_osd(small, config="--psm 0 -c min_characters_to_try=20")
        info: dict[str, str] = {}
        for line in osd.splitlines():
            if ":" in line:
                k, _, v = line.partition(":")
                info[k.strip()] = v.strip()
        confidence = float(info.get("Orientation confidence", "0"))
        rotate = int(info.get("Rotate", "0"))
        if confidence >= min_confidence and rotate in (90, 180, 270):
            return rotate
    except Exception as exc:  # no tesseract, no osd data, too little text...
        log.debug("OSD orientation unavailable: %s", type(exc).__name__)
    return 0


def rotate_90(gray_or_rgb: np.ndarray, clockwise_deg: int) -> np.ndarray:
    if clockwise_deg == 90:
        return cv2.rotate(gray_or_rgb, cv2.ROTATE_90_CLOCKWISE)
    if clockwise_deg == 180:
        return cv2.rotate(gray_or_rgb, cv2.ROTATE_180)
    if clockwise_deg == 270:
        return cv2.rotate(gray_or_rgb, cv2.ROTATE_90_COUNTERCLOCKWISE)
    return gray_or_rgb


def estimate_skew(gray: np.ndarray, max_angle: float = 15.0) -> float:
    """Find the rotation (degrees, as used by ``cv2.getRotationMatrix2D``: positive = counter-clockwise)
    that straightens the page, using a projection-profile search.

    Text lines are horizontal when the variance of the row sums is largest, so the angle that maximises
    that variance is the correction to apply with :func:`apply_skew`.
    """
    longest = max(gray.shape)
    if longest > 1000:
        scale = 1000 / longest
        gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    if int(binary.sum() / 255) < 200:  # essentially blank page
        return 0.0
    h, w = binary.shape
    center = (w / 2, h / 2)

    def score(angle: float) -> float:
        m = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(binary, m, (w, h), flags=cv2.INTER_NEAREST, borderValue=0)
        return float(np.var(rotated.sum(axis=1, dtype=np.float64)))

    coarse = np.arange(-max_angle, max_angle + 0.01, 1.0)
    best = max(coarse, key=score)
    fine = np.arange(best - 0.9, best + 0.91, 0.1)
    best = max(fine, key=score)
    if abs(best) < 0.3:
        return 0.0
    return float(best)


def apply_skew(img: np.ndarray, angle: float, border: int | tuple = 255) -> np.ndarray:
    h, w = img.shape[:2]
    m = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    return cv2.warpAffine(img, m, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT, borderValue=border)


def enhance_contrast(gray: np.ndarray) -> np.ndarray:
    return cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)


def reduce_noise(gray: np.ndarray) -> np.ndarray:
    return cv2.medianBlur(gray, 3)


def adaptive_binarize(gray: np.ndarray) -> np.ndarray:
    return cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 15)


# ----------------------------------------------------------------------
def preprocess_image(original: Image.Image, cfg: PreprocessSettings) -> PreprocessResult:
    """Return the image to OCR. ``original`` is not modified."""
    original = original.convert("RGB")
    if not cfg.enabled:
        return PreprocessResult(original, [], True)

    gray0 = pil_to_gray(original)
    score_before = readability_score(gray0)
    steps: list[str] = []
    rotation = 0
    skew = 0.0

    work = gray0
    if cfg.auto_orient:
        rotation = detect_rotation_osd(work)
        if rotation:
            work = rotate_90(work, rotation)
            steps.append(f"orient{rotation}")

    if max(work.shape) < 1000:
        work = cv2.resize(work, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
        steps.append("upscale")

    if cfg.deskew:
        skew = estimate_skew(work, cfg.max_deskew_angle)
        if skew:
            work = apply_skew(work, skew)
            steps.append(f"deskew{skew:+.1f}")

    if cfg.denoise:
        work = reduce_noise(work)
        steps.append("denoise")
    if cfg.contrast:
        work = enhance_contrast(work)
        steps.append("contrast")
    if cfg.adaptive_threshold:
        work = adaptive_binarize(work)
        steps.append("threshold")
    if cfg.grayscale:
        steps.insert(0, "grayscale")

    if not steps:
        return PreprocessResult(original, [], True, score_before=score_before, score_after=score_before)

    score_after = readability_score(work)
    # Geometry fixes (rotation / deskew) are corrections, not cosmetics: only compare pure filter output.
    only_filters = rotation == 0 and skew == 0.0 and "upscale" not in steps
    if cfg.keep_original_if_worse and only_filters and score_after < score_before * 0.9:
        log.info("Preprocessing reduced readability (%.3f -> %.3f); keeping original", score_before, score_after)
        return PreprocessResult(original, ["kept_original"], True, score_before=score_before, score_after=score_after)

    return PreprocessResult(
        gray_to_pil(work), steps, False, rotation_deg=rotation, skew_deg=skew,
        score_before=score_before, score_after=score_after,
    )
