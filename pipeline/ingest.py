"""File validation, safe storage and page rendering.

Rules enforced here:
* only PDF / JPG / JPEG / PNG, and the file *content* must match the extension;
* size, page-count and image-dimension limits come from settings;
* the original filename is only ever used for display. Files are stored under a name derived
  from their SHA-256 hash, so a hostile filename cannot influence any path;
* uploaded files are never executed or opened by anything other than image/PDF decoders.
"""

from __future__ import annotations

import hashlib
import logging
import os
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

from PIL import Image, ImageOps

from config import Settings

log = logging.getLogger(__name__)

SUPPORTED_EXTENSIONS = {".pdf": "pdf", ".jpg": "jpeg", ".jpeg": "jpeg", ".png": "png"}
_PNG_MAGIC = b"\x89PNG\r\n\x1a\n"


class IngestError(Exception):
    """A file was rejected. The message is safe to show to the user."""


@dataclass(frozen=True)
class IngestedFile:
    doc_id: str          # full SHA-256 of the file content
    filename: str        # sanitized, display only
    ext: str             # '.pdf' | '.jpg' | '.jpeg' | '.png'
    kind: str            # 'pdf' | 'jpeg' | 'png'
    size_bytes: int
    page_count: int
    stored_path: Path

    @property
    def short_id(self) -> str:
        return self.doc_id[:32]


# ----------------------------------------------------------------------
def sanitize_filename(name: str | None) -> str:
    """Keep only a harmless display name: no directories, no control characters, bounded length."""
    base = (name or "").replace("\\", "/").split("/")[-1]
    base = re.sub(r"[\x00-\x1f\x7f]", "", base).strip().strip(".")
    return base[:150] or "file"


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def detect_kind(data: bytes) -> str | None:
    head = data[:1024]
    if b"%PDF-" in head:
        return "pdf"
    if data[:3] == b"\xff\xd8\xff":
        return "jpeg"
    if data[:8] == _PNG_MAGIC:
        return "png"
    return None


# ----------------------------------------------------------------------
# PDF backends
# ----------------------------------------------------------------------
def _pymupdf():
    """The PyMuPDF module (``pymupdf`` in current releases, ``fitz`` in older ones), or None."""
    try:
        import pymupdf  # type: ignore

        return pymupdf
    except ImportError:
        try:
            import fitz  # type: ignore

            return fitz
        except ImportError:
            return None


def _have_pymupdf() -> bool:
    return _pymupdf() is not None


def _have_pdf2image() -> bool:
    try:
        import pdf2image  # noqa: F401

        return True
    except ImportError:
        return False


def pick_pdf_backend(preferred: str = "auto") -> str:
    if preferred in ("auto", "pymupdf") and _have_pymupdf():
        return "pymupdf"
    if preferred in ("auto", "pdf2image") and _have_pdf2image():
        return "pdf2image"
    raise IngestError(
        "No PDF renderer is available. Install PyMuPDF (pip install PyMuPDF) or pdf2image plus Poppler "
        "(see README, 'PDF support')."
    )


def pdf_page_count(path: Path, backend: str = "auto") -> int:
    chosen = pick_pdf_backend(backend)
    try:
        if chosen == "pymupdf":
            with _pymupdf().open(path) as doc:
                if doc.needs_pass:
                    raise IngestError("This PDF is password-protected. Remove the password and import it again.")
                return doc.page_count
        from pdf2image import pdfinfo_from_path

        info = pdfinfo_from_path(str(path))
        if info.get("Encrypted", "no").lower().startswith("yes") and "Pages" not in info:
            raise IngestError("This PDF is password-protected. Remove the password and import it again.")
        return int(info["Pages"])
    except IngestError:
        raise
    except Exception as exc:  # corrupted file, missing Poppler, ...
        log.warning("PDF page count failed (%s): %s", chosen, type(exc).__name__)
        hint = " Poppler may not be installed or on PATH." if chosen == "pdf2image" else ""
        raise IngestError(f"Could not read this PDF. It may be damaged or password-protected.{hint}") from exc


def _fit_dpi(width_pt: float, height_pt: float, dpi: int, max_pixels: int) -> int:
    pixels = (width_pt / 72 * dpi) * (height_pt / 72 * dpi)
    if pixels <= max_pixels:
        return dpi
    scale = (max_pixels / pixels) ** 0.5
    return max(50, int(dpi * scale))


def render_pdf_page(path: Path, page_no: int, settings: Settings) -> Image.Image:
    """Render one 1-based PDF page to an RGB image."""
    backend = pick_pdf_backend(settings.processing.pdf_backend)
    dpi = settings.limits.pdf_dpi
    try:
        if backend == "pymupdf":
            with _pymupdf().open(path) as doc:
                page = doc[page_no - 1]
                dpi = _fit_dpi(page.rect.width, page.rect.height, dpi, settings.limits.max_image_pixels)
                pix = page.get_pixmap(dpi=dpi, alpha=False)
                return Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
        from pdf2image import convert_from_path

        images = convert_from_path(str(path), dpi=dpi, first_page=page_no, last_page=page_no)
        if not images:
            raise IngestError(f"Page {page_no} could not be rendered.")
        img = images[0].convert("RGB")
        if img.width * img.height > settings.limits.max_image_pixels:
            raise IngestError(f"Page {page_no} is larger than the configured limit.")
        return img
    except IngestError:
        raise
    except Exception as exc:
        log.warning("PDF render failed (%s) page %s: %s", backend, page_no, type(exc).__name__)
        raise IngestError(f"Page {page_no} of the PDF could not be rendered (damaged file?).") from exc


def load_image(path: Path, settings: Settings) -> Image.Image:
    """Open a JPG/PNG safely: dimension checks before decoding, EXIF orientation applied."""
    try:
        with Image.open(path) as probe:
            width, height = probe.size
            if width * height > settings.limits.max_image_pixels or max(width, height) > settings.limits.max_image_side:
                raise IngestError(
                    f"Image is {width}x{height}px, above the configured limit "
                    f"({settings.limits.max_image_side}px per side / {settings.limits.max_image_pixels:,} pixels)."
                )
            img = ImageOps.exif_transpose(probe)
            img.load()
    except IngestError:
        raise
    except Exception as exc:
        raise IngestError("This image could not be decoded. It may be damaged.") from exc
    return to_rgb(img)


def to_rgb(img: Image.Image) -> Image.Image:
    if img.mode in ("RGBA", "LA") or (img.mode == "P" and "transparency" in img.info):
        rgba = img.convert("RGBA")
        background = Image.new("RGB", rgba.size, (255, 255, 255))
        background.paste(rgba, mask=rgba.split()[-1])
        return background
    return img.convert("RGB")


def load_page_image(file_path: Path, kind: str, page_no: int, settings: Settings) -> Image.Image:
    if kind == "pdf":
        return render_pdf_page(file_path, page_no, settings)
    return load_image(file_path, settings)


# ----------------------------------------------------------------------
# Ingestion
# ----------------------------------------------------------------------
def ingest_bytes(name: str, data: bytes, settings: Settings) -> IngestedFile:
    """Validate and store one uploaded file. Raises :class:`IngestError` with a friendly reason."""
    display = sanitize_filename(name)
    ext = Path(display).suffix.lower()
    if ext not in SUPPORTED_EXTENSIONS:
        raise IngestError(f"'{display}': unsupported file type '{ext or 'none'}'. Use PDF, JPG, JPEG or PNG.")
    if not data:
        raise IngestError(f"'{display}' is empty.")
    max_bytes = int(settings.limits.max_file_mb * 1024 * 1024)
    if len(data) > max_bytes:
        raise IngestError(f"'{display}' is {len(data) / 1024 / 1024:.1f} MB, above the {settings.limits.max_file_mb:g} MB limit.")

    kind = detect_kind(data)
    expected = SUPPORTED_EXTENSIONS[ext]
    if kind is None or kind != expected:
        raise IngestError(f"'{display}': the file content does not match its '{ext}' extension, so it was rejected.")

    doc_id = sha256_bytes(data)
    settings.ensure_dirs()
    stored = settings.uploads_dir / f"{doc_id[:32]}{ext}"
    if not stored.exists():
        fd, tmp_name = tempfile.mkstemp(dir=settings.tmp_dir, suffix=".part")
        try:
            with os.fdopen(fd, "wb") as fh:
                fh.write(data)
            os.replace(tmp_name, stored)
        finally:
            if os.path.exists(tmp_name):
                os.unlink(tmp_name)

    try:
        if kind == "pdf":
            pages = pdf_page_count(stored, settings.processing.pdf_backend)
            if pages < 1:
                raise IngestError(f"'{display}' has no pages.")
        else:
            load_image(stored, settings)  # validates dimensions and decodability
            pages = 1
        if pages > settings.limits.max_pages:
            raise IngestError(f"'{display}' has {pages} pages, above the {settings.limits.max_pages}-page limit.")
    except IngestError as exc:
        _remove_if_unreferenced(stored)
        message = str(exc)
        raise IngestError(message if message.startswith(f"'{display}'") else f"'{display}': {message}") from exc

    return IngestedFile(doc_id, display, ext, kind, len(data), pages, stored)


def _remove_if_unreferenced(path: Path) -> None:
    try:
        path.unlink(missing_ok=True)
    except OSError:
        pass


def ingest_path(path: Path | str, settings: Settings) -> IngestedFile:
    """Ingest a file from disk. Symlinks and non-regular files are refused."""
    p = Path(path)
    if p.is_symlink() or not p.is_file():
        raise IngestError(f"'{p.name}' is not a regular file.")
    max_bytes = int(settings.limits.max_file_mb * 1024 * 1024)
    try:
        size = p.stat().st_size
        if size > max_bytes:
            raise IngestError(f"'{p.name}' is {size / 1024 / 1024:.1f} MB, above the {settings.limits.max_file_mb:g} MB limit.")
        data = p.read_bytes()
    except PermissionError as exc:
        raise IngestError(f"'{p.name}': permission denied. Check the file's permissions.") from exc
    except OSError as exc:
        raise IngestError(f"'{p.name}' could not be read ({exc.strerror or 'I/O error'}).") from exc
    return ingest_bytes(p.name, data, settings)


def list_folder(folder: Path | str, recursive: bool = False) -> list[Path]:
    """List supported files in an explicitly chosen folder (symlinks skipped, sorted by name)."""
    root = Path(folder).expanduser()
    if not root.is_dir():
        raise IngestError(f"'{folder}' is not a folder, or it does not exist.")
    iterator = root.rglob("*") if recursive else root.iterdir()
    found: list[Path] = []
    try:
        for p in iterator:
            if p.is_symlink() or not p.is_file():
                continue
            if p.suffix.lower() in SUPPORTED_EXTENSIONS:
                found.append(p)
    except PermissionError as exc:
        raise IngestError(f"Permission denied while reading '{folder}'.") from exc
    return sorted(found, key=lambda x: str(x).lower())


# ----------------------------------------------------------------------
# Page images on disk (the original, unprocessed render used for display and OCR input)
# ----------------------------------------------------------------------
def page_image_path(settings: Settings, doc_id: str, page_no: int, suffix: str = "") -> Path:
    return settings.pages_dir / doc_id[:32] / f"p{page_no}{suffix}.png"


def render_and_store_page(f_path: Path, kind: str, doc_id: str, page_no: int, settings: Settings) -> tuple[Image.Image, Path]:
    """Render a page, save it as PNG for the review screen, return (image, saved path)."""
    img = load_page_image(f_path, kind, page_no, settings)
    out = page_image_path(settings, doc_id, page_no)
    out.parent.mkdir(parents=True, exist_ok=True)
    if not out.exists():
        img.save(out, format="PNG")
    return img, out


def iter_pages(f_path: Path, kind: str, page_count: int, settings: Settings) -> Iterator[tuple[int, Image.Image]]:
    for n in range(1, page_count + 1):
        yield n, load_page_image(f_path, kind, n, settings)
