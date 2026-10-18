"""Application settings.

Settings come from three places, later ones win:
1. ``config/settings.json`` (defaults shipped with the app)
2. a ``.env`` file in the project root (optional)
3. real environment variables (``GST_*``)

Nothing here talks to the network. The only network host the app is allowed to call is a
loopback Ollama server (``127.0.0.1`` / ``localhost`` / ``::1``).
"""

from __future__ import annotations

import json
import os
from decimal import Decimal
from pathlib import Path
from typing import Any, Mapping
from urllib.parse import urlparse

from pydantic import BaseModel, ConfigDict, Field, field_validator

PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_SETTINGS_FILE = PROJECT_ROOT / "config" / "settings.json"
STATE_CODES_FILE = PROJECT_ROOT / "config" / "gst_state_codes.json"
RATES_FILE = PROJECT_ROOT / "config" / "gst_rates.json"

LOOPBACK_HOSTS = {"127.0.0.1", "localhost", "::1"}


class ConfigError(ValueError):
    """Raised when settings are invalid or unsafe."""


class _Section(BaseModel):
    model_config = ConfigDict(extra="forbid")


class OllamaSettings(_Section):
    host: str = "http://127.0.0.1:11434"
    model: str = "qwen2.5:7b-instruct"
    timeout_s: float = 180.0
    num_ctx: int = 8192
    temperature: float = 0.0
    max_ocr_chars: int = 12000
    allow_remote: bool = False

    @field_validator("host")
    @classmethod
    def _strip(cls, v: str) -> str:
        return v.strip().rstrip("/")


class OcrSettings(_Section):
    engine: str = "paddleocr"
    fallback_engine: str | None = "tesseract"
    lang: str = "en"
    tesseract_cmd: str | None = None
    # Tesseract page-segmentation mode. 4 = "single column of text of variable sizes", which keeps large
    # headings such as the vendor name. 6 ("one uniform block") was measured to drop them.
    tesseract_psm: int = 4
    low_confidence_threshold: float = 0.8
    retry_with_original_below: float = 0.6

    @field_validator("tesseract_psm")
    @classmethod
    def _psm_ok(cls, v: int) -> int:
        if not 0 <= int(v) <= 13:
            raise ValueError("tesseract_psm must be between 0 and 13")
        return int(v)

    @field_validator("engine", "fallback_engine")
    @classmethod
    def _engine_ok(cls, v: str | None) -> str | None:
        if v is None or v == "" or v.lower() == "none":
            return None
        v = v.lower()
        if v not in {"paddleocr", "tesseract"}:
            raise ValueError("OCR engine must be 'paddleocr' or 'tesseract'")
        return v


class PreprocessSettings(_Section):
    enabled: bool = True
    auto_orient: bool = True
    grayscale: bool = True
    contrast: bool = True
    denoise: bool = True
    deskew: bool = True
    adaptive_threshold: bool = False
    max_deskew_angle: float = 15.0
    keep_original_if_worse: bool = True


class LimitSettings(_Section):
    max_file_mb: float = 25
    max_pages: int = 20
    max_image_pixels: int = 40_000_000
    max_image_side: int = 8000
    pdf_dpi: int = 200


class ProcessingSettings(_Section):
    pdf_mode: str = "per_file"
    batch_size: int = 10
    pdf_backend: str = "auto"

    @field_validator("pdf_mode")
    @classmethod
    def _mode_ok(cls, v: str) -> str:
        if v not in {"per_file", "per_page"}:
            raise ValueError("pdf_mode must be 'per_file' or 'per_page'")
        return v

    @field_validator("pdf_backend")
    @classmethod
    def _backend_ok(cls, v: str) -> str:
        if v not in {"auto", "pymupdf", "pdf2image"}:
            raise ValueError("pdf_backend must be 'auto', 'pymupdf' or 'pdf2image'")
        return v


class ValidationSettings(_Section):
    amount_tolerance: Decimal = Decimal("1.00")
    date_day_first: bool = True
    # Dates like 08/03/2026 can be read two ways. Default: flag them for a person to confirm. Turn this off
    # if all your invoices use day/month/year and the extra review work is not worth it.
    flag_ambiguous_dates: bool = True
    warn_on_future_date: bool = True
    rate_check_enabled: bool = False
    rate_tolerance_pct: Decimal = Decimal("0.15")
    required_fields: list[str] = Field(
        default_factory=lambda: [
            "vendor_name",
            "vendor_gstin",
            "invoice_number",
            "invoice_date",
            "total_amount",
        ]
    )


class LoggingSettings(_Section):
    level: str = "INFO"


class Settings(_Section):
    data_dir: Path = Path("data")
    ollama: OllamaSettings = Field(default_factory=OllamaSettings)
    ocr: OcrSettings = Field(default_factory=OcrSettings)
    preprocess: PreprocessSettings = Field(default_factory=PreprocessSettings)
    limits: LimitSettings = Field(default_factory=LimitSettings)
    processing: ProcessingSettings = Field(default_factory=ProcessingSettings)
    validation: ValidationSettings = Field(default_factory=ValidationSettings)
    logging: LoggingSettings = Field(default_factory=LoggingSettings)

    # ---- derived paths -------------------------------------------------
    @property
    def data_path(self) -> Path:
        p = Path(self.data_dir)
        return p if p.is_absolute() else PROJECT_ROOT / p

    @property
    def db_path(self) -> Path:
        return self.data_path / "gst_sorter.sqlite3"

    @property
    def uploads_dir(self) -> Path:
        return self.data_path / "uploads"

    @property
    def pages_dir(self) -> Path:
        return self.data_path / "pages"

    @property
    def tmp_dir(self) -> Path:
        return self.data_path / "tmp"

    @property
    def logs_dir(self) -> Path:
        return self.data_path / "logs"

    @property
    def log_file(self) -> Path:
        return self.logs_dir / "app.log"

    def ensure_dirs(self) -> None:
        for d in (self.data_path, self.uploads_dir, self.pages_dir, self.tmp_dir, self.logs_dir):
            d.mkdir(parents=True, exist_ok=True)


# ----------------------------------------------------------------------
def parse_dotenv(text: str) -> dict[str, str]:
    """Tiny .env parser (KEY=VALUE, '#' comments, optional quotes)."""
    out: dict[str, str] = {}
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        out[key.strip()] = value
    return out


def _deep_merge(base: dict[str, Any], extra: Mapping[str, Any]) -> dict[str, Any]:
    for key, value in extra.items():
        if isinstance(value, Mapping) and isinstance(base.get(key), dict):
            _deep_merge(base[key], value)
        else:
            base[key] = value
    return base


_ENV_MAP: dict[str, tuple[str, ...]] = {
    "GST_DATA_DIR": ("data_dir",),
    "GST_OLLAMA_HOST": ("ollama", "host"),
    "GST_OLLAMA_MODEL": ("ollama", "model"),
    "GST_OLLAMA_TIMEOUT_S": ("ollama", "timeout_s"),
    "GST_OLLAMA_NUM_CTX": ("ollama", "num_ctx"),
    "GST_OCR_ENGINE": ("ocr", "engine"),
    "GST_OCR_FALLBACK_ENGINE": ("ocr", "fallback_engine"),
    "GST_TESSERACT_CMD": ("ocr", "tesseract_cmd"),
    "GST_TESSERACT_PSM": ("ocr", "tesseract_psm"),
    "GST_AMOUNT_TOLERANCE": ("validation", "amount_tolerance"),
    "GST_PDF_MODE": ("processing", "pdf_mode"),
    "GST_PDF_BACKEND": ("processing", "pdf_backend"),
    "GST_LOG_LEVEL": ("logging", "level"),
}


def _check_ollama_host(host: str, allow_remote: bool) -> None:
    parsed = urlparse(host)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise ConfigError(f"Invalid Ollama host '{host}'. Expected something like http://127.0.0.1:11434")
    if parsed.hostname not in LOOPBACK_HOSTS and not allow_remote:
        raise ConfigError(
            f"Ollama host '{parsed.hostname}' is not a loopback address. Invoice text would leave this computer. "
            "Use http://127.0.0.1:11434, or set GST_ALLOW_REMOTE_OLLAMA=1 if you really intend this."
        )


def load_settings(
    settings_file: Path | str | None = None,
    env: Mapping[str, str] | None = None,
    dotenv_file: Path | str | None = None,
    overrides: Mapping[str, Any] | None = None,
) -> Settings:
    """Build :class:`Settings` from the JSON file, optional .env, and environment variables."""
    data: dict[str, Any] = {}
    path = Path(settings_file) if settings_file else DEFAULT_SETTINGS_FILE
    if path.exists():
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ConfigError(f"{path} is not valid JSON: {exc}") from exc

    environ: dict[str, str] = {}
    env_path = Path(dotenv_file) if dotenv_file else PROJECT_ROOT / ".env"
    if env_path.exists():
        environ.update(parse_dotenv(env_path.read_text(encoding="utf-8")))
    environ.update(dict(os.environ if env is None else env))

    for env_key, target in _ENV_MAP.items():
        if env_key in environ and environ[env_key] != "":
            node = data
            for part in target[:-1]:
                node = node.setdefault(part, {})
            node[target[-1]] = environ[env_key]

    if environ.get("GST_ALLOW_REMOTE_OLLAMA") == "1":
        data.setdefault("ollama", {})["allow_remote"] = True

    if overrides:
        _deep_merge(data, overrides)

    try:
        settings = Settings.model_validate(data)
    except ValueError as exc:
        raise ConfigError(f"Invalid settings: {exc}") from exc

    _check_ollama_host(settings.ollama.host, settings.ollama.allow_remote)
    return settings


def load_state_codes(path: Path | str | None = None) -> dict[str, dict[str, Any]]:
    """Return {'27': {'name': 'Maharashtra'}, ...} from the configurable mapping file."""
    p = Path(path) if path else STATE_CODES_FILE
    return json.loads(p.read_text(encoding="utf-8"))["codes"]


def load_rate_periods(path: Path | str | None = None) -> list[dict[str, Any]]:
    """Return the configurable list of GST rate periods (reference data, not legal advice)."""
    p = Path(path) if path else RATES_FILE
    if not p.exists():
        return []
    return json.loads(p.read_text(encoding="utf-8")).get("periods", [])
