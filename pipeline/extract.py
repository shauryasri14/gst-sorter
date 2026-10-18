"""Structured invoice extraction with a *local* Ollama model.

Only loopback hosts are allowed (enforced in :mod:`config`). The model is asked to copy facts that are
printed in the OCR text; it is never asked to validate GST rules, and nothing it returns is trusted until
the deterministic checks in :mod:`pipeline.validate` and a human review have seen it.
"""

from __future__ import annotations

import json
import logging
import re
import time
from dataclasses import dataclass, field
from decimal import Decimal
from pathlib import Path
from typing import Any, Protocol

import requests
from pydantic import ValidationError

from config import PROJECT_ROOT, OllamaSettings
from pipeline.schemas import EXTRACTION_FIELDS, ExtractedInvoice

log = logging.getLogger(__name__)

PROMPT_FILE = PROJECT_ROOT / "config" / "extraction_prompt.txt"

DEFAULT_PROMPT = """You extract data from OCR text of an Indian GST tax invoice or receipt.

STRICT RULES
1. Use ONLY information that is explicitly present in the OCR text. Never guess, never calculate, never use outside knowledge.
2. If a value is missing, unreadable or unclear, use null. Never use 0 for an unknown amount.
3. Copy text exactly as printed. Do not "fix" a GSTIN, an invoice number or a name, even if it looks wrong. If something looks wrong, describe it in extraction_notes.
4. vendor_* is the SELLER / supplier who issued the invoice. buyer_* is the customer ("Bill To", "Buyer", "Consignee").
5. A GSTIN is a 15-character tax id (for example 27ABCDE1234F1Z5). Do not confuse it with a PAN (10 characters) or a phone number.
6. taxable_amount is the value before tax (taxable value / sub total). cgst, sgst and igst are the tax AMOUNTS in rupees, not the percentages. If only a percentage is printed, use null.
7. total_amount is the final invoice total (grand total / amount payable).
8. Amounts are plain numbers with a dot for decimals: no currency symbol, no thousands separators.
9. invoice_date: copy the date exactly as printed (for example 14/09/2026). Do not reformat it.
10. currency is "INR" unless the document clearly shows another currency.
11. extraction_notes: short strings about anything uncertain (partly unreadable GSTIN, reverse charge, several GSTINs, etc.). Use [] if there is nothing to note.

Return ONLY one JSON object with exactly these keys:
vendor_name, vendor_gstin, buyer_name, buyer_gstin, invoice_number, invoice_date, taxable_amount, cgst, sgst, igst, total_amount, currency, extraction_notes

OCR TEXT:
<<<
{ocr_text}
>>>
"""

REPAIR_PROMPT = """Your previous reply could not be used: {problem}

Reply again with ONLY one valid JSON object containing exactly the keys:
vendor_name, vendor_gstin, buyer_name, buyer_gstin, invoice_number, invoice_date, taxable_amount, cgst, sgst, igst, total_amount, currency, extraction_notes
Use null for anything not explicitly present in the OCR text. Amounts must be plain numbers. No explanation, no markdown.

Your previous reply was:
{previous}
"""


class OllamaError(RuntimeError):
    """Base class: the message is safe to show to the user."""


class OllamaUnavailableError(OllamaError):
    pass


class OllamaModelMissingError(OllamaError):
    pass


class ChatClient(Protocol):
    def chat(self, messages: list[dict[str, str]], schema: dict[str, Any] | None = None) -> str: ...


# ----------------------------------------------------------------------
def invoice_json_schema() -> dict[str, Any]:
    nullable_text = {"anyOf": [{"type": "string"}, {"type": "null"}]}
    nullable_num = {"anyOf": [{"type": "number"}, {"type": "null"}]}
    props: dict[str, Any] = {k: nullable_text for k in
                             ("vendor_name", "vendor_gstin", "buyer_name", "buyer_gstin", "invoice_number", "invoice_date", "currency")}
    props.update({k: nullable_num for k in ("taxable_amount", "cgst", "sgst", "igst", "total_amount")})
    props["extraction_notes"] = {"type": "array", "items": {"type": "string"}}
    return {"type": "object", "properties": props, "required": list(props), "additionalProperties": False}


class OllamaClient:
    """Minimal HTTP client for a local Ollama server. Environment proxies are ignored on purpose."""

    def __init__(self, cfg: OllamaSettings, session: requests.Session | None = None) -> None:
        self.cfg = cfg
        self.session = session or requests.Session()
        self.session.trust_env = False  # never route invoice text through HTTP(S)_PROXY

    def _url(self, path: str) -> str:
        return f"{self.cfg.host}{path}"

    def list_models(self) -> list[str]:
        try:
            resp = self.session.get(self._url("/api/tags"), timeout=5)
            resp.raise_for_status()
            return [m.get("name", "") for m in resp.json().get("models", [])]
        except requests.ConnectionError as exc:
            raise OllamaUnavailableError(_not_running(self.cfg.host)) from exc
        except requests.Timeout as exc:
            raise OllamaUnavailableError(f"Ollama at {self.cfg.host} did not answer within 5 seconds.") from exc
        except (requests.RequestException, ValueError) as exc:
            raise OllamaError(f"Unexpected reply from Ollama: {type(exc).__name__}") from exc

    def chat(self, messages: list[dict[str, str]], schema: dict[str, Any] | None = None) -> str:
        payload: dict[str, Any] = {
            "model": self.cfg.model,
            "messages": messages,
            "stream": False,
            "options": {"temperature": self.cfg.temperature, "num_ctx": self.cfg.num_ctx},
        }
        if schema is not None:
            payload["format"] = schema
        try:
            resp = self.session.post(self._url("/api/chat"), json=payload, timeout=self.cfg.timeout_s)
        except requests.ConnectionError as exc:
            raise OllamaUnavailableError(_not_running(self.cfg.host)) from exc
        except requests.Timeout as exc:
            raise OllamaError(
                f"The model did not answer within {self.cfg.timeout_s:g} seconds. Try a smaller model "
                "(see README, 'Choosing a model') or raise GST_OLLAMA_TIMEOUT_S."
            ) from exc
        if resp.status_code == 404:
            raise OllamaModelMissingError(_model_missing(self.cfg.model))
        if resp.status_code >= 400:
            detail = ""
            try:
                detail = str(resp.json().get("error", ""))[:200]
            except ValueError:
                pass
            if "not found" in detail.lower():
                raise OllamaModelMissingError(_model_missing(self.cfg.model))
            raise OllamaError(f"Ollama returned an error (HTTP {resp.status_code}). {detail}".strip())
        try:
            return str(resp.json()["message"]["content"])
        except (KeyError, ValueError, TypeError) as exc:
            raise OllamaError("Ollama sent a reply in an unexpected format.") from exc


def _not_running(host: str) -> str:
    return (
        f"Ollama is not running at {host}. Start it by opening the Ollama app, or run `ollama serve` "
        "in a terminal, then try again."
    )


def _model_missing(model: str) -> str:
    return f"The model '{model}' is not installed in Ollama. Download it once with:  ollama pull {model}"


@dataclass
class OllamaStatus:
    running: bool
    model_present: bool
    models: list[str] = field(default_factory=list)
    message: str = ""

    @property
    def ready(self) -> bool:
        return self.running and self.model_present


def check_ollama(cfg: OllamaSettings, client: Any | None = None) -> OllamaStatus:
    """Report whether Ollama is reachable and the configured model is installed."""
    client = client or OllamaClient(cfg)
    try:
        models = client.list_models()
    except OllamaUnavailableError as exc:
        return OllamaStatus(False, False, [], str(exc))
    except OllamaError as exc:
        return OllamaStatus(False, False, [], str(exc))
    wanted = cfg.model
    present = wanted in models or (":" not in wanted and f"{wanted}:latest" in models)
    if present:
        return OllamaStatus(True, True, models, f"Ollama is running and '{wanted}' is installed.")
    others = f" Installed models: {', '.join(models)}." if models else " No models are installed yet."
    return OllamaStatus(True, False, models, _model_missing(wanted) + others)


# ----------------------------------------------------------------------
_FENCE = re.compile(r"^```(?:json)?\s*|\s*```$", re.IGNORECASE)


def parse_llm_json(text: str) -> dict[str, Any]:
    """Parse a model reply into a dict. Numbers become exact Decimals (no float rounding)."""
    cleaned = _FENCE.sub("", text.strip())
    start, end = cleaned.find("{"), cleaned.rfind("}")
    if start == -1 or end <= start:
        raise ValueError("the reply did not contain a JSON object")
    try:
        data = json.loads(cleaned[start:end + 1], parse_float=Decimal, parse_constant=_reject_constant)
    except json.JSONDecodeError as exc:
        raise ValueError(f"the reply was not valid JSON ({exc.msg} at position {exc.pos})") from exc
    if not isinstance(data, dict):
        raise ValueError("the reply JSON was not an object")
    return data


def _reject_constant(name: str) -> Any:
    raise ValueError(f"'{name}' is not allowed in JSON")


def build_invoice(data: dict[str, Any]) -> tuple[ExtractedInvoice, list[str]]:
    """Validate a parsed dict. Unknown keys are dropped (and reported), never trusted."""
    allowed = set(EXTRACTION_FIELDS) | {"extraction_notes"}
    dropped = sorted(k for k in data if k not in allowed)
    cleaned = {k: v for k, v in data.items() if k in allowed}
    return ExtractedInvoice.model_validate(cleaned), dropped


def _describe(exc: Exception) -> str:
    if isinstance(exc, ValidationError):
        parts = [f"{'.'.join(map(str, e['loc']))}: {e['msg']}" for e in exc.errors()[:5]]
        return "invalid values - " + "; ".join(parts)
    return str(exc)


def load_prompt_template() -> str:
    if PROMPT_FILE.exists():
        text = PROMPT_FILE.read_text(encoding="utf-8")
        if "{ocr_text}" in text:
            return text
        log.warning("config/extraction_prompt.txt has no {ocr_text} placeholder; using the built-in prompt")
    return DEFAULT_PROMPT


@dataclass
class ExtractionResult:
    invoice: ExtractedInvoice | None
    raw_response: str = ""
    error: str | None = None
    retried: bool = False
    elapsed_ms: int = 0
    model: str = ""
    dropped_keys: list[str] = field(default_factory=list)
    ocr_truncated: bool = False

    @property
    def ok(self) -> bool:
        return self.invoice is not None and self.error is None


def extract_invoice(
    ocr_text: str,
    cfg: OllamaSettings,
    client: ChatClient | None = None,
    prompt_template: str | None = None,
) -> ExtractionResult:
    """Ask the local model for structured fields. At most one repair retry; no fabricated fallback.

    Raises :class:`OllamaUnavailableError` / :class:`OllamaModelMissingError` so callers can stop a batch
    early; malformed model output is returned as ``ExtractionResult.error`` instead.
    """
    if not ocr_text.strip():
        return ExtractionResult(None, error="There is no OCR text to extract from.", model=cfg.model)

    truncated = len(ocr_text) > cfg.max_ocr_chars
    text = ocr_text[: cfg.max_ocr_chars]
    template = prompt_template or load_prompt_template()
    prompt = template.replace("{ocr_text}", text)
    schema = invoice_json_schema()
    client = client or OllamaClient(cfg)

    messages: list[dict[str, str]] = [{"role": "user", "content": prompt}]
    start = time.perf_counter()
    raw = client.chat(messages, schema)
    retried = False
    try:
        invoice, dropped = build_invoice(parse_llm_json(raw))
    except (ValueError, ValidationError) as first_error:
        problem = _describe(first_error)
        log.info("LLM output unusable (%s); retrying once", type(first_error).__name__)
        retried = True
        repair = REPAIR_PROMPT.format(problem=problem, previous=raw[:2000])
        raw = client.chat(messages + [{"role": "assistant", "content": raw[:2000]}, {"role": "user", "content": repair}], schema)
        try:
            invoice, dropped = build_invoice(parse_llm_json(raw))
        except (ValueError, ValidationError) as second_error:
            return ExtractionResult(
                None, raw_response=raw, retried=True, model=cfg.model, ocr_truncated=truncated,
                elapsed_ms=int((time.perf_counter() - start) * 1000),
                error=f"The model's reply could not be used even after one retry ({_describe(second_error)}). "
                      "Try again, or use a larger model.",
            )

    notes = list(invoice.extraction_notes)
    if dropped:
        notes.append(f"Ignored unexpected keys from the model: {', '.join(dropped)}")
    if truncated:
        notes.append(f"OCR text was longer than {cfg.max_ocr_chars} characters and was cut before extraction")
    invoice = invoice.model_copy(update={"extraction_notes": notes})
    return ExtractionResult(
        invoice, raw_response=raw, retried=retried, model=cfg.model, dropped_keys=dropped, ocr_truncated=truncated,
        elapsed_ms=int((time.perf_counter() - start) * 1000),
    )


def write_default_prompt(path: Path = PROMPT_FILE) -> None:
    path.write_text(DEFAULT_PROMPT, encoding="utf-8")
