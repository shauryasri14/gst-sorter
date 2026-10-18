"""Local LLM extraction: JSON parsing, the single repair retry, Ollama failures, localhost-only config."""

from __future__ import annotations

import json
from decimal import Decimal

import pytest
import requests

from config import ConfigError, OllamaSettings, load_settings
from pipeline.extract import (
    DEFAULT_PROMPT,
    OllamaClient,
    OllamaError,
    OllamaModelMissingError,
    OllamaUnavailableError,
    build_invoice,
    check_ollama,
    extract_invoice,
    invoice_json_schema,
    parse_llm_json,
)
from pipeline.schemas import ExtractedInvoice, parse_amount
from tests.conftest import FakeChat

CFG = OllamaSettings()
GOOD = {
    "vendor_name": "Demo Sharma Electronics", "vendor_gstin": "27AAPFU0939F1ZV", "buyer_name": None, "buyer_gstin": None,
    "invoice_number": "INV-2041", "invoice_date": "14/09/2026", "taxable_amount": 10000.0, "cgst": 900, "sgst": "900.00",
    "igst": None, "total_amount": 11800.5, "currency": "INR", "extraction_notes": [],
}


# ---- parsing ---------------------------------------------------------------------------
def test_parses_plain_and_fenced_json():
    assert parse_llm_json('{"a": 1}') == {"a": 1}
    assert parse_llm_json('```json\n{"a": 1}\n```') == {"a": 1}
    assert parse_llm_json('Sure! Here it is: {"a": 1} Hope that helps') == {"a": 1}


@pytest.mark.parametrize("text", ["", "no json here", "{broken", '["a", "b"]', '{"a": NaN}', '{"a": Infinity}'])
def test_unusable_replies_raise_value_error(text):
    with pytest.raises(ValueError):
        parse_llm_json(text)


def test_numbers_become_exact_decimals_without_float_rounding():
    data = parse_llm_json('{"total_amount": 1234.10, "cgst": 0.1}')
    assert data["total_amount"] == Decimal("1234.10") and str(data["total_amount"]) == "1234.10"
    inv, _ = build_invoice(data)
    assert inv.total_amount == Decimal("1234.10") and inv.cgst == Decimal("0.1")


def test_float_input_does_not_pick_up_binary_noise():
    assert ExtractedInvoice(total_amount=0.1 + 0.2).total_amount == Decimal("0.30000000000000004")  # what the float really is
    assert ExtractedInvoice(total_amount=11800.5).total_amount == Decimal("11800.5")


@pytest.mark.parametrize("raw, expected", [("₹1,200.50", Decimal("1200.50")), ("Rs. 1,00,000", Decimal("100000")),
                                           ("INR 99", Decimal("99")), ("(250.00)", Decimal("-250.00")), (7, Decimal("7")),
                                           (None, None), ("null", None), ("N/A", None), ("", None), ("-", None)])
def test_amount_parsing(raw, expected):
    assert parse_amount(raw) == expected


@pytest.mark.parametrize("bad", ["abc", "12.3.4", True, float("nan"), "1e999999999999"])
def test_bad_amounts_raise(bad):
    with pytest.raises((ValueError, Exception)):
        ExtractedInvoice(total_amount=bad)


def test_missing_values_stay_none_never_zero():
    inv, _ = build_invoice({"cgst": None, "sgst": "null", "igst": "N/A", "total_amount": ""})
    assert inv.cgst is None and inv.sgst is None and inv.igst is None and inv.total_amount is None
    assert inv.vendor_name is None and inv.currency == "INR"


def test_unknown_keys_are_dropped_and_reported():
    inv, dropped = build_invoice({**GOOD, "hacker": "x", "discount": 5})
    assert dropped == ["discount", "hacker"] and not hasattr(inv, "hacker")


def test_schema_demands_every_field_and_forbids_extras():
    schema = invoice_json_schema()
    assert schema["additionalProperties"] is False
    assert set(schema["required"]) == set(GOOD)


def test_model_rejects_extra_fields_directly():
    with pytest.raises(Exception):
        ExtractedInvoice(vendor_name="x", surprise=1)


def test_prompt_forbids_guessing_and_zero_for_unknown():
    assert "ONLY information that is explicitly present" in DEFAULT_PROMPT
    assert "Never use 0 for an unknown amount" in DEFAULT_PROMPT and "null" in DEFAULT_PROMPT
    assert "{ocr_text}" in DEFAULT_PROMPT


# ---- extract_invoice -------------------------------------------------------------------
def test_good_reply_is_used_without_retry():
    chat = FakeChat(GOOD)
    res = extract_invoice("some ocr text", CFG, chat)
    assert res.ok and not res.retried and len(chat.calls) == 1
    assert res.invoice.total_amount == Decimal("11800.5") and res.invoice.sgst == Decimal("900.00")
    assert "some ocr text" in chat.calls[0][0]["content"]


def test_malformed_json_gets_exactly_one_repair_retry_that_can_succeed():
    chat = FakeChat("this is not json", GOOD)
    res = extract_invoice("ocr", CFG, chat)
    assert res.ok and res.retried and len(chat.calls) == 2
    repair = chat.calls[1][-1]["content"]
    assert "could not be used" in repair and "this is not json" in repair


def test_two_bad_replies_fail_cleanly_after_one_retry_only():
    chat = FakeChat("nope", "still nope", GOOD)
    res = extract_invoice("ocr", CFG, chat)
    assert not res.ok and res.invoice is None and res.retried
    assert len(chat.calls) == 2                                # never a third attempt
    assert "even after one retry" in (res.error or "")


def test_invalid_field_values_also_trigger_the_retry():
    chat = FakeChat({**GOOD, "total_amount": "lots"}, GOOD)
    res = extract_invoice("ocr", CFG, chat)
    assert res.ok and res.retried
    assert "total_amount" in chat.calls[1][-1]["content"]


def test_empty_ocr_text_never_calls_the_model():
    chat = FakeChat(GOOD)
    res = extract_invoice("   \n ", CFG, chat)
    assert not res.ok and chat.calls == [] and "no OCR text" in res.error


def test_overlong_ocr_text_is_truncated_and_noted():
    cfg = OllamaSettings(max_ocr_chars=50)
    chat = FakeChat(GOOD)
    res = extract_invoice("x" * 500, cfg, chat)
    assert res.ocr_truncated and any("cut before extraction" in n for n in res.invoice.extraction_notes)
    assert len(chat.calls[0][0]["content"]) < 2500


def test_unknown_keys_from_the_model_are_noted_not_trusted():
    res = extract_invoice("ocr", CFG, FakeChat({**GOOD, "evil": "x"}))
    assert res.ok and res.dropped_keys == ["evil"]
    assert any("evil" in n for n in res.invoice.extraction_notes)


def test_connection_errors_propagate_so_a_batch_can_stop():
    with pytest.raises(OllamaUnavailableError):
        extract_invoice("ocr", CFG, FakeChat(OllamaUnavailableError("Ollama is not running")))


# ---- Ollama HTTP client (no real server: a scripted session) ------------------------------
class FakeResponse:
    def __init__(self, status=200, payload=None, text=""):
        self.status_code, self._payload = status, payload

    def json(self):
        if self._payload is None:
            raise ValueError("no json")
        return self._payload

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(str(self.status_code))


class FakeSession:
    def __init__(self, get=None, post=None):
        self._get, self._post, self.trust_env, self.posted = get, post, True, []

    def get(self, url, **kw):
        if isinstance(self._get, Exception):
            raise self._get
        return self._get

    def post(self, url, **kw):
        self.posted.append((url, kw))
        if isinstance(self._post, Exception):
            raise self._post
        return self._post


def test_missing_ollama_service_gives_an_actionable_message():
    client = OllamaClient(CFG, FakeSession(get=requests.ConnectionError("refused"), post=requests.ConnectionError("refused")))
    with pytest.raises(OllamaUnavailableError) as exc:
        client.chat([{"role": "user", "content": "hi"}])
    assert "ollama serve" in str(exc.value) and "127.0.0.1:11434" in str(exc.value)
    status = check_ollama(CFG, client)
    assert not status.running and not status.ready and "not running" in status.message


def test_model_not_installed_tells_the_user_how_to_pull_it():
    client = OllamaClient(CFG, FakeSession(post=FakeResponse(404, {"error": "model not found"})))
    with pytest.raises(OllamaModelMissingError) as exc:
        client.chat([{"role": "user", "content": "hi"}])
    assert f"ollama pull {CFG.model}" in str(exc.value)


def test_check_ollama_reports_installed_and_missing_models():
    tags = FakeResponse(200, {"models": [{"name": "llama3.2:3b"}]})
    status = check_ollama(CFG, OllamaClient(CFG, FakeSession(get=tags)))
    assert status.running and not status.model_present and "ollama pull qwen2.5:7b-instruct" in status.message
    assert "llama3.2:3b" in status.message
    ok = check_ollama(OllamaSettings(model="llama3.2:3b"), OllamaClient(OllamaSettings(model="llama3.2:3b"), FakeSession(get=tags)))
    assert ok.ready


def test_timeout_message_suggests_a_smaller_model():
    client = OllamaClient(CFG, FakeSession(post=requests.Timeout("slow")))
    with pytest.raises(OllamaError) as exc:
        client.chat([{"role": "user", "content": "hi"}])
    assert "smaller model" in str(exc.value)


def test_unexpected_reply_shape_is_an_error_not_a_crash():
    client = OllamaClient(CFG, FakeSession(post=FakeResponse(200, {"unexpected": True})))
    with pytest.raises(OllamaError):
        client.chat([{"role": "user", "content": "hi"}])


def test_request_goes_to_the_configured_local_host_with_structured_output_and_no_env_proxy():
    session = FakeSession(post=FakeResponse(200, {"message": {"content": "{}"}}))
    client = OllamaClient(CFG, session)
    client.chat([{"role": "user", "content": "hi"}], invoice_json_schema())
    url, kw = session.posted[0]
    assert url == "http://127.0.0.1:11434/api/chat"
    assert kw["json"]["stream"] is False and kw["json"]["format"]["type"] == "object"
    assert kw["json"]["options"]["temperature"] == 0.0
    assert session.trust_env is False                      # HTTP(S)_PROXY must never carry invoice text


# ---- configuration: localhost only ----------------------------------------------------------
@pytest.mark.parametrize("host", ["http://127.0.0.1:11434", "http://localhost:11434", "http://[::1]:11434"])
def test_loopback_hosts_are_accepted(host, tmp_path):
    s = load_settings(env={"GST_OLLAMA_HOST": host}, dotenv_file=tmp_path / "none.env")
    assert s.ollama.host == host


@pytest.mark.parametrize("host", ["http://192.168.1.20:11434", "https://api.example.com", "http://ollama.internal:11434"])
def test_remote_hosts_are_refused_unless_explicitly_allowed(host, tmp_path):
    with pytest.raises(ConfigError, match="not a loopback"):
        load_settings(env={"GST_OLLAMA_HOST": host}, dotenv_file=tmp_path / "none.env")
    ok = load_settings(env={"GST_OLLAMA_HOST": host, "GST_ALLOW_REMOTE_OLLAMA": "1"}, dotenv_file=tmp_path / "none.env")
    assert ok.ollama.allow_remote


def test_garbage_host_is_a_config_error(tmp_path):
    with pytest.raises(ConfigError):
        load_settings(env={"GST_OLLAMA_HOST": "not a url"}, dotenv_file=tmp_path / "none.env")


def test_model_name_is_configurable_from_environment(tmp_path):
    s = load_settings(env={"GST_OLLAMA_MODEL": "qwen2.5:3b-instruct"}, dotenv_file=tmp_path / "none.env")
    assert s.ollama.model == "qwen2.5:3b-instruct"


def test_absurd_exponent_amount_from_the_model_is_rejected_and_retried():
    chat = FakeChat('{"total_amount": 1e999999999999, "extraction_notes": []}', GOOD)
    res = extract_invoice("ocr", CFG, chat)
    assert res.ok and res.retried and res.invoice.total_amount == Decimal("11800.5")
