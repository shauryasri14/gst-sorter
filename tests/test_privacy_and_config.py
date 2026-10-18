"""Static checks for the privacy promises: local-only, no telemetry, no cloud SDKs."""

from __future__ import annotations

import ast
import re
import tomllib
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
SOURCES = [p for p in ROOT.rglob("*.py") if not any(part in {".venv", "venv", "tests", "__pycache__", "data"} for part in p.parts)]
FORBIDDEN_IMPORTS = {"openai", "anthropic", "boto3", "botocore", "google", "azure", "sentry_sdk", "mixpanel", "segment", "analytics",
                     "posthog", "amplitude", "datadog", "newrelic", "bugsnag", "rollbar", "cohere", "vertexai", "ocrspace", "textract"}
NETWORK_MODULES = {"requests", "urllib", "urllib3", "http", "httpx", "aiohttp", "socket", "websockets", "ftplib", "smtplib"}


def imports_of(path: Path) -> set[str]:
    found: set[str] = set()
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if isinstance(node, ast.Import):
            found |= {a.name.split(".")[0] for a in node.names}
        elif isinstance(node, ast.ImportFrom) and node.module:
            found.add(node.module.split(".")[0])
    return found


def test_there_are_sources_to_check():
    assert len(SOURCES) >= 15


def test_no_cloud_ai_or_telemetry_sdk_is_imported():
    bad = {str(p.relative_to(ROOT)): imports_of(p) & FORBIDDEN_IMPORTS for p in SOURCES if imports_of(p) & FORBIDDEN_IMPORTS}
    assert bad == {}


def test_only_the_ollama_client_touches_the_network():
    users = {str(p.relative_to(ROOT)) for p in SOURCES if imports_of(p) & NETWORK_MODULES}
    # config.py only *parses* the host URL (urllib.parse); extract.py is the single place that sends requests.
    assert users == {"pipeline/extract.py", "config.py"}, users
    assert "requests" not in imports_of(ROOT / "config.py")


def test_no_hardcoded_external_urls_in_the_code():
    allowed = ("127.0.0.1", "localhost", "[::1]", "example.com", "example.org")
    offenders = []
    for p in SOURCES:
        for url in re.findall(r"https?://[^\s\"')>]+", p.read_text(encoding="utf-8")):
            if not any(a in url for a in allowed):
                offenders.append((p.name, url))
    # documentation links in comments/help text are fine; nothing may be requested at runtime.
    assert all(("docs" in u or "github" in u or "ollama.com" in u or "paddlepaddle" in u or "tesseract" in u.lower() or "ubmannheim" in u.lower()
                or "wikipedia" in u or "gst" in u or "cbic" in u) for _, u in offenders), offenders


def test_streamlit_is_bound_to_localhost_with_telemetry_off():
    cfg = tomllib.loads((ROOT / ".streamlit" / "config.toml").read_text(encoding="utf-8"))
    assert cfg["server"]["address"] == "127.0.0.1"
    assert cfg["browser"]["gatherUsageStats"] is False


def test_default_ollama_host_is_loopback():
    from config import OllamaSettings, load_settings

    assert OllamaSettings().host.startswith("http://127.0.0.1")
    assert load_settings(env={}, dotenv_file=ROOT / "nope.env").ollama.host.startswith("http://127.0.0.1")


def test_private_data_is_git_ignored():
    ignore = (ROOT / ".gitignore").read_text(encoding="utf-8")
    for needle in ("data/*", "*.sqlite3", ".env", "*.xlsx"):
        assert needle in ignore
    assert "!data/.gitkeep" in ignore and (ROOT / "data" / ".gitkeep").exists()


def test_env_example_has_no_secrets_and_only_known_keys():
    from config import _ENV_MAP

    keys = set(re.findall(r"^#?\s*(GST_[A-Z_]+)\s*=", (ROOT / ".env.example").read_text(), flags=re.M))
    assert keys, "expected at least one example setting"
    assert keys <= set(_ENV_MAP) | {"GST_ALLOW_REMOTE_OLLAMA"}, keys - set(_ENV_MAP)


def test_settings_file_loads_and_matches_the_models():
    from config import load_settings

    s = load_settings(env={}, dotenv_file=ROOT / "nope.env")
    assert s.validation.amount_tolerance == 1 and s.ocr.engine == "paddleocr" and s.ollama.model.startswith("qwen2.5")


def test_state_code_table_has_every_gst_state_and_ut_prefix():
    from config import load_state_codes

    codes = load_state_codes()
    for must in ("07", "09", "27", "29", "33", "36", "37", "38"):
        assert must in codes, must
    assert all(v.get("name") for v in codes.values())


def test_rate_table_is_data_not_code():
    from config import load_rate_periods

    periods = load_rate_periods()
    assert periods and all({"from", "rates_pct"} <= set(p) for p in periods)
    src = (ROOT / "pipeline" / "validate.py").read_text(encoding="utf-8")
    assert not re.search(r"\b(18|28)\s*(%|,)", src.split("def check_rate")[0]), "rates must not be hard-coded in validate.py"
