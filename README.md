# GST Invoice & Receipt Sorter

Turns a folder of Indian GST invoices (PDF, JPG, PNG) into one validated spreadsheet. **Everything runs on your
computer**: OCR, the language model, validation, storage and export. No cloud AI, no paid API, no telemetry.

```
Files → ingest → render PDF pages → preprocess → OCR → local LLM (Ollama) → deterministic GST checks
      → human review & correction → Excel / CSV
```

The model only *copies* what is printed. All GST checks (GSTIN checksum, tax arithmetic, state codes, dates,
duplicates) are plain Python in `pipeline/validate.py`. Extraction success is never shown as "verified": only a person
can mark an invoice **Reviewed**.

## 1. Prerequisites
| Need | Notes |
|---|---|
| Python 3.11 or 3.12 | Recommended: PaddleOCR/PaddlePaddle wheels lag behind the newest Python. The code and tests also ran on 3.13 with Tesseract. |
| [Ollama](https://ollama.com) | Local model server. |
| 8 GB RAM minimum | The default 7B model needs roughly 5-6 GB. Use a smaller model if needed (section 4). |
| Internet, **once** | To install packages, download the model and (PaddleOCR) its OCR models. Afterwards invoice processing works offline. |
| Tesseract (optional) | OCR fallback. |

## 2. Install
**Windows (PowerShell)**
```powershell
cd gst-sorter
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1          # if blocked: Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
pip install -r requirements.txt
```
**Linux / macOS**
```bash
cd gst-sorter
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```
Nothing is installed automatically by the app. `PyMuPDF` renders PDFs (AGPL-licensed; check this if you redistribute).
If you prefer `pdf2image`, install Poppler and set `GST_PDF_BACKEND=pdf2image`.

## 3. OCR engines
* **PaddleOCR** (primary): installed by `requirements.txt`. Downloads its models on first use (internet needed once).
* **Tesseract** (fallback, also usable alone with `GST_OCR_ENGINE=tesseract`):
  * Windows: install the UB Mannheim build, then `GST_TESSERACT_CMD=C:\Program Files\Tesseract-OCR\tesseract.exe` if it is not on PATH.
  * macOS: `brew install tesseract`  ·  Linux: `sudo apt install tesseract-ocr`

## 4. Ollama and the model
```powershell
ollama serve                       # skip if the Ollama app is already running
ollama pull qwen2.5:7b-instruct    # default model, one-time download
ollama list                        # confirm it is installed
```
Smaller machine? Use a smaller model and tell the app:
```powershell
ollama pull qwen2.5:3b-instruct
$env:GST_OLLAMA_MODEL = "qwen2.5:3b-instruct"     # Linux/macOS: export GST_OLLAMA_MODEL=qwen2.5:3b-instruct
```
Or copy `.env.example` to `.env` and edit it. The app refuses any non-localhost Ollama address (invoice text would leave
your computer) unless you set `GST_ALLOW_REMOTE_OLLAMA=1`. The Dashboard shows whether Ollama and the model are ready.
Smaller models make more mistakes; the validation layer and your review are what catch them.

## 5. Run
```powershell
streamlit run app.py
```
Opens http://127.0.0.1:8501 (bound to this computer only; `.streamlit/config.toml`).

1. **Import documents**: upload files or give a folder path → **Start processing** (pause / cancel / retry failed).
2. **Invoice review**: original page beside editable fields, every warning with its exact reason, low-OCR-confidence
   highlighting, duplicate candidates, filters, "Re-run extraction", "Retry OCR for this page". Corrections are saved in SQLite.
3. **Export**: `.xlsx` (Invoices, Validation, Summary sheets), CSV, validation report. Text starting with `= + - @` is
   neutralised so spreadsheets cannot run it as a formula. Amounts are exact and never adjusted to make totals add up.
4. **Delete local data** (Export page) wipes the database, stored copies, page images and temp files.

## 6. Tests
```powershell
pytest                              # 300 tests, about a minute; no network and no Ollama needed
```
They use test doubles for the LLM and (mostly) OCR, plus real Tesseract tests that skip themselves if it is not installed.

## 7. Synthetic test batch and evaluation
All synthetic invoices are **fictitious** and stamped "SAMPLE - FICTITIOUS".
```powershell
python eval/synthetic.py --out data/synthetic_batch --count 20 --seed 7          # images/PDFs + ground_truth.csv

# a) OCR only (no Ollama needed): is the true value present in the OCR text at all?
python eval/evaluate.py --truth data/synthetic_batch/ground_truth.csv --ocr-only data/synthetic_batch --ocr tesseract

# b) full pipeline (needs OCR + running Ollama): accuracy, flags, timings
python eval/evaluate.py --truth data/synthetic_batch/ground_truth.csv --run data/synthetic_batch

# c) score an export you made in the app
python eval/evaluate.py --truth ground_truth.csv --predictions gst_invoices.csv --json report.json
```
Reports: exact-match and within-tolerance accuracy per field, GSTIN accuracy, invoice-level success, % flagged for
review, wrong-but-not-flagged invoices, and time per page / per invoice. Validation never edits values, so accuracy
is the same before and after it; the report shows what it *catches*.

### Measurements actually taken while building this
Machine: a cloud sandbox CPU, 20 synthetic invoices (5 degradation types, 4 as PDF), Tesseract 5.3.4. **Not measured:**
PaddleOCR and any Ollama model (neither was available there), so there are **no LLM accuracy figures yet**. Run (b) on your machine.

| Field | Found in Tesseract OCR text |
|---|---|
| vendor / buyer name, invoice no., date, total, CGST, SGST | 100% |
| taxable amount, IGST | 90% |
| vendor GSTIN | 70% |
| buyer GSTIN | 55% |

OCR time: mean 2.5 s per page (median 2.4 s, p95 3.1 s).

* **GSTINs are the weak spot for OCR** (letter O read for digit 0, I for 1, S for 5, dropped characters). Of 40 GSTINs
  copied from the OCR text, 17 were misread; the format/checksum check flagged 15. The other 2 were not corruption: the vendor's
  GSTIN was unreadable and my simple test script paired the buyer's GSTIN with the vendor role, which a checksum cannot detect.
  (This used a regex copier, not an LLM, so it measures the validator only.)
* Tesseract `--psm 6` found the large vendor-name heading on 4 of 10 invoices; `--psm 4` found it on 10 of 10. The default is now 4 (`ocr.tesseract_psm`).

## 8. Configuration
`config/settings.json` (defaults), overridden by `.env`, overridden by `GST_*` environment variables.
Key options: `validation.amount_tolerance` (₹1), `validation.date_day_first`, `validation.flag_ambiguous_dates`
(on by default: any `08/03/2026`-style date is flagged; turn off if all your invoices are day-first),
`validation.rate_check_enabled` (off; warning-only), `processing.pdf_mode` (`per_file` or `per_page`), `limits.*`,
`preprocess.*`. State codes live in `config/gst_state_codes.json` and reference rates in `config/gst_rates.json`
(**editable data, not legal advice; verify against official notifications**).

## 9. Troubleshooting
| Symptom | Fix |
|---|---|
| "Ollama is not running" | Start the Ollama app or `ollama serve`. The batch pauses and files stay Pending. |
| "model ... is not installed" | `ollama pull <model>` (name shown in the message). |
| "model did not answer within N seconds" | Smaller model, or `GST_OLLAMA_TIMEOUT_S=600`. |
| "PaddleOCR is not installed / could not start" | `pip install paddleocr paddlepaddle`; first run needs internet. Or `GST_OCR_ENGINE=tesseract`. |
| "Tesseract program was not found" | Install it and set `GST_TESSERACT_CMD`. |
| PDF cannot be read | Damaged or password-protected: remove the password. With pdf2image, install Poppler. |
| Many "Needs Review" for dates | See `flag_ambiguous_dates` above. |
| Windows script blocked | `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` |

Logs: `data/logs/app.log` (rotating). They hold stage names, short ids, durations and error types, never invoice text, GSTINs or amounts (tested).

## 10. Privacy: what is and is not promised
* OCR, model, validation, storage and export run locally; the only network call in the code is to the Ollama address, and it must be loopback. Streamlit telemetry is off. Static tests check that no cloud/telemetry SDK is imported.
* Initial setup (packages, model, PaddleOCR models) needs internet; routine processing works offline.
* Local does **not** mean encrypted: the SQLite database, stored copies of your files and page images sit unencrypted in `data/`. Use disk encryption (BitLocker/FileVault) and the delete button.
* `data/` is git-ignored. Do not commit real invoices.

## 11. Known limitations
* Not tested here: PaddleOCR and real LLM accuracy (see section 7). Treat accuracy on your own documents as unknown until you run the evaluation.
* Handwritten bills, very low-quality photos, non-English text and multi-invoice pages are unsupported or unreliable.
* A valid GSTIN check character only means well-formed. The app cannot tell whether a GSTIN is registered or active (that needs the GST portal).
* Place of supply is not derived from GSTINs; state mismatches are warnings, not legal conclusions.
* Line-item level GST rates are not checked; the optional rate check looks at the blended rate only.
* Pause/cancel take effect between files, not mid-file. One batch runs at a time.
* A multi-page PDF is one invoice by default; use `GST_PDF_MODE=per_page` for scanned bundles of separate invoices.
