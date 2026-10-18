"""Synthetic GST invoice generator.

Everything produced here is FICTITIOUS: made-up vendors, made-up buyers, made-up numbers. GSTINs are built
to be well-formed (the check character comes from the same checksum the app validates), which means a
generated GSTIN could by coincidence equal a real one. They are for testing only and every image is
stamped "SAMPLE - FICTITIOUS".

Usage:
    python eval/synthetic.py --out data/synthetic_batch --count 20 --seed 7

Writes invoice images / PDFs plus ``ground_truth.csv`` (the correct values) into --out.
"""

from __future__ import annotations

import argparse
import csv
import random
import sys
from dataclasses import dataclass, field
from datetime import date, timedelta
from decimal import Decimal
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pipeline.validate import GSTIN_CHARS, gstin_checksum_char  # noqa: E402

VENDORS = [
    "Demo Sharma Electronics", "Sample Gupta Traders", "Test Iyer Stationers", "Example Rao Hardware",
    "Fictional Mehta Foods", "Mock Khan Textiles", "Dummy Patel Pharma Supplies", "Sample Nair Spices",
    "Test Bose Books", "Demo Singh Auto Parts", "Example Das Printers", "Fictional Reddy Agro",
]
BUYERS = ["Alpha Test Services LLP", "Beta Sample Studio", "Gamma Demo Works", "Delta Mock Consulting"]
ITEMS = ["Laptop sleeve", "Printer paper A4", "LED bulb pack", "Steel shelf", "Notebook set", "USB cable", "Desk lamp", "Ink cartridge"]
STATE_CODES = ["07", "09", "27", "29", "33", "24", "06", "19", "08", "36"]
RATES = [5, 12, 18, 28]
VARIANTS = ["clean", "rotated", "blurry", "low_contrast", "noisy"]

GROUND_TRUTH_COLUMNS = [
    "file", "vendor_name", "vendor_gstin", "buyer_name", "buyer_gstin", "invoice_number", "invoice_date",
    "taxable_amount", "cgst", "sgst", "igst", "total_amount", "variant",
]


@dataclass
class InvoiceSpec:
    vendor_name: str
    vendor_gstin: str
    buyer_name: str
    buyer_gstin: str
    invoice_number: str
    invoice_date: date
    taxable_amount: Decimal
    cgst: Decimal | None
    sgst: Decimal | None
    igst: Decimal | None
    total_amount: Decimal
    rate: int
    items: list[tuple[str, int, Decimal]] = field(default_factory=list)

    def truth_row(self, filename: str, variant: str) -> dict[str, str]:
        def money(v: Decimal | None) -> str:
            return "" if v is None else format(v, "f")

        return {
            "file": filename, "vendor_name": self.vendor_name, "vendor_gstin": self.vendor_gstin,
            "buyer_name": self.buyer_name, "buyer_gstin": self.buyer_gstin, "invoice_number": self.invoice_number,
            "invoice_date": self.invoice_date.isoformat(), "taxable_amount": money(self.taxable_amount),
            "cgst": money(self.cgst), "sgst": money(self.sgst), "igst": money(self.igst),
            "total_amount": money(self.total_amount), "variant": variant,
        }


def make_gstin(rng: random.Random, state: str) -> str:
    """A well-formed, checksum-valid GSTIN with a fictitious PAN part."""
    pan = "".join(rng.choice("ABCDEFGHJKLMNPQRSTUVWXYZ") for _ in range(5)) + f"{rng.randint(0, 9999):04d}" + rng.choice("ABCDEFGHJKLMNPQRSTUVWXYZ")
    first14 = f"{state}{pan}{rng.choice('123456789')}Z"
    return first14 + (gstin_checksum_char(first14) or "0")


def make_spec(rng: random.Random, today: date | None = None) -> InvoiceSpec:
    today = today or date(2026, 10, 1)
    seller_state = rng.choice(STATE_CODES)
    inter_state = rng.random() < 0.35
    buyer_state = rng.choice([s for s in STATE_CODES if s != seller_state]) if inter_state else seller_state
    rate = rng.choice(RATES)
    lines: list[tuple[str, int, Decimal]] = []
    for name in rng.sample(ITEMS, rng.randint(1, 3)):
        lines.append((name, rng.randint(1, 9), Decimal(rng.randint(50, 4000)) + Decimal(rng.choice(["0", "0.50", "0.25"]))))
    taxable = sum((q * p for _, q, p in lines), Decimal("0"))
    if inter_state:
        igst = (taxable * rate / 100).quantize(Decimal("0.01"))
        cgst = sgst = None
        total = taxable + igst
    else:
        half = (taxable * rate / 200).quantize(Decimal("0.01"))
        cgst = sgst = half
        igst = None
        total = taxable + half + half
    return InvoiceSpec(
        vendor_name=rng.choice(VENDORS), vendor_gstin=make_gstin(rng, seller_state),
        buyer_name=rng.choice(BUYERS), buyer_gstin=make_gstin(rng, buyer_state),
        invoice_number=f"{rng.choice(['INV', 'TI', 'SB'])}-{rng.randint(1000, 99999)}",
        invoice_date=today - timedelta(days=rng.randint(10, 400)),
        taxable_amount=taxable, cgst=cgst, sgst=sgst, igst=igst, total_amount=total, rate=rate, items=lines,
    )


# ----------------------------------------------------------------------
# Rendering
# ----------------------------------------------------------------------
def _font(size: int, bold: bool = False) -> ImageFont.ImageFont:
    for path in (
        f"/usr/share/fonts/truetype/dejavu/DejaVuSans{'-Bold' if bold else ''}.ttf",
        "C:/Windows/Fonts/arialbd.ttf" if bold else "C:/Windows/Fonts/arial.ttf",
        "/Library/Fonts/Arial.ttf",
    ):
        try:
            return ImageFont.truetype(path, size)
        except OSError:
            continue
    return ImageFont.load_default(size)


def _inr(value: Decimal) -> str:
    return f"{value:,.2f}"


def render_invoice(spec: InvoiceSpec) -> Image.Image:
    img = Image.new("RGB", (1240, 1754), "white")
    d = ImageDraw.Draw(img)
    h1, h2, body, small = _font(46, True), _font(30, True), _font(28), _font(22)
    d.text((70, 60), spec.vendor_name.upper(), font=h1, fill="black")
    d.text((70, 125), f"GSTIN: {spec.vendor_gstin}", font=body, fill="black")
    d.text((70, 165), "TAX INVOICE", font=h2, fill="black")
    d.text((70, 215), "SAMPLE - FICTITIOUS DOCUMENT FOR SOFTWARE TESTING", font=small, fill="#555555")
    d.line((70, 255, 1170, 255), fill="black", width=3)
    d.text((70, 280), f"Invoice No: {spec.invoice_number}", font=body, fill="black")
    d.text((700, 280), f"Date: {spec.invoice_date.strftime('%d/%m/%Y')}", font=body, fill="black")
    d.text((70, 340), "Bill To:", font=h2, fill="black")
    d.text((70, 385), spec.buyer_name, font=body, fill="black")
    d.text((70, 425), f"GSTIN: {spec.buyer_gstin}", font=body, fill="black")
    y = 520
    d.rectangle((70, y, 1170, y + 50), outline="black", width=2)
    for x, label in ((85, "Item"), (620, "Qty"), (760, "Rate"), (960, "Amount")):
        d.text((x, y + 8), label, font=h2, fill="black")
    y += 50
    for name, qty, price in spec.items:
        d.rectangle((70, y, 1170, y + 50), outline="#888888", width=1)
        d.text((85, y + 8), name, font=body, fill="black")
        d.text((620, y + 8), str(qty), font=body, fill="black")
        d.text((760, y + 8), _inr(price), font=body, fill="black")
        d.text((960, y + 8), _inr(price * qty), font=body, fill="black")
        y += 50
    y += 40
    rows = [("Taxable Value", spec.taxable_amount)]
    half = spec.rate / 2
    if spec.igst is not None:
        rows.append((f"IGST @ {spec.rate}%", spec.igst))
    else:
        rows += [(f"CGST @ {half:g}%", spec.cgst), (f"SGST @ {half:g}%", spec.sgst)]
    for label, value in rows:
        d.text((620, y), label, font=body, fill="black")
        d.text((960, y), _inr(value), font=body, fill="black")  # type: ignore[arg-type]
        y += 44
    d.line((620, y + 5, 1170, y + 5), fill="black", width=2)
    d.text((620, y + 20), "Grand Total", font=h2, fill="black")
    d.text((960, y + 20), _inr(spec.total_amount), font=h2, fill="black")
    d.text((70, 1650), "Computer generated sample invoice. Not a real tax document.", font=small, fill="#555555")
    return img


def degrade(img: Image.Image, variant: str, rng: random.Random) -> Image.Image:
    """Apply a camera/scan style degradation to a rendered invoice."""
    if variant == "clean":
        return img
    arr = np.asarray(img)
    if variant == "rotated":
        angle = rng.choice([-4.0, -2.5, 2.5, 4.0])
        h, w = arr.shape[:2]
        m = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
        arr = cv2.warpAffine(arr, m, (w, h), flags=cv2.INTER_LINEAR, borderValue=(255, 255, 255))
    elif variant == "blurry":
        arr = cv2.GaussianBlur(arr, (0, 0), 1.6)
    elif variant == "low_contrast":
        arr = (arr.astype(np.float32) * 0.45 + 120).clip(0, 255).astype(np.uint8)
    elif variant == "noisy":
        noise = np.random.default_rng(rng.randint(0, 2**31 - 1)).normal(0, 22, arr.shape)
        arr = (arr.astype(np.float32) + noise).clip(0, 255).astype(np.uint8)
    else:
        raise ValueError(f"unknown variant {variant}")
    return Image.fromarray(arr)


def make_batch(out_dir: Path, count: int = 20, seed: int = 7, pdfs: int = 3, variants: list[str] | None = None) -> list[dict[str, str]]:
    """Write ``count`` invoices (PNG/JPG, plus ``pdfs`` of them as PDF) and ground_truth.csv. Returns the truth rows."""
    rng = random.Random(seed)
    out_dir.mkdir(parents=True, exist_ok=True)
    variants = variants or VARIANTS
    rows: list[dict[str, str]] = []
    for i in range(count):
        spec = make_spec(rng)
        variant = variants[i % len(variants)]
        img = degrade(render_invoice(spec), variant, rng)
        if i < pdfs:
            name = f"synthetic_{i + 1:03d}_{variant}.pdf"
            img.save(out_dir / name, "PDF", resolution=150.0)
        elif i % 2:
            name = f"synthetic_{i + 1:03d}_{variant}.jpg"
            img.save(out_dir / name, "JPEG", quality=88)
        else:
            name = f"synthetic_{i + 1:03d}_{variant}.png"
            img.save(out_dir / name, "PNG")
        rows.append(spec.truth_row(name, variant))
    with (out_dir / "ground_truth.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=GROUND_TRUTH_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    return rows


def make_multipage_pdf(path: Path, specs: list[InvoiceSpec]) -> None:
    """One PDF with one page per spec (used by the multi-page tests)."""
    pages = [render_invoice(s) for s in specs]
    pages[0].save(path, "PDF", save_all=True, append_images=pages[1:], resolution=150.0)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate FICTITIOUS test invoices and ground_truth.csv")
    parser.add_argument("--out", type=Path, default=ROOT / "data" / "synthetic_batch")
    parser.add_argument("--count", type=int, default=20)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--pdfs", type=int, default=3, help="how many of the invoices to write as PDF")
    args = parser.parse_args()
    rows = make_batch(args.out, args.count, args.seed, args.pdfs)
    print(f"Wrote {len(rows)} fictitious invoices and ground_truth.csv to {args.out}")


if __name__ == "__main__":
    main()
