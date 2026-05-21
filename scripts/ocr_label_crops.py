#!/usr/bin/env python3
"""Run EasyOCR over crop images and fill the `text` column in crop_labels.csv.

Usage: python scripts/ocr_label_crops.py --dataset data/indian_plate_train
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

try:
    import easyocr
    from PIL import Image
except Exception:
    print("Missing dependencies: please install easyocr and pillow in your environment")
    raise


def clean_text(t: str) -> str:
    # Keep alphanumerics and uppercase, strip spaces
    import re
    if not t:
        return ""
    s = re.sub(r"[^A-Za-z0-9]", "", t)
    return s.upper()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--lang", default="en")
    parser.add_argument("--min-conf", type=float, default=0.3)
    args = parser.parse_args()

    root = Path(args.dataset)
    csv_path = root / "crop_labels.csv"
    if not csv_path.exists():
        print("crop_labels.csv not found at", csv_path)
        sys.exit(1)

    reader = easyocr.Reader([args.lang], gpu=False)
    rows = []
    with csv_path.open(newline='', encoding='utf-8') as fh:
        dr = csv.DictReader(fh)
        fieldnames = dr.fieldnames
        if "text" not in fieldnames:
            print("CSV missing 'text' column")
            sys.exit(1)
        for r in dr:
            rows.append(r)

    filled = 0
    total = len(rows)
    for i, r in enumerate(rows):
        crop_rel = r.get('crop_path')
        if not crop_rel:
            continue
        crop_path = root / crop_rel
        if not crop_path.exists():
            continue
        # Skip if already populated
        if r.get('text') and r.get('text').strip():
            continue
        try:
            res = reader.readtext(str(crop_path))
            # res is list of (bbox, text, conf)
            if not res:
                ocr_text = ""
            else:
                # choose highest confidence
                best = max(res, key=lambda x: x[2])
                ocr_text = best[1] if best else ""
            clean = clean_text(ocr_text)
            if clean:
                r['text'] = clean
                filled += 1
        except Exception as e:
            print(f"OCR error {crop_path}: {e}")

    # Backup
    bak = csv_path.with_suffix('.csv.bak')
    if not bak.exists():
        csv_path.replace(bak)
        src = bak
    else:
        src = csv_path

    # Write updated CSV
    out_path = root / "crop_labels.csv"
    with out_path.open('w', newline='', encoding='utf-8') as fh:
        dw = csv.DictWriter(fh, fieldnames=fieldnames)
        dw.writeheader()
        for r in rows:
            dw.writerow(r)

    print(f"OCR pass complete. Filled {filled}/{total} text fields. Backup at {bak}")


if __name__ == '__main__':
    main()
