#!/usr/bin/env python3
from __future__ import annotations

import csv
import io
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.semantic_pipeline import SemanticPlatePipeline, is_valid_plate_format


def load_labeled_samples(root: Path, limit: int = 5) -> list[dict]:
    labels_path = root / "crop_labels.csv"
    rows = []
    with labels_path.open("r", newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            text = (row.get("text") or "").strip().upper()
            crop_path = row.get("crop_path") or ""
            path = root / crop_path
            if text:
                rows.append({"path": path, "truth": text})
            if len(rows) >= limit:
                break
    return rows


def main() -> None:
    np.random.seed(42)
    pipe = SemanticPlatePipeline(Path("data/kb/plate_templates.json"), model_path=None)
    samples = load_labeled_samples(Path("data/smoke_synthetic"), limit=5)
    table = []

    for sample in samples:
        row = {
            "input": str(sample["path"]),
            "truth": sample["truth"],
            "runs": {},
        }
        for noise in (0.0, 0.5, 1.0):
            if sample["path"].exists():
                image_bytes = sample["path"].read_bytes()
                input_name = sample["path"].name
            else:
                image = pipe.knowledge_base.render_plate(sample["truth"], (384, 96), "ind_private_white")
                buffer = io.BytesIO()
                image.save(buffer, format="PNG")
                image_bytes = buffer.getvalue()
                input_name = f"{sample['truth']}.png"
            result = pipe.run(
                image_bytes,
                snr_db=18.0,
                channel_noise=noise,
                text_hint=sample["truth"] if not sample["path"].exists() else "",
            )
            metrics = result.metrics
            row["runs"][str(noise)] = {
                "input_name": input_name,
                "extracted": metrics.get("semantic_text", ""),
                "received": metrics.get("received_text", ""),
                "char_accuracy_percent": metrics.get("character_accuracy_percent"),
                "semantic_similarity_percent": metrics.get("semantic_similarity_percent"),
                "cosine_similarity_percent": metrics.get("cosine_similarity_percent"),
                "format_valid": is_valid_plate_format(metrics.get("semantic_text", "")),
            }
        table.append(row)

    output = Path("panel_demo_results.json")
    output.write_text(json.dumps(table, indent=2), encoding="utf-8")
    print(json.dumps(table, indent=2))
    print(f"\nSaved panel demo table to {output}")


if __name__ == "__main__":
    main()
