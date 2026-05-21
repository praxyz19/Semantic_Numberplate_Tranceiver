#!/usr/bin/env python3
from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.image_utils import image_from_bytes
from src.semantic_pipeline import try_qwen_vlm_ocr


def main() -> None:
    if len(sys.argv) < 2:
        raise SystemExit("Usage: py -3.10 -B scripts/test_qwen_vlm_plate.py <image_path>")

    os.environ.setdefault("USE_QWEN_VLM", "1")
    image_path = Path(sys.argv[1])
    plate = image_from_bytes(image_path.read_bytes())
    text = try_qwen_vlm_ocr(plate)
    print(text or "Qwen VLM did not return a plate sequence.")


if __name__ == "__main__":
    main()
