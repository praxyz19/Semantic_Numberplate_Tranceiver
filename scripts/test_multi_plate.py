from __future__ import annotations

import io
import random
from pathlib import Path

from PIL import Image

from src.semantic_pipeline import SemanticPlatePipeline


def synthesize_multi_plate_frame(left: Image.Image, right: Image.Image) -> Image.Image:
    pad = 12
    height = max(left.height, right.height) + pad * 2
    width = left.width + right.width + pad * 3
    background = Image.new("RGB", (width, height), (226, 226, 226))

    left_y = (height - left.height) // 2
    right_y = (height - right.height) // 2

    background.paste(left, (pad, left_y))
    background.paste(right, (left.width + pad * 2, right_y))
    return background


def main() -> None:
    crop_dir = Path("data/smoke_synthetic/crops")
    crops = sorted(crop_dir.glob("*.png"))
    if len(crops) < 2:
        raise SystemExit("Need at least two crops in data/smoke_synthetic/crops")

    random.seed(7)
    left_path, right_path = random.sample(crops, 2)
    left_img = Image.open(left_path).convert("RGB")
    right_img = Image.open(right_path).convert("RGB")

    composite = synthesize_multi_plate_frame(left_img, right_img)
    buffer = io.BytesIO()
    composite.save(buffer, format="PNG")

    pipeline = SemanticPlatePipeline(
        Path("data/kb/plate_templates.json"),
        model_path=Path("artifacts_scratch/semantic_lpr_async_fl.pt"),
    )
    result = pipeline.run_multi(buffer.getvalue(), max_plates=2, snr_db=18.0, channel_noise=0.0)

    print(f"Composite image: {composite.width}x{composite.height}")
    for plate in result.plates:
        metrics = plate.metrics
        print(
            f"{metrics.get('task_id', 'task')} | OCR={metrics.get('ocr_text', '')} | "
            f"Semantic={metrics.get('semantic_text', '')} | "
            f"BBox={metrics.get('bbox', '')}"
        )

    out_dir = Path("data/smoke_synthetic/multi_frames")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "multi_plate_sample.png"
    composite.save(out_path)
    print(f"Saved composite frame to {out_path}")


if __name__ == "__main__":
    main()
