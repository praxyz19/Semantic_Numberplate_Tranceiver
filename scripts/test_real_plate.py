import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.semantic_pipeline import SemanticPlatePipeline

pipeline = SemanticPlatePipeline(
    kb_path=Path("data/kb/plate_templates.json"),
    model_path=Path("artifacts/semantic_lpr_async_fl.pt"),
)

for img_path in sorted(Path("data/synthetic_plates/crops").glob("*.png"))[:5]:
    img_bytes = img_path.read_bytes()
    result = pipeline.run(img_bytes, snr_db=20.0)
    m = result.metrics
    print(
        f"{img_path.name:25s} | OCR={m['ocr_text']:12s} | "
        f"Cosine={m['image_cosine_similarity_percent']:5.1f}% | "
        f"CharAcc={m['character_accuracy_percent']:5.1f}% | "
        f"PSNR={m['psnr_db']:.1f} dB"
    )
