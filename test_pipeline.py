"""Quick test of the full pipeline."""
from pathlib import Path
from src.semantic_pipeline import SemanticPlatePipeline

p = SemanticPlatePipeline(
    Path("data/kb/plate_templates.json"),
    model_path=Path("artifacts/semantic_lpr_async_fl.pt"),
)

crop_dir = Path("data/synthetic_plates/crops")
test_img = sorted(crop_dir.glob("*.png"))[0]
print(f"Testing with: {test_img}")

result = p.run(test_img.read_bytes(), snr_db=18.0, channel_noise=0.0)
m = result.metrics
print(f"Input image data URL length: {len(result.input_image)}")
print(f"Extracted plate data URL length: {len(result.extracted_plate)}")
print(f"Received map data URL length: {len(result.received_semantic_map)}")
print(f"Reconstructed plate data URL length: {len(result.reconstructed_plate)}")
print(f"Reconstructed scene data URL length: {len(result.reconstructed_scene)}")
print(f"OCR text: {m.get('ocr_text', 'N/A')}")
print(f"Semantic text: {m.get('semantic_text', 'N/A')}")
print(f"Received text: {m.get('received_text', 'N/A')}")
print(f"PSNR: {m.get('psnr_db', 'N/A')}")
print(f"SSIM: {m.get('ssim', 'N/A')}")
print(f"Char accuracy: {m.get('character_accuracy_percent', 'N/A')}%")
print(f"Cosine similarity: {m.get('cosine_similarity_percent', 'N/A')}%")
print(f"Image cosine: {m.get('image_cosine_similarity_percent', 'N/A')}%")
print(f"Compression ratio: {m.get('compression_ratio', 'N/A')}x")
print("PIPELINE TEST PASSED")
