import sys
import os
from pathlib import Path
from PIL import Image

# Set debug environment variables so it prints candidates and scores
os.environ["SCORING_DEBUG_PRINT"] = "1"

# Adjust path to import src modules
sys.path.append(str(Path(".").resolve()))
from src.semantic_pipeline import SemanticPlatePipeline

print("--- START FULL PIPELINE DIAGNOSIS ---")
image_path = Path(r"C:\Users\PRAKRUTI M SHETTI\Downloads\fancy_number_plate_bfbc501f34.jpg")
if not image_path.exists():
    print("Error: Image path does not exist!")
    sys.exit(1)

# Initialize Pipeline
p = SemanticPlatePipeline(
    Path("data/kb/plate_templates.json"),
    model_path=Path("artifacts/semantic_lpr_async_fl.pt"),
)

# Load Image
img = Image.open(image_path)
print(f"Loaded image size: {img.size}")

# Run pick_best_plate
print("\n--- RUNNING pick_best_plate ---")
bbox, confidence, plate, plate_text, model_semantics, model_recon = p.pick_best_plate(img)
print("\n--- RESULTS ---")
print(f"Final Chosen BBox: {bbox}")
print(f"Final Chosen Plate Text: '{plate_text}'")
if model_semantics:
    print(f"Model Text Predicted: '{model_semantics.get('text', '')}'")
    print(f"Model Text Confidence: {model_semantics.get('text_confidence', 0.0)}")

print("--- END DIAGNOSIS ---")
