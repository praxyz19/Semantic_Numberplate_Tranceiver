#!/usr/bin/env python3
"""Quick test of format-aware OCR"""

import json
from pathlib import Path
from src.semantic_pipeline import SemanticPlatePipeline
from src.semantic_pipeline import is_valid_plate_format

def main():
    kb_path = Path('data/kb/plate_templates.json')
    pipe = SemanticPlatePipeline(kb_path=kb_path, model_path=None)
    
    # Test on first few smoke crops
    test_crops = sorted(Path('data/smoke_synthetic/crops').glob('*.png'))[:5]
    
    print(f"\n{'='*80}")
    print("FORMAT-AWARE OCR TEST RESULTS")
    print(f"{'='*80}\n")
    
    for i, crop_path in enumerate(test_crops, 1):
        print(f"\n[Test {i}] {crop_path.name}")
        print("-" * 80)
        
        try:
            data = crop_path.read_bytes()
            result = pipe.run(data, include_scene_context=False, snr_db=18.0, channel_noise=0.0)
            
            ocr_text = result.metrics.get('ocr_text', '?')
            semantic_text = result.metrics.get('semantic_text', '?')
            char_acc = result.metrics.get('character_accuracy_percent', 0)
            
            print(f"  OCR Text:       {ocr_text}")
            print(f"  Semantic Text:  {semantic_text}")
            print(f"  Char Accuracy:  {char_acc:.1f}%")
            print(f"  Format Valid:   {'✓ YES' if pipe.is_valid_plate_format(semantic_text) else '✗ NO'}")
            is_valid = is_valid_plate_format(semantic_text)
            
        except Exception as e:
            print(f"  ERROR: {e}")
    
    print(f"\n{'='*80}\n")

    valid_count = 0
    for i, crop_path in enumerate(test_crops, 1):
        if is_valid:
            valid_count += 1
if __name__ == '__main__':
    main()
