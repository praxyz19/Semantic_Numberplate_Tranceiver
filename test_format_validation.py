#!/usr/bin/env python3
"""Test format-aware OCR validation"""

from pathlib import Path
from src.semantic_pipeline import SemanticPlatePipeline, is_valid_plate_format

def test_ocr():
    kb_path = Path('data/kb/plate_templates.json')
    pipe = SemanticPlatePipeline(kb_path=kb_path, model_path=None)
    
    test_crops = sorted(Path('data/smoke_synthetic/crops').glob('*.png'))[:5]
    
    print(f"\n{'='*80}")
    print("FORMAT-AWARE OCR VALIDATION TEST")
    print(f"{'='*80}\n")
    
    valid_count = 0
    for i, crop_path in enumerate(test_crops, 1):
        print(f"[Test {i}] {crop_path.name}")
        try:
            data = crop_path.read_bytes()
            result = pipe.run(data, include_scene_context=False, snr_db=18.0, channel_noise=0.0)
            
            ocr_text = result.metrics.get('ocr_text', '?')
            semantic_text = result.metrics.get('semantic_text', '?')
            is_valid = is_valid_plate_format(semantic_text)
            
            status = '✓ VALID' if is_valid else '✗ INVALID'
            print(f"  OCR: {ocr_text:15} → {semantic_text:15} [{status}]")
            
            if is_valid:
                valid_count += 1
                
        except Exception as e:
            print(f"  ERROR: {e}")
    
    print(f"\n{'='*80}")
    print(f"Result: {valid_count}/{len(test_crops)} have valid format")
    print(f"{'='*80}\n")

if __name__ == '__main__':
    test_ocr()
