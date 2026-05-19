#!/usr/bin/env python3
"""
Comprehensive accuracy evaluation on held-out validation set.
Runs after training to measure:
- Exact match %
- Format validity %
- Character-level Levenshtein similarity
"""

import json
from pathlib import Path
from collections import defaultdict
from src.semantic_pipeline import SemanticPlatePipeline, is_valid_plate_format
from difflib import SequenceMatcher

def evaluate_accuracy():
    kb_path = Path('data/kb/plate_templates.json')
    pipe = SemanticPlatePipeline(kb_path=kb_path, model_path=None)
    
    # Collect test crops from smoke_synthetic
    test_crops = sorted(Path('data/smoke_synthetic/crops').glob('*.png'))
    if not test_crops:
        print("ERROR: No test crops found in data/smoke_synthetic/crops/")
        return
    
    print(f"\n{'='*80}")
    print("COMPREHENSIVE OCR ACCURACY EVALUATION")
    print(f"{'='*80}\n")
    print(f"Test set: {len(test_crops)} crops from smoke_synthetic")
    print(f"{'='*80}\n")
    
    results = {
        'exact_match': 0,
        'format_valid': 0,
        'char_similarity': []
    }
    
    for i, crop_path in enumerate(test_crops, 1):
        # Extract ground truth from filename (format: <plate_id>_<text>.png)
        # If no ground truth in filename, try to extract from nearby labels file
        try:
            data = crop_path.read_bytes()
            result = pipe.run(data, include_scene_context=False, snr_db=18.0, channel_noise=0.0)
            
            semantic_text = result.metrics.get('semantic_text', '?')
            is_valid = is_valid_plate_format(semantic_text)
            
            # Try to get ground truth
            ground_truth = None
            if '_' in crop_path.stem:
                # Extract from filename: <id>_<text>
                parts = crop_path.stem.split('_')
                if len(parts) >= 2:
                    ground_truth = '_'.join(parts[1:]).upper()
            
            # Calculate character similarity if we have ground truth
            char_sim = 0.0
            if ground_truth:
                char_sim = SequenceMatcher(None, semantic_text, ground_truth).ratio()
                results['char_similarity'].append(char_sim)
            
            # Check for exact match
            if ground_truth and semantic_text == ground_truth:
                results['exact_match'] += 1
            
            # Check format validity
            if is_valid:
                results['format_valid'] += 1
            
            # Print progress every 10 samples
            if i % 10 == 0 or i == len(test_crops):
                status = f"{'✓' if is_valid else '✗'} {semantic_text:15} | Sim: {char_sim:.2f}"
                print(f"[{i:3d}/{len(test_crops)}] {status}")
        
        except Exception as e:
            print(f"[{i:3d}/{len(test_crops)}] ERROR: {crop_path.name} - {e}")
    
    # Calculate statistics
    avg_char_sim = sum(results['char_similarity']) / len(results['char_similarity']) if results['char_similarity'] else 0.0
    exact_match_pct = (results['exact_match'] / len(test_crops)) * 100
    format_valid_pct = (results['format_valid'] / len(test_crops)) * 100
    
    print(f"\n{'='*80}")
    print("RESULTS SUMMARY")
    print(f"{'='*80}\n")
    print(f"  Total test samples:       {len(test_crops)}")
    print(f"  Exact match:              {results['exact_match']}/{len(test_crops)} ({exact_match_pct:.1f}%)")
    print(f"  Format valid:             {results['format_valid']}/{len(test_crops)} ({format_valid_pct:.1f}%)")
    print(f"  Avg char similarity:      {avg_char_sim:.3f} ({avg_char_sim*100:.1f}%)")
    print(f"\n{'='*80}\n")
    
    # Compare with previous results if available
    results_file = Path('evaluation_results.json')
    if results_file.exists():
        prev_results = json.loads(results_file.read_text())
        print("IMPROVEMENT OVER PREVIOUS RUN:")
        print(f"  Exact match: {prev_results['exact_match_pct']:.1f}% → {exact_match_pct:.1f}% ({exact_match_pct - prev_results['exact_match_pct']:+.1f}%)")
        print(f"  Format valid: {prev_results['format_valid_pct']:.1f}% → {format_valid_pct:.1f}% ({format_valid_pct - prev_results['format_valid_pct']:+.1f}%)")
        print(f"  Char similarity: {prev_results['avg_char_sim']:.3f} → {avg_char_sim:.3f} ({avg_char_sim - prev_results['avg_char_sim']:+.3f})")
        print(f"\n{'='*80}\n")
    
    # Save results
    results_file.write_text(json.dumps({
        'exact_match_pct': exact_match_pct,
        'format_valid_pct': format_valid_pct,
        'avg_char_sim': avg_char_sim,
        'total_samples': len(test_crops)
    }, indent=2))
    
    return {
        'exact_match_pct': exact_match_pct,
        'format_valid_pct': format_valid_pct,
        'avg_char_sim': avg_char_sim
    }

if __name__ == '__main__':
    evaluate_accuracy()
