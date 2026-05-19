#!/usr/bin/env python3
"""
Corrected evaluation script that loads ground truth from crop_labels.csv
and runs accuracy metrics on the specified checkpoint.
"""
import argparse
import csv
import json
from pathlib import Path
from difflib import SequenceMatcher

from src.semantic_pipeline import SemanticPlatePipeline, is_valid_plate_format


def run_eval(model_path: Path, out_file: Path | None = None):
    kb_path = Path('data/kb/plate_templates.json')
    
    # Initialize pipeline
    # If model_path exists, load it, otherwise None
    model_to_load = model_path if model_path and model_path.exists() else None
    pipe = SemanticPlatePipeline(kb_path=kb_path, model_path=model_to_load)

    # Load true labels from crop_labels.csv
    labels_file = Path('data/smoke_synthetic/crop_labels.csv')
    if not labels_file.exists():
        print(f"ERROR: Missing labels file at {labels_file}")
        return None

    true_labels = {}
    with open(labels_file, mode='r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            # key is the filename (e.g. 000000_0.png)
            filename = Path(row['crop_path']).name
            true_labels[filename] = row['text'].strip().upper()

    test_crops = sorted(Path('data/smoke_synthetic/crops').glob('*.png'))
    if not test_crops:
        print("ERROR: No test crops found in data/smoke_synthetic/crops/")
        return None

    results = {'exact_match': 0, 'format_valid': 0, 'char_similarity': []}

    print(f"\n{'='*80}")
    print(f"EVALUATING MODEL: {model_path if model_to_load else 'Classical OCR Only'}")
    print(f"{'='*80}\n")

    for i, crop_path in enumerate(test_crops, 1):
        try:
            data = crop_path.read_bytes()
            result = pipe.run(data, include_scene_context=False, snr_db=18.0, channel_noise=0.0)
            
            semantic_text = result.metrics.get('semantic_text', '?')
            is_valid = is_valid_plate_format(semantic_text)

            # Get the correct ground truth from csv
            ground_truth = true_labels.get(crop_path.name)

            char_sim = 0.0
            if ground_truth:
                char_sim = SequenceMatcher(None, semantic_text, ground_truth).ratio()
                results['char_similarity'].append(char_sim)

            is_exact = ground_truth and semantic_text == ground_truth
            if is_exact:
                results['exact_match'] += 1
            if is_valid:
                results['format_valid'] += 1

            status_symbol = '✓' if is_exact else ('⚠' if is_valid else '✗')
            gt_str = f"GT: {ground_truth}" if ground_truth else "GT: Unknown"
            print(f"[{i:2d}/{len(test_crops)}] {status_symbol} Predicted: {semantic_text:15} | {gt_str:15} | Sim: {char_sim:.2f}")

        except Exception as e:
            print(f"[{i:2d}/{len(test_crops)}] ERROR on {crop_path.name}: {e}")

    avg_char_sim = sum(results['char_similarity']) / len(results['char_similarity']) if results['char_similarity'] else 0.0
    exact_match_pct = (results['exact_match'] / len(test_crops)) * 100
    format_valid_pct = (results['format_valid'] / len(test_crops)) * 100

    out = {
        'exact_match_pct': exact_match_pct,
        'format_valid_pct': format_valid_pct,
        'avg_char_sim': avg_char_sim,
        'total_samples': len(test_crops)
    }

    print('\n' + '='*80)
    print('RESULTS SUMMARY')
    print('='*80)
    print(f"  Model Path:          {model_path if model_to_load else 'Classical OCR Only'}")
    print(f"  Total test samples:  {out['total_samples']}")
    print(f"  Exact match:         {results['exact_match']}/{out['total_samples']} ({out['exact_match_pct']:.1f}%)")
    print(f"  Format valid:        {results['format_valid']}/{out['total_samples']} ({out['format_valid_pct']:.1f}%)")
    print(f"  Avg char similarity: {out['avg_char_sim']:.3f} ({out['avg_char_sim']*100:.1f}%)")
    print('='*80 + '\n')

    if out_file:
        out_file.write_text(json.dumps(out, indent=2))

    return out


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='artifacts_scratch/semantic_lpr_async_fl.pt')
    parser.add_argument('--out', default='quick_eval_results_correct.json')
    args = parser.parse_args()
    model_path = Path(args.model)
    out_file = Path(args.out)
    run_eval(model_path, out_file)
