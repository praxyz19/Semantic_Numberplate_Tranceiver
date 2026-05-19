#!/usr/bin/env python3
"""
Runs the semantic transmission pipeline on all 13 test plates,
and generates a premium visual HTML verification dashboard showing
the exact extractions and HSRP reconstructions side-by-side.
"""
import csv
import json
from pathlib import Path
from difflib import SequenceMatcher

from src.semantic_pipeline import SemanticPlatePipeline, is_valid_plate_format


def generate_gallery():
    kb_path = Path('data/kb/plate_templates.json')
    model_path = Path('artifacts/semantic_lpr_async_fl.pt')
    
    print("Initializing pipeline...")
    pipe = SemanticPlatePipeline(kb_path=kb_path, model_path=model_path)

    # Load true labels
    labels_file = Path('data/smoke_synthetic/crop_labels.csv')
    if not labels_file.exists():
        print(f"ERROR: Missing labels file at {labels_file}")
        return

    true_labels = {}
    with open(labels_file, mode='r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            filename = Path(row['crop_path']).name
            true_labels[filename] = row['text'].strip().upper()

    test_crops = sorted(Path('data/smoke_synthetic/crops').glob('*.png'))
    if not test_crops:
        print("ERROR: No test crops found in data/smoke_synthetic/crops/")
        return

    plates_data = []
    exact_matches = 0
    valid_formats = 0
    total_sim = 0.0

    print(f"Processing {len(test_crops)} plates through semantic communication pipeline...")

    for i, crop_path in enumerate(test_crops, 1):
        try:
            data = crop_path.read_bytes()
            # Run at 18 dB SNR (standard noisy channel simulation)
            result = pipe.run(data, include_scene_context=False, snr_db=18.0, channel_noise=0.0)
            
            semantic_text = result.metrics.get('semantic_text', '?')
            ground_truth = true_labels.get(crop_path.name, '')
            
            is_exact = (semantic_text == ground_truth)
            is_valid = is_valid_plate_format(semantic_text)
            char_sim = SequenceMatcher(None, semantic_text, ground_truth).ratio()
            
            if is_exact:
                exact_matches += 1
            if is_valid:
                valid_formats += 1
            total_sim += char_sim

            plates_data.append({
                'index': i,
                'filename': crop_path.name,
                'ground_truth': ground_truth,
                'predicted': semantic_text,
                'is_exact': is_exact,
                'is_valid': is_valid,
                'similarity': round(char_sim * 100, 1),
                'input_image': result.extracted_plate,
                'semantic_map': result.received_semantic_map,
                'reconstructed_plate': result.reconstructed_plate,
                'compression': round(result.metrics.get('compression_ratio', 1.0), 1),
            })
            
            status = '✓ EXACT' if is_exact else ('⚠ VALID FORMAT' if is_valid else '✗ FAIL')
            print(f"[{i:2d}/{len(test_crops)}] {status} - GT: {ground_truth:10} | Pred: {semantic_text:10} | Sim: {char_sim:.2f}")

        except Exception as e:
            print(f"[{i:2d}/{len(test_crops)}] ERROR on {crop_path.name}: {e}")

    total = len(test_crops)
    exact_match_pct = (exact_matches / total) * 100
    format_valid_pct = (valid_formats / total) * 100
    avg_similarity = (total_sim / total) * 100

    # HTML Template
    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Semantic License Plate Verification Gallery</title>
    <link href="https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;800&family=JetBrains+Mono:wght@400;700&display=swap" rel="stylesheet">
    <style>
        :root {{
            --bg-primary: #0b0f19;
            --bg-secondary: #161e31;
            --bg-card: rgba(30, 41, 59, 0.45);
            --border-glow: rgba(56, 189, 248, 0.2);
            --accent-blue: #38bdf8;
            --accent-green: #34d399;
            --accent-yellow: #fbbf24;
            --accent-red: #f87171;
            --text-main: #f8fafc;
            --text-muted: #94a3b8;
        }}

        * {{
            box-sizing: border-box;
            margin: 0;
            padding: 0;
        }}

        body {{
            background-color: var(--bg-primary);
            color: var(--text-main);
            font-family: 'Outfit', sans-serif;
            padding: 2.5rem 1.5rem;
            line-height: 1.6;
        }}

        .container {{
            max-width: 1400px;
            margin: 0 auto;
        }}

        header {{
            text-align: center;
            margin-bottom: 3rem;
            background: linear-gradient(135deg, rgba(30, 41, 59, 0.5), rgba(15, 23, 42, 0.8));
            padding: 2rem;
            border-radius: 1.5rem;
            border: 1px solid var(--border-glow);
            box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
            backdrop-filter: blur(10px);
        }}

        header h1 {{
            font-size: 2.8rem;
            font-weight: 800;
            letter-spacing: -0.05em;
            background: linear-gradient(to right, #38bdf8, #818cf8);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 0.5rem;
        }}

        header p {{
            color: var(--text-muted);
            font-size: 1.1rem;
            max-width: 800px;
            margin: 0 auto 1.5rem auto;
        }}

        /* Metrics Widget */
        .metrics-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(240px, 1fr));
            gap: 1.5rem;
            margin-top: 1rem;
        }}

        .metric-card {{
            background: rgba(15, 23, 42, 0.6);
            border: 1px solid var(--border-glow);
            border-radius: 1rem;
            padding: 1.25rem;
            text-align: center;
            position: relative;
            overflow: hidden;
            transition: all 0.3s ease;
        }}

        .metric-card:hover {{
            transform: translateY(-4px);
            border-color: var(--accent-blue);
        }}

        .metric-card .value {{
            font-size: 2.5rem;
            font-weight: 800;
            margin-bottom: 0.25rem;
        }}

        .metric-card.exact .value {{ color: var(--accent-green); }}
        .metric-card.format .value {{ color: var(--accent-blue); }}
        .metric-card.sim .value {{ color: var(--accent-yellow); }}
        .metric-card.total .value {{ color: var(--text-main); }}

        .metric-card .label {{
            font-size: 0.9rem;
            font-weight: 600;
            color: var(--text-muted);
            text-transform: uppercase;
            letter-spacing: 0.05em;
        }}

        /* Plate Grid */
        .plates-grid {{
            display: grid;
            grid-template-columns: 1fr;
            gap: 2rem;
        }}

        @media (min-width: 900px) {{
            .plates-grid {{
                grid-template-columns: repeat(2, 1fr);
            }}
        }}

        .plate-card {{
            background: var(--bg-card);
            border: 1px solid var(--border-glow);
            border-radius: 1.25rem;
            padding: 1.5rem;
            display: flex;
            flex-direction: column;
            gap: 1.25rem;
            box-shadow: 0 4px 20px 0 rgba(0, 0, 0, 0.15);
            backdrop-filter: blur(12px);
            transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
            position: relative;
            overflow: hidden;
        }}

        .plate-card:hover {{
            transform: translateY(-6px);
            border-color: rgba(56, 189, 248, 0.45);
            box-shadow: 0 10px 30px 0 rgba(56, 189, 248, 0.12);
        }}

        /* Header Info inside Card */
        .card-header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            border-bottom: 1px solid rgba(148, 163, 184, 0.1);
            padding-bottom: 0.75rem;
        }}

        .card-title {{
            font-size: 1.15rem;
            font-weight: 600;
            color: var(--accent-blue);
        }}

        .status-badge {{
            padding: 0.25rem 0.75rem;
            border-radius: 9999px;
            font-size: 0.8rem;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 0.05em;
        }}

        .status-badge.exact {{
            background-color: rgba(52, 211, 153, 0.15);
            color: var(--accent-green);
            border: 1px solid rgba(52, 211, 153, 0.3);
        }}

        .status-badge.format {{
            background-color: rgba(251, 191, 36, 0.15);
            color: var(--accent-yellow);
            border: 1px solid rgba(251, 191, 36, 0.3);
        }}

        .status-badge.fail {{
            background-color: rgba(248, 113, 113, 0.15);
            color: var(--accent-red);
            border: 1px solid rgba(248, 113, 113, 0.3);
        }}

        /* Image Comparer Section */
        .image-comparer {{
            display: grid;
            grid-template-columns: repeat(3, 1fr);
            gap: 1rem;
            align-items: center;
            text-align: center;
        }}

        .img-box {{
            display: flex;
            flex-direction: column;
            align-items: center;
            gap: 0.5rem;
        }}

        .img-box span {{
            font-size: 0.8rem;
            font-weight: 600;
            color: var(--text-muted);
            text-transform: uppercase;
            letter-spacing: 0.05em;
        }}

        .img-box img {{
            width: 100%;
            max-width: 192px;
            height: auto;
            border-radius: 0.5rem;
            border: 2px solid rgba(148, 163, 184, 0.15);
            background-color: rgba(0, 0, 0, 0.4);
            image-rendering: pixelated;
        }}

        .img-box.reconstructed img {{
            border-color: var(--accent-blue);
            box-shadow: 0 0 10px rgba(56, 189, 248, 0.15);
        }}

        /* Alphanumeric Text Display */
        .text-display {{
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 1rem;
            background-color: rgba(15, 23, 42, 0.5);
            padding: 1rem;
            border-radius: 0.75rem;
            border: 1px solid rgba(148, 163, 184, 0.08);
        }}

        .text-box {{
            display: flex;
            flex-direction: column;
            gap: 0.25rem;
        }}

        .text-box .label {{
            font-size: 0.75rem;
            font-weight: 600;
            color: var(--text-muted);
            text-transform: uppercase;
            letter-spacing: 0.05em;
        }}

        .text-box .val {{
            font-family: 'JetBrains Mono', monospace;
            font-size: 1.35rem;
            font-weight: 700;
            letter-spacing: 0.05em;
            color: var(--text-main);
        }}

        .text-box.gt .val {{
            color: var(--text-main);
        }}

        .text-box.pred .val {{
            color: var(--accent-blue);
        }}

        .plate-card.exact-match-card .text-box.pred .val {{
            color: var(--accent-green);
        }}

        /* Secondary Metrics inside Card */
        .card-metrics {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            font-size: 0.85rem;
            color: var(--text-muted);
            background-color: rgba(15, 23, 42, 0.25);
            padding: 0.5rem 0.75rem;
            border-radius: 0.5rem;
        }}

        .card-metrics span strong {{
            color: var(--text-main);
        }}

        /* Glowing background blobs */
        .blob {{
            position: absolute;
            width: 150px;
            height: 150px;
            border-radius: 50%;
            background: radial-gradient(circle, rgba(56, 189, 248, 0.08) 0%, rgba(56, 189, 248, 0) 70%);
            top: -50px;
            right: -50px;
            pointer-events: none;
        }}

        .plate-card.exact-match-card .blob {{
            background: radial-gradient(circle, rgba(52, 211, 153, 0.08) 0%, rgba(52, 211, 153, 0) 70%);
        }}

        .plate-card.exact-match-card {{
            border-color: rgba(52, 211, 153, 0.2);
        }}

        .plate-card.exact-match-card:hover {{
            border-color: rgba(52, 211, 153, 0.55);
            box-shadow: 0 10px 30px 0 rgba(52, 211, 153, 0.12);
        }}
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>Semantic Plate Verification Gallery</h1>
            <p>
                Visual comparison demonstrating exact character extraction and HSRP visual reconstruction on the 13 held-out test crops at 18 dB SNR (noisy channel). Reconstructed images are built entirely on the receiver side using the shared Knowledge Base blended with deep-space symbols.
            </p>
            
            <div class="metrics-grid">
                <div class="metric-card exact">
                    <div class="value">{exact_match_pct:.1f}%</div>
                    <div class="label">Exact Match Rate</div>
                </div>
                <div class="metric-card format">
                    <div class="value">{format_valid_pct:.1f}%</div>
                    <div class="label">Format Compliance</div>
                </div>
                <div class="metric-card sim">
                    <div class="value">{avg_similarity:.1f}%</div>
                    <div class="label">Avg Char Similarity</div>
                </div>
                <div class="metric-card total">
                    <div class="value">{total}</div>
                    <div class="label">Total Test Plates</div>
                </div>
            </div>
        </header>

        <div class="plates-grid">
"""

    for item in plates_data:
        status_class = 'exact' if item['is_exact'] else ('format' if item['is_valid'] else 'fail')
        status_text = 'Exact Match' if item['is_exact'] else ('Valid Format (1-off)' if item['is_valid'] else 'Extraction Fail')
        card_exact_class = 'exact-match-card' if item['is_exact'] else ''

        html_content += f"""
            <!-- Plate Card #{item['index']} -->
            <div class="plate-card {card_exact_class}">
                <div class="blob"></div>
                <div class="card-header">
                    <span class="card-title">Plate {item['index']:02d}: {item['filename']}</span>
                    <span class="status-badge {status_class}">{status_text}</span>
                </div>
                
                <div class="image-comparer">
                    <div class="img-box">
                        <span>Original Crop</span>
                        <img src="{item['input_image']}" alt="Original Crop" />
                    </div>
                    <div class="img-box">
                        <span>Semantic Map</span>
                        <img src="{item['semantic_map']}" alt="Semantic Map" />
                    </div>
                    <div class="img-box reconstructed">
                        <span>Reconstructed HSRP</span>
                        <img src="{item['reconstructed_plate']}" alt="Reconstructed HSRP" />
                    </div>
                </div>

                <div class="text-display">
                    <div class="text-box gt">
                        <span class="label">Ground Truth</span>
                        <span class="val">{item['ground_truth']}</span>
                    </div>
                    <div class="text-box pred">
                        <span class="label">Extracted Alphanumeric</span>
                        <span class="val">{item['predicted']}</span>
                    </div>
                </div>

                <div class="card-metrics">
                    <span>Similarity: <strong>{item['similarity']}%</strong></span>
                    <span>Semantic Compression: <strong>{item['compression']}x</strong></span>
                </div>
            </div>
"""

    html_content += """
        </div>
    </div>
</body>
</html>
"""

    output_path = Path('verification_gallery.html')
    output_path.write_text(html_content, encoding='utf-8')
    print(f"\nSUCCESS! Verification gallery written to {output_path.resolve()}")
    print("You can open this file in any web browser to see the live extractions and premium visual reconstructions.")


if __name__ == '__main__':
    generate_gallery()
