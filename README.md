# Semantic Licence Plate Communication Prototype

This project implements a local GUI for the research-paper flow:

`input image -> licence plate semantic extraction -> semantic packet transmission -> receiver shared knowledge base -> reconstructed plate/scene`.

It also includes a synthetic multi-number-plate dataset generator and an asynchronous federated learning trainer for a small semantic LPR model. The first version is intentionally dependency-light so it can run locally on CPU.

## What Is Implemented

- Flask GUI with input, extraction, transmission, receiver KB, and reconstruction panels.
- Classical plate-region extractor using Pillow/NumPy, with optional OCR if Tesseract is installed.
- Semantic packet containing only essential plate meaning: character sequence, character positions, minimal geometry, KB template ID, and optional compact scene context.
- BPSK + AWGN channel simulation with receiver SNR and channel-noise controls using `y = x + n`.
- Bandwidth optimization metrics and reconstruction similarity metrics.
- Semantic extraction panel renders a character/position/confidence map instead of retransmitting the plate crop.
- Indian registration prior corrects common confusions such as `TM -> TN/TS`, `O -> 0`, `B -> 8`, `S -> 5` based on expected plate format.
- Number-plate semantic image map uses compact luminance, edge-structure, and text-mask channels inspired by the semantic image transmission projects.
- Receiver-side shared knowledge base in `data/kb/plate_templates.json`.
- PyTorch `SemanticLPRNet`: semantic encoder, latent vector, decoder reconstruction head, and fixed-position plate text head.
- Asynchronous FL simulation with client-local updates and staleness-weighted server aggregation.
- Synthetic multi-plate dataset generator with scene labels and crop labels.

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Run the GUI:

```powershell
python app.py
```

Open:

```text
http://127.0.0.1:5000
```

In the GUI:

- Increase **Receiver SNR** for cleaner semantic symbol delivery.
- Decrease **Receiver SNR** to see character corruption and lower semantic similarity.
- Increase **Channel noise** to corrupt the transmitted character bits.
- The transmitted data is intentionally compact: it sends the character sequence, reconstruction metadata, and a low-resolution semantic image map, not the full input image.

## Generate The Training Dataset

This creates scenes with 1 to 3 licence plates and also saves plate crops for model training.

```powershell
python scripts/generate_synthetic_dataset.py --output data/synthetic_plates --samples 1200 --clients 4 --max-plates 3
```

Generated files:

- `data/synthetic_plates/images/`: full synthetic traffic-like scenes.
- `data/synthetic_plates/crops/`: extracted plate crops.
- `data/synthetic_plates/labels.csv`: scene image path, plate text, bbox, and client ID.
- `data/synthetic_plates/crop_labels.csv`: crop path, plate text, and client ID.

## Train The Asynchronous FL Model

```powershell
python scripts/train_async_fl.py --dataset data/synthetic_plates --clients 4 --rounds 5 --local-epochs 1 --batch-size 32
```

The trained shared semantic model is saved to:

```text
artifacts/semantic_lpr_async_fl.pt
```

Restart `python app.py` after training. The GUI will automatically load that artifact and include neural semantic latents in the transmitted packet.

## Notes On Reconstruction Quality

A receiver cannot perfectly recreate an arbitrary full input image from only symbolic plate semantics. To make the demo honest and useful:

- Plate reconstruction is semantic: the receiver renders a clean, readable plate from the character sequence and shared KB template.
- Full-scene reconstruction is approximate unless you enable “Include compact scene context” in the GUI.
- The trained FL model improves the shared semantic representation, but exact pixel matching requires transmitting more visual context.
- For real roadside blur, install OCR support and train the FL model on a labelled plate dataset. The app tries EasyOCR first, then Tesseract, then a lightweight local template fallback.

## Requirements For Proper Training

For the strong backend version, provide or install:

- A labelled number-plate dataset with images and plate text labels.
- Preferably bounding boxes for each plate. If boxes are unavailable, use cropped plate images.
- Python packages from `requirements.txt`.
- Optional OCR packages from `optional-requirements.txt`:

```powershell
pip install -r optional-requirements.txt
```

Recommended dataset CSV format:

```text
image_path,text,x1,y1,x2,y2,client_id
images/img001.jpg,TN88F4089,12,20,280,150,0
```

For federated learning, split `client_id` by camera/device/location. The trainer will treat each client as one asynchronous FL participant.

## Current Training Status

I generated a small mixed Indian-style training set at:

```text
data/indian_plate_train
```

and trained the asynchronous FL model to:

```text
artifacts/semantic_lpr_async_fl.pt
```

This verifies the FL code path and recognizer integration. It is still not a substitute for your real labelled dataset; presentation-grade recognition needs real Indian plate images with blur, perspective, yellow/white plates, and correct labels.

## Project Layout

```text
app.py                         Flask backend
templates/index.html           GUI
static/css/style.css           GUI styling
static/js/app.js               GUI API calls
src/semantic_pipeline.py       extraction, packetization, transmission, reconstruction
src/kb.py                      receiver knowledge base renderer
src/model.py                   semantic LPR neural model
src/dataset.py                 plate crop dataset utilities
src/fl_async.py                asynchronous federated trainer
scripts/generate_synthetic_dataset.py
scripts/train_async_fl.py
data/kb/plate_templates.json
```
