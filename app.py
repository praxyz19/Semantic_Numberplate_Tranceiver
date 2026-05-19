from __future__ import annotations

import json
from pathlib import Path

from flask import Flask, jsonify, render_template, request

from src.semantic_pipeline import SemanticPlatePipeline


BASE_DIR = Path(__file__).resolve().parent

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 12 * 1024 * 1024

pipeline = SemanticPlatePipeline(
    BASE_DIR / "data" / "kb" / "plate_templates.json",
    model_path=BASE_DIR / "artifacts" / "semantic_lpr_async_fl.pt",
)


@app.get("/")
def index():
    return render_template("index.html")


@app.get("/verification")
def verification_dashboard():
    p = BASE_DIR / "verification_gallery.html"
    if p.exists():
        return p.read_text(encoding="utf-8")
    return "Verification gallery not generated yet. Run generate_verification_gallery.py first.", 404


@app.get("/api/health")
def health():
    return jsonify({"status": "ok"})


@app.get("/api/kb")
def knowledge_base():
    return jsonify(pipeline.knowledge_base.public_summary())


@app.post("/api/process")
def process_image():
    if "image" not in request.files:
        return jsonify({"error": "Upload an image field named 'image'."}), 400

    image_file = request.files["image"]
    if not image_file.filename:
        return jsonify({"error": "Choose an image first."}), 400

    include_scene_context = request.form.get("include_scene_context", "false") == "true"
    snr_db = float(request.form.get("snr_db", "18"))
    snr_db = max(-5.0, min(snr_db, 35.0))
    channel_noise = float(request.form.get("channel_noise", "0.00"))
    channel_noise = max(0.0, min(channel_noise, 1.0))
    multi_plate = request.form.get("multi_plate", "false") == "true"
    max_plates = int(request.form.get("max_plates", "2"))
    max_plates = max(1, min(max_plates, 6))

    try:
        if multi_plate:
            result = pipeline.run_multi(
                image_file.read(),
                include_scene_context=include_scene_context,
                snr_db=snr_db,
                channel_noise=channel_noise,
                max_plates=max_plates,
            )
        else:
            result = pipeline.run(
                image_file.read(),
                include_scene_context=include_scene_context,
                snr_db=snr_db,
                channel_noise=channel_noise,
            )
    except Exception as exc:  # GUI-friendly error while keeping the server alive.
        return jsonify({"error": str(exc)}), 500

    response = result.as_dict()
    if multi_plate:
        for plate in response.get("plates", []):
            plate["semantic_packet_pretty"] = json.dumps(
                {
                    "transmitted_semantics": plate.get("semantic_packet", {}),
                    "received_over_awgn_channel": plate.get("received_packet", {}),
                },
                indent=2,
            )
        return jsonify(response)

    response["semantic_packet_pretty"] = json.dumps(
        {
            "transmitted_semantics": response["semantic_packet"],
            "received_over_awgn_channel": response["received_packet"],
        },
        indent=2,
    )
    return jsonify(response)


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)
