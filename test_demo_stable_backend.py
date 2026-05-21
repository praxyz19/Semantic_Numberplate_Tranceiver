from pathlib import Path

import numpy as np

from src.channel import transmit_plate_text_awgn
from src.semantic_pipeline import (
    SemanticPlatePipeline,
    build_plate_semantic_map,
    semantic_similarity_metrics,
)


def test_zero_noise_preserves_ordered_sequence():
    np.random.seed(7)
    text = "MH12AB1234"
    rx_chars, report = transmit_plate_text_awgn(text, snr_db=18.0, channel_noise=0.0)
    rx_text = "".join(item["rx"] for item in rx_chars)

    assert rx_text == text
    assert len(rx_text) == len(text)
    assert report["measured_ber"] == 0.0
    assert report["channel_capacity_bps_per_hz"] > 0
    assert "shannon_min_snr_db" in report


def test_high_noise_degrades_but_keeps_length_and_confusions():
    np.random.seed(3)
    text = "MH12AB1234"
    rx_chars, report = transmit_plate_text_awgn(text, snr_db=18.0, channel_noise=1.0)
    rx_text = "".join(item["rx"] for item in rx_chars)

    assert len(rx_text) == len(text)
    assert rx_text != text
    assert report["symbol_errors"] > 0

    np.random.seed(5)
    rx_chars, _ = transmit_plate_text_awgn("SOBNM", snr_db=18.0, channel_noise=1.0)
    pairs = {(item["tx"], item["rx"]) for item in rx_chars if item["tx"] != item["rx"]}
    assert ("S", "5") in pairs or ("O", "0") in pairs or ("B", "8") in pairs or ("N", "M") in pairs or ("M", "N") in pairs


def test_packet_transmission_and_reconstruction_use_received_text():
    np.random.seed(11)
    pipe = SemanticPlatePipeline(Path("data/kb/plate_templates.json"), model_path=None)
    text = "MH12AB1234"
    plate = pipe.knowledge_base.render_plate(text, (384, 96), "ind_private_white")
    source = plate.copy()
    packet = pipe.extract_semantics(
        source,
        plate,
        (0, 0, plate.width, plate.height),
        0.99,
        text,
        include_scene_context=False,
    )
    compact_map = build_plate_semantic_map(plate)
    received, received_map = pipe.transmit(packet, compact_map, snr_db=18.0, channel_noise=0.0)
    reconstructed = pipe.reconstruct_plate(received, received_map)
    metrics = semantic_similarity_metrics(plate, reconstructed, packet, received, compact_map, received_map)

    assert packet["plate_sequence"]["text"] == text
    assert received["received_sequence"]["repaired_text"] == text
    assert metrics["character_accuracy"] == 1.0
