from __future__ import annotations

import math

import numpy as np


PLATE_ALPHABET = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ?"


# ── helpers ────────────────────────────────────────────────────────────

def text_to_bits(text: str) -> np.ndarray:
    data = text.encode("ascii", errors="replace")
    bits: list[int] = []
    for byte in data:
        bits.extend(int(bit) for bit in f"{byte:08b}")
    return np.array(bits, dtype=np.int8)


def bits_to_plate_text(bits: np.ndarray, expected_len: int) -> str:
    chars = []
    usable = bits[: expected_len * 8]
    if len(usable) < expected_len * 8:
        usable = np.pad(usable, (0, expected_len * 8 - len(usable)))
    for idx in range(0, expected_len * 8, 8):
        byte = int("".join(str(int(bit)) for bit in usable[idx : idx + 8]), 2)
        char = chr(byte).upper()
        chars.append(char if char in PLATE_ALPHABET[:-1] else "?")
    return "".join(chars)


def bpsk_mod(bits: np.ndarray) -> np.ndarray:
    return 2.0 * bits.astype(np.float32) - 1.0


def bpsk_demod(signal: np.ndarray) -> np.ndarray:
    return (signal >= 0).astype(np.int8)


# ── BPSK text channel ─────────────────────────────────────────────────

def transmit_plate_text_awgn(
    text: str, snr_db: float, channel_noise: float
) -> tuple[list[dict], dict]:
    """Transmit plate characters using BPSK over AWGN.

    *channel_noise* is a UI-facing degradation knob that lowers effective Eb/N0.
    """
    text = text.upper()
    bits = text_to_bits(text)
    if bits.size == 0:
        return [], awgn_report(snr_db, channel_noise, 0.0, 0, 0)

    effective_ebn0_db = float(snr_db) - float(channel_noise) * 45.0
    ebn0_linear = 10.0 ** (effective_ebn0_db / 10.0)
    noise_variance = 1.0 / (2.0 * max(ebn0_linear, 1e-9))
    tx_signal = bpsk_mod(bits)
    noise = np.sqrt(noise_variance) * np.random.randn(*tx_signal.shape)
    rx_signal = tx_signal + noise
    rx_bits = bpsk_demod(rx_signal)
    rx_text = bits_to_plate_text(rx_bits, len(text))
    bit_errors = int(np.sum(bits != rx_bits[: bits.size]))
    ber = bit_errors / max(int(bits.size), 1)

    received = []
    symbol_errors = 0
    for idx, (tx_char, rx_char) in enumerate(zip(text, rx_text)):
        if tx_char != rx_char:
            symbol_errors += 1
        received.append(
            {
                "position": idx,
                "tx": tx_char,
                "rx": rx_char,
                "status": "ok" if tx_char == rx_char else "error",
            }
        )

    return received, awgn_report(
        snr_db, channel_noise, ber, bit_errors, symbol_errors,
        effective_ebn0_db, int(bits.size),
    )


def awgn_report(
    snr_db: float,
    channel_noise: float,
    ber: float,
    bit_errors: int,
    symbol_errors: int,
    effective_ebn0_db: float | None = None,
    bit_count: int = 0,
) -> dict:
    effective = float(snr_db) if effective_ebn0_db is None else effective_ebn0_db
    snr_linear = 10.0 ** (float(snr_db) / 10.0)
    ebn0_linear = 10.0 ** (effective / 10.0)
    theoretical_ber = 0.5 * math.erfc(math.sqrt(max(ebn0_linear, 1e-12)))
    return {
        "mode": "BPSK_AWGN",
        "equation": "y = x + n, n ~ N(0, N0/2)",
        "snr_db": round(float(snr_db), 2),
        "channel_noise": round(float(channel_noise), 3),
        "effective_ebn0_db": round(effective, 2),
        "snr_linear": round(snr_linear, 5),
        "channel_capacity_bps_per_hz": round(math.log2(1.0 + snr_linear), 4),
        "theoretical_ber_bpsk": theoretical_ber,
        "measured_ber": round(float(ber), 6),
        "bit_errors": bit_errors,
        "bit_count": bit_count,
        "symbol_errors": symbol_errors,
    }


# ── Neural latent AWGN channel (numpy, for pipeline inference) ─────────

def apply_awgn_to_tensor_np(
    signal: np.ndarray, snr_db: float, channel_noise: float = 0.0,
) -> tuple[np.ndarray, dict]:
    """Add AWGN to a numpy array of channel symbols.

    Returns (noisy_signal, report_dict).
    """
    effective_snr_db = float(snr_db) - float(channel_noise) * 35.0
    snr_linear = 10.0 ** (effective_snr_db / 10.0)
    signal_power = float(np.mean(signal ** 2))
    noise_power = signal_power / max(snr_linear, 1e-9)
    noise = np.random.randn(*signal.shape).astype(np.float32) * math.sqrt(
        max(noise_power, 0.0)
    )
    received = signal + noise
    mse = float(np.mean((signal - received) ** 2))

    # cosine similarity between tx and rx
    tx_flat = signal.reshape(-1).astype(np.float64)
    rx_flat = received.reshape(-1).astype(np.float64)
    denom = float(np.linalg.norm(tx_flat) * np.linalg.norm(rx_flat))
    cosine_sim = float(np.dot(tx_flat, rx_flat) / max(denom, 1e-12))

    return received, {
        "effective_snr_db": round(effective_snr_db, 2),
        "signal_power": round(signal_power, 6),
        "noise_power": round(noise_power, 6),
        "mse": round(mse, 8),
        "cosine_similarity": round(max(0.0, min(1.0, cosine_sim)), 6),
    }


# ── Semantic map AWGN (for the image-domain compact map) ──────────────

def transmit_semantic_map_awgn(
    semantic_map_arr: np.ndarray, snr_db: float, channel_noise: float = 0.0,
) -> tuple[np.ndarray, dict]:
    """Add AWGN to a uint8 semantic map image (H, W, C).

    Returns (received_uint8, report).
    """
    signal = semantic_map_arr.astype(np.float32) / 255.0
    effective_snr_db = float(snr_db) - float(channel_noise) * 35.0
    snr_linear = 10.0 ** (effective_snr_db / 10.0)
    signal_power = float(np.mean(signal ** 2))
    noise_power = signal_power / max(snr_linear, 1e-9)
    noise = np.random.randn(*signal.shape).astype(np.float32) * math.sqrt(
        max(noise_power, 0.0)
    )
    received = np.clip(signal + noise, 0.0, 1.0)
    mse = float(np.mean((signal - received) ** 2))
    return (received * 255.0).astype(np.uint8), {
        "noise_power": noise_power,
        "map_mse": mse,
        "effective_snr_db": effective_snr_db,
    }
