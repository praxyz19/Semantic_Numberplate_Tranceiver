from __future__ import annotations

import math

import torch
from torch import nn
import torch.nn.functional as F

from .dataset import CHARSET, MAX_PLATE_LEN, PAD_TOKEN


# ---------------------------------------------------------------------------
# AWGN Channel Layer (differentiable, used inside the model during training)
# ---------------------------------------------------------------------------

class AWGNChannelLayer(nn.Module):
    """Additive White Gaussian Noise channel.

    During training the SNR is sampled uniformly from *snr_range_db* so the
    model learns to be robust across channel conditions.  At inference the
    caller can fix a specific SNR.
    """

    def __init__(self, snr_range_db: tuple[float, float] = (0.0, 20.0)):
        super().__init__()
        self.snr_low, self.snr_high = snr_range_db
        self._fixed_snr: float | None = None

    def set_snr(self, snr_db: float | None) -> None:
        self._fixed_snr = snr_db

    def forward(self, signal: torch.Tensor) -> tuple[torch.Tensor, float]:
        """Apply AWGN to *signal* ([B, D]).

        Returns (noisy_signal, noise_power).
        """
        if self._fixed_snr is not None:
            snr_db = self._fixed_snr
        elif self.training:
            snr_db = float(torch.empty(1).uniform_(self.snr_low, self.snr_high).item())
        else:
            snr_db = (self.snr_low + self.snr_high) / 2.0

        signal_power = torch.mean(signal ** 2)
        snr_linear = 10.0 ** (snr_db / 10.0)
        noise_power = signal_power / max(snr_linear, 1e-9)

        noise = torch.randn_like(signal) * torch.sqrt(noise_power.clamp(min=1e-12))
        noisy = signal + noise
        return noisy, float(noise_power.detach())


# ---------------------------------------------------------------------------
# Power-normalisation helper
# ---------------------------------------------------------------------------

class PowerNormalize(nn.Module):
    """Normalise transmit symbols so average power ≈ 1."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        power = torch.mean(x ** 2, dim=-1, keepdim=True).clamp(min=1e-8)
        return x / torch.sqrt(power)


# ---------------------------------------------------------------------------
# Semantic LPR Network — autoencoder with channel
# ---------------------------------------------------------------------------

class SemanticLPRNet(nn.Module):
    """End-to-end semantic communication model for licence-plate images.

    Architecture
    ============
    Encoder (transmitter):
        Conv backbone  →  latent feature map  →  flatten  →  channel symbols
    Channel:
        Power-normalise  →  AWGN  →  received symbols
    Decoder (receiver):
        Reshape received symbols  →  ConvTranspose backbone  →  reconstructed image
    Text head:
        From latent vector predict plate characters.
    """

    def __init__(
        self,
        latent_dim: int = 128,
        channel_dim: int = 64,
        max_len: int = MAX_PLATE_LEN,
        num_chars: int = len(CHARSET) + 1,
        snr_range_db: tuple[float, float] = (0.0, 20.0),
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.channel_dim = channel_dim
        self.max_len = max_len
        self.num_chars = num_chars

        # --- Encoder (semantic extractor / transmitter) ---
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2, padding=1),   # → 32×32×96
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(32, 64, 3, stride=2, padding=1),  # → 64×16×48
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(64, 128, 3, stride=2, padding=1), # → 128×8×24
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(128, 256, 3, stride=2, padding=1),# → 256×4×12
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
        )

        # Flatten encoder output and project to latent
        self._enc_flat = 256 * 4 * 12  # 12288
        self.to_latent = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self._enc_flat, latent_dim),
            nn.ReLU(inplace=True),
        )

        # --- Channel encoder / decoder ---
        self.channel_enc = nn.Sequential(
            nn.Linear(latent_dim, channel_dim),
        )
        self.power_norm = PowerNormalize()
        self.channel = AWGNChannelLayer(snr_range_db)
        self.channel_dec = nn.Sequential(
            nn.Linear(channel_dim, latent_dim),
            nn.ReLU(inplace=True),
        )

        # --- Decoder (receiver / reconstructor) ---
        self.from_latent = nn.Sequential(
            nn.Linear(latent_dim, self._enc_flat),
            nn.ReLU(inplace=True),
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1),  # → 128×8×24
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),

            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),   # → 64×16×48
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),

            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),    # → 32×32×96
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2, inplace=True),

            nn.ConvTranspose2d(32, 3, 4, stride=2, padding=1),     # → 3×64×192
            nn.Sigmoid(),
        )

        # --- Text classification head ---
        self.text_head = nn.Linear(latent_dim, max_len * num_chars)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, x: torch.Tensor, snr_db: float | None = None) -> dict[str, torch.Tensor]:
        """Full forward pass: encode → channel → decode.

        Parameters
        ----------
        x : [B, 3, 64, 192]  plate crop tensor in [0, 1]
        snr_db : override SNR (dB).  None = sample during train / use mid during eval.
        """
        if snr_db is not None:
            self.channel.set_snr(snr_db)

        # Encode
        features = self.encoder(x)                       # [B, 256, 4, 12]
        latent = self.to_latent(features)                # [B, latent_dim]

        # Text head (from clean latent)
        text_logits = self.text_head(latent).view(-1, self.max_len, self.num_chars)

        # Channel
        tx_symbols = self.channel_enc(latent)            # [B, channel_dim]
        tx_symbols = self.power_norm(tx_symbols)
        rx_symbols, noise_power = self.channel(tx_symbols)

        # Decode
        rx_latent = self.channel_dec(rx_symbols)         # [B, latent_dim]
        decoded_flat = self.from_latent(rx_latent)       # [B, 256*4*12]
        reconstruction = self.decoder(decoded_flat.view(-1, 256, 4, 12))

        # Reset fixed SNR so training samples random again
        if snr_db is not None:
            self.channel.set_snr(None)

        return {
            "latent": latent,
            "tx_symbols": tx_symbols,
            "rx_symbols": rx_symbols,
            "rx_latent": rx_latent,
            "text_logits": text_logits,
            "reconstruction": reconstruction,
            "noise_power": noise_power,
        }

    # ------------------------------------------------------------------
    # Inference helpers (used by the pipeline)
    # ------------------------------------------------------------------

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encoder only — returns latent vector."""
        features = self.encoder(x)
        return self.to_latent(features)

    def encode_to_symbols(self, x: torch.Tensor) -> torch.Tensor:
        """Encode + channel-encode (no noise)."""
        latent = self.encode(x)
        return self.power_norm(self.channel_enc(latent))

    def decode_from_symbols(self, rx_symbols: torch.Tensor) -> torch.Tensor:
        """Channel-decode + image-decode."""
        rx_latent = self.channel_dec(rx_symbols)
        decoded_flat = self.from_latent(rx_latent)
        return self.decoder(decoded_flat.view(-1, 256, 4, 12))

    def latent_feature_map(self, x: torch.Tensor) -> torch.Tensor:
        """Return the 256×4×12 feature map before flattening (for visualisation)."""
        return self.encoder(x)


# ---------------------------------------------------------------------------
# Loss function
# ---------------------------------------------------------------------------

def edge_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    # Simple Sobel filters for edge detection
    kx = torch.tensor([[1, 0, -1], [2, 0, -2], [1, 0, -1]], dtype=torch.float32, device=pred.device).view(1, 1, 3, 3)
    ky = torch.tensor([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype=torch.float32, device=pred.device).view(1, 1, 3, 3)
    
    # Apply to each channel independently
    pred_edges_x = F.conv2d(pred.view(-1, 1, pred.shape[2], pred.shape[3]), kx, padding=1)
    pred_edges_y = F.conv2d(pred.view(-1, 1, pred.shape[2], pred.shape[3]), ky, padding=1)
    target_edges_x = F.conv2d(target.view(-1, 1, target.shape[2], target.shape[3]), kx, padding=1)
    target_edges_y = F.conv2d(target.view(-1, 1, target.shape[2], target.shape[3]), ky, padding=1)
    
    loss_x = F.l1_loss(pred_edges_x, target_edges_x)
    loss_y = F.l1_loss(pred_edges_y, target_edges_y)
    return loss_x + loss_y

def semantic_lpr_loss(
    outputs: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor],
    recon_weight: float = 1.0,
    text_weight: float = 1.0,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Combined loss: reconstruction (L1 + Edge) + text classification (CE)."""
    # L1 Loss for better sharpness than MSE
    l1_recon = F.l1_loss(outputs["reconstruction"], batch["target_image"])
    
    # Edge loss to enforce sharp character boundaries
    e_loss = edge_loss(outputs["reconstruction"], batch["target_image"])
    
    recon_loss = l1_recon + 0.5 * e_loss

    text_logits = outputs["text_logits"].reshape(-1, len(CHARSET) + 1)
    text_target = batch["text"].reshape(-1)
    text_loss = F.cross_entropy(text_logits, text_target, ignore_index=PAD_TOKEN)

    loss = recon_weight * recon_loss + text_weight * text_loss
    return loss, {
        "loss": float(loss.detach()),
        "text_loss": float(text_loss.detach()),
        "recon_loss": float(recon_loss.detach()),
    }

