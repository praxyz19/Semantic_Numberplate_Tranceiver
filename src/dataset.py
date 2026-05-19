from __future__ import annotations

import csv
import hashlib
import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageEnhance, ImageFilter
from torch.utils.data import Dataset


CHARSET = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"
PAD_TOKEN = len(CHARSET)
MAX_PLATE_LEN = 10
IMAGE_SIZE = (192, 64)  # (width, height)

# CCPD helpers
CCPD_ALPHABETS = list("ABCDEFGHJKLMNPQRSTUVWXYZ") + ["O"]
CCPD_ADS = list("ABCDEFGHJKLMNPQRSTUVWXYZ") + list("0123456789") + ["O"]


# ── text encoding ─────────────────────────────────────────────────────

def encode_text(text: str, max_len: int = MAX_PLATE_LEN) -> torch.Tensor:
    values = [CHARSET.index(ch) if ch in CHARSET else PAD_TOKEN for ch in text.upper()[:max_len]]
    values += [PAD_TOKEN] * (max_len - len(values))
    return torch.tensor(values, dtype=torch.long)


def decode_text(values: torch.Tensor) -> str:
    chars = []
    for idx in values.detach().cpu().tolist():
        if idx == PAD_TOKEN:
            continue
        if 0 <= idx < len(CHARSET):
            chars.append(CHARSET[idx])
    return "".join(chars)


# ── image processing ──────────────────────────────────────────────────

def pil_to_tensor(image: Image.Image) -> torch.Tensor:
    image = image.convert("RGB").resize(IMAGE_SIZE, Image.Resampling.BILINEAR)
    arr = np.asarray(image, dtype=np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1)  # [3, H, W]


def augment_plate(image: Image.Image, rng: random.Random | None = None) -> Image.Image:
    """Apply random augmentations to make training more robust."""
    if rng is None:
        rng = random.Random()

    # Random blur
    if rng.random() < 0.35:
        radius = rng.uniform(0.3, 1.8)
        image = image.filter(ImageFilter.GaussianBlur(radius=radius))

    # Random brightness / contrast
    if rng.random() < 0.4:
        factor = rng.uniform(0.7, 1.3)
        image = ImageEnhance.Brightness(image).enhance(factor)
    if rng.random() < 0.4:
        factor = rng.uniform(0.7, 1.3)
        image = ImageEnhance.Contrast(image).enhance(factor)

    # Random rotation (small)
    if rng.random() < 0.3:
        angle = rng.uniform(-5, 5)
        image = image.rotate(angle, expand=False, fillcolor=(220, 225, 226))

    # Random color jitter
    if rng.random() < 0.25:
        factor = rng.uniform(0.8, 1.2)
        image = ImageEnhance.Color(image).enhance(factor)

    return image


# ── datasets ──────────────────────────────────────────────────────────

class PlateCropDataset(Dataset):
    def __init__(self, root: str | Path, labels_file: str = "crop_labels.csv", client_id: int | None = None, augment: bool = False):
        self.root = Path(root)
        self.augment = augment
        labels_path = self.root / labels_file
        if not labels_path.exists():
            raise FileNotFoundError(f"Missing {labels_path}. Run scripts/generate_synthetic_dataset.py first.")

        with labels_path.open("r", newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))

        if client_id is not None:
            rows = [row for row in rows if int(row["client_id"]) == int(client_id)]
        self.rows = rows

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str]:
        row = self.rows[index]
        image = Image.open(self.root / row["crop_path"]).convert("RGB")
        if self.augment:
            image = augment_plate(image)
        tensor = pil_to_tensor(image)
        return {
            "image": tensor,
            "target_image": tensor.clone(),
            "text": encode_text(row["text"]),
            "raw_text": row["text"],
            "client_id": torch.tensor(int(row["client_id"]), dtype=torch.long),
        }


def discover_crop_datasets(base_dir: Path) -> list[Path]:
    if not base_dir.exists():
        return []
    if (base_dir / "crop_labels.csv").exists():
        return [base_dir]
    return sorted([item for item in base_dir.iterdir() if item.is_dir() and (item / "crop_labels.csv").exists()])


def stable_client_id(key: str, client_count: int) -> int:
    if client_count <= 0:
        return 0
    digest = hashlib.md5(key.encode("utf-8")).hexdigest()
    return int(digest[:8], 16) % client_count


def parse_ccpd_filename(name: str) -> tuple[tuple[int, int, int, int], list[int]] | None:
    stem = Path(name).stem
    parts = stem.split("-")
    if len(parts) < 5:
        return None
    bbox_raw = parts[2]
    if "_" not in bbox_raw or "&" not in bbox_raw:
        return None
    try:
        left, right = bbox_raw.split("_", 1)
        x1, y1 = [int(v) for v in left.split("&")]
        x2, y2 = [int(v) for v in right.split("&")]
        indices = [int(v) for v in parts[4].split("_")]
    except ValueError:
        return None
    return (x1, y1, x2, y2), indices


def decode_ccpd_plate(indices: list[int]) -> str:
    if len(indices) < 7:
        return ""
    text = []
    alpha_idx = indices[1]
    if 0 <= alpha_idx < len(CCPD_ALPHABETS):
        text.append(CCPD_ALPHABETS[alpha_idx])
    for idx in indices[2:7]:
        if 0 <= idx < len(CCPD_ADS):
            text.append(CCPD_ADS[idx])
        else:
            text.append("?")
    return "".join(text)


class UnifiedPlateDataset(Dataset):
    """Dataset that merges synthetic plates, manually labelled data, and optional CCPD."""

    def __init__(
        self,
        data_root: str | Path,
        client_count: int = 1,
        client_id: int | None = None,
        ccpd_root: str | Path | None = None,
        ccpd_splits: list[str] | None = None,
        augment: bool = True,
    ) -> None:
        self.samples: list[dict] = []
        self.augment = augment
        data_root_path = Path(data_root)
        for root in discover_crop_datasets(data_root_path):
            labels_path = root / "crop_labels.csv"
            with labels_path.open("r", newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            for idx, row in enumerate(rows):
                if not row.get("crop_path"):
                    continue
                client_value = row.get("client_id")
                if client_value is None or client_value == "":
                    client_value = stable_client_id(f"{labels_path}:{idx}", client_count)
                sample = {
                    "image_path": root / row["crop_path"],
                    "crop_box": None,
                    "text": row.get("text", ""),
                    "client_id": int(client_value) % max(client_count, 1),
                }
                self.samples.append(sample)

        if ccpd_root is not None:
            ccpd_root_path = Path(ccpd_root)
            split_dir = ccpd_root_path / "split"
            split_files = ccpd_splits or ["train.txt", "ccpd_blur.txt", "ccpd_rotate.txt", "ccpd_tilt.txt"]
            seen = set()
            for split_name in split_files:
                split_path = split_dir / split_name
                if not split_path.exists():
                    continue
                for raw in split_path.read_text(encoding="utf-8").splitlines():
                    rel = raw.strip()
                    if not rel or rel in seen:
                        continue
                    seen.add(rel)
                    sample_path = ccpd_root_path / rel
                    if not sample_path.exists():
                        continue
                    parsed = parse_ccpd_filename(rel)
                    if not parsed:
                        continue
                    crop_box, indices = parsed
                    text = decode_ccpd_plate(indices)
                    assigned_client = stable_client_id(rel, client_count)
                    self.samples.append(
                        {
                            "image_path": sample_path,
                            "crop_box": crop_box,
                            "text": text,
                            "client_id": assigned_client,
                        }
                    )

        if client_id is not None:
            self.samples = [item for item in self.samples if int(item["client_id"]) == int(client_id)]

        if not self.samples:
            raise FileNotFoundError("No dataset samples found. Ensure crop_labels.csv or CCPD images are available.")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str]:
        sample = self.samples[index]
        image = Image.open(sample["image_path"]).convert("RGB")
        crop_box = sample.get("crop_box")
        if crop_box:
            x1, y1, x2, y2 = [max(0, int(v)) for v in crop_box]
            image = image.crop((x1, y1, max(x2, x1 + 1), max(y2, y1 + 1)))
        if self.augment:
            image = augment_plate(image)
        tensor = pil_to_tensor(image)
        return {
            "image": tensor,
            "target_image": tensor.clone(),
            "text": encode_text(sample.get("text", "")),
            "raw_text": sample.get("text", ""),
            "client_id": torch.tensor(int(sample.get("client_id", 0)), dtype=torch.long),
        }
