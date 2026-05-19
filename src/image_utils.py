from __future__ import annotations

import base64
import io
from typing import Iterable

from PIL import Image, ImageOps


def image_from_bytes(data: bytes) -> Image.Image:
    image = Image.open(io.BytesIO(data))
    return ImageOps.exif_transpose(image).convert("RGB")


def image_to_data_url(image: Image.Image, fmt: str = "PNG", quality: int = 90) -> str:
    buffer = io.BytesIO()
    save_kwargs = {}
    if fmt.upper() in {"JPG", "JPEG", "WEBP"}:
        save_kwargs["quality"] = quality
    image.save(buffer, format=fmt, **save_kwargs)
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    mime = "jpeg" if fmt.upper() in {"JPG", "JPEG"} else fmt.lower()
    return f"data:image/{mime};base64,{encoded}"


def data_url_payload(data_url: str) -> bytes:
    return base64.b64decode(data_url.split(",", 1)[1])


def resize_for_display(image: Image.Image, max_side: int = 900) -> Image.Image:
    copy = image.copy()
    copy.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)
    return copy


def clamp_box(box: Iterable[int], width: int, height: int) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = [int(v) for v in box]
    x1 = max(0, min(x1, width - 1))
    y1 = max(0, min(y1, height - 1))
    x2 = max(x1 + 1, min(x2, width))
    y2 = max(y1 + 1, min(y2, height))
    return x1, y1, x2, y2

