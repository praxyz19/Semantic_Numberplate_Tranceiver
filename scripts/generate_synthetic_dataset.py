from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter, ImageFont


# ── Multi-country plate text generators ───────────────────────────────

INDIAN_STATES = ["KA", "MH", "DL", "TN", "KL", "AP", "TS", "GJ", "RJ", "UP", "HR", "MP", "WB", "PB", "BR"]
EU_COUNTRIES = ["D", "F", "NL", "B", "E", "I", "PL", "CZ", "A", "CH"]
US_STATES = ["CA", "TX", "NY", "FL", "IL", "PA", "OH", "GA", "NC", "MI"]
LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
DIGITS = "0123456789"


def random_indian_plate() -> tuple[str, str]:
    """Returns (text, format_name)."""
    text = (
        random.choice(INDIAN_STATES)
        + "".join(random.choice(DIGITS) for _ in range(2))
        + "".join(random.choice(LETTERS) for _ in range(2))
        + "".join(random.choice(DIGITS) for _ in range(4))
    )
    return text, "indian"


def random_eu_plate() -> tuple[str, str]:
    """European-style: 2 letters + 3 digits + 2 letters."""
    text = (
        "".join(random.choice(LETTERS) for _ in range(2))
        + "".join(random.choice(DIGITS) for _ in range(3))
        + "".join(random.choice(LETTERS) for _ in range(2))
    )
    return text, "european"


def random_us_plate() -> tuple[str, str]:
    """US-style: 3 letters + 4 digits."""
    text = (
        "".join(random.choice(LETTERS) for _ in range(3))
        + "".join(random.choice(DIGITS) for _ in range(4))
    )
    return text, "us"


def random_generic_plate() -> tuple[str, str]:
    """Generic alphanumeric plate."""
    length = random.randint(5, 8)
    text = "".join(random.choice(LETTERS + DIGITS) for _ in range(length))
    return text, "generic"


def random_plate_text() -> tuple[str, str]:
    """Pick a random plate format."""
    r = random.random()
    if r < 0.35:
        return random_indian_plate()
    elif r < 0.55:
        return random_eu_plate()
    elif r < 0.75:
        return random_us_plate()
    else:
        return random_generic_plate()


# ── Font helper ───────────────────────────────────────────────────────

def get_font(size: int) -> ImageFont.ImageFont:
    for path in [
        "C:/Windows/Fonts/arialbd.ttf",
        "C:/Windows/Fonts/arial.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    ]:
        if Path(path).exists():
            return ImageFont.truetype(path, size=size)
    return ImageFont.load_default()


# ── Plate rendering ──────────────────────────────────────────────────

PLATE_STYLES = {
    "indian_white": {"bg": (246, 247, 242), "fg": (15, 23, 42), "border": (20, 20, 20)},
    "indian_yellow": {"bg": (238, 197, 27), "fg": (15, 23, 42), "border": (15, 23, 42)},
    "eu_white": {"bg": (255, 255, 255), "fg": (0, 0, 0), "border": (0, 51, 153)},
    "us_white": {"bg": (248, 250, 252), "fg": (20, 30, 48), "border": (30, 60, 120)},
    "dark": {"bg": (31, 41, 55), "fg": (248, 250, 252), "border": (226, 232, 240)},
    "generic": {"bg": (240, 240, 235), "fg": (20, 28, 36), "border": (31, 41, 55)},
}


def draw_plate(
    text: str, width: int, height: int, style_name: str = "generic", two_line: bool = False,
) -> Image.Image:
    style = PLATE_STYLES.get(style_name, PLATE_STYLES["generic"])
    bg, fg, border_color = style["bg"], style["fg"], style["border"]

    image = Image.new("RGB", (width, height), bg)
    draw = ImageDraw.Draw(image)
    bw = max(2, height // 16)
    draw.rounded_rectangle(
        (bw, bw, width - bw, height - bw),
        radius=max(3, height // 10),
        outline=border_color,
        width=bw,
    )

    if two_line and len(text) >= 6:
        mid = len(text) // 2
        lines = [text[:mid], text[mid:]]
        font = get_font(int(height * 0.34))
        boxes = [draw.textbbox((0, 0), line, font=font) for line in lines]
        total_h = sum(box[3] - box[1] for box in boxes) + max(2, height // 28)
        y = (height - total_h) / 2 - height * 0.02
        for line, bbox in zip(lines, boxes):
            draw.text(
                ((width - (bbox[2] - bbox[0])) / 2, y), line, font=font, fill=fg,
            )
            y += bbox[3] - bbox[1] + max(2, height // 28)
    else:
        # Add spacing
        if len(text) >= 7:
            if len(text) == 10:  # Indian style: KA 01 AB 1234
                spaced = f"{text[:2]} {text[2:4]} {text[4:6]} {text[6:]}"
            elif len(text) == 7:  # EU / US style
                spaced = f"{text[:3]} {text[3:]}"
            else:
                spaced = text
        else:
            spaced = text
        font = get_font(int(height * 0.52))
        bbox = draw.textbbox((0, 0), spaced, font=font)
        x = (width - (bbox[2] - bbox[0])) / 2
        y = (height - (bbox[3] - bbox[1])) / 2 - height * 0.05
        draw.text((x, y), spaced, font=font, fill=fg)

    return image


def choose_plate_style(format_name: str) -> str:
    if format_name == "indian":
        return random.choice(["indian_white", "indian_white", "indian_white", "indian_yellow"])
    elif format_name == "european":
        return random.choice(["eu_white", "eu_white", "dark"])
    elif format_name == "us":
        return random.choice(["us_white", "us_white", "dark"])
    else:
        return random.choice(list(PLATE_STYLES.keys()))


# ── Background generator ─────────────────────────────────────────────

def background(width: int, height: int) -> Image.Image:
    base = Image.new(
        "RGB", (width, height),
        random.choice([(186, 198, 203), (74, 91, 105), (116, 130, 126), (205, 199, 188), (160, 170, 175)]),
    )
    draw = ImageDraw.Draw(base)
    for _ in range(10):
        color = tuple(random.randint(45, 220) for _ in range(3))
        x1 = random.randint(0, width)
        y1 = random.randint(0, height)
        x2 = min(width, x1 + random.randint(80, 260))
        y2 = min(height, y1 + random.randint(40, 170))
        draw.rectangle((x1, y1, x2, y2), fill=color)
    return base.filter(ImageFilter.GaussianBlur(radius=1.2))


# ── Main generation ──────────────────────────────────────────────────

def generate(args: argparse.Namespace) -> None:
    random.seed(args.seed)
    root = Path(args.output)
    images_dir = root / "images"
    crops_dir = root / "crops"
    images_dir.mkdir(parents=True, exist_ok=True)
    crops_dir.mkdir(parents=True, exist_ok=True)
    scene_rows = []
    crop_rows = []

    for idx in range(args.samples):
        scene = background(args.width, args.height)
        plate_count = random.randint(1, args.max_plates)
        for plate_idx in range(plate_count):
            text, format_name = random_plate_text()
            style_name = choose_plate_style(format_name)
            yellow = "yellow" in style_name
            two_line = yellow and random.random() < args.two_line_ratio

            if two_line:
                plate_w = random.randint(145, 215)
                plate_h = int(plate_w / random.uniform(1.65, 2.25))
            else:
                plate_w = random.randint(170, 280)
                plate_h = int(plate_w / random.uniform(4.0, 4.9))

            plate = draw_plate(text, plate_w, plate_h, style_name=style_name, two_line=two_line)

            # Random augmentations
            if random.random() < args.blur_ratio:
                plate = plate.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.4, 1.6)))

            angle = random.uniform(-args.max_rotation, args.max_rotation)
            rotated = plate.rotate(angle, expand=True, fillcolor=(220, 225, 226))

            x = random.randint(10, max(10, args.width - rotated.width - 10))
            y = random.randint(args.height // 3, max(args.height // 3 + 1, args.height - rotated.height - 10))
            scene.paste(rotated, (x, y))
            x1, y1, x2, y2 = x, y, x + rotated.width, y + rotated.height

            crop_path = f"crops/{idx:06d}_{plate_idx}.png"
            rotated.save(root / crop_path)
            client_id = idx % args.clients
            crop_rows.append({"crop_path": crop_path, "text": text, "client_id": client_id})
            scene_rows.append(
                {
                    "image_path": f"images/{idx:06d}.jpg",
                    "plate_index": plate_idx,
                    "text": text,
                    "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                    "client_id": client_id,
                }
            )

        scene.save(images_dir / f"{idx:06d}.jpg", quality=90)

    write_csv(root / "labels.csv", scene_rows)
    write_csv(root / "crop_labels.csv", crop_rows)
    print(f"Generated {args.samples} scenes and {len(crop_rows)} plate crops in {root}")


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate a synthetic multi-country licence-plate dataset.")
    parser.add_argument("--output", default="data/synthetic_plates")
    parser.add_argument("--samples", type=int, default=2000)
    parser.add_argument("--clients", type=int, default=4)
    parser.add_argument("--max-plates", type=int, default=3)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=360)
    parser.add_argument("--max-rotation", type=float, default=8.0)
    parser.add_argument("--two-line-ratio", type=float, default=0.75)
    parser.add_argument("--blur-ratio", type=float, default=0.35)
    parser.add_argument("--seed", type=int, default=7)
    return parser


if __name__ == "__main__":
    generate(build_parser().parse_args())
