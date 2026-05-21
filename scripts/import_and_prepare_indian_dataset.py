from pathlib import Path
import argparse
import csv
from PIL import Image


def prepare(src: Path, dst: Path) -> None:
    labels_dir = src / "labels"
    images_dir = src / "images"
    dst.mkdir(parents=True, exist_ok=True)
    crops_dir = dst / "crops"
    crops_dir.mkdir(parents=True, exist_ok=True)
    rows = []

    for label_path in sorted(labels_dir.glob("*.txt")):
        stem = label_path.stem
        # try common image extensions
        img_path = None
        for ext in (".jpg", ".jpeg", ".png"):
            candidate = images_dir / (stem + ext)
            if candidate.exists():
                img_path = candidate
                break
        if img_path is None:
            continue

        with label_path.open("r", encoding="utf-8") as fh:
            lines = [l.strip() for l in fh.readlines() if l.strip()]
        if not lines:
            continue

        try:
            image = Image.open(img_path).convert("RGB")
        except Exception:
            continue

        w_img, h_img = image.size
        for idx, line in enumerate(lines):
            parts = line.split()
            if len(parts) < 5:
                continue
            try:
                _, xc, yc, w, h = parts[:5]
                xc = float(xc)
                yc = float(yc)
                w = float(w)
                h = float(h)
            except ValueError:
                continue

            x1 = int((xc - w / 2.0) * w_img)
            y1 = int((yc - h / 2.0) * h_img)
            x2 = int((xc + w / 2.0) * w_img)
            y2 = int((yc + h / 2.0) * h_img)

            # expand a little
            pad_w = int(0.06 * (x2 - x1))
            pad_h = int(0.10 * (y2 - y1))
            x1 = max(0, x1 - pad_w)
            y1 = max(0, y1 - pad_h)
            x2 = min(w_img, x2 + pad_w)
            y2 = min(h_img, y2 + pad_h)

            crop = image.crop((x1, y1, x2, y2))
            out_name = f"{stem}_{idx}.png"
            out_path = crops_dir / out_name
            crop.save(out_path)
            rows.append((f"crops/{out_name}", "", "0"))

    # write CSV
    csv_path = dst / "crop_labels.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["crop_path", "text", "client_id"])
        for row in rows:
            writer.writerow(row)

    print(f"Prepared {len(rows)} crops -> {csv_path}")


def build_parser():
    p = argparse.ArgumentParser()
    p.add_argument("--src", required=True)
    p.add_argument("--dst", required=True)
    return p


def main():
    args = build_parser().parse_args()
    src = Path(args.src)
    dst = Path(args.dst)
    if not src.exists():
        raise SystemExit(f"Source not found: {src}")
    prepare(src, dst)


if __name__ == "__main__":
    main()
