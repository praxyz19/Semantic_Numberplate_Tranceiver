from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


@dataclass(frozen=True)
class PlateTemplate:
    template_id: str
    country: str
    background: tuple[int, int, int]
    foreground: tuple[int, int, int]
    border: tuple[int, int, int]
    aspect_ratio: float
    notes: str


class PlateKnowledgeBase:
    """Receiver-side shared knowledge used to reconstruct a semantic plate packet."""

    def __init__(self, path: Path):
        self.path = path
        self.templates = self._load_templates(path)

    def _load_templates(self, path: Path) -> dict[str, PlateTemplate]:
        data = json.loads(path.read_text(encoding="utf-8"))
        templates: dict[str, PlateTemplate] = {}
        for raw in data["templates"]:
            templates[raw["template_id"]] = PlateTemplate(
                template_id=raw["template_id"],
                country=raw["country"],
                background=tuple(raw["background"]),
                foreground=tuple(raw["foreground"]),
                border=tuple(raw["border"]),
                aspect_ratio=float(raw["aspect_ratio"]),
                notes=raw["notes"],
            )
        return templates

    def choose_template(self, features: dict, plate_text: str | None = None) -> PlateTemplate:
        rgb = features.get("background_rgb", [245, 246, 242])
        if rgb[0] > 150 and rgb[1] > 120 and rgb[2] < 120:
            return self.templates.get("yellow_plate", next(iter(self.templates.values())))
        if features.get("background_luma", 255) < 120:
            return self.templates.get("dark_plate", next(iter(self.templates.values())))
        if plate_text and len(plate_text) >= 8:
            return self.templates.get("ind_private_white", next(iter(self.templates.values())))
        return self.templates.get("generic_white", next(iter(self.templates.values())))

    def render_plate(self, text: str, size: tuple[int, int], template_id: str | None = None) -> Image.Image:
        width, height = size
        template = self.templates.get(template_id or "generic_white", next(iter(self.templates.values())))
        image = Image.new("RGB", (width, height), template.background)
        draw = ImageDraw.Draw(image)
        
        # Rounded border
        border_w = max(2, height // 18)
        draw.rounded_rectangle(
            (border_w, border_w, width - border_w, height - border_w),
            radius=max(4, height // 12),
            outline=template.border,
            width=border_w,
        )

        clean = (text or "UNKNOWN").upper().replace(" ", "")
        
        # --- Draw HSRP Indian Plate Elements if it's an Indian template ---
        is_indian = (template.country.lower() == "india" or template.template_id in {"ind_private_white", "generic_white", "yellow_plate"})
        hsrp_width = 0
        if is_indian and width / max(height, 1) > 2.6: # Don't draw HSRP strip on two-row square plates
            hsrp_width = int(width * 0.08)
            # Draw blue vertical strip on the left
            draw.rounded_rectangle(
                (border_w + 1, border_w + 1, border_w + hsrp_width, height - border_w - 1),
                radius=max(1, height // 24),
                fill=(0, 102, 204), # Premium HSRP Blue
            )
            # Draw yellow chromium hologram at the top of the blue strip
            holo_h = int(height * 0.16)
            holo_w = int(hsrp_width * 0.7)
            holo_x = border_w + (hsrp_width - holo_w) // 2
            holo_y = border_w + int(height * 0.12)
            draw.rounded_rectangle(
                (holo_x, holo_y, holo_x + holo_w, holo_y + holo_h),
                radius=max(1, holo_h // 5),
                fill=(255, 215, 0), # Gold/Chromium Hologram
                outline=(204, 163, 0),
                width=1
            )
            # Draw 'IND' text in white in the blue strip
            try:
                ind_font = self._font_for(hsrp_width, int(height * 0.15), "IND")
                ind_bbox = draw.textbbox((0, 0), "IND", font=ind_font)
                ind_w = ind_bbox[2] - ind_bbox[0]
                ind_h = ind_bbox[3] - ind_bbox[1]
                ind_x = border_w + (hsrp_width - ind_w) // 2
                ind_y = height - border_w - ind_h - int(height * 0.12)
                draw.text((ind_x, ind_y), "IND", fill=(255, 255, 255), font=ind_font)
            except Exception:
                pass

        # Adjust text rendering box to exclude the left blue strip
        text_area_start_x = border_w + hsrp_width + max(2, int(width * 0.02))
        text_area_width = width - border_w - text_area_start_x
        
        lines = [clean]
        if width / max(height, 1) < 2.6 and len(clean) >= 8:
            lines = [clean[:-4], clean[-4:]]
        elif len(clean) > 6:
            # Standard Indian formatting, e.g. MH 12 AB 1234
            if len(clean) == 10:
                clean = f"{clean[:2]} {clean[2:4]} {clean[4:6]} {clean[6:]}"
            elif len(clean) == 9:
                clean = f"{clean[:2]} {clean[2:4]} {clean[4:5]} {clean[5:]}"
            lines = [clean]

        font_height = height * (0.36 if len(lines) > 1 else 0.58)
        font = self._font_for(text_area_width, int(font_height * 2), max(lines, key=len))
        total_h = 0
        boxes = []
        for line in lines:
            bbox = draw.textbbox((0, 0), line, font=font)
            boxes.append(bbox)
            total_h += bbox[3] - bbox[1]
        gap = max(2, int(height * 0.05)) if len(lines) > 1 else 0
        y = (height - total_h - gap * (len(lines) - 1)) / 2 - height * 0.02
        
        for line, bbox in zip(lines, boxes):
            tw = bbox[2] - bbox[0]
            th = bbox[3] - bbox[1]
            # Center horizontally within the remaining text area
            x = text_area_start_x + (text_area_width - tw) / 2
            draw.text((x, y), line, font=font, fill=template.foreground)
            y += th + gap
        return image

    def public_summary(self) -> dict:
        return {
            "templates": [
                {
                    "template_id": item.template_id,
                    "country": item.country,
                    "aspect_ratio": item.aspect_ratio,
                    "notes": item.notes,
                }
                for item in self.templates.values()
            ]
        }

    @staticmethod
    def _font_for(width: int, height: int, text: str) -> ImageFont.ImageFont:
        candidates = [
            "C:/Windows/Fonts/arialbd.ttf",
            "C:/Windows/Fonts/arial.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        ]
        size = max(14, int(height * 0.55))
        for font_path in candidates:
            if Path(font_path).exists():
                while size > 10:
                    font = ImageFont.truetype(font_path, size=size)
                    box = ImageDraw.Draw(Image.new("RGB", (width, height))).textbbox((0, 0), text, font=font)
                    if box[2] - box[0] <= width * 0.86:
                        return font
                    size -= 2
        return ImageFont.load_default()
