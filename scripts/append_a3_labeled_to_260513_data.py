import argparse
import base64
import csv
import json
import re
import shutil
from io import BytesIO
from pathlib import Path

import numpy as np
from openpyxl import load_workbook
from PIL import Image, ImageDraw


IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
FIELDNAMES = [
    "split",
    "id",
    "original_case_dir",
    "original_image_path",
    "original_label_path",
    "new_image_path",
    "new_label_path",
    "new_mask_path",
    "new_overlay_path",
    "original_image_name",
    "original_label_name",
    "new_image_name",
    "new_label_name",
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Append nested LabelMe A3 cases to data/260513_data raw layout."
    )
    parser.add_argument(
        "--source-root",
        type=Path,
        required=True,
        help="Nested case directory containing */A3.json and A3 images.",
    )
    parser.add_argument(
        "--target-root",
        type=Path,
        default=Path("./data/260513_data"),
        help="Existing raw 260513_data directory.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only report what would be appended.",
    )
    return parser.parse_args()


def is_image(path: Path):
    return path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES


def next_labeled_index(image_dir: Path):
    max_index = 0
    for path in image_dir.iterdir():
        match = re.fullmatch(r"labeled_(\d+)\.[^.]+", path.name)
        if match:
            max_index = max(max_index, int(match.group(1)))
    return max_index + 1


def load_annotation(json_path: Path):
    return json.loads(json_path.read_text(encoding="utf-8"))


def candidate_image_paths(json_path: Path, annotation: dict):
    image_path = annotation.get("imagePath")
    candidates = []
    if image_path:
        raw = json_path.parent / image_path
        candidates.append(raw)
        stem = raw.stem
    else:
        stem = json_path.stem

    for suffix in sorted(IMAGE_SUFFIXES):
        candidates.append(json_path.parent / f"{stem}{suffix}")
        candidates.append(json_path.parent / f"{stem}{suffix.upper()}")
        candidates.append(json_path.parent / f"A3{suffix}")
        candidates.append(json_path.parent / f"A3{suffix.upper()}")
    return candidates


def find_image_file(json_path: Path, annotation: dict):
    for candidate in candidate_image_paths(json_path, annotation):
        if is_image(candidate):
            return candidate
    return None


def read_image_from_labelme(annotation: dict):
    image_data = annotation.get("imageData")
    if not image_data:
        return None
    with Image.open(BytesIO(base64.b64decode(image_data))) as image:
        return image.convert("RGB")


def draw_shape(draw: ImageDraw.ImageDraw, shape: dict, fill_value: int):
    points = shape.get("points") or []
    if not points:
        return
    xy = [(float(x), float(y)) for x, y in points]
    shape_type = shape.get("shape_type") or "polygon"

    if shape_type in {"polygon", "linestrip"} and len(xy) >= 3:
        draw.polygon(xy, fill=fill_value)
    elif shape_type == "rectangle" and len(xy) >= 2:
        draw.rectangle([xy[0], xy[1]], fill=fill_value)
    elif shape_type == "circle" and len(xy) >= 2:
        cx, cy = xy[0]
        ex, ey = xy[1]
        radius = ((cx - ex) ** 2 + (cy - ey) ** 2) ** 0.5
        draw.ellipse([cx - radius, cy - radius, cx + radius, cy + radius], fill=fill_value)
    elif shape_type in {"line", "linestrip"} and len(xy) >= 2:
        draw.line(xy, fill=fill_value, width=3)
    elif len(xy) >= 3:
        draw.polygon(xy, fill=fill_value)


def build_mask(annotation: dict, width: int, height: int):
    mask_image = Image.new("L", (width, height), 0)
    draw = ImageDraw.Draw(mask_image)
    label_counts = {}
    for shape in annotation.get("shapes", []):
        label = str(shape.get("label"))
        label_counts[label] = label_counts.get(label, 0) + 1
        if not label.isdigit():
            continue
        draw_shape(draw, shape, int(label))
    return mask_image, label_counts


def make_overlay(image: Image.Image, mask: Image.Image):
    overlay = image.convert("RGB").copy()
    base = np.asarray(overlay).astype(np.float32)
    mask_arr = np.asarray(mask)
    colors = {
        1: np.array([255, 0, 0], dtype=np.float32),
        2: np.array([0, 255, 0], dtype=np.float32),
    }
    for value, color in colors.items():
        region = mask_arr == value
        base[region] = base[region] * 0.55 + color * 0.45
    return Image.fromarray(np.clip(base, 0, 255).astype(np.uint8), mode="RGB")


def discover_samples(source_root: Path):
    samples = []
    skipped = []
    for case_dir in sorted(path for path in source_root.iterdir() if path.is_dir()):
        json_path = case_dir / "A3.json"
        if not json_path.exists():
            skipped.append({"case_dir": case_dir.name, "reason": "missing_A3_json"})
            continue

        annotation = load_annotation(json_path)
        image_path = find_image_file(json_path, annotation)
        if image_path is not None:
            with Image.open(image_path) as image:
                image_rgb = image.convert("RGB")
                width, height = image_rgb.size
            original_image_name = image_path.name
            image_origin = "file"
            image_suffix = image_path.suffix.lower()
        else:
            image_rgb = read_image_from_labelme(annotation)
            if image_rgb is None:
                skipped.append({"case_dir": case_dir.name, "reason": "missing_image"})
                continue
            width, height = image_rgb.size
            original_image_name = annotation.get("imagePath") or "A3.png"
            image_origin = "imageData"
            image_suffix = Path(original_image_name).suffix.lower() or ".png"

        mask, label_counts = build_mask(annotation, width, height)
        mask_arr = np.asarray(mask)
        if not np.any(mask_arr > 0):
            skipped.append({"case_dir": case_dir.name, "reason": "empty_mask"})
            continue

        samples.append(
            {
                "case_dir": case_dir,
                "json_path": json_path,
                "image_path": image_path,
                "image_origin": image_origin,
                "image_rgb": image_rgb,
                "image_suffix": image_suffix,
                "original_image_name": original_image_name,
                "mask": mask,
                "label_counts": label_counts,
                "positive_pixels": int(np.count_nonzero(mask_arr)),
            }
        )
    return samples, skipped


def read_existing_mapping(csv_path: Path):
    with csv_path.open(newline="", encoding="utf-8-sig") as csv_file:
        reader = csv.DictReader(csv_file)
        rows = list(reader)
        fieldnames = reader.fieldnames or FIELDNAMES
    return rows, fieldnames


def append_csv(csv_path: Path, rows):
    existing, fieldnames = read_existing_mapping(csv_path)
    merged = existing + rows
    with csv_path.open("w", newline="", encoding="utf-8-sig") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(merged)


def append_xlsx(xlsx_path: Path, rows):
    workbook = load_workbook(xlsx_path)
    sheet = workbook.active
    headers = [cell.value for cell in sheet[1]]
    for row in rows:
        sheet.append([row.get(header, "") for header in headers])
    workbook.save(xlsx_path)


def main():
    args = parse_args()
    source_root = args.source_root.resolve()
    target_root = args.target_root.resolve()
    labeled_root = target_root / "labeled"
    image_dir = labeled_root / "images"
    label_dir = labeled_root / "label"
    mask_dir = labeled_root / "masks"
    overlay_dir = labeled_root / "overlays"
    csv_path = target_root / "filename_mapping.csv"
    xlsx_path = target_root / "filename_mapping.xlsx"

    samples, skipped = discover_samples(source_root)
    start_index = next_labeled_index(image_dir)

    duplicate_case_dirs = set()
    if csv_path.exists():
        existing_rows, _ = read_existing_mapping(csv_path)
        duplicate_case_dirs = {
            row.get("original_case_dir", "")
            for row in existing_rows
            if str(row.get("original_case_dir", "")).startswith(source_root.name + "\\")
        }
    samples = [sample for sample in samples if f"{source_root.name}\\{sample['case_dir'].name}" not in duplicate_case_dirs]

    rows = []
    planned = []
    for offset, sample in enumerate(samples):
        index = start_index + offset
        sample_id = f"labeled_{index:06d}"
        image_suffix = sample["image_suffix"] if sample["image_suffix"] in IMAGE_SUFFIXES else ".png"
        new_image_name = f"{sample_id}{image_suffix}"
        new_label_name = f"{sample_id}.json"
        new_mask_name = f"{sample_id}.png"
        new_overlay_name = f"{sample_id}.png"

        row = {
            "split": "labeled",
            "id": sample_id,
            "original_case_dir": f"{source_root.name}\\{sample['case_dir'].name}",
            "original_image_path": f"{source_root.name}\\{sample['case_dir'].name}\\{sample['original_image_name']}",
            "original_label_path": f"{source_root.name}\\{sample['case_dir'].name}\\A3.json",
            "new_image_path": f"labeled/images/{new_image_name}",
            "new_label_path": f"labeled/label/{new_label_name}",
            "new_mask_path": f"labeled/masks/{new_mask_name}",
            "new_overlay_path": f"labeled/overlays/{new_overlay_name}",
            "original_image_name": sample["original_image_name"],
            "original_label_name": "A3.json",
            "new_image_name": new_image_name,
            "new_label_name": new_label_name,
        }
        rows.append(row)
        planned.append(
            {
                "id": sample_id,
                "case_dir": sample["case_dir"].name,
                "image_origin": sample["image_origin"],
                "label_counts": sample["label_counts"],
                "positive_pixels": sample["positive_pixels"],
            }
        )

        if args.dry_run:
            continue

        target_image = image_dir / new_image_name
        target_label = label_dir / new_label_name
        target_mask = mask_dir / new_mask_name
        target_overlay = overlay_dir / new_overlay_name
        for target in (target_image, target_label, target_mask, target_overlay):
            if target.exists():
                raise FileExistsError(f"Refusing to overwrite existing file: {target}")

        if sample["image_path"] is not None:
            shutil.copy2(sample["image_path"], target_image)
        else:
            sample["image_rgb"].save(target_image)
        shutil.copy2(sample["json_path"], target_label)
        sample["mask"].save(target_mask)
        make_overlay(sample["image_rgb"], sample["mask"]).save(target_overlay)

    if not args.dry_run and rows:
        append_csv(csv_path, rows)
        append_xlsx(xlsx_path, rows)

    print(
        json.dumps(
            {
                "source_root": str(source_root),
                "target_root": str(target_root),
                "start_index": start_index,
                "planned_count": len(planned),
                "skipped": skipped,
                "already_present_skipped": len(duplicate_case_dirs),
                "planned": planned,
                "dry_run": args.dry_run,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
