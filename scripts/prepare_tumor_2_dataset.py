import argparse
import base64
import csv
import hashlib
import json
import random
import shutil
from io import BytesIO
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")


def parse_args():
    parser = argparse.ArgumentParser(description="Prepare tumor_2 for label-1 A3 fissure segmentation.")
    parser.add_argument("--output-root", type=Path, default=Path("./SampleData/tumor_2"))
    parser.add_argument(
        "--labeled-source",
        type=Path,
        default=Path("./data/刘凡A3zxx标注_xiulabel"),
        help="Nested Labelme case directory. A3.json is treated as annotated.",
    )
    parser.add_argument(
        "--unlabeled-sources",
        nargs="+",
        type=Path,
        default=[
            Path("./data/A3"),
            Path("./data/颅脑正常病例已标注"),
            Path("./data/颅脑正常病例A3刘凡用_xiu"),
        ],
        help="Flat or nested A3 image directories used only as unlabeled training images.",
    )
    parser.add_argument("--target-label", type=str, default="1")
    parser.add_argument("--val-count", type=int, default=5)
    parser.add_argument("--test-count", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--csv-name", type=str, default="split_record.csv")
    parser.add_argument("--manifest-name", type=str, default="split_manifest.json")
    return parser.parse_args()


def is_image(path: Path):
    return path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES


def read_rgb(path: Path):
    with Image.open(path) as image:
        return np.asarray(image.convert("RGB"))


def read_rgb_from_labelme(annotation: dict):
    image_data = annotation.get("imageData")
    if not image_data:
        return None
    with Image.open(BytesIO(base64.b64decode(image_data))) as image:
        return np.asarray(image.convert("RGB"))


def write_png(path: Path, image_rgb: np.ndarray):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(image_rgb.astype(np.uint8), mode="RGB").save(path)


def write_mask(path: Path, mask: np.ndarray):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(mask.astype(np.uint8), mode="L").save(path)


def image_hash(image_rgb: np.ndarray):
    digest = hashlib.md5(image_rgb.tobytes()).hexdigest()
    height, width = image_rgb.shape[:2]
    return f"{digest}_{height}x{width}x3"


def matching_image_for_json(json_path: Path, annotation: dict):
    image_path = annotation.get("imagePath")
    candidates = []
    if image_path:
        raw = json_path.parent / image_path
        candidates.append(raw)
        stem = raw.stem
    else:
        stem = json_path.stem

    for suffix in IMAGE_SUFFIXES + tuple(s.upper() for s in IMAGE_SUFFIXES):
        candidates.append(json_path.parent / f"{stem}{suffix}")
        candidates.append(json_path.parent / f"A3{suffix}")

    for candidate in candidates:
        if candidate.exists() and is_image(candidate):
            return candidate
    return None


def draw_shape(draw: ImageDraw.ImageDraw, shape: dict):
    points = shape.get("points") or []
    if not points:
        return
    xy = [(float(x), float(y)) for x, y in points]
    shape_type = shape.get("shape_type") or "polygon"

    if shape_type in {"polygon", "linestrip"} and len(xy) >= 3:
        draw.polygon(xy, fill=255)
    elif shape_type == "rectangle" and len(xy) >= 2:
        draw.rectangle([xy[0], xy[1]], fill=255)
    elif shape_type == "circle" and len(xy) >= 2:
        cx, cy = xy[0]
        ex, ey = xy[1]
        radius = ((cx - ex) ** 2 + (cy - ey) ** 2) ** 0.5
        draw.ellipse([cx - radius, cy - radius, cx + radius, cy + radius], fill=255)
    elif shape_type in {"line", "linestrip"} and len(xy) >= 2:
        draw.line(xy, fill=255, width=3)
    elif len(xy) >= 3:
        draw.polygon(xy, fill=255)


def build_mask(annotation: dict, image_shape, target_label: str):
    height, width = image_shape[:2]
    mask_image = Image.new("L", (width, height), 0)
    draw = ImageDraw.Draw(mask_image)
    for shape in annotation.get("shapes", []):
        if str(shape.get("label")) != target_label:
            continue
        draw_shape(draw, shape)
    return np.asarray(mask_image, dtype=np.uint8)


def discover_labeled_samples(root: Path, target_label: str):
    samples = []
    skipped = []
    label_counts = {}
    shape_type_counts = {}

    for json_path in sorted(root.glob("*/A3.json")):
        annotation = json.loads(json_path.read_text(encoding="utf-8"))
        for shape in annotation.get("shapes", []):
            label = str(shape.get("label"))
            label_counts[label] = label_counts.get(label, 0) + 1
            shape_type = str(shape.get("shape_type") or "polygon")
            shape_type_counts[shape_type] = shape_type_counts.get(shape_type, 0) + 1

        image_path = matching_image_for_json(json_path, annotation)
        if image_path is not None:
            image = read_rgb(image_path)
            source_image = image_path.name
            image_origin = "file"
        else:
            image = read_rgb_from_labelme(annotation)
            source_image = annotation.get("imagePath") or ""
            image_origin = "imageData"

        if image is None:
            skipped.append({"source_parent": json_path.parent.name, "reason": "missing_image"})
            continue

        mask = build_mask(annotation, image.shape, target_label)
        positive_pixels = int((mask > 0).sum())
        if positive_pixels == 0:
            skipped.append({"source_parent": json_path.parent.name, "reason": f"no_label_{target_label}"})
            continue

        samples.append(
            {
                "source_dir": root,
                "source_parent": json_path.parent.name,
                "source_image": source_image,
                "source_annotation": json_path.name,
                "image": image,
                "mask": mask,
                "positive_pixels": positive_pixels,
                "image_origin": image_origin,
                "discovery_mode": "labelme_a3",
            }
        )

    return samples, skipped, label_counts, shape_type_counts


def exact_a3_image(case_dir: Path):
    for path in sorted(case_dir.iterdir()):
        if is_image(path) and path.stem.upper() == "A3":
            return path
    return None


def discover_unlabeled_images(input_dir: Path):
    case_dirs = [path for path in sorted(input_dir.iterdir()) if path.is_dir()]
    if case_dirs:
        samples = []
        for case_dir in case_dirs:
            image_path = exact_a3_image(case_dir)
            if image_path is None:
                continue
            samples.append(
                {
                    "source_dir": input_dir,
                    "source_parent": case_dir.name,
                    "source_image": image_path.name,
                    "source_annotation": "",
                    "image": read_rgb(image_path),
                    "discovery_mode": "nested_exact_a3",
                    "notes": "unlabeled_exact_A3",
                }
            )
        return samples

    return [
        {
            "source_dir": input_dir,
            "source_parent": input_dir.name,
            "source_image": image_path.name,
            "source_annotation": "",
            "image": read_rgb(image_path),
            "discovery_mode": "flat",
            "notes": "unlabeled_flat_image",
        }
        for image_path in sorted(input_dir.iterdir())
        if is_image(image_path)
    ]


def ensure_clean_output(output_root: Path):
    if output_root.exists():
        shutil.rmtree(output_root)
    for split in ("labeled", "val", "test"):
        (output_root / split / "image").mkdir(parents=True, exist_ok=True)
        (output_root / split / "mask").mkdir(parents=True, exist_ok=True)
    (output_root / "unlabeled" / "image").mkdir(parents=True, exist_ok=True)


def new_record(**kwargs):
    return {
        "status": kwargs.get("status", "added"),
        "split": kwargs.get("split", ""),
        "new_name": kwargs.get("new_name", ""),
        "source_dir": str(kwargs.get("source_dir", "")),
        "source_parent": kwargs.get("source_parent", ""),
        "source_image": kwargs.get("source_image", ""),
        "source_annotation": kwargs.get("source_annotation", ""),
        "target_label": kwargs.get("target_label", ""),
        "height": kwargs.get("height", ""),
        "width": kwargs.get("width", ""),
        "positive_pixels": kwargs.get("positive_pixels", ""),
        "hash": kwargs.get("hash", ""),
        "discovery_mode": kwargs.get("discovery_mode", ""),
        "notes": kwargs.get("notes", ""),
    }


def assign_annotated_splits(samples, val_count: int, test_count: int, seed: int):
    if val_count + test_count >= len(samples):
        raise ValueError(
            f"val-count + test-count must be smaller than annotated sample count ({len(samples)})."
        )
    shuffled = list(samples)
    random.Random(seed).shuffle(shuffled)
    test_start = len(shuffled) - test_count
    val_start = test_start - val_count
    return (
        [("labeled", sample) for sample in shuffled[:val_start]]
        + [("val", sample) for sample in shuffled[val_start:test_start]]
        + [("test", sample) for sample in shuffled[test_start:]]
    )


def main():
    args = parse_args()
    output_root = args.output_root.resolve()
    labeled_source = args.labeled_source.resolve()
    unlabeled_sources = [path.resolve() for path in args.unlabeled_sources]
    ensure_clean_output(output_root)

    labeled_samples, skipped_labeled, label_counts, shape_type_counts = discover_labeled_samples(
        labeled_source,
        args.target_label,
    )
    assigned = assign_annotated_splits(labeled_samples, args.val_count, args.test_count, args.seed)

    manifest = {
        "output_root": str(output_root),
        "seed": args.seed,
        "target_label": args.target_label,
        "labeled_source": str(labeled_source),
        "unlabeled_sources": [str(path) for path in unlabeled_sources],
        "label_counts_in_labeled_json": label_counts,
        "shape_type_counts_in_labeled_json": shape_type_counts,
        "skipped_labeled": skipped_labeled,
        "split_counts": {},
        "samples": [],
    }
    records = []
    seen_hashes = {}
    case_index = 1

    for split, sample in assigned:
        case_name = f"case_{case_index:04d}.png"
        case_index += 1
        image = sample["image"]
        mask = sample["mask"]
        digest = image_hash(image)

        write_png(output_root / split / "image" / case_name, image)
        write_mask(output_root / split / "mask" / case_name, mask)
        seen_hashes.setdefault(digest, case_name)

        record = new_record(
            split=split,
            new_name=case_name,
            source_dir=sample["source_dir"],
            source_parent=sample["source_parent"],
            source_image=sample["source_image"],
            source_annotation=sample["source_annotation"],
            target_label=args.target_label,
            height=int(image.shape[0]),
            width=int(image.shape[1]),
            positive_pixels=sample["positive_pixels"],
            hash=digest,
            discovery_mode=sample["discovery_mode"],
            notes=f"image_origin={sample['image_origin']}",
        )
        records.append(record)
        manifest["samples"].append(record)

    for input_dir in unlabeled_sources:
        for sample in discover_unlabeled_images(input_dir):
            image = sample.pop("image")
            digest = image_hash(image)
            height, width = image.shape[:2]

            if digest in seen_hashes:
                records.append(
                    new_record(
                        status="duplicate_skipped",
                        split="unlabeled",
                        source_dir=input_dir,
                        source_parent=sample["source_parent"],
                        source_image=sample["source_image"],
                        source_annotation=sample["source_annotation"],
                        height=int(height),
                        width=int(width),
                        hash=digest,
                        discovery_mode=sample["discovery_mode"],
                        notes=f"duplicate_of_{seen_hashes[digest]}",
                    )
                )
                continue

            case_name = f"case_{case_index:04d}.png"
            case_index += 1
            write_png(output_root / "unlabeled" / "image" / case_name, image)
            seen_hashes[digest] = case_name

            record = new_record(
                split="unlabeled",
                new_name=case_name,
                source_dir=input_dir,
                source_parent=sample["source_parent"],
                source_image=sample["source_image"],
                source_annotation=sample["source_annotation"],
                height=int(height),
                width=int(width),
                hash=digest,
                discovery_mode=sample["discovery_mode"],
                notes=sample["notes"],
            )
            records.append(record)
            manifest["samples"].append(record)

    split_counts = {
        "labeled": len(list((output_root / "labeled" / "image").glob("*.png"))),
        "unlabeled": len(list((output_root / "unlabeled" / "image").glob("*.png"))),
        "val": len(list((output_root / "val" / "image").glob("*.png"))),
        "test": len(list((output_root / "test" / "image").glob("*.png"))),
    }
    manifest["split_counts"] = split_counts

    csv_path = output_root / args.csv_name
    with csv_path.open("w", newline="", encoding="utf-8-sig") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=list(records[0].keys()))
        writer.writeheader()
        writer.writerows(records)

    manifest_path = output_root / args.manifest_name
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    print(
        json.dumps(
            {
                "target_label": args.target_label,
                "labeled_json_label_counts": label_counts,
                "split_counts": split_counts,
                "csv_path": str(csv_path),
                "manifest_path": str(manifest_path),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
