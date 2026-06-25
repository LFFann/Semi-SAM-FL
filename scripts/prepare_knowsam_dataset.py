import argparse
import json
import random
import shutil
import re
from pathlib import Path

import cv2
import numpy as np


IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")


def parse_args():
    parser = argparse.ArgumentParser(description="Convert Labelme annotations to KnowSAM dataset layout.")
    parser.add_argument("--input-dir", type=Path, default=Path("./data/labeled"),
                        help="Directory containing Labelme image/json pairs.")
    parser.add_argument("--output-root", type=Path, default=Path("./SampleData/tumor_1"),
                        help="Output dataset root for KnowSAM.")
    parser.add_argument("--append", action="store_true",
                        help="Append samples into an existing output dataset instead of replacing it.")
    parser.add_argument("--nested-a3", action="store_true",
                        help="Discover A3 image/json pairs from nested case directories.")
    parser.add_argument("--labeled-count", type=int, default=15,
                        help="Number of samples to place into labeled split.")
    parser.add_argument("--unlabeled-count", type=int, default=3,
                        help="Number of samples to place into unlabeled split.")
    parser.add_argument("--val-count", type=int, default=2,
                        help="Number of samples to place into val split.")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed used for split shuffling.")
    return parser.parse_args()


def imread_unicode(image_path: Path):
    data = np.fromfile(str(image_path), dtype=np.uint8)
    image = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Failed to read image: {image_path}")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def imwrite_unicode(image_path: Path, image: np.ndarray):
    image_path.parent.mkdir(parents=True, exist_ok=True)
    suffix = image_path.suffix.lower() or ".png"
    success, encoded = cv2.imencode(suffix, image)
    if not success:
        raise ValueError(f"Failed to encode image: {image_path}")
    encoded.tofile(str(image_path))


def discover_samples(input_dir: Path):
    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory does not exist: {input_dir}")

    samples = []
    for json_path in sorted(input_dir.glob("*.json")):
        stem = json_path.stem
        image_path = None
        for suffix in IMAGE_SUFFIXES:
            candidate = input_dir / f"{stem}{suffix}"
            if candidate.exists():
                image_path = candidate
                break
        if image_path is None:
            raise FileNotFoundError(f"Image file not found for annotation: {json_path}")
        samples.append((image_path, json_path))

    if not samples:
        raise ValueError(f"No Labelme json files found under {input_dir}")
    return samples


def discover_nested_a3_samples(input_dir: Path):
    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory does not exist: {input_dir}")

    samples = []
    missing_pairs = []
    for case_dir in sorted([path for path in input_dir.iterdir() if path.is_dir()]):
        json_path = case_dir / "A3.json"
        image_path = None
        for suffix in IMAGE_SUFFIXES:
            for candidate in (case_dir / f"A3{suffix}", case_dir / f"A3{suffix.upper()}"):
                if candidate.exists():
                    image_path = candidate
                    break
            if image_path is not None:
                break
        if json_path.exists() and image_path is not None:
            samples.append((image_path, json_path))
        else:
            missing_pairs.append(case_dir.name)

    if not samples:
        raise ValueError(f"No nested A3 image/json pairs found under {input_dir}")
    return samples, missing_pairs


def draw_shape(mask: np.ndarray, shape: dict):
    shape_type = shape.get("shape_type", "polygon")
    points = np.array(shape.get("points", []), dtype=np.float32)
    if len(points) == 0:
        return

    points_int = np.round(points).astype(np.int32)
    if shape_type in {"polygon", "linestrip"}:
        cv2.fillPoly(mask, [points_int], 255)
    elif shape_type == "rectangle":
        top_left = tuple(points_int[0])
        bottom_right = tuple(points_int[1])
        cv2.rectangle(mask, top_left, bottom_right, 255, thickness=-1)
    elif shape_type == "circle":
        center = points[0]
        edge = points[1]
        radius = int(round(np.linalg.norm(center - edge)))
        cv2.circle(mask, tuple(np.round(center).astype(np.int32)), radius, 255, thickness=-1)
    else:
        cv2.fillPoly(mask, [points_int], 255)


def build_mask(annotation: dict, image_shape):
    height, width = image_shape[:2]
    mask = np.zeros((height, width), dtype=np.uint8)
    for shape in annotation.get("shapes", []):
        if str(shape.get("label")) != "1":
            continue
        draw_shape(mask, shape)
    return mask


def ensure_clean_output(output_root: Path):
    if output_root.exists():
        shutil.rmtree(output_root)
    for split in ("labeled", "unlabeled", "val"):
        (output_root / split / "image").mkdir(parents=True, exist_ok=True)
        if split != "unlabeled":
            (output_root / split / "mask").mkdir(parents=True, exist_ok=True)


def ensure_output_dirs(output_root: Path):
    for split in ("labeled", "unlabeled", "val"):
        (output_root / split / "image").mkdir(parents=True, exist_ok=True)
        if split != "unlabeled":
            (output_root / split / "mask").mkdir(parents=True, exist_ok=True)


def assign_splits(samples, labeled_count, unlabeled_count, val_count, seed):
    total_requested = labeled_count + unlabeled_count + val_count
    if total_requested != len(samples):
        raise ValueError(
            f"Split counts ({total_requested}) must match sample count ({len(samples)})."
        )

    shuffled = list(samples)
    random.Random(seed).shuffle(shuffled)

    split_names = (
        ["labeled"] * labeled_count
        + ["unlabeled"] * unlabeled_count
        + ["val"] * val_count
    )
    return list(zip(shuffled, split_names))


def next_case_index(output_root: Path):
    pattern = re.compile(r"case_(\d+)\.png$")
    max_index = 0
    for image_path in output_root.glob("*/*/case_*.png"):
        match = pattern.search(image_path.name)
        if match:
            max_index = max(max_index, int(match.group(1)))
    return max_index + 1


def load_manifest(manifest_path: Path):
    if manifest_path.exists():
        return json.loads(manifest_path.read_text(encoding="utf-8"))
    return {"samples": []}


def main():
    args = parse_args()
    if args.nested_a3:
        samples, missing_pairs = discover_nested_a3_samples(args.input_dir)
    else:
        samples = discover_samples(args.input_dir)
        missing_pairs = []

    assigned = assign_splits(
        samples,
        labeled_count=args.labeled_count,
        unlabeled_count=args.unlabeled_count,
        val_count=args.val_count,
        seed=args.seed,
    )

    manifest_path = args.output_root / "split_manifest.json"
    if args.append:
        ensure_output_dirs(args.output_root)
        manifest = load_manifest(manifest_path)
        start_index = next_case_index(args.output_root)
    else:
        ensure_clean_output(args.output_root)
        manifest = {"samples": []}
        start_index = 1

    manifest.update({
        "input_dir": str(args.input_dir.resolve()),
        "output_root": str(args.output_root.resolve()),
        "seed": args.seed,
        "split_counts": {
            "labeled": args.labeled_count,
            "unlabeled": args.unlabeled_count,
            "val": args.val_count,
        },
        "append_mode": args.append,
        "nested_a3": args.nested_a3,
        "missing_pairs": missing_pairs,
    })

    for index, ((image_path, json_path), split) in enumerate(assigned, start=start_index):
        case_name = f"case_{index:04d}.png"
        annotation = json.loads(json_path.read_text(encoding="utf-8"))
        image = imread_unicode(image_path)
        mask = build_mask(annotation, image.shape)

        output_image = args.output_root / split / "image" / case_name
        imwrite_unicode(output_image, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
        if split != "unlabeled":
            output_mask = args.output_root / split / "mask" / case_name
            imwrite_unicode(output_mask, mask)

        manifest["samples"].append({
            "split": split,
            "new_name": case_name,
            "source_parent": image_path.parent.name,
            "source_image": image_path.name,
            "source_annotation": json_path.name,
            "height": int(image.shape[0]),
            "width": int(image.shape[1]),
            "positive_pixels": int((mask > 0).sum()),
        })

    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Prepared {len(assigned)} samples at {args.output_root}")
    if missing_pairs:
        print(f"Skipped {len(missing_pairs)} folders without A3 image/json pairs: {missing_pairs}")
    print(f"Manifest written to {manifest_path}")


if __name__ == "__main__":
    main()
