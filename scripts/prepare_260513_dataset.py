import argparse
import csv
import json
import random
import shutil
from pathlib import Path

import numpy as np
from PIL import Image


IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Prepare data/260513_data into the KnowSAM SampleData layout."
    )
    parser.add_argument("--source-root", type=Path, default=Path("./data/260513_data"))
    parser.add_argument("--output-root", type=Path, default=Path("./SampleData/260513_data"))
    parser.add_argument(
        "--target-label",
        type=int,
        default=None,
        help="Mask label value to export as foreground. Omit to export all non-zero labels as foreground.",
    )
    parser.add_argument(
        "--multi-class",
        action="store_true",
        help="Preserve original integer mask labels instead of exporting a binary foreground mask.",
    )
    parser.add_argument("--val-count", type=int, default=None)
    parser.add_argument("--test-count", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--manifest-name", type=str, default="split_manifest.json")
    parser.add_argument("--csv-name", type=str, default="split_record.csv")
    return parser.parse_args()


def is_image(path: Path):
    return path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES


def ensure_clean_output(output_root: Path):
    if output_root.exists():
        shutil.rmtree(output_root)
    for split in ("labeled", "val", "test"):
        (output_root / split / "image").mkdir(parents=True, exist_ok=True)
        (output_root / split / "mask").mkdir(parents=True, exist_ok=True)
    (output_root / "unlabeled" / "image").mkdir(parents=True, exist_ok=True)


def write_png(source: Path, target: Path, mode: str):
    target.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(source) as image:
        image.convert(mode).save(target)


def write_binary_mask(source: Path, target: Path, target_label: int | None):
    target.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(source) as image:
        mask = np.array(image.convert("L"))
    if target_label is None:
        mask = mask > 0
    else:
        mask = mask == target_label
    mask = mask.astype(np.uint8) * 255
    Image.fromarray(mask, mode="L").save(target)


def write_multiclass_mask(source: Path, target: Path):
    target.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(source) as image:
        mask = np.array(image.convert("L"), dtype=np.uint8)
    Image.fromarray(mask, mode="L").save(target)


def discover_labeled(source_root: Path):
    image_dir = source_root / "labeled" / "images"
    mask_dir = source_root / "labeled" / "masks"
    images = sorted(path for path in image_dir.iterdir() if is_image(path))
    masks_by_stem = {path.stem: path for path in mask_dir.iterdir() if is_image(path)}

    samples = []
    missing_masks = []
    for image_path in images:
        mask_path = masks_by_stem.get(image_path.stem)
        if mask_path is None:
            missing_masks.append(str(image_path))
            continue
        samples.append({"image": image_path, "mask": mask_path})

    if missing_masks:
        raise FileNotFoundError(
            "Missing masks for labeled images:\n" + "\n".join(missing_masks[:20])
        )
    return samples


def discover_unlabeled(source_root: Path):
    unlabeled_dir = source_root / "unlabelled"
    return sorted(path for path in unlabeled_dir.iterdir() if is_image(path))


def split_labeled(samples, val_count, test_count, seed):
    if val_count is None:
        val_count = round(len(samples) * 0.10)
    if test_count is None:
        test_count = round(len(samples) * 0.10)
    if val_count + test_count >= len(samples):
        raise ValueError(
            f"val-count + test-count must be smaller than labeled sample count ({len(samples)})."
        )

    shuffled = list(samples)
    random.Random(seed).shuffle(shuffled)
    test_start = len(shuffled) - test_count
    val_start = test_start - val_count
    return {
        "labeled": shuffled[:val_start],
        "val": shuffled[val_start:test_start],
        "test": shuffled[test_start:],
    }


def record(split, case_name, image_path, mask_path=None):
    return {
        "split": split,
        "new_name": case_name,
        "source_image": str(image_path),
        "source_mask": str(mask_path) if mask_path else "",
    }


def main():
    args = parse_args()
    source_root = args.source_root.resolve()
    output_root = args.output_root.resolve()
    labeled_samples = discover_labeled(source_root)
    unlabeled_samples = discover_unlabeled(source_root)
    split_samples = split_labeled(labeled_samples, args.val_count, args.test_count, args.seed)

    ensure_clean_output(output_root)

    records = []
    case_index = 1
    for split in ("labeled", "val", "test"):
        for sample in split_samples[split]:
            case_name = f"case_{case_index:04d}.png"
            case_index += 1
            write_png(sample["image"], output_root / split / "image" / case_name, "RGB")
            if args.multi_class:
                write_multiclass_mask(sample["mask"], output_root / split / "mask" / case_name)
            else:
                write_binary_mask(sample["mask"], output_root / split / "mask" / case_name, args.target_label)
            records.append(record(split, case_name, sample["image"], sample["mask"]))

    for image_path in unlabeled_samples:
        case_name = f"case_{case_index:04d}.png"
        case_index += 1
        write_png(image_path, output_root / "unlabeled" / "image" / case_name, "RGB")
        records.append(record("unlabeled", case_name, image_path))

    split_counts = {
        "labeled": len(split_samples["labeled"]),
        "unlabeled": len(unlabeled_samples),
        "val": len(split_samples["val"]),
        "test": len(split_samples["test"]),
    }
    manifest = {
        "source_root": str(source_root),
        "output_root": str(output_root),
        "seed": args.seed,
        "target_label": args.target_label,
        "multi_class": args.multi_class,
        "split_policy": "shuffle labeled samples, then 80% train / 10% val / 10% test by default",
        "split_counts": split_counts,
        "samples": records,
    }

    with (output_root / args.csv_name).open("w", newline="", encoding="utf-8-sig") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=["split", "new_name", "source_image", "source_mask"])
        writer.writeheader()
        writer.writerows(records)

    (output_root / args.manifest_name).write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps({"output_root": str(output_root), "split_counts": split_counts}, indent=2))


if __name__ == "__main__":
    main()
