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
        description="Prepare 260513_data semi-supervised datasets with labeled training ratios."
    )
    parser.add_argument("--source-root", type=Path, default=Path("./data/260513_data"))
    parser.add_argument("--sampledata-root", type=Path, default=Path("./SampleData"))
    parser.add_argument(
        "--output-prefix",
        type=str,
        default="260513_data_labeled",
        help="Output folders become <output-prefix><ratio>pct, e.g. 260513_data_labeled10pct.",
    )
    parser.add_argument(
        "--ratios",
        nargs="+",
        type=float,
        default=[0.10, 0.20, 0.30],
        help="Labeled ratio inside the annotated training pool after val/test are held out.",
    )
    parser.add_argument("--val-ratio", type=float, default=0.10)
    parser.add_argument("--test-ratio", type=float, default=0.10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--mask-mode",
        choices=("multiclass", "foreground", "target"),
        default="multiclass",
        help="multiclass preserves 0/1/2; foreground exports all nonzero labels as 255; target exports one label as 255.",
    )
    parser.add_argument("--target-label", type=int, default=1)
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


def write_image(source: Path, target: Path):
    with Image.open(source) as image:
        image.convert("RGB").save(target)


def write_mask(source: Path, target: Path, mode: str, target_label: int):
    with Image.open(source) as image:
        mask = np.asarray(image.convert("L"), dtype=np.uint8)
    if mode == "foreground":
        mask = (mask > 0).astype(np.uint8) * 255
    elif mode == "target":
        mask = (mask == target_label).astype(np.uint8) * 255
    Image.fromarray(mask.astype(np.uint8), mode="L").save(target)


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
        samples.append({"source_image": image_path, "source_mask": mask_path})
    if missing_masks:
        raise FileNotFoundError("Missing masks:\n" + "\n".join(missing_masks[:20]))
    return samples


def discover_unlabeled(source_root: Path):
    unlabeled_dir = source_root / "unlabelled"
    return sorted(path for path in unlabeled_dir.iterdir() if is_image(path))


def ratio_tag(ratio: float):
    value = ratio * 100
    if abs(value - round(value)) < 1e-8:
        return str(int(round(value)))
    return f"{value:g}".replace(".", "p")


def record(split, case_name, source_image, source_mask=None, source_pool=""):
    return {
        "split": split,
        "new_name": case_name,
        "source_image": str(source_image),
        "source_mask": str(source_mask) if source_mask else "",
        "source_pool": source_pool,
    }


def materialize_dataset(
    output_root: Path,
    labeled_train,
    annotated_unlabeled_train,
    raw_unlabeled_train,
    val_samples,
    test_samples,
    args,
):
    ensure_clean_output(output_root)
    records = []
    case_index = 1

    def next_case_name():
        nonlocal case_index
        case_name = f"case_{case_index:04d}.png"
        case_index += 1
        return case_name

    for split, samples in (("labeled", labeled_train), ("val", val_samples), ("test", test_samples)):
        for sample in samples:
            case_name = next_case_name()
            write_image(sample["source_image"], output_root / split / "image" / case_name)
            write_mask(sample["source_mask"], output_root / split / "mask" / case_name, args.mask_mode, args.target_label)
            records.append(record(split, case_name, sample["source_image"], sample["source_mask"], "annotated"))

    for sample in annotated_unlabeled_train:
        case_name = next_case_name()
        write_image(sample["source_image"], output_root / "unlabeled" / "image" / case_name)
        records.append(record("unlabeled", case_name, sample["source_image"], sample["source_mask"], "annotated_train_not_labeled"))

    for image_path in raw_unlabeled_train:
        case_name = next_case_name()
        write_image(image_path, output_root / "unlabeled" / "image" / case_name)
        records.append(record("unlabeled", case_name, image_path, None, "raw_unlabelled"))

    counts = {
        "labeled": len(labeled_train),
        "unlabeled": len(annotated_unlabeled_train) + len(raw_unlabeled_train),
        "val": len(val_samples),
        "test": len(test_samples),
    }
    manifest = {
        "source_root": str(args.source_root.resolve()),
        "output_root": str(output_root.resolve()),
        "seed": args.seed,
        "mask_mode": args.mask_mode,
        "target_label": args.target_label if args.mask_mode == "target" else None,
        "split_policy": (
            "Hold out val/test from annotated samples, then sample labeled ratio "
            "inside the remaining annotated training pool; other annotated training "
            "samples are image-only unlabeled training samples."
        ),
        "split_counts": counts,
        "samples": records,
    }

    with (output_root / "split_record.csv").open("w", newline="", encoding="utf-8-sig") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=["split", "new_name", "source_image", "source_mask", "source_pool"])
        writer.writeheader()
        writer.writerows(records)
    (output_root / "split_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return counts


def main():
    args = parse_args()
    source_root = args.source_root.resolve()
    sampledata_root = args.sampledata_root.resolve()
    labeled_samples = discover_labeled(source_root)
    raw_unlabeled = discover_unlabeled(source_root)

    shuffled = list(labeled_samples)
    random.Random(args.seed).shuffle(shuffled)
    val_count = round(len(shuffled) * args.val_ratio)
    test_count = round(len(shuffled) * args.test_ratio)
    if val_count + test_count >= len(shuffled):
        raise ValueError("val/test ratios leave no annotated training samples.")

    test_start = len(shuffled) - test_count
    val_start = test_start - val_count
    train_pool = shuffled[:val_start]
    val_samples = shuffled[val_start:test_start]
    test_samples = shuffled[test_start:]

    results = []
    for ratio in args.ratios:
        if not 0 < ratio <= 1:
            raise ValueError(f"Ratio must be in (0, 1], got {ratio}.")
        labeled_count = round(len(train_pool) * ratio)
        labeled_count = max(1, min(labeled_count, len(train_pool)))
        labeled_train = train_pool[:labeled_count]
        annotated_unlabeled_train = train_pool[labeled_count:]
        output_root = sampledata_root / f"{args.output_prefix}{ratio_tag(ratio)}pct"
        counts = materialize_dataset(
            output_root,
            labeled_train,
            annotated_unlabeled_train,
            raw_unlabeled,
            val_samples,
            test_samples,
            args,
        )
        results.append(
            {
                "output_root": str(output_root),
                "ratio": ratio,
                "annotated_total": len(labeled_samples),
                "annotated_train_pool": len(train_pool),
                "raw_unlabeled": len(raw_unlabeled),
                "split_counts": counts,
            }
        )

    print(json.dumps({"results": results}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
