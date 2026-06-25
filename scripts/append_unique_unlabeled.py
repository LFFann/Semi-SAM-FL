import argparse
import hashlib
import json
import re
from pathlib import Path

import cv2
import numpy as np


IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")


def parse_args():
    parser = argparse.ArgumentParser(description="Append unique A3 images into KnowSAM unlabeled split.")
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("./SampleData/260513_data_label1"),
        help="KnowSAM dataset root.",
    )
    parser.add_argument(
        "--input-dirs",
        nargs="+",
        type=Path,
        required=True,
        help="Input directories. Flat directories add all images; nested directories add exact A3 slices from case folders.",
    )
    parser.add_argument(
        "--report-name",
        type=str,
        default="unlabeled_append_report.json",
        help="Report filename written under output root.",
    )
    return parser.parse_args()


def read_rgb(image_path: Path):
    data = np.fromfile(str(image_path), dtype=np.uint8)
    image = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Failed to read image: {image_path}")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def write_png(image_path: Path, image_rgb: np.ndarray):
    image_path.parent.mkdir(parents=True, exist_ok=True)
    image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    success, encoded = cv2.imencode(".png", image_bgr)
    if not success:
        raise ValueError(f"Failed to encode image for {image_path}")
    encoded.tofile(str(image_path))


def image_digest(image_rgb: np.ndarray):
    return hashlib.md5(image_rgb.tobytes()).hexdigest() + f"_{image_rgb.shape[0]}x{image_rgb.shape[1]}x{image_rgb.shape[2]}"


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


def build_existing_hashes(output_root: Path):
    existing = {}
    for split in ("labeled", "unlabeled", "val"):
        image_dir = output_root / split / "image"
        if not image_dir.exists():
            continue
        for image_path in sorted(image_dir.glob("*")):
            image_rgb = read_rgb(image_path)
            existing.setdefault(image_digest(image_rgb), []).append(str(image_path))
    return existing


def discover_flat_images(input_dir: Path):
    samples = []
    for image_path in sorted(input_dir.iterdir()):
        if image_path.is_file() and image_path.suffix.lower() in IMAGE_SUFFIXES:
            samples.append(
                {
                    "source_dir": str(input_dir),
                    "source_parent": input_dir.name,
                    "source_image": image_path.name,
                    "image_path": image_path,
                    "discovery_mode": "flat",
                }
            )
    return samples


def discover_nested_a3_images(input_dir: Path):
    samples = []
    for case_dir in sorted([path for path in input_dir.iterdir() if path.is_dir()]):
        image_path = None
        for suffix in IMAGE_SUFFIXES + tuple(s.upper() for s in IMAGE_SUFFIXES):
            candidate = case_dir / f"A3{suffix}"
            if candidate.exists():
                image_path = candidate
                break
        if image_path is None:
            continue
        samples.append(
            {
                "source_dir": str(input_dir),
                "source_parent": case_dir.name,
                "source_image": image_path.name,
                "image_path": image_path,
                "discovery_mode": "nested_a3",
            }
        )
    return samples


def discover_candidates(input_dirs):
    candidates = []
    for input_dir in input_dirs:
        if not input_dir.exists():
            raise FileNotFoundError(f"Input directory not found: {input_dir}")
        nested_candidates = discover_nested_a3_images(input_dir)
        if nested_candidates:
            candidates.extend(nested_candidates)
        else:
            candidates.extend(discover_flat_images(input_dir))
    return candidates


def main():
    args = parse_args()
    output_root = args.output_root.resolve()
    unlabeled_dir = output_root / "unlabeled" / "image"
    unlabeled_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_root / "split_manifest.json"
    report_path = output_root / args.report_name

    existing_hashes = build_existing_hashes(output_root)
    manifest = load_manifest(manifest_path)
    candidates = discover_candidates([path.resolve() for path in args.input_dirs])

    next_index = next_case_index(output_root)
    seen_new_hashes = set()
    report = {
        "output_root": str(output_root),
        "candidate_count": len(candidates),
        "added_count": 0,
        "existing_count": 0,
        "duplicate_within_new_count": 0,
        "added_samples": [],
        "existing_samples": [],
        "duplicate_within_new_samples": [],
    }

    for candidate in candidates:
        image_rgb = read_rgb(candidate["image_path"])
        digest = image_digest(image_rgb)
        if digest in existing_hashes:
            report["existing_count"] += 1
            report["existing_samples"].append(
                {
                    **candidate,
                    "image_path": str(candidate["image_path"]),
                    "matches": existing_hashes[digest],
                    "hash": digest,
                }
            )
            continue
        if digest in seen_new_hashes:
            report["duplicate_within_new_count"] += 1
            report["duplicate_within_new_samples"].append(
                {
                    **candidate,
                    "image_path": str(candidate["image_path"]),
                    "hash": digest,
                }
            )
            continue

        case_name = f"case_{next_index:04d}.png"
        next_index += 1
        destination = unlabeled_dir / case_name
        write_png(destination, image_rgb)
        seen_new_hashes.add(digest)
        existing_hashes.setdefault(digest, []).append(str(destination))

        sample_record = {
            "split": "unlabeled",
            "new_name": case_name,
            "source_dir": candidate["source_dir"],
            "source_parent": candidate["source_parent"],
            "source_image": candidate["source_image"],
            "height": int(image_rgb.shape[0]),
            "width": int(image_rgb.shape[1]),
            "hash": digest,
            "discovery_mode": candidate["discovery_mode"],
        }
        manifest.setdefault("samples", []).append(sample_record)
        report["added_count"] += 1
        report["added_samples"].append(sample_record)

    manifest["last_unlabeled_append"] = {
        "added_count": report["added_count"],
        "existing_count": report["existing_count"],
        "duplicate_within_new_count": report["duplicate_within_new_count"],
        "report_path": str(report_path),
    }

    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps({
        "candidate_count": report["candidate_count"],
        "added_count": report["added_count"],
        "existing_count": report["existing_count"],
        "duplicate_within_new_count": report["duplicate_within_new_count"],
        "manifest_path": str(manifest_path),
        "report_path": str(report_path),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
