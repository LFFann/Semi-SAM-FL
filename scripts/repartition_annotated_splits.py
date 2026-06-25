import argparse
import json
import random
import shutil
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description="Repartition labeled/val splits within the annotated pool.")
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("./SampleData/260513_data_label1"),
        help="KnowSAM dataset root.",
    )
    parser.add_argument(
        "--labeled-count",
        type=int,
        required=True,
        help="Target number of annotated samples in labeled split.",
    )
    parser.add_argument(
        "--val-count",
        type=int,
        required=True,
        help="Target number of annotated samples in val split.",
    )
    parser.add_argument(
        "--test-count",
        type=int,
        default=0,
        help="Target number of annotated samples in test split.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for repartition.",
    )
    parser.add_argument(
        "--report-name",
        type=str,
        default="annotated_repartition_report.json",
        help="Report filename written under output root.",
    )
    return parser.parse_args()


def move_case(output_root: Path, case_name: str, source_split: str, target_split: str):
    if source_split == target_split:
        return
    for subdir in ("image", "mask"):
        source_path = output_root / source_split / subdir / case_name
        target_path = output_root / target_split / subdir / case_name
        if not source_path.exists():
            continue
        target_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(source_path), str(target_path))


def main():
    args = parse_args()
    output_root = args.output_root.resolve()
    manifest_path = output_root / "split_manifest.json"
    report_path = output_root / args.report_name

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    annotated_cases = []
    for split in ("labeled", "val", "test"):
        image_dir = output_root / split / "image"
        if not image_dir.exists():
            continue
        for image_path in sorted(image_dir.glob("case_*.png")):
            annotated_cases.append({"case_name": image_path.name, "current_split": split})

    total_annotated = len(annotated_cases)
    if args.labeled_count + args.val_count + args.test_count != total_annotated:
        raise ValueError(
            f"Target counts ({args.labeled_count} + {args.val_count} + {args.test_count}) do not match "
            f"annotated pool size ({total_annotated})."
        )

    shuffled = list(annotated_cases)
    random.Random(args.seed).shuffle(shuffled)
    labeled_cases = {item["case_name"] for item in shuffled[:args.labeled_count]}
    val_start = args.labeled_count
    val_end = val_start + args.val_count
    val_cases = {item["case_name"] for item in shuffled[val_start:val_end]}
    test_cases = {item["case_name"] for item in shuffled[val_end:]}

    moved_cases = []
    for item in annotated_cases:
        case_name = item["case_name"]
        source_split = item["current_split"]
        if case_name in labeled_cases:
            target_split = "labeled"
        elif case_name in val_cases:
            target_split = "val"
        else:
            target_split = "test"
        move_case(output_root, case_name, source_split, target_split)
        if source_split != target_split:
            moved_cases.append({
                "case_name": case_name,
                "from": source_split,
                "to": target_split,
            })

    for sample in manifest.get("samples", []):
        if sample.get("new_name") in labeled_cases:
            sample["split"] = "labeled"
        elif sample.get("new_name") in val_cases:
            sample["split"] = "val"
        elif sample.get("new_name") in test_cases:
            sample["split"] = "test"

    current_counts = {
        "labeled": args.labeled_count,
        "unlabeled": len(list((output_root / "unlabeled" / "image").glob("case_*.png"))),
        "val": args.val_count,
        "test": args.test_count,
    }
    report = {
        "seed": args.seed,
        "annotated_total": total_annotated,
        "current_split_counts": current_counts,
        "moved_count": len(moved_cases),
        "moved_cases": moved_cases,
    }
    manifest["current_split_counts"] = current_counts
    manifest["annotated_repartition"] = report

    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({
        "current_split_counts": current_counts,
        "moved_count": len(moved_cases),
        "manifest_path": str(manifest_path),
        "report_path": str(report_path),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
