import argparse
import csv
import logging
import os
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Optional

import cv2
import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Model.model import KnowSAM
from utils.lateral_fissure_measurement import (
    annotate_lateral_fissure_measurement,
    measure_lateral_fissure,
    measurement_to_row,
    parse_pixel_spacing,
)


IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")
GENERATED_SUFFIXES = (
    "_pred_mask",
    "_pred_color",
    "_overlay",
    "_measurement",
    "_prob_class",
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Batch inference for A3 ultrasound image folders. "
                    "Input root should contain patient subfolders with images."
    )
    parser.add_argument(
        "--input-root",
        required=True,
        help="Root folder. Patient folders under this root contain A3 images.",
    )
    parser.add_argument(
        "--output-root",
        default=None,
        help="Optional output root. If omitted, results are written beside each image.",
    )
    parser.add_argument(
        "--model-path",
        default="./Results/train_260513_data_label1_v100_semi_106_117_13_13/SGDL_best_model.pth",
        help="KnowSAM/SGDL checkpoint path, or PASS checkpoint when --variant a3_pass.",
    )
    parser.add_argument(
        "--variant",
        choices=("knowsam", "a3_pass"),
        default="knowsam",
        help="Model variant to run.",
    )
    parser.add_argument(
        "--a3-pass-dir",
        default="./variants/A3_PASS_KnowSAM",
        help="A3-PASS variant folder containing state_modules.py.",
    )
    parser.add_argument(
        "--head",
        choices=("pass", "sgdl"),
        default="pass",
        help="For --variant a3_pass, choose PASS output or SGDL fusion output.",
    )
    parser.add_argument("--num-classes", type=int, default=2)
    parser.add_argument("--in-channels", type=int, default=3)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument(
        "--device",
        default="auto",
        help="cuda, cpu, cuda:0, or auto.",
    )
    parser.add_argument(
        "--include-keyword",
        default="",
        help="Only process images whose filename contains this keyword, e.g. A3. Empty means all images.",
    )
    parser.add_argument(
        "--save-prob",
        action="store_true",
        help="Also save foreground probability maps as uint8 PNG files.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing mask/overlay outputs.",
    )
    parser.add_argument(
        "--pixel-spacing",
        default="",
        help="Optional pixel spacing in mm, either one value or row,col.",
    )
    parser.add_argument(
        "--disable-measurement",
        action="store_true",
        help="Disable lateral fissure width/depth measurement outputs.",
    )
    parser.add_argument("--pass-state-size", type=int, default=64)
    parser.add_argument("--pass-state-dim", type=int, default=64)
    parser.add_argument("--pass-base-channels", type=int, default=32)
    return parser.parse_args()


def setup_logger(summary_dir: Path):
    summary_dir.mkdir(parents=True, exist_ok=True)
    log_path = summary_dir / "batch_inference.log"
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s.%(msecs)03d] %(message)s",
        datefmt="%H:%M:%S",
        handlers=[
            logging.FileHandler(log_path, encoding="utf-8"),
            logging.StreamHandler(),
        ],
        force=True,
    )
    return log_path


def resolve_device(device_arg: str):
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)


def imread_unicode(path: Path):
    data = np.fromfile(str(path), dtype=np.uint8)
    image = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Failed to read image: {path}")
    return image


def imwrite_unicode(path: Path, image: np.ndarray):
    path.parent.mkdir(parents=True, exist_ok=True)
    ext = path.suffix if path.suffix else ".png"
    ok, encoded = cv2.imencode(ext, image)
    if not ok:
        raise OSError(f"Failed to encode image: {path}")
    encoded.tofile(str(path))


def should_skip_image(path: Path, include_keyword: str):
    if path.suffix.lower() not in IMAGE_EXTENSIONS:
        return True
    stem = path.stem.lower()
    if any(stem.endswith(suffix) for suffix in GENERATED_SUFFIXES):
        return True
    if "_prob_class" in stem:
        return True
    if include_keyword and include_keyword.lower() not in path.name.lower():
        return True
    return False


def collect_images(input_root: Path, include_keyword: str):
    return sorted(
        path for path in input_root.rglob("*")
        if path.is_file() and not should_skip_image(path, include_keyword)
    )


def build_model_args(args):
    return SimpleNamespace(
        num_classes=args.num_classes,
        in_channels=args.in_channels,
        image_size=args.image_size,
        point_nums=5,
        box_nums=1,
        mod="sam_adpt",
        model_type="vit_b",
        thd=False,
        multimask=False,
        encoder_adapter=True,
        pass_state_size=args.pass_state_size,
        pass_state_dim=args.pass_state_dim,
        pass_base_channels=args.pass_base_channels,
    )


def normalize_checkpoint(checkpoint):
    if isinstance(checkpoint, dict):
        for key in ("state_dict", "model_state_dict", "model"):
            if key in checkpoint and isinstance(checkpoint[key], dict):
                checkpoint = checkpoint[key]
                break
    if isinstance(checkpoint, dict) and any(key.startswith("module.") for key in checkpoint):
        checkpoint = {key.replace("module.", "", 1): value for key, value in checkpoint.items()}
    return checkpoint


def build_model(args, device):
    model_args = build_model_args(args)
    if args.variant == "knowsam":
        model = KnowSAM(model_args, bilinear=False)
    else:
        a3_pass_dir = (REPO_ROOT / args.a3_pass_dir).resolve()
        if not (a3_pass_dir / "state_modules.py").exists():
            raise FileNotFoundError(f"state_modules.py not found in: {a3_pass_dir}")
        sys.path.insert(0, str(a3_pass_dir))
        from state_modules import A3PASSNet
        model = A3PASSNet(model_args)

    checkpoint = torch.load(args.model_path, map_location=device)
    model.load_state_dict(normalize_checkpoint(checkpoint))
    model.to(device)
    model.eval()
    return model


def preprocess_image(image_bgr: np.ndarray, image_size: int):
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(image_rgb, (image_size, image_size), interpolation=cv2.INTER_NEAREST)
    tensor = resized.astype(np.float32) / 255.0
    tensor = np.transpose(tensor, (2, 0, 1))[None, ...]
    return torch.from_numpy(tensor)


def infer_softmax(model, image_tensor: torch.Tensor, variant: str, head: str):
    if variant == "knowsam":
        _, _, _, _, fusion_logits = model(image_tensor)
        return torch.softmax(fusion_logits, dim=1)

    out = model(image_tensor)
    if head == "sgdl":
        return out["fusion_soft"]
    return out["pass_soft"]


def colorize_mask(mask: np.ndarray):
    colors = np.array(
        [
            [0, 0, 0],
            [0, 0, 255],
            [0, 255, 0],
            [255, 0, 0],
            [0, 255, 255],
            [255, 0, 255],
        ],
        dtype=np.uint8,
    )
    color = np.zeros((*mask.shape, 3), dtype=np.uint8)
    for class_idx in range(1, int(mask.max()) + 1):
        color[mask == class_idx] = colors[class_idx % len(colors)]
    return color


def overlay_mask(image_bgr: np.ndarray, mask: np.ndarray, alpha=0.35):
    color_layer = colorize_mask(mask)
    return cv2.addWeighted(image_bgr.copy(), 1.0, color_layer, alpha, 0)


def postprocess_prediction(prob_map: torch.Tensor, args, original_size):
    prob = prob_map.squeeze(0).detach().cpu().numpy()
    if args.num_classes == 2:
        foreground = prob[1]
        foreground = cv2.resize(
            foreground,
            original_size,
            interpolation=cv2.INTER_LINEAR,
        )
        mask = (foreground > args.threshold).astype(np.uint8)
        prob_outputs = {1: (np.clip(foreground, 0.0, 1.0) * 255).astype(np.uint8)}
    else:
        channels = [
            cv2.resize(prob[class_idx], original_size, interpolation=cv2.INTER_LINEAR)
            for class_idx in range(args.num_classes)
        ]
        resized_prob = np.stack(channels, axis=0)
        mask = np.argmax(resized_prob, axis=0).astype(np.uint8)
        prob_outputs = {
            class_idx: (np.clip(resized_prob[class_idx], 0.0, 1.0) * 255).astype(np.uint8)
            for class_idx in range(1, args.num_classes)
        }
    return mask, prob_outputs


def output_dir_for_image(image_path: Path, input_root: Path, output_root: Optional[Path]):
    if output_root is None:
        return image_path.parent
    relative_parent = image_path.parent.relative_to(input_root)
    return output_root / relative_parent


def output_paths(image_path: Path, out_dir: Path):
    return {
        "mask": out_dir / f"{image_path.stem}_pred_mask.png",
        "color": out_dir / f"{image_path.stem}_pred_color.png",
        "overlay": out_dir / f"{image_path.stem}_overlay.png",
        "measurement": out_dir / f"{image_path.stem}_measurement.png",
    }


def main():
    args = parse_args()
    input_root = Path(args.input_root).resolve()
    if not input_root.exists():
        raise FileNotFoundError(f"Input root not found: {input_root}")

    output_root = Path(args.output_root).resolve() if args.output_root else None
    summary_dir = output_root if output_root else input_root
    log_path = setup_logger(summary_dir)
    device = resolve_device(args.device)
    pixel_spacing = parse_pixel_spacing(args.pixel_spacing)

    logging.info("Input root: %s", input_root)
    logging.info("Output root: %s", output_root if output_root else "same folder as each image")
    logging.info("Device: %s", device)
    logging.info("Variant: %s", args.variant)
    logging.info("Checkpoint: %s", Path(args.model_path).resolve())

    image_paths = collect_images(input_root, args.include_keyword)
    logging.info("Found %d image(s)", len(image_paths))
    if not image_paths:
        raise RuntimeError("No images found. Check --input-root, --include-keyword, and supported extensions.")

    model = build_model(args, device)
    rows = []

    with torch.no_grad():
        for index, image_path in enumerate(image_paths, start=1):
            image_bgr = imread_unicode(image_path)
            height, width = image_bgr.shape[:2]
            out_dir = output_dir_for_image(image_path, input_root, output_root)
            paths = output_paths(image_path, out_dir)

            expected_outputs = [paths["mask"], paths["overlay"]]
            if not args.disable_measurement:
                expected_outputs.append(paths["measurement"])
            if not args.overwrite and all(path.exists() for path in expected_outputs):
                logging.info("[%d/%d] skip existing: %s", index, len(image_paths), image_path)
                rows.append({
                    "image_path": str(image_path),
                    "output_dir": str(out_dir),
                    "status": "skipped_existing",
                    "positive_pixels": "",
                    "fissure_measurement_status": "",
                    "fissure_component_count": "",
                    "fissure_area_px": "",
                    "fissure_depth_px": "",
                    "fissure_width_px": "",
                    "fissure_mean_width_px": "",
                    "fissure_orientation_deg": "",
                })
                continue

            image_tensor = preprocess_image(image_bgr, args.image_size).to(device)
            prob_map = infer_softmax(model, image_tensor, args.variant, args.head)
            mask, prob_outputs = postprocess_prediction(prob_map, args, (width, height))

            if args.num_classes == 2:
                mask_to_save = (mask * 255).astype(np.uint8)
            else:
                mask_to_save = mask.astype(np.uint8)
            imwrite_unicode(paths["mask"], mask_to_save)
            imwrite_unicode(paths["color"], colorize_mask(mask))
            imwrite_unicode(paths["overlay"], overlay_mask(image_bgr, mask))

            measurement_row = {}
            if not args.disable_measurement:
                measurement = measure_lateral_fissure(mask, pixel_spacing=pixel_spacing)
                measurement_row = measurement_to_row(measurement)
                measurement_overlay = annotate_lateral_fissure_measurement(
                    image_bgr,
                    mask,
                    measurement=measurement,
                    pixel_spacing=pixel_spacing,
                )
                imwrite_unicode(paths["measurement"], measurement_overlay)

            if args.save_prob:
                for class_idx, prob_uint8 in prob_outputs.items():
                    prob_path = out_dir / f"{image_path.stem}_prob_class{class_idx}.png"
                    imwrite_unicode(prob_path, prob_uint8)

            positive_pixels = int((mask > 0).sum())
            rows.append({
                "image_path": str(image_path),
                "output_dir": str(out_dir),
                "status": "ok",
                "positive_pixels": positive_pixels,
                **measurement_row,
            })
            logging.info(
                "[%d/%d] saved mask/overlay for %s positive_pixels=%d fissure_width_px=%s fissure_depth_px=%s",
                index,
                len(image_paths),
                image_path,
                positive_pixels,
                measurement_row.get("fissure_width_px", ""),
                measurement_row.get("fissure_depth_px", ""),
            )

    summary_path = summary_dir / "batch_inference_summary.csv"
    with open(summary_path, "w", newline="", encoding="utf-8-sig") as csv_file:
        fieldnames = sorted({key for row in rows for key in row.keys()})
        ordered = ["image_path", "output_dir", "status", "positive_pixels"]
        fieldnames = ordered + [key for key in fieldnames if key not in ordered]
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    logging.info("Summary CSV: %s", summary_path)
    logging.info("Log file: %s", log_path)
    print(f"processed_or_checked={len(rows)}")
    print(f"summary_csv={summary_path}")
    print(f"log_file={log_path}")


if __name__ == "__main__":
    main()
