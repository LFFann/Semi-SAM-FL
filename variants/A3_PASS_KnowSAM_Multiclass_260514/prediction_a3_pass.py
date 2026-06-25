import argparse
import csv
import json
import logging
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "..", ".."))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)
if REPO_ROOT not in sys.path:
    sys.path.insert(1, REPO_ROOT)

from dataloader.dataset import build_Dataset
from dataloader.transforms import build_transforms
from state_modules import A3PASSNet
from utils.lateral_fissure_measurement import (
    annotate_lateral_fissure_measurement,
    measure_lateral_fissure,
    measurement_to_row,
    parse_pixel_spacing,
)
from utils.training_monitor import save_evaluation_artifacts
from utils.utils import multiclass_segmentation_metrics


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="./SampleData")
    parser.add_argument("--dataset", type=str, default="/260513_data_multiclass")
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--num_classes", type=int, default=3)
    parser.add_argument("--in_channels", type=int, default=3)
    parser.add_argument("-lr", type=float, default=1e-4)
    parser.add_argument("-UNet_lr", type=float, default=0.0025)
    parser.add_argument("-VNet_lr", type=float, default=0.0025)
    parser.add_argument("--image_size", type=int, default=256)
    parser.add_argument("--point_nums", type=int, default=5)
    parser.add_argument("--box_nums", type=int, default=1)
    parser.add_argument("--mod", type=str, default="sam_adpt")
    parser.add_argument("--model_type", type=str, default="vit_b")
    parser.add_argument("-thd", type=bool, default=False)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--labeled_bs", type=int, default=1)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--multimask", type=bool, default=False)
    parser.add_argument("--encoder_adapter", type=bool, default=True)
    parser.add_argument("--sam_checkpoint", type=str, default="./sam_vit_b_01ec64.pth")
    parser.add_argument("--num_workers", type=int, default=0)

    parser.add_argument("--PASS_model_path", type=str, default="./Results/A3_PASS_KnowSAM_Multiclass_V100_bs32_10k_106_117_13_13_260514/fold_0/PASS_best_model.pth")
    parser.add_argument("--save_dir", type=str, default="./Results/A3_PASS_KnowSAM_Multiclass_V100_bs32_10k_106_117_13_13_260514/fold_0/prediction_test")

    parser.add_argument("--pass_state_size", type=int, default=64)
    parser.add_argument("--pass_state_dim", type=int, default=64)
    parser.add_argument("--pass_base_channels", type=int, default=32)
    parser.add_argument("--pass_state_lr", type=float, default=0.001)
    parser.add_argument("--measurement_class", type=int, default=1,
                        help="class index used for lateral fissure measurement")
    parser.add_argument("--pixel_spacing", type=str, default="",
                        help="optional pixel spacing in mm, either one value or row,col")
    parser.add_argument("--disable_measurement", action="store_true",
                        help="disable lateral fissure width/depth measurement outputs")
    return parser.parse_args()


def setup_logger(save_dir: Path):
    save_dir.mkdir(parents=True, exist_ok=True)
    log_path = save_dir / "prediction.log"
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s.%(msecs)03d] %(message)s",
        datefmt="%H:%M:%S",
        handlers=[logging.FileHandler(log_path, encoding="utf-8"), logging.StreamHandler()],
        force=True,
    )
    return log_path


def to_uint8_rgb(image_tensor: torch.Tensor):
    image = image_tensor.detach().cpu().numpy()
    image = np.transpose(image, (1, 2, 0))
    image = np.clip(image, 0, 255).astype(np.uint8)
    return image


CLASS_COLORS = np.array([
    [0, 0, 0],
    [0, 0, 255],
    [0, 255, 0],
    [255, 0, 0],
    [0, 255, 255],
    [255, 0, 255],
], dtype=np.uint8)


def label_from_softmax(prob_map: torch.Tensor):
    return torch.argmax(prob_map, dim=1).squeeze(0).detach().cpu().numpy().astype(np.uint8)


def colorize_mask(mask: np.ndarray):
    color = np.zeros((*mask.shape, 3), dtype=np.uint8)
    for class_idx in range(1, int(mask.max()) + 1):
        color[mask == class_idx] = CLASS_COLORS[class_idx % len(CLASS_COLORS)]
    return color


def overlay_multiclass(image_bgr: np.ndarray, mask: np.ndarray, alpha=0.35):
    color_layer = colorize_mask(mask)
    return cv2.addWeighted(image_bgr.copy(), 1.0, color_layer, alpha, 0)


def safe_eval(test_label, prob_map, num_classes):
    try:
        return multiclass_segmentation_metrics(test_label, prob_map, num_classes)
    except Exception as exc:
        logging.warning("Metric computation failed: %s", exc)
        result = {}
        for class_idx in range(1, num_classes):
            result[f"class_{class_idx}_dice"] = float("nan")
            result[f"class_{class_idx}_iou"] = float("nan")
            result[f"class_{class_idx}_hd95"] = float("nan")
        result["avg_dice"] = float("nan")
        result["avg_iou"] = float("nan")
        result["avg_hd95"] = float("nan")
        return result


def prefixed_metrics(prefix, metrics):
    renamed = {}
    for key, value in metrics.items():
        renamed[f"{prefix}_{key}"] = float(value)
    return renamed


def format_metrics(prefix, metrics, num_classes):
    parts = [
        "%s_avg_dice=%.6f" % (prefix, metrics["avg_dice"]),
        "%s_avg_iou=%.6f" % (prefix, metrics["avg_iou"]),
        "%s_avg_hd95=%.6f" % (prefix, metrics["avg_hd95"]),
    ]
    for class_idx in range(1, num_classes):
        parts.extend([
            "%s_class_%d_dice=%.6f" % (prefix, class_idx, metrics[f"class_{class_idx}_dice"]),
            "%s_class_%d_iou=%.6f" % (prefix, class_idx, metrics[f"class_{class_idx}_iou"]),
            "%s_class_%d_hd95=%.6f" % (prefix, class_idx, metrics[f"class_{class_idx}_hd95"]),
        ])
    return " ".join(parts)


def save_case_outputs(save_dir: Path, case_name: str, ori_image: np.ndarray, gt_mask: np.ndarray,
                      pass_mask: np.ndarray, sgdl_mask: np.ndarray,
                      measurement_overlay_pass: np.ndarray = None,
                      measurement_overlay_sgdl: np.ndarray = None):
    dirs = {
        "original": save_dir / "original",
        "gt_mask": save_dir / "gt_mask",
        "gt_color": save_dir / "gt_color",
        "pred_mask_pass": save_dir / "pred_mask_pass",
        "pred_mask_sgdl": save_dir / "pred_mask_sgdl",
        "pred_color_pass": save_dir / "pred_color_pass",
        "pred_color_sgdl": save_dir / "pred_color_sgdl",
        "overlay_pass": save_dir / "overlay_pass",
        "overlay_sgdl": save_dir / "overlay_sgdl",
    }
    if measurement_overlay_pass is not None:
        dirs["measurement_overlay_pass"] = save_dir / "measurement_overlay_pass"
    if measurement_overlay_sgdl is not None:
        dirs["measurement_overlay_sgdl"] = save_dir / "measurement_overlay_sgdl"
    for directory in dirs.values():
        directory.mkdir(parents=True, exist_ok=True)

    cv2.imwrite(str(dirs["original"] / case_name), ori_image)
    cv2.imwrite(str(dirs["gt_mask"] / case_name), gt_mask)
    cv2.imwrite(str(dirs["gt_color"] / case_name), colorize_mask(gt_mask))
    cv2.imwrite(str(dirs["pred_mask_pass"] / case_name), pass_mask)
    cv2.imwrite(str(dirs["pred_mask_sgdl"] / case_name), sgdl_mask)
    cv2.imwrite(str(dirs["pred_color_pass"] / case_name), colorize_mask(pass_mask))
    cv2.imwrite(str(dirs["pred_color_sgdl"] / case_name), colorize_mask(sgdl_mask))
    cv2.imwrite(str(dirs["overlay_pass"] / case_name), overlay_multiclass(ori_image, pass_mask))
    cv2.imwrite(str(dirs["overlay_sgdl"] / case_name), overlay_multiclass(ori_image, sgdl_mask))
    if measurement_overlay_pass is not None:
        cv2.imwrite(str(dirs["measurement_overlay_pass"] / case_name), measurement_overlay_pass)
    if measurement_overlay_sgdl is not None:
        cv2.imwrite(str(dirs["measurement_overlay_sgdl"] / case_name), measurement_overlay_sgdl)


def add_measurement_summary(summary: dict, case_metrics: list, prefix: str):
    status_key = f"{prefix}_measurement_status"
    width_key = f"{prefix}_width_px"
    depth_key = f"{prefix}_depth_px"
    mean_width_key = f"{prefix}_mean_width_px"
    measurable = [item for item in case_metrics if item.get(status_key) == "ok"]
    summary[f"num_measurable_{prefix}s"] = len(measurable)
    summary[f"avg_{prefix}_width_px"] = float(np.mean([item[width_key] for item in measurable])) if measurable else float("nan")
    summary[f"avg_{prefix}_depth_px"] = float(np.mean([item[depth_key] for item in measurable])) if measurable else float("nan")
    summary[f"avg_{prefix}_mean_width_px"] = float(np.mean([item[mean_width_key] for item in measurable])) if measurable else float("nan")
    mm_width_key = f"{prefix}_width_mm"
    if measurable and mm_width_key in measurable[0]:
        mm_depth_key = f"{prefix}_depth_mm"
        mm_mean_width_key = f"{prefix}_mean_width_mm"
        summary[f"avg_{prefix}_width_mm"] = float(np.mean([item[mm_width_key] for item in measurable]))
        summary[f"avg_{prefix}_depth_mm"] = float(np.mean([item[mm_depth_key] for item in measurable]))
        summary[f"avg_{prefix}_mean_width_mm"] = float(np.mean([item[mm_mean_width_key] for item in measurable]))


def main():
    args = parse_args()
    save_dir = Path(args.save_dir)
    log_path = setup_logger(save_dir)
    logging.info("Prediction arguments: %s", args)
    pixel_spacing = parse_pixel_spacing(args.pixel_spacing)

    data_transforms = build_transforms(args)
    test_dataset = build_Dataset(
        args=args,
        data_dir=args.data_path + args.dataset,
        split=args.split,
        transform=data_transforms["valid_test"],
    )
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=args.num_workers)
    logging.info("Loaded split '%s' with %d samples", args.split, len(test_dataset))

    model = A3PASSNet(args).to(args.device)
    checkpoint = torch.load(args.PASS_model_path, map_location=args.device)
    model.load_state_dict(checkpoint)
    model.eval()
    logging.info("Loaded PASS weights from %s", args.PASS_model_path)

    case_metrics = []
    with torch.no_grad():
        for i_batch, sampled_batch in enumerate(test_loader):
            test_image = sampled_batch["image"].to(args.device)
            test_label = sampled_batch["label"].to(args.device)
            ori_image = to_uint8_rgb(sampled_batch["ori_image"][0]).copy()
            case_name = sampled_batch["case_name"][0]

            out = model(test_image)
            pass_metrics = safe_eval(test_label, out["pass_soft"], args.num_classes)
            sgdl_metrics = safe_eval(test_label, out["fusion_soft"], args.num_classes)
            pass_mask = label_from_softmax(out["pass_soft"])
            sgdl_mask = label_from_softmax(out["fusion_soft"])
            gt_mask = test_label.squeeze(0).detach().cpu().numpy().astype(np.uint8)

            measurement_row = {}
            measurement_overlay_pass = None
            measurement_overlay_sgdl = None
            if not args.disable_measurement:
                pass_fissure_mask = (pass_mask == args.measurement_class).astype(np.uint8)
                sgdl_fissure_mask = (sgdl_mask == args.measurement_class).astype(np.uint8)
                pass_measurement = measure_lateral_fissure(pass_fissure_mask, pixel_spacing=pixel_spacing)
                sgdl_measurement = measure_lateral_fissure(sgdl_fissure_mask, pixel_spacing=pixel_spacing)
                measurement_row.update(measurement_to_row(pass_measurement, prefix="fissure"))
                measurement_row.update(measurement_to_row(sgdl_measurement, prefix="sgdl_fissure"))
                measurement_overlay_pass = annotate_lateral_fissure_measurement(
                    ori_image,
                    pass_fissure_mask,
                    measurement=pass_measurement,
                    pixel_spacing=pixel_spacing,
                )
                measurement_overlay_sgdl = annotate_lateral_fissure_measurement(
                    ori_image,
                    sgdl_fissure_mask,
                    measurement=sgdl_measurement,
                    pixel_spacing=pixel_spacing,
                )

            save_case_outputs(
                save_dir,
                case_name,
                ori_image,
                gt_mask,
                pass_mask,
                sgdl_mask,
                measurement_overlay_pass,
                measurement_overlay_sgdl,
            )

            case_info = {
                "index": i_batch,
                "case_name": case_name,
                "dice": pass_metrics["avg_dice"],
                "iou": pass_metrics["avg_iou"],
                "hd95": pass_metrics["avg_hd95"],
                **prefixed_metrics("pass", pass_metrics),
                **prefixed_metrics("sgdl", sgdl_metrics),
                **measurement_row,
            }
            for class_idx in range(1, args.num_classes):
                case_info[f"pass_pred_class_{class_idx}_pixels"] = int((pass_mask == class_idx).sum())
                case_info[f"sgdl_pred_class_{class_idx}_pixels"] = int((sgdl_mask == class_idx).sum())
                case_info[f"gt_class_{class_idx}_pixels"] = int((gt_mask == class_idx).sum())
            case_metrics.append(case_info)
            logging.info(
                "case=%s idx=%d %s %s",
                case_name,
                i_batch,
                format_metrics("pass", pass_metrics, args.num_classes),
                format_metrics("sgdl", sgdl_metrics, args.num_classes),
            )

    summary = {
        "split": args.split,
        "pass_model_path": os.path.abspath(args.PASS_model_path),
        "save_dir": str(save_dir.resolve()),
        "log_path": str(log_path.resolve()),
        "num_cases": len(case_metrics),
        "pass_avg_dice": float(np.nanmean([item["pass_avg_dice"] for item in case_metrics])) if case_metrics else float("nan"),
        "pass_avg_iou": float(np.nanmean([item["pass_avg_iou"] for item in case_metrics])) if case_metrics else float("nan"),
        "pass_avg_hd95": float(np.nanmean([item["pass_avg_hd95"] for item in case_metrics])) if case_metrics else float("nan"),
        "sgdl_avg_dice": float(np.nanmean([item["sgdl_avg_dice"] for item in case_metrics])) if case_metrics else float("nan"),
        "sgdl_avg_iou": float(np.nanmean([item["sgdl_avg_iou"] for item in case_metrics])) if case_metrics else float("nan"),
        "sgdl_avg_hd95": float(np.nanmean([item["sgdl_avg_hd95"] for item in case_metrics])) if case_metrics else float("nan"),
    }
    for prefix in ("pass", "sgdl"):
        for class_idx in range(1, args.num_classes):
            for metric_name in ("dice", "iou", "hd95"):
                key = f"{prefix}_class_{class_idx}_{metric_name}"
                summary[f"{key}_avg"] = float(np.nanmean([item[key] for item in case_metrics])) if case_metrics else float("nan")
    if not args.disable_measurement:
        summary["measurement_class"] = args.measurement_class
        add_measurement_summary(summary, case_metrics, "fissure")
        add_measurement_summary(summary, case_metrics, "sgdl_fissure")

    csv_path = save_dir / "case_metrics.csv"
    fieldnames = list(case_metrics[0].keys()) if case_metrics else [
        "index", "case_name", "dice", "iou", "hd95",
        "pass_avg_dice", "pass_avg_iou", "pass_avg_hd95",
        "sgdl_avg_dice", "sgdl_avg_iou", "sgdl_avg_hd95",
    ]
    with open(csv_path, "w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(case_metrics)

    summary_path = save_dir / "summary.json"
    summary_path.write_text(
        json.dumps({"summary": summary, "cases": case_metrics}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    monitor_dir = save_evaluation_artifacts(
        save_dir,
        case_metrics,
        {
            "split": summary["split"],
            "num_cases": summary["num_cases"],
            "avg_dice": summary["pass_avg_dice"],
            "avg_iou": summary["pass_avg_iou"],
            "avg_hd95": summary["pass_avg_hd95"],
        },
        metadata={
            "split": args.split,
            "pass_model_path": os.path.abspath(args.PASS_model_path),
            "log_path": str(log_path.resolve()),
            "root_summary_path": str(summary_path.resolve()),
            "root_case_metrics_path": str(csv_path.resolve()),
        },
    )

    logging.info("Prediction summary: %s", summary)
    logging.info(
        "PASS macro metrics: avg_dice=%.6f avg_iou=%.6f avg_hd95=%.6f",
        summary["pass_avg_dice"],
        summary["pass_avg_iou"],
        summary["pass_avg_hd95"],
    )
    logging.info(
        "SGDL macro metrics: avg_dice=%.6f avg_iou=%.6f avg_hd95=%.6f",
        summary["sgdl_avg_dice"],
        summary["sgdl_avg_iou"],
        summary["sgdl_avg_hd95"],
    )
    for prefix in ("pass", "sgdl"):
        for class_idx in range(1, args.num_classes):
            logging.info(
                "%s class_%d metrics: dice=%.6f iou=%.6f hd95=%.6f",
                prefix.upper(),
                class_idx,
                summary[f"{prefix}_class_{class_idx}_dice_avg"],
                summary[f"{prefix}_class_{class_idx}_iou_avg"],
                summary[f"{prefix}_class_{class_idx}_hd95_avg"],
            )
    logging.info("Evaluation monitor saved to: %s", monitor_dir.resolve())
    print(f"{args.split} :")
    print("pass_avg_dice: ", summary["pass_avg_dice"])
    print("pass_avg_iou: ", summary["pass_avg_iou"])
    print("pass_avg_hd95: ", summary["pass_avg_hd95"])
    print("sgdl_avg_dice: ", summary["sgdl_avg_dice"])
    print("sgdl_avg_iou: ", summary["sgdl_avg_iou"])
    print("sgdl_avg_hd95: ", summary["sgdl_avg_hd95"])
    print("outputs saved to:", save_dir.resolve())


if __name__ == "__main__":
    main()
