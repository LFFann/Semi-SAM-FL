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
from utils.utils import eval


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="./SampleData")
    parser.add_argument("--dataset", type=str, default="/260513_data_label1")
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--num_classes", type=int, default=2)
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

    parser.add_argument("--PASS_model_path", type=str, default="./Results/A3_PASS_KnowSAM_V100_label1_106_117_13_13/fold_0/PASS_best_model.pth")
    parser.add_argument("--save_dir", type=str, default="./Results/A3_PASS_KnowSAM_V100_label1_106_117_13_13/fold_0/prediction_test")

    parser.add_argument("--pass_state_size", type=int, default=64)
    parser.add_argument("--pass_state_dim", type=int, default=64)
    parser.add_argument("--pass_base_channels", type=int, default=32)
    parser.add_argument("--pass_state_lr", type=float, default=0.001)
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


def binary_mask_from_softmax(prob_map: torch.Tensor, threshold=0.5):
    pred_mask = (prob_map > threshold).to(torch.float32).squeeze(0)[1].detach().cpu().numpy()
    return (pred_mask * 255).astype(np.uint8)


def overlay_mask(image_bgr: np.ndarray, mask_uint8: np.ndarray, color=(0, 0, 255), alpha=0.35):
    color_layer = np.zeros_like(image_bgr, dtype=np.uint8)
    color_layer[mask_uint8 > 0] = color
    return cv2.addWeighted(image_bgr.copy(), 1.0, color_layer, alpha, 0)


def safe_eval(test_label, prob_map):
    try:
        dice, iou, hd95 = eval(test_label, prob_map, thr=0.5)
        return float(dice), float(iou), float(hd95)
    except Exception as exc:
        logging.warning("Metric computation failed: %s", exc)
        return float("nan"), float("nan"), float("nan")


def save_case_outputs(save_dir: Path, case_name: str, ori_image: np.ndarray, gt_mask: np.ndarray,
                      pass_mask: np.ndarray, sgdl_mask: np.ndarray, measurement_overlay: np.ndarray = None):
    dirs = {
        "original": save_dir / "original",
        "gt_mask": save_dir / "gt_mask",
        "pred_mask": save_dir / "pred_mask",
        "overlay": save_dir / "overlay",
        "pred_mask_pass": save_dir / "pred_mask_pass",
        "pred_mask_sgdl": save_dir / "pred_mask_sgdl",
        "overlay_pass": save_dir / "overlay_pass",
        "overlay_sgdl": save_dir / "overlay_sgdl",
    }
    if measurement_overlay is not None:
        dirs["measurement_overlay"] = save_dir / "measurement_overlay"
    for directory in dirs.values():
        directory.mkdir(parents=True, exist_ok=True)

    cv2.imwrite(str(dirs["original"] / case_name), ori_image)
    cv2.imwrite(str(dirs["gt_mask"] / case_name), gt_mask)
    cv2.imwrite(str(dirs["pred_mask"] / case_name), pass_mask)
    cv2.imwrite(str(dirs["overlay"] / case_name), overlay_mask(ori_image, pass_mask, color=(0, 0, 255)))
    cv2.imwrite(str(dirs["pred_mask_pass"] / case_name), pass_mask)
    cv2.imwrite(str(dirs["pred_mask_sgdl"] / case_name), sgdl_mask)
    cv2.imwrite(str(dirs["overlay_pass"] / case_name), overlay_mask(ori_image, pass_mask, color=(0, 0, 255)))
    cv2.imwrite(str(dirs["overlay_sgdl"] / case_name), overlay_mask(ori_image, sgdl_mask, color=(0, 255, 0)))
    if measurement_overlay is not None:
        cv2.imwrite(str(dirs["measurement_overlay"] / case_name), measurement_overlay)


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
            pass_dice, pass_iou, pass_hd95 = safe_eval(test_label, out["pass_soft"])
            sgdl_dice, sgdl_iou, sgdl_hd95 = safe_eval(test_label, out["fusion_soft"])
            pass_mask = binary_mask_from_softmax(out["pass_soft"])
            sgdl_mask = binary_mask_from_softmax(out["fusion_soft"])
            gt_mask = (test_label.squeeze(0).detach().cpu().numpy() * 255).astype(np.uint8)

            measurement_row = {}
            measurement_overlay = None
            if not args.disable_measurement:
                measurement = measure_lateral_fissure(pass_mask, pixel_spacing=pixel_spacing)
                measurement_row = measurement_to_row(measurement)
                measurement_overlay = annotate_lateral_fissure_measurement(
                    ori_image,
                    pass_mask,
                    measurement=measurement,
                    pixel_spacing=pixel_spacing,
                )

            save_case_outputs(save_dir, case_name, ori_image, gt_mask, pass_mask, sgdl_mask, measurement_overlay)

            case_info = {
                "index": i_batch,
                "case_name": case_name,
                "dice": pass_dice,
                "iou": pass_iou,
                "hd95": pass_hd95,
                "pass_dice": pass_dice,
                "pass_iou": pass_iou,
                "pass_hd95": pass_hd95,
                "sgdl_dice": sgdl_dice,
                "sgdl_iou": sgdl_iou,
                "sgdl_hd95": sgdl_hd95,
                "pass_pred_positive_pixels": int((pass_mask > 0).sum()),
                "sgdl_pred_positive_pixels": int((sgdl_mask > 0).sum()),
                "gt_positive_pixels": int((gt_mask > 0).sum()),
                **measurement_row,
            }
            case_metrics.append(case_info)
            logging.info(
                "case=%s idx=%d pass_dice=%.6f sgdl_dice=%.6f",
                case_name,
                i_batch,
                pass_dice,
                sgdl_dice,
            )

    valid_metrics = [item for item in case_metrics if not np.isnan(item["pass_dice"])]
    summary = {
        "split": args.split,
        "pass_model_path": os.path.abspath(args.PASS_model_path),
        "save_dir": str(save_dir.resolve()),
        "log_path": str(log_path.resolve()),
        "num_cases": len(case_metrics),
        "pass_avg_dice": float(np.mean([item["pass_dice"] for item in valid_metrics])) if valid_metrics else float("nan"),
        "pass_avg_iou": float(np.mean([item["pass_iou"] for item in valid_metrics])) if valid_metrics else float("nan"),
        "pass_avg_hd95": float(np.mean([item["pass_hd95"] for item in valid_metrics])) if valid_metrics else float("nan"),
        "sgdl_avg_dice": float(np.mean([item["sgdl_dice"] for item in valid_metrics])) if valid_metrics else float("nan"),
        "sgdl_avg_iou": float(np.mean([item["sgdl_iou"] for item in valid_metrics])) if valid_metrics else float("nan"),
        "sgdl_avg_hd95": float(np.mean([item["sgdl_hd95"] for item in valid_metrics])) if valid_metrics else float("nan"),
    }
    if not args.disable_measurement:
        measurable = [item for item in case_metrics if item.get("fissure_measurement_status") == "ok"]
        summary.update({
            "num_measurable_fissures": len(measurable),
            "avg_fissure_width_px": float(np.mean([item["fissure_width_px"] for item in measurable])) if measurable else float("nan"),
            "avg_fissure_depth_px": float(np.mean([item["fissure_depth_px"] for item in measurable])) if measurable else float("nan"),
            "avg_fissure_mean_width_px": float(np.mean([item["fissure_mean_width_px"] for item in measurable])) if measurable else float("nan"),
        })
        if measurable and "fissure_width_mm" in measurable[0]:
            summary.update({
                "avg_fissure_width_mm": float(np.mean([item["fissure_width_mm"] for item in measurable])),
                "avg_fissure_depth_mm": float(np.mean([item["fissure_depth_mm"] for item in measurable])),
                "avg_fissure_mean_width_mm": float(np.mean([item["fissure_mean_width_mm"] for item in measurable])),
            })

    csv_path = save_dir / "case_metrics.csv"
    fieldnames = list(case_metrics[0].keys()) if case_metrics else [
        "index", "case_name", "dice", "iou", "hd95",
        "pass_dice", "pass_iou", "pass_hd95",
        "sgdl_dice", "sgdl_iou", "sgdl_hd95",
        "pass_pred_positive_pixels", "sgdl_pred_positive_pixels", "gt_positive_pixels",
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
