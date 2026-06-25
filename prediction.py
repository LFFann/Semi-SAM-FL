import argparse
import csv
import json
import logging
import os
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader

from dataloader.dataset import build_Dataset
from dataloader.transforms import build_transforms
from Model.model import KnowSAM
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
    parser.add_argument('--data_path', type=str, default='./SampleData',
                        help='dataset root path')
    parser.add_argument('--dataset', type=str, default='/260513_data_label1',
                        help='dataset name')
    parser.add_argument('--split', type=str, default='test',
                        help='dataset split to evaluate, e.g. test, val, or dataset-specific test split')
    parser.add_argument('--num_classes', type=int, default=2,
                        help='output channel of network')
    parser.add_argument('--in_channels', type=int, default=3,
                        help='input channel of network')
    parser.add_argument('--image_size', type=int, default=256,
                        help='patch size of network input')
    parser.add_argument('--point_nums', type=int, default=10, help='points number')
    parser.add_argument('--box_nums', type=int, default=1, help='boxes number')
    parser.add_argument('--mod', type=str, default='sam_adpt', help='mod type')
    parser.add_argument("--model_type", type=str, default="vit_b", help="sam model_type")
    parser.add_argument('--thd', type=bool, default=False, help='3d or not')
    parser.add_argument('--SGDL_model_path', type=str,
                        default="./Results/train_260513_data_label1_v100_semi_106_117_13_13/SGDL_best_model.pth",
                        help='SGDL model weight path')
    parser.add_argument('--save_dir', type=str,
                        default="./Results/train_260513_data_label1_v100_semi_106_117_13_13/prediction_test",
                        help='directory to save logs and visualizations')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--pixel_spacing', type=str, default='',
                        help='optional pixel spacing in mm, either one value or row,col')
    parser.add_argument('--disable_measurement', action='store_true',
                        help='disable lateral fissure width/depth measurement outputs')
    return parser.parse_args()


def setup_logger(save_dir: Path):
    save_dir.mkdir(parents=True, exist_ok=True)
    log_path = save_dir / "prediction.log"
    logging.basicConfig(
        level=logging.INFO,
        format='[%(asctime)s.%(msecs)03d] %(message)s',
        datefmt='%H:%M:%S',
        handlers=[
            logging.FileHandler(log_path, encoding="utf-8"),
            logging.StreamHandler(),
        ],
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
    overlay = image_bgr.copy()
    color_layer = np.zeros_like(image_bgr, dtype=np.uint8)
    color_layer[mask_uint8 > 0] = color
    overlay = cv2.addWeighted(overlay, 1.0, color_layer, alpha, 0)
    return overlay


def safe_eval(test_label, fusion_map_soft):
    try:
        dice, iou, hd95 = eval(test_label, fusion_map_soft, thr=0.5)
        return float(dice), float(iou), float(hd95)
    except Exception as exc:
        logging.warning("Metric computation failed: %s", exc)
        return float("nan"), float("nan"), float("nan")


def save_case_outputs(save_dir: Path, case_name: str, ori_image: np.ndarray, gt_mask: np.ndarray,
                      pred_mask: np.ndarray, measurement_overlay: np.ndarray = None):
    original_dir = save_dir / "original"
    gt_dir = save_dir / "gt_mask"
    pred_dir = save_dir / "pred_mask"
    overlay_dir = save_dir / "overlay"
    measurement_dir = save_dir / "measurement_overlay"
    directories = [original_dir, gt_dir, pred_dir, overlay_dir]
    if measurement_overlay is not None:
        directories.append(measurement_dir)
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)

    pred_overlay = overlay_mask(ori_image, pred_mask, color=(0, 0, 255))

    cv2.imwrite(str(original_dir / case_name), ori_image)
    cv2.imwrite(str(gt_dir / case_name), gt_mask)
    cv2.imwrite(str(pred_dir / case_name), pred_mask)
    cv2.imwrite(str(overlay_dir / case_name), pred_overlay)
    if measurement_overlay is not None:
        cv2.imwrite(str(measurement_dir / case_name), measurement_overlay)


def main():
    args = parse_args()
    save_dir = Path(args.save_dir)
    log_path = setup_logger(save_dir)
    logging.info("Prediction arguments: %s", args)
    pixel_spacing = parse_pixel_spacing(args.pixel_spacing)

    data_transforms = build_transforms(args)
    test_dataset = build_Dataset(args, data_dir=args.data_path + args.dataset, split=args.split,
                                 transform=data_transforms["valid_test"])
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=args.num_workers)
    logging.info("Loaded split '%s' with %d samples", args.split, len(test_dataset))

    model = KnowSAM(args, bilinear=False).to(args.device).train()
    checkpoint = torch.load(args.SGDL_model_path, map_location=args.device)
    model.load_state_dict(checkpoint)
    model.eval()
    logging.info("Loaded model weights from %s", args.SGDL_model_path)

    case_metrics = []
    with torch.no_grad():
        for i_batch, sampled_batch in enumerate(test_loader):
            test_image = sampled_batch["image"].to(args.device)
            test_label = sampled_batch["label"].to(args.device)
            ori_image = to_uint8_rgb(sampled_batch["ori_image"][0]).copy()
            case_name = sampled_batch["case_name"][0]

            pred_unet, pred_vnet, pred_unet_soft, pred_vnet_soft, fusion_map = model(test_image)
            fusion_map_soft = torch.softmax(fusion_map, dim=1)

            dice, iou, hd95 = safe_eval(test_label, fusion_map_soft)
            pred_mask = binary_mask_from_softmax(fusion_map_soft)
            gt_mask = (test_label.squeeze(0).detach().cpu().numpy() * 255).astype(np.uint8)

            measurement = None
            measurement_row = {}
            measurement_overlay = None
            if not args.disable_measurement:
                measurement = measure_lateral_fissure(pred_mask, pixel_spacing=pixel_spacing)
                measurement_row = measurement_to_row(measurement)
                measurement_overlay = annotate_lateral_fissure_measurement(
                    ori_image,
                    pred_mask,
                    measurement=measurement,
                    pixel_spacing=pixel_spacing,
                )

            save_case_outputs(save_dir, case_name, ori_image, gt_mask, pred_mask, measurement_overlay)

            case_info = {
                "index": i_batch,
                "case_name": case_name,
                "dice": dice,
                "iou": iou,
                "hd95": hd95,
                "pred_positive_pixels": int((pred_mask > 0).sum()),
                "gt_positive_pixels": int((gt_mask > 0).sum()),
                **measurement_row,
            }
            case_metrics.append(case_info)
            logging.info(
                "case=%s idx=%d dice=%.6f iou=%.6f hd95=%.6f pred_pixels=%d gt_pixels=%d fissure_width_px=%s fissure_depth_px=%s",
                case_name, i_batch, dice, iou, hd95,
                case_info["pred_positive_pixels"], case_info["gt_positive_pixels"],
                case_info.get("fissure_width_px", ""),
                case_info.get("fissure_depth_px", ""),
            )

    valid_metrics = [item for item in case_metrics if not np.isnan(item["dice"])]
    summary = {
        "split": args.split,
        "model_path": os.path.abspath(args.SGDL_model_path),
        "save_dir": str(save_dir.resolve()),
        "log_path": str(log_path.resolve()),
        "num_cases": len(case_metrics),
        "avg_dice": float(np.mean([item["dice"] for item in valid_metrics])) if valid_metrics else float("nan"),
        "avg_iou": float(np.mean([item["iou"] for item in valid_metrics])) if valid_metrics else float("nan"),
        "avg_hd95": float(np.mean([item["hd95"] for item in valid_metrics])) if valid_metrics else float("nan"),
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
    with open(csv_path, "w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=list(case_metrics[0].keys()) if case_metrics else [
            "index", "case_name", "dice", "iou", "hd95", "pred_positive_pixels", "gt_positive_pixels"
        ])
        writer.writeheader()
        writer.writerows(case_metrics)

    summary_path = save_dir / "summary.json"
    summary_path.write_text(
        json.dumps({"summary": summary, "cases": case_metrics}, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )
    monitor_dir = save_evaluation_artifacts(
        save_dir,
        case_metrics,
        summary,
        metadata={
            "split": args.split,
            "model_path": os.path.abspath(args.SGDL_model_path),
            "log_path": str(log_path.resolve()),
            "root_summary_path": str(summary_path.resolve()),
            "root_case_metrics_path": str(csv_path.resolve()),
        },
    )

    logging.info("Prediction summary: %s", summary)
    logging.info("Evaluation monitor saved to: %s", monitor_dir.resolve())
    print(f"{args.split} :")
    print("avg_dice: ", summary["avg_dice"])
    print("avg_iou: ", summary["avg_iou"])
    print("avg_hd95: ", summary["avg_hd95"])
    print("outputs saved to:", save_dir.resolve())


if __name__ == '__main__':
    main()
