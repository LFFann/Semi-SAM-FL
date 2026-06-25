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
from medpy import metric
from torch.utils.data import DataLoader

try:
    from hausdorff import hausdorff_distance
except Exception:
    hausdorff_distance = None

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from dataloader.dataset import build_Dataset
from dataloader.transforms import build_transforms
from Model.model import KnowSAM
from models.dynamic_graph import DynamicPrototypeGraph
from models.reliability_head import ReliabilityHead
from utils.lateral_fissure_measurement import (
    annotate_lateral_fissure_measurement,
    measure_lateral_fissure,
    measurement_to_row,
    parse_pixel_spacing,
)
from utils.training_monitor import save_evaluation_artifacts


CLASS_COLORS = {
    0: (0, 0, 0),
    1: (0, 0, 255),
    2: (0, 255, 0),
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="./SampleData")
    parser.add_argument("--dataset", type=str, default="/260513_data_multiclass")
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--num_classes", type=int, default=3)
    parser.add_argument("--in_channels", type=int, default=3)
    parser.add_argument("--image_size", type=int, default=256)
    parser.add_argument("--point_nums", type=int, default=10)
    parser.add_argument("--box_nums", type=int, default=1)
    parser.add_argument("--mod", type=str, default="sam_adpt")
    parser.add_argument("--model_type", type=str, default="vit_b")
    parser.add_argument("--thd", type=bool, default=False)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--multimask", type=bool, default=False)
    parser.add_argument("--encoder_adapter", type=bool, default=True)
    parser.add_argument("--SGDL_model_path", type=str,
                        default="./Results/Multiclass_KnowSAM_V100_bs32_10k_106_117_13_13/SGDL_best_model.pth")
    parser.add_argument("--DPG_model_path", type=str, default="",
                        help="DPG checkpoint for SRG-SAM++-Lite deploy/graph prediction")
    parser.add_argument("--reliability_model_path", type=str, default="",
                        help="Reliability Head checkpoint for SRG-SAM++-Lite deploy prediction")
    parser.add_argument("--prediction_mode", type=str, default="deploy",
                        choices=["deploy", "final", "student", "graph"],
                        help="which logits to evaluate")
    parser.add_argument("--use_dpg_deploy", type=lambda x: str(x).lower() not in ("0", "false", "no"), default=True)
    parser.add_argument("--use_reliability_head", type=lambda x: str(x).lower() not in ("0", "false", "no"), default=True)
    parser.add_argument("--alpha_graph", type=float, default=0.3)
    parser.add_argument("--srg_graph_alpha", type=float, default=0.5)
    parser.add_argument("--srg_affinity_size", type=int, default=16)
    parser.add_argument("--save_dir", type=str,
                        default="./Results/Multiclass_KnowSAM_V100_bs32_10k_106_117_13_13/prediction_test")
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--measurement_class", type=int, default=1,
                        help="class index used for lateral fissure measurement")
    parser.add_argument("--pixel_spacing", type=str, default="",
                        help="optional pixel spacing in mm, either one value or row,col")
    parser.add_argument("--disable_measurement", action="store_true",
                        help="disable lateral fissure width/depth measurement outputs")
    args = parser.parse_args()

    if args.prediction_mode == "final":
        args.prediction_mode = "deploy"
    return args


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
    return np.clip(image, 0, 255).astype(np.uint8)


def colorize_mask(mask: np.ndarray):
    color = np.zeros((*mask.shape, 3), dtype=np.uint8)
    for class_idx, bgr in CLASS_COLORS.items():
        color[mask == class_idx] = bgr
    return color


def overlay_multiclass(image_bgr: np.ndarray, mask: np.ndarray, alpha=0.35):
    color = colorize_mask(mask)
    fg = mask > 0
    overlay = image_bgr.copy()
    blended = cv2.addWeighted(image_bgr, 1.0, color, alpha, 0)
    overlay[fg] = blended[fg]
    return overlay


def class_metrics(gt: np.ndarray, pred: np.ndarray, class_idx: int):
    gt_c = gt == class_idx
    pred_c = pred == class_idx
    gt_sum = gt_c.sum()
    pred_sum = pred_c.sum()
    if gt_sum == 0 and pred_sum == 0:
        return 1.0, 1.0, 0.0
    if gt_sum == 0 or pred_sum == 0:
        return 0.0, 0.0, float("nan")
    if hausdorff_distance is None:
        hd95 = float(metric.binary.hd95(pred_c, gt_c))
    else:
        hd95 = float(hausdorff_distance(gt_c.astype(np.uint8), pred_c.astype(np.uint8)) * 0.95)
    return (
        float(metric.binary.dc(pred_c, gt_c)),
        float(metric.binary.jc(pred_c, gt_c)),
        hd95,
    )


def evaluate_multiclass(gt: np.ndarray, pred: np.ndarray, num_classes: int):
    result = {}
    rows = []
    for class_idx in range(1, num_classes):
        dice, iou, hd95 = class_metrics(gt, pred, class_idx)
        result[f"class_{class_idx}_dice"] = dice
        result[f"class_{class_idx}_iou"] = iou
        result[f"class_{class_idx}_hd95"] = hd95
        rows.append((dice, iou, hd95))
    metrics_array = np.array(rows, dtype=np.float32)
    result["dice"] = float(np.nanmean(metrics_array[:, 0]))
    result["iou"] = float(np.nanmean(metrics_array[:, 1]))
    result["hd95"] = float(np.nanmean(metrics_array[:, 2]))
    return result


def format_multiclass_metrics(metrics: dict, num_classes: int):
    parts = [
        "avg_dice=%.6f" % metrics["dice"],
        "avg_iou=%.6f" % metrics["iou"],
        "avg_hd95=%.6f" % metrics["hd95"],
    ]
    for class_idx in range(1, num_classes):
        parts.extend([
            "class_%d_dice=%.6f" % (class_idx, metrics[f"class_{class_idx}_dice"]),
            "class_%d_iou=%.6f" % (class_idx, metrics[f"class_{class_idx}_iou"]),
            "class_%d_hd95=%.6f" % (class_idx, metrics[f"class_{class_idx}_hd95"]),
        ])
    return " ".join(parts)


def save_case_outputs(
    save_dir: Path,
    case_name: str,
    ori_image: np.ndarray,
    gt: np.ndarray,
    pred: np.ndarray,
    measurement_overlay: np.ndarray = None,
):
    dirs = {
        "original": save_dir / "original",
        "gt_mask": save_dir / "gt_mask",
        "pred_mask": save_dir / "pred_mask",
        "gt_color": save_dir / "gt_color",
        "pred_color": save_dir / "pred_color",
        "overlay": save_dir / "overlay",
    }
    if measurement_overlay is not None:
        dirs["measurement_overlay"] = save_dir / "measurement_overlay"
    for path in dirs.values():
        path.mkdir(parents=True, exist_ok=True)

    cv2.imwrite(str(dirs["original"] / case_name), ori_image)
    cv2.imwrite(str(dirs["gt_mask"] / case_name), gt.astype(np.uint8))
    cv2.imwrite(str(dirs["pred_mask"] / case_name), pred.astype(np.uint8))
    cv2.imwrite(str(dirs["gt_color"] / case_name), colorize_mask(gt))
    cv2.imwrite(str(dirs["pred_color"] / case_name), colorize_mask(pred))
    cv2.imwrite(str(dirs["overlay"] / case_name), overlay_multiclass(ori_image, pred))
    if measurement_overlay is not None:
        cv2.imwrite(str(dirs["measurement_overlay"] / case_name), measurement_overlay)


def main():
    args = parse_args()
    save_dir = Path(args.save_dir)
    log_path = setup_logger(save_dir)
    logging.info("Prediction arguments: %s", args)
    pixel_spacing = parse_pixel_spacing(args.pixel_spacing)

    transforms = build_transforms(args)
    test_dataset = build_Dataset(args, data_dir=args.data_path + args.dataset, split=args.split,
                                 transform=transforms["valid_test"])
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=args.num_workers)
    logging.info("Loaded split '%s' with %d samples", args.split, len(test_dataset))

    model = KnowSAM(args, bilinear=False).to(args.device)
    graph = DynamicPrototypeGraph(
        num_classes=args.num_classes,
        feature_dim=32,
        alpha=args.srg_graph_alpha,
        affinity_size=args.srg_affinity_size,
    ).to(args.device)
    reliability_head = ReliabilityHead(in_channels=32).to(args.device)

    checkpoint = torch.load(args.SGDL_model_path, map_location=args.device)
    model.load_state_dict(checkpoint)
    if args.prediction_mode == "graph" or (args.prediction_mode == "deploy" and args.use_dpg_deploy):
        if not args.DPG_model_path or not os.path.isfile(args.DPG_model_path):
            raise FileNotFoundError("DPG_model_path is required for deploy/graph prediction.")
        graph.load_state_dict(torch.load(args.DPG_model_path, map_location=args.device))
    if args.prediction_mode == "deploy" and args.use_reliability_head:
        if not args.reliability_model_path or not os.path.isfile(args.reliability_model_path):
            raise FileNotFoundError("reliability_model_path is required for deploy prediction.")
        reliability_head.load_state_dict(torch.load(args.reliability_model_path, map_location=args.device))
    model.eval()
    graph.eval()
    reliability_head.eval()

    case_metrics = []
    with torch.no_grad():
        for index, batch in enumerate(test_loader):
            image = batch["image"].to(args.device)
            label = batch["label"].to(args.device)
            case_name = batch["case_name"][0]
            ori_image = to_uint8_rgb(batch["ori_image"][0]).copy()

            _, _, _, _, fusion_map, features = model(image, return_features=True)
            graph_logits = graph(features)
            if graph_logits.shape[-2:] != fusion_map.shape[-2:]:
                graph_logits = torch.nn.functional.interpolate(
                    graph_logits, size=fusion_map.shape[-2:], mode="bilinear", align_corners=False
                )
            if args.use_reliability_head:
                R_hat = reliability_head(features, out_size=fusion_map.shape[-2:])
            else:
                R_hat = fusion_map.new_ones(image.shape[0], 1, *fusion_map.shape[-2:])
            if args.use_dpg_deploy:
                deploy_logits = fusion_map + args.alpha_graph * R_hat * graph_logits
            else:
                deploy_logits = fusion_map
            logits_by_mode = {
                "deploy": deploy_logits,
                "student": fusion_map,
                "graph": graph_logits,
            }
            pred_logits = logits_by_mode[args.prediction_mode]
            pred = torch.argmax(torch.softmax(pred_logits, dim=1), dim=1).squeeze(0).cpu().numpy().astype(np.uint8)
            gt = label.squeeze(0).cpu().numpy().astype(np.uint8)
            metrics = evaluate_multiclass(gt, pred, args.num_classes)
            measurement_row = {}
            measurement_overlay = None
            if not args.disable_measurement:
                fissure_mask = (pred == args.measurement_class).astype(np.uint8)
                measurement = measure_lateral_fissure(fissure_mask, pixel_spacing=pixel_spacing)
                measurement_row = measurement_to_row(measurement)
                measurement_overlay = annotate_lateral_fissure_measurement(
                    ori_image,
                    fissure_mask,
                    measurement=measurement,
                    pixel_spacing=pixel_spacing,
                )
            save_case_outputs(save_dir, case_name, ori_image, gt, pred, measurement_overlay)

            row = {
                "index": index,
                "case_name": case_name,
                **metrics,
                "pred_class_1_pixels": int((pred == 1).sum()),
                "pred_class_2_pixels": int((pred == 2).sum()),
                "gt_class_1_pixels": int((gt == 1).sum()),
                "gt_class_2_pixels": int((gt == 2).sum()),
                **measurement_row,
            }
            case_metrics.append(row)
            logging.info("case=%s %s", case_name, format_multiclass_metrics(metrics, args.num_classes))

    valid = [row for row in case_metrics if not np.isnan(row["dice"])]
    summary = {
        "split": args.split,
        "model_path": os.path.abspath(args.SGDL_model_path),
        "dpg_model_path": os.path.abspath(args.DPG_model_path) if args.DPG_model_path else "",
        "reliability_model_path": os.path.abspath(args.reliability_model_path) if args.reliability_model_path else "",
        "prediction_mode": args.prediction_mode,
        "save_dir": str(save_dir.resolve()),
        "log_path": str(log_path.resolve()),
        "num_cases": len(case_metrics),
        "avg_dice": float(np.nanmean([row["dice"] for row in valid])) if valid else float("nan"),
        "avg_iou": float(np.nanmean([row["iou"] for row in valid])) if valid else float("nan"),
        "avg_hd95": float(np.nanmean([row["hd95"] for row in valid])) if valid else float("nan"),
    }
    for class_idx in range(1, args.num_classes):
        summary[f"class_{class_idx}_avg_dice"] = float(np.nanmean([row[f"class_{class_idx}_dice"] for row in case_metrics]))
        summary[f"class_{class_idx}_avg_iou"] = float(np.nanmean([row[f"class_{class_idx}_iou"] for row in case_metrics]))
        summary[f"class_{class_idx}_avg_hd95"] = float(np.nanmean([row[f"class_{class_idx}_hd95"] for row in case_metrics]))
    if not args.disable_measurement:
        measurable = [row for row in case_metrics if row.get("fissure_measurement_status") == "ok"]
        summary.update({
            "measurement_class": args.measurement_class,
            "num_measurable_fissures": len(measurable),
            "avg_fissure_width_px": float(np.mean([row["fissure_width_px"] for row in measurable])) if measurable else float("nan"),
            "avg_fissure_depth_px": float(np.mean([row["fissure_depth_px"] for row in measurable])) if measurable else float("nan"),
            "avg_fissure_mean_width_px": float(np.mean([row["fissure_mean_width_px"] for row in measurable])) if measurable else float("nan"),
        })
        if measurable and "fissure_width_mm" in measurable[0]:
            summary.update({
                "avg_fissure_width_mm": float(np.mean([row["fissure_width_mm"] for row in measurable])),
                "avg_fissure_depth_mm": float(np.mean([row["fissure_depth_mm"] for row in measurable])),
                "avg_fissure_mean_width_mm": float(np.mean([row["fissure_mean_width_mm"] for row in measurable])),
            })

    csv_path = save_dir / "case_metrics.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=list(case_metrics[0].keys()) if case_metrics else [])
        writer.writeheader()
        writer.writerows(case_metrics)

    summary_path = save_dir / "summary.json"
    summary_path.write_text(json.dumps({"summary": summary, "cases": case_metrics}, ensure_ascii=False, indent=2),
                            encoding="utf-8")
    save_evaluation_artifacts(
        save_dir,
        case_metrics,
        summary,
        metadata={
            "split": args.split,
            "model_path": os.path.abspath(args.SGDL_model_path),
            "dpg_model_path": os.path.abspath(args.DPG_model_path) if args.DPG_model_path else "",
            "reliability_model_path": os.path.abspath(args.reliability_model_path) if args.reliability_model_path else "",
            "prediction_mode": args.prediction_mode,
            "log_path": str(log_path.resolve()),
            "root_summary_path": str(summary_path.resolve()),
            "root_case_metrics_path": str(csv_path.resolve()),
        },
    )

    logging.info("Prediction summary: %s", summary)
    logging.info(
        "Prediction macro metrics: avg_dice=%.6f avg_iou=%.6f avg_hd95=%.6f",
        summary["avg_dice"],
        summary["avg_iou"],
        summary["avg_hd95"],
    )
    for class_idx in range(1, args.num_classes):
        logging.info(
            "Prediction class_%d metrics: dice=%.6f iou=%.6f hd95=%.6f",
            class_idx,
            summary[f"class_{class_idx}_avg_dice"],
            summary[f"class_{class_idx}_avg_iou"],
            summary[f"class_{class_idx}_avg_hd95"],
        )
    print(f"{args.split} :")
    print("avg_dice: ", summary["avg_dice"])
    print("avg_iou: ", summary["avg_iou"])
    print("avg_hd95: ", summary["avg_hd95"])
    print("outputs saved to:", save_dir.resolve())


if __name__ == "__main__":
    main()
