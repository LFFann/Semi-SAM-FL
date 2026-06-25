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
import torch.nn.functional as F
from torch.utils.data import DataLoader

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "..", ".."))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)
if REPO_ROOT not in sys.path:
    sys.path.insert(1, REPO_ROOT)

from dataloader.dataset import build_Dataset
from dataloader.transforms import build_transforms
from utils.lateral_fissure_measurement import (
    annotate_lateral_fissure_measurement,
    measure_lateral_fissure,
    measurement_to_row,
    parse_pixel_spacing,
)
from utils.training_monitor import save_evaluation_artifacts
from utils.utils import eval
from trainer_a3_rcp import Trainer


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='./SampleData', help='dataset root path')
    parser.add_argument('--dataset', type=str, default='/260513_data_label1', help='dataset name')
    parser.add_argument('--split', type=str, default='test', help='dataset split')
    parser.add_argument('--num_classes', type=int, default=2, help='output channel of network')
    parser.add_argument('--in_channels', type=int, default=3, help='input channel of network')
    parser.add_argument('-lr', type=float, default=1e-4, help='sam optimizer lr placeholder for trainer init')
    parser.add_argument('-UNet_lr', type=float, default=0.0025, help='sgdl optimizer lr placeholder for trainer init')
    parser.add_argument('-VNet_lr', type=float, default=0.0025, help='unused placeholder for compatibility')
    parser.add_argument('--image_size', type=int, default=256, help='network input image size')
    parser.add_argument('--point_nums', type=int, default=5, help='points number')
    parser.add_argument('--box_nums', type=int, default=1, help='boxes number')
    parser.add_argument('--mod', type=str, default='sam_adpt', help='mod type')
    parser.add_argument("--model_type", type=str, default="vit_b", help="sam model_type")
    parser.add_argument('-thd', type=bool, default=False, help='3d or not')
    parser.add_argument('--batch_size', type=int, default=1, help='eval batch size')
    parser.add_argument('--labeled_bs', type=int, default=1, help='placeholder for trainer init')
    parser.add_argument('--seed', type=int, default=42, help='random seed')
    parser.add_argument('--mixed_iterations', type=int, default=12000, help='placeholder')
    parser.add_argument('--max_iterations', type=int, default=50000, help='placeholder')
    parser.add_argument('--val_interval', type=int, default=200, help='placeholder')
    parser.add_argument('--n_fold', type=int, default=1, help='placeholder')
    parser.add_argument('--consistency', type=float, default=0.1, help='placeholder')
    parser.add_argument('--consistency_rampup', type=float, default=200.0, help='placeholder')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument("--multimask", type=bool, default=False, help="output multimask")
    parser.add_argument("--encoder_adapter", type=bool, default=True, help="use adapter")
    parser.add_argument("--sam_checkpoint", type=str, default="./sam_vit_b_01ec64.pth", help="sam checkpoint")
    parser.add_argument('--SGDL_model_path', type=str,
                        default="./Results/A3_RCP_KnowSAM_V100_label1_106_117_13_13/fold_0/SGDL_best_model.pth",
                        help='SGDL model weight path')
    parser.add_argument('--sam_model_path', type=str,
                        default="./Results/A3_RCP_KnowSAM_V100_label1_106_117_13_13/fold_0/sam_best_model.pth",
                        help='SAM model weight path')
    parser.add_argument('--save_dir', type=str,
                        default="./Results/A3_RCP_KnowSAM_V100_label1_106_117_13_13/fold_0/prediction_test",
                        help='directory to save logs and visualizations')
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--uckd_alpha', type=float, default=2.0)
    parser.add_argument('--uckd_min_weight', type=float, default=0.15)
    parser.add_argument('--qapl_min_weight', type=float, default=0.20)
    parser.add_argument('--rcp_alpha', type=float, default=2.0)
    parser.add_argument('--rcp_min_weight', type=float, default=0.10)
    parser.add_argument('--rcp_sharpen', type=float, default=1.5)
    parser.add_argument('--sap_boundary_weight', type=float, default=0.10)
    parser.add_argument('--sap_shape_weight', type=float, default=0.05)
    parser.add_argument('--sap_area_lower', type=float, default=0.001)
    parser.add_argument('--sap_area_upper', type=float, default=0.08)
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
    overlay = image_bgr.copy()
    color_layer = np.zeros_like(image_bgr, dtype=np.uint8)
    color_layer[mask_uint8 > 0] = color
    return cv2.addWeighted(overlay, 1.0, color_layer, alpha, 0)


def safe_eval(test_label, prob_map):
    try:
        dice, iou, hd95 = eval(test_label, prob_map, thr=0.5)
        return float(dice), float(iou), float(hd95)
    except Exception as exc:
        logging.warning("Metric computation failed: %s", exc)
        return float("nan"), float("nan"), float("nan")


def save_case_outputs(save_dir: Path, case_name: str, ori_image: np.ndarray, gt_mask: np.ndarray,
                      sgdl_mask: np.ndarray, sam_mask: np.ndarray, measurement_overlay: np.ndarray = None):
    original_dir = save_dir / "original"
    gt_dir = save_dir / "gt_mask"
    # 兼容原版 monitor 的默认目录约定：pred_mask / overlay
    pred_dir = save_dir / "pred_mask"
    overlay_dir = save_dir / "overlay"
    sgdl_dir = save_dir / "pred_mask_sgdl"
    sam_dir = save_dir / "pred_mask_sam"
    sgdl_overlay_dir = save_dir / "overlay_sgdl"
    sam_overlay_dir = save_dir / "overlay_sam"
    measurement_dir = save_dir / "measurement_overlay"
    directories = [original_dir, gt_dir, pred_dir, overlay_dir, sgdl_dir, sam_dir, sgdl_overlay_dir, sam_overlay_dir]
    if measurement_overlay is not None:
        directories.append(measurement_dir)
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)

    cv2.imwrite(str(original_dir / case_name), ori_image)
    cv2.imwrite(str(gt_dir / case_name), gt_mask)
    # 原版兼容口径默认使用 SGDL 结果
    cv2.imwrite(str(pred_dir / case_name), sgdl_mask)
    cv2.imwrite(str(overlay_dir / case_name), overlay_mask(ori_image, sgdl_mask, color=(0, 0, 255)))
    cv2.imwrite(str(sgdl_dir / case_name), sgdl_mask)
    cv2.imwrite(str(sam_dir / case_name), sam_mask)
    cv2.imwrite(str(sgdl_overlay_dir / case_name), overlay_mask(ori_image, sgdl_mask, color=(0, 0, 255)))
    cv2.imwrite(str(sam_overlay_dir / case_name), overlay_mask(ori_image, sam_mask, color=(0, 255, 0)))
    if measurement_overlay is not None:
        cv2.imwrite(str(measurement_dir / case_name), measurement_overlay)


def forward_v1(trainer: Trainer, image: torch.Tensor):
    image_embeddings = trainer.sam_model.image_encoder(image)
    pred_unet, pred_vnet, pred_unet_soft, pred_vnet_soft, fusion_map = trainer.SGDL(image)
    fusion_map_soft = torch.softmax(fusion_map, dim=1)
    prompt_logits, prompt_reliability = trainer.build_a3_consensus_prompt(
        pred_unet_soft,
        pred_vnet_soft,
        fusion_map_soft,
    )
    _, boxes_embedding, _ = trainer.sam_model.super_prompt(image_embeddings)
    low_res_masks_all = torch.empty(
        (image.shape[0], 0, int(trainer.args.image_size / 4), int(trainer.args.image_size / 4)),
        device=trainer.args.device,
    )
    for i in range(trainer.args.num_classes):
        sparse_embeddings, dense_embeddings = trainer.sam_model.prompt_encoder(
            points=None,
            boxes=boxes_embedding[i],
            masks=F.interpolate(prompt_logits[:, i, ...].unsqueeze(1), size=(64, 64), mode='bilinear'),
        )
        low_res_masks, _ = trainer.sam_model.mask_decoder(
            image_embeddings=image_embeddings,
            image_pe=trainer.sam_model.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=trainer.args.multimask,
        )
        low_res_masks_all = torch.cat((low_res_masks_all, low_res_masks), dim=1)

    pred_sam = F.interpolate(
        low_res_masks_all,
        size=(trainer.args.image_size, trainer.args.image_size),
        mode="bilinear",
        align_corners=False,
    )
    pred_sam_soft = torch.softmax(pred_sam, dim=1)
    return fusion_map_soft, pred_sam_soft, prompt_reliability


def main():
    args = parse_args()
    save_dir = Path(args.save_dir)
    log_path = setup_logger(save_dir)
    logging.info("Prediction arguments: %s", args)
    pixel_spacing = parse_pixel_spacing(args.pixel_spacing)

    data_transforms = build_transforms(args)
    test_dataset = build_Dataset(args=args, data_dir=args.data_path + args.dataset, split=args.split,
                                 transform=data_transforms["valid_test"])
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=args.num_workers)
    logging.info("Loaded split '%s' with %d samples", args.split, len(test_dataset))

    trainer = Trainer(args)
    sgdl_checkpoint = torch.load(args.SGDL_model_path, map_location=args.device)
    sam_checkpoint = torch.load(args.sam_model_path, map_location=args.device)
    trainer.SGDL.load_state_dict(sgdl_checkpoint)
    trainer.sam_model.load_state_dict(sam_checkpoint)
    trainer.SGDL.eval()
    trainer.sam_model.eval()
    logging.info("Loaded SGDL weights from %s", args.SGDL_model_path)
    logging.info("Loaded SAM weights from %s", args.sam_model_path)

    case_metrics = []
    with torch.no_grad():
        for i_batch, sampled_batch in enumerate(test_loader):
            test_image = sampled_batch["image"].to(args.device)
            test_label = sampled_batch["label"].to(args.device)
            ori_image = to_uint8_rgb(sampled_batch["ori_image"][0]).copy()
            case_name = sampled_batch["case_name"][0]

            fusion_map_soft, pred_sam_soft, prompt_reliability = forward_v1(trainer, test_image)
            sgdl_dice, sgdl_iou, sgdl_hd95 = safe_eval(test_label, fusion_map_soft)
            sam_dice, sam_iou, sam_hd95 = safe_eval(test_label, pred_sam_soft)
            sgdl_mask = binary_mask_from_softmax(fusion_map_soft)
            sam_mask = binary_mask_from_softmax(pred_sam_soft)
            gt_mask = (test_label.squeeze(0).detach().cpu().numpy() * 255).astype(np.uint8)

            measurement_row = {}
            measurement_overlay = None
            if not args.disable_measurement:
                measurement = measure_lateral_fissure(sgdl_mask, pixel_spacing=pixel_spacing)
                measurement_row = measurement_to_row(measurement)
                measurement_overlay = annotate_lateral_fissure_measurement(
                    ori_image,
                    sgdl_mask,
                    measurement=measurement,
                    pixel_spacing=pixel_spacing,
                )

            save_case_outputs(save_dir, case_name, ori_image, gt_mask, sgdl_mask, sam_mask, measurement_overlay)

            case_info = {
                "index": i_batch,
                "case_name": case_name,
                # 兼容原版 monitor 协议：默认 dice/iou/hd95 指代 SGDL 结果
                "dice": sgdl_dice,
                "iou": sgdl_iou,
                "hd95": sgdl_hd95,
                "sgdl_dice": sgdl_dice,
                "sgdl_iou": sgdl_iou,
                "sgdl_hd95": sgdl_hd95,
                "sam_dice": sam_dice,
                "sam_iou": sam_iou,
                "sam_hd95": sam_hd95,
                "prompt_weight_mean": float(prompt_reliability.mean().item()),
                "sgdl_pred_positive_pixels": int((sgdl_mask > 0).sum()),
                "sam_pred_positive_pixels": int((sam_mask > 0).sum()),
                "gt_positive_pixels": int((gt_mask > 0).sum()),
                **measurement_row,
            }
            case_metrics.append(case_info)
            logging.info(
                "case=%s idx=%d sgdl_dice=%.6f sam_dice=%.6f prompt_weight=%.6f",
                case_name, i_batch, sgdl_dice, sam_dice, case_info["prompt_weight_mean"]
            )

    valid_metrics = [item for item in case_metrics if not np.isnan(item["sgdl_dice"])]
    summary = {
        "split": args.split,
        "sgdl_model_path": os.path.abspath(args.SGDL_model_path),
        "sam_model_path": os.path.abspath(args.sam_model_path),
        "save_dir": str(save_dir.resolve()),
        "log_path": str(log_path.resolve()),
        "num_cases": len(case_metrics),
        "sgdl_avg_dice": float(np.mean([item["sgdl_dice"] for item in valid_metrics])) if valid_metrics else float("nan"),
        "sgdl_avg_iou": float(np.mean([item["sgdl_iou"] for item in valid_metrics])) if valid_metrics else float("nan"),
        "sgdl_avg_hd95": float(np.mean([item["sgdl_hd95"] for item in valid_metrics])) if valid_metrics else float("nan"),
        "sam_avg_dice": float(np.mean([item["sam_dice"] for item in valid_metrics])) if valid_metrics else float("nan"),
        "sam_avg_iou": float(np.mean([item["sam_iou"] for item in valid_metrics])) if valid_metrics else float("nan"),
        "sam_avg_hd95": float(np.mean([item["sam_hd95"] for item in valid_metrics])) if valid_metrics else float("nan"),
        "prompt_weight_mean": float(np.mean([item["prompt_weight_mean"] for item in valid_metrics])) if valid_metrics else float("nan"),
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
            "index", "case_name", "dice", "iou", "hd95",
            "sgdl_dice", "sgdl_iou", "sgdl_hd95",
            "sam_dice", "sam_iou", "sam_hd95", "prompt_weight_mean",
            "sgdl_pred_positive_pixels", "sam_pred_positive_pixels", "gt_positive_pixels"
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
        {
            "split": summary["split"],
            "num_cases": summary["num_cases"],
            "avg_dice": summary["sgdl_avg_dice"],
            "avg_iou": summary["sgdl_avg_iou"],
            "avg_hd95": summary["sgdl_avg_hd95"],
        },
        metadata={
            "split": args.split,
            "sgdl_model_path": os.path.abspath(args.SGDL_model_path),
            "sam_model_path": os.path.abspath(args.sam_model_path),
            "log_path": str(log_path.resolve()),
            "root_summary_path": str(summary_path.resolve()),
            "root_case_metrics_path": str(csv_path.resolve()),
        },
    )

    logging.info("Prediction summary: %s", summary)
    logging.info("Evaluation monitor saved to: %s", monitor_dir.resolve())
    print(f"{args.split} :")
    print("sgdl_avg_dice: ", summary["sgdl_avg_dice"])
    print("sgdl_avg_iou: ", summary["sgdl_avg_iou"])
    print("sgdl_avg_hd95: ", summary["sgdl_avg_hd95"])
    print("sam_avg_dice: ", summary["sam_avg_dice"])
    print("sam_avg_iou: ", summary["sam_avg_iou"])
    print("sam_avg_hd95: ", summary["sam_avg_hd95"])
    print("outputs saved to:", save_dir.resolve())


if __name__ == '__main__':
    main()
