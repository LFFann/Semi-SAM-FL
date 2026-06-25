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
from trainer_ra import Trainer
from utils.training_monitor import save_evaluation_artifacts
from utils.utils import eval


def add_common_args(parser):
    parser.add_argument("--data_path", type=str, default="./SampleData")
    parser.add_argument("--dataset", type=str, default="/260513_data_label1")
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--num_classes", type=int, default=2)
    parser.add_argument("--in_channels", type=int, default=3)
    parser.add_argument("-lr", type=float, default=1e-4)
    parser.add_argument("-UNet_lr", type=float, default=0.01)
    parser.add_argument("-VNet_lr", type=float, default=0.01)
    parser.add_argument("--image_size", type=int, default=256)
    parser.add_argument("--point_nums", type=int, default=5)
    parser.add_argument("--box_nums", type=int, default=1)
    parser.add_argument("--mod", type=str, default="sam_adpt")
    parser.add_argument("--model_type", type=str, default="vit_b")
    parser.add_argument("-thd", type=bool, default=False)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--labeled_bs", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--mixed_iterations", type=int, default=12000)
    parser.add_argument("--max_iterations", type=int, default=50000)
    parser.add_argument("--val_interval", type=int, default=200)
    parser.add_argument("--n_fold", type=int, default=1)
    parser.add_argument("--consistency", type=float, default=0.1)
    parser.add_argument("--consistency_rampup", type=float, default=200.0)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--multimask", type=bool, default=False)
    parser.add_argument("--encoder_adapter", type=bool, default=True)
    parser.add_argument("--sam_checkpoint", type=str, default="./sam_vit_b_01ec64.pth")
    parser.add_argument("--SGDL_model_path", type=str, default="./Results/RA_SAM_SSL/fold_0/SGDL_best_model.pth")
    parser.add_argument("--sam_model_path", type=str, default="./Results/RA_SAM_SSL/fold_0/sam_best_model.pth")
    parser.add_argument("--save_dir", type=str, default="./Results/RA_SAM_SSL/fold_0/prediction_test")
    parser.add_argument("--num_workers", type=int, default=0)

    parser.add_argument("--ra_enabled", type=int, default=1)
    parser.add_argument("--ra_delta_pixels", type=int, default=2)
    parser.add_argument("--ra_delta_levels", type=str, default="-4,-2,0,2,4")
    parser.add_argument("--ra_tube_radius", type=int, default=5)
    parser.add_argument("--ra_intervention_kernel", type=int, default=9)
    parser.add_argument("--ra_intervention_strength", type=float, default=0.25)
    parser.add_argument("--ra_beta_esr", type=float, default=6.0)
    parser.add_argument("--ra_beta_prompt", type=float, default=2.0)
    parser.add_argument("--ra_beta_jump", type=float, default=1.0)
    parser.add_argument("--ra_esr_mid", type=float, default=0.05)
    parser.add_argument("--ra_min_area_change", type=float, default=0.005)
    parser.add_argument("--ra_max_area_change", type=float, default=0.40)
    parser.add_argument("--ra_saturation_iou", type=float, default=0.995)
    parser.add_argument("--ra_prompt_logit", type=float, default=6.0)
    parser.add_argument("--ra_disable_esr", type=int, default=0)
    parser.add_argument("--ra_disable_prompt", type=int, default=0)
    parser.add_argument("--ra_no_intervention", type=int, default=0)
    parser.add_argument("--ra_baseline", type=str, default="response_audit", choices=["response_audit", "prompt_ensemble"])
    parser.add_argument("--ra_intervention_mode", type=str, default="boundary", choices=["boundary", "random", "interior", "none"])
    parser.add_argument("--ra_prompt_high", type=float, default=0.20)
    parser.add_argument("--ra_prompt_low", type=float, default=0.08)
    parser.add_argument("--ra_esr_high", type=float, default=0.08)
    parser.add_argument("--ra_esr_low", type=float, default=0.03)
    parser.add_argument("--ra_jump_high", type=float, default=0.25)
    parser.add_argument("--ra_min_inside_change", type=float, default=0.002)
    parser.add_argument("--ra_max_inside_change", type=float, default=0.20)
    parser.add_argument("--ra_max_outside_change", type=float, default=0.002)
    parser.add_argument("--ra_enforce_intervention_validity", type=int, default=1)


def parse_args():
    parser = argparse.ArgumentParser()
    add_common_args(parser)
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


def to_uint8_rgb(image_tensor):
    image = image_tensor.detach().cpu().numpy()
    image = np.transpose(image, (1, 2, 0))
    return np.clip(image, 0, 255).astype(np.uint8)


def binary_mask_from_softmax(prob_map, threshold=0.5):
    if prob_map.shape[1] > 2:
        pred_mask = torch.argmax(prob_map, dim=1).squeeze(0).detach().cpu().numpy()
        return np.clip(pred_mask * (255 // max(prob_map.shape[1] - 1, 1)), 0, 255).astype(np.uint8)
    pred_mask = (prob_map > threshold).to(torch.float32).squeeze(0)[1].detach().cpu().numpy()
    return (pred_mask * 255).astype(np.uint8)


def overlay_mask(image_bgr, mask_uint8, color=(0, 0, 255), alpha=0.35):
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


def save_case_outputs(save_dir, case_name, ori_image, gt_mask, sgdl_mask, sam_mask):
    dirs = {
        "original": save_dir / "original",
        "gt_mask": save_dir / "gt_mask",
        "pred_mask": save_dir / "pred_mask",
        "overlay": save_dir / "overlay",
        "pred_mask_sgdl": save_dir / "pred_mask_sgdl",
        "pred_mask_sam": save_dir / "pred_mask_sam",
        "overlay_sgdl": save_dir / "overlay_sgdl",
        "overlay_sam": save_dir / "overlay_sam",
    }
    for directory in dirs.values():
        directory.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(dirs["original"] / case_name), ori_image)
    cv2.imwrite(str(dirs["gt_mask"] / case_name), gt_mask)
    cv2.imwrite(str(dirs["pred_mask"] / case_name), sgdl_mask)
    cv2.imwrite(str(dirs["overlay"] / case_name), overlay_mask(ori_image, sgdl_mask))
    cv2.imwrite(str(dirs["pred_mask_sgdl"] / case_name), sgdl_mask)
    cv2.imwrite(str(dirs["pred_mask_sam"] / case_name), sam_mask)
    cv2.imwrite(str(dirs["overlay_sgdl"] / case_name), overlay_mask(ori_image, sgdl_mask, color=(0, 0, 255)))
    cv2.imwrite(str(dirs["overlay_sam"] / case_name), overlay_mask(ori_image, sam_mask, color=(0, 255, 0)))


def forward_ra(trainer, image):
    image_embeddings = trainer.sam_model.image_encoder(image)
    pred_unet, pred_vnet, pred_unet_soft, pred_vnet_soft, fusion_map = trainer.SGDL(image)
    pred_sam = trainer.forward_sam_from_logits(image_embeddings, fusion_map)
    return torch.softmax(fusion_map, dim=1), torch.softmax(pred_sam, dim=1)


def main():
    args = parse_args()
    save_dir = Path(args.save_dir)
    log_path = setup_logger(save_dir)
    logging.info("Prediction arguments: %s", args)

    data_transforms = build_transforms(args)
    test_dataset = build_Dataset(args=args, data_dir=args.data_path + args.dataset, split=args.split, transform=data_transforms["valid_test"])
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=args.num_workers)
    logging.info("Loaded split '%s' with %d samples", args.split, len(test_dataset))

    trainer = Trainer(args)
    trainer.SGDL.load_state_dict(torch.load(args.SGDL_model_path, map_location=args.device))
    trainer.sam_model.load_state_dict(torch.load(args.sam_model_path, map_location=args.device))
    trainer.SGDL.eval()
    trainer.sam_model.eval()
    logging.info("Loaded SGDL weights from %s", args.SGDL_model_path)
    logging.info("Loaded SAM weights from %s", args.sam_model_path)

    case_metrics = []
    with torch.no_grad():
        for index, sampled_batch in enumerate(test_loader):
            test_image = sampled_batch["image"].to(args.device)
            test_label = sampled_batch["label"].to(args.device)
            ori_image = to_uint8_rgb(sampled_batch["ori_image"][0]).copy()
            case_name = sampled_batch["case_name"][0]
            fusion_soft, sam_soft = forward_ra(trainer, test_image)

            sgdl_dice, sgdl_iou, sgdl_hd95 = safe_eval(test_label, fusion_soft)
            sam_dice, sam_iou, sam_hd95 = safe_eval(test_label, sam_soft)
            sgdl_mask = binary_mask_from_softmax(fusion_soft)
            sam_mask = binary_mask_from_softmax(sam_soft)
            gt_mask = (test_label.squeeze(0).detach().cpu().numpy() * 255).astype(np.uint8)
            save_case_outputs(save_dir, case_name, ori_image, gt_mask, sgdl_mask, sam_mask)

            case_info = {
                "index": index,
                "case_name": case_name,
                "dice": sgdl_dice,
                "iou": sgdl_iou,
                "hd95": sgdl_hd95,
                "sgdl_dice": sgdl_dice,
                "sgdl_iou": sgdl_iou,
                "sgdl_hd95": sgdl_hd95,
                "sam_dice": sam_dice,
                "sam_iou": sam_iou,
                "sam_hd95": sam_hd95,
                "sgdl_pred_positive_pixels": int((sgdl_mask > 0).sum()),
                "sam_pred_positive_pixels": int((sam_mask > 0).sum()),
                "gt_positive_pixels": int((gt_mask > 0).sum()),
            }
            case_metrics.append(case_info)
            logging.info("case=%s idx=%d sgdl_dice=%.6f sam_dice=%.6f", case_name, index, sgdl_dice, sam_dice)

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
    }

    csv_path = save_dir / "case_metrics.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as csv_file:
        fieldnames = list(case_metrics[0].keys()) if case_metrics else ["index", "case_name", "dice", "iou", "hd95"]
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(case_metrics)

    summary_path = save_dir / "summary.json"
    summary_path.write_text(json.dumps({"summary": summary, "cases": case_metrics}, ensure_ascii=False, indent=2), encoding="utf-8")
    monitor_dir = save_evaluation_artifacts(
        save_dir,
        case_metrics,
        {"split": summary["split"], "num_cases": summary["num_cases"], "avg_dice": summary["sgdl_avg_dice"], "avg_iou": summary["sgdl_avg_iou"], "avg_hd95": summary["sgdl_avg_hd95"]},
        metadata={"log_path": str(log_path.resolve()), "root_summary_path": str(summary_path.resolve()), "root_case_metrics_path": str(csv_path.resolve())},
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


if __name__ == "__main__":
    main()

