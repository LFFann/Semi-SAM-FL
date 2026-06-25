import argparse
import csv
import json
import logging
import os
import sys
from pathlib import Path

import cv2
import matplotlib
import numpy as np
import torch
from torch.utils.data import DataLoader

matplotlib.use("Agg")
import matplotlib.pyplot as plt

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "..", ".."))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)
if REPO_ROOT not in sys.path:
    sys.path.insert(1, REPO_ROOT)

from dataloader.dataset import build_Dataset
from dataloader.transforms import build_transforms
from ra_modules import foreground_boundary_tube, logits_to_hard_label, normalized_entropy_map
from prediction_ra import add_common_args
from trainer_ra import Trainer


def parse_args():
    parser = argparse.ArgumentParser()
    add_common_args(parser)
    parser.set_defaults(split="val", save_dir="./Results/RA_SAM_SSL/fold_0/diagnosis_val")
    parser.add_argument("--max_cases", type=int, default=0)
    return parser.parse_args()


def setup_logger(save_dir: Path):
    save_dir.mkdir(parents=True, exist_ok=True)
    log_path = save_dir / "diagnosis.log"
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s.%(msecs)03d] %(message)s",
        datefmt="%H:%M:%S",
        handlers=[logging.FileHandler(log_path, encoding="utf-8"), logging.StreamHandler()],
        force=True,
    )
    return log_path


def to_numpy_map(x):
    return x.detach().float().squeeze().cpu().numpy()


def normalize01(x):
    x = np.asarray(x, dtype=np.float32)
    finite = np.isfinite(x)
    if not finite.any():
        return np.zeros_like(x, dtype=np.float32)
    lo = np.nanpercentile(x[finite], 1)
    hi = np.nanpercentile(x[finite], 99)
    if hi <= lo:
        return np.zeros_like(x, dtype=np.float32)
    return np.clip((x - lo) / (hi - lo), 0.0, 1.0)


def save_heatmap(path: Path, value_map, base_image=None):
    heat = (normalize01(value_map) * 255).astype(np.uint8)
    heat = cv2.applyColorMap(heat, cv2.COLORMAP_JET)
    if base_image is not None:
        base = np.clip(base_image, 0, 255).astype(np.uint8)
        if base.ndim == 2:
            base = cv2.cvtColor(base, cv2.COLOR_GRAY2BGR)
        heat = cv2.addWeighted(base, 0.55, heat, 0.45, 0)
    cv2.imwrite(str(path), heat)


def binary_boundary(mask_np, radius=2):
    mask = torch.from_numpy(mask_np.astype(np.float32)).view(1, 1, *mask_np.shape)
    return foreground_boundary_tube(mask, radius).squeeze().numpy() > 0.5


def rankdata(values):
    order = np.argsort(values)
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(len(values), dtype=np.float64)
    return ranks


def spearman(x, y):
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    valid = np.isfinite(x) & np.isfinite(y)
    x = x[valid]
    y = y[valid]
    if len(x) < 3:
        return float("nan")
    rx = rankdata(x)
    ry = rankdata(y)
    if np.std(rx) == 0 or np.std(ry) == 0:
        return float("nan")
    return float(np.corrcoef(rx, ry)[0, 1])


def binary_auc(labels, scores):
    labels = np.asarray(labels).astype(np.int32)
    scores = np.asarray(scores).astype(np.float64)
    valid = np.isfinite(scores)
    labels = labels[valid]
    scores = scores[valid]
    pos = labels == 1
    neg = labels == 0
    if pos.sum() == 0 or neg.sum() == 0:
        return float("nan")
    ranks = rankdata(scores) + 1.0
    return float((ranks[pos].sum() - pos.sum() * (pos.sum() + 1) / 2.0) / (pos.sum() * neg.sum()))


def binary_aupr(labels, scores):
    labels = np.asarray(labels).astype(np.int32)
    scores = np.asarray(scores).astype(np.float64)
    valid = np.isfinite(scores)
    labels = labels[valid]
    scores = scores[valid]
    if labels.sum() == 0:
        return float("nan")
    order = np.argsort(-scores)
    labels = labels[order]
    tp = np.cumsum(labels == 1)
    fp = np.cumsum(labels == 0)
    precision = tp / np.maximum(tp + fp, 1)
    recall = tp / max(int((labels == 1).sum()), 1)
    precision = np.concatenate([[1.0], precision])
    recall = np.concatenate([[0.0], recall])
    return float(np.trapz(precision, recall))


def safe_mean_on_mask(value, mask):
    if mask.sum() <= 0:
        return float("nan")
    return float(np.nanmean(value[mask]))


def curve_means(ra_output):
    tube = ra_output.boundary_tube > 0.5
    orig_prob = torch.softmax(ra_output.orig_curve_logits, dim=2)[:, :, 1:2]
    atten_prob = torch.softmax(ra_output.atten_curve_logits, dim=2)[:, :, 1:2]
    orig_values = []
    atten_values = []
    for idx in range(orig_prob.shape[0]):
        if tube.sum() > 0:
            orig_values.append(float(orig_prob[idx][tube].mean().item()))
            atten_values.append(float(atten_prob[idx][tube].mean().item()))
        else:
            orig_values.append(float("nan"))
            atten_values.append(float("nan"))
    return orig_values, atten_values


def save_curve(path: Path, deltas, orig_values, atten_values):
    plt.figure(figsize=(5, 3.2))
    plt.plot(deltas, orig_values, marker="o", label="original")
    plt.plot(deltas, atten_values, marker="s", label="attenuated")
    plt.xlabel("prompt dose delta")
    plt.ylabel("mean boundary foreground response")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def main():
    args = parse_args()
    save_dir = Path(args.save_dir)
    log_path = setup_logger(save_dir)
    logging.info("Diagnosis arguments: %s", args)

    data_transforms = build_transforms(args)
    dataset = build_Dataset(args=args, data_dir=args.data_path + args.dataset, split=args.split, transform=data_transforms["valid_test"])
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=args.num_workers)
    logging.info("Loaded split '%s' with %d samples", args.split, len(dataset))

    trainer = Trainer(args)
    trainer.SGDL.load_state_dict(torch.load(args.SGDL_model_path, map_location=args.device))
    trainer.sam_model.load_state_dict(torch.load(args.sam_model_path, map_location=args.device))
    trainer.SGDL.eval()
    trainer.sam_model.eval()

    heatmap_dir = save_dir / "heatmaps"
    curve_dir = save_dir / "response_curves"
    heatmap_dir.mkdir(parents=True, exist_ok=True)
    curve_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    pooled = {
        "admission": [],
        "pc_neg": [],
        "esr": [],
        "entropy_neg": [],
        "agreement": [],
        "edge": [],
        "reliable": [],
        "error": [],
    }

    with torch.no_grad():
        for index, sampled_batch in enumerate(loader):
            if args.max_cases > 0 and index >= args.max_cases:
                break
            image = sampled_batch["image"].to(args.device)
            label = sampled_batch["label"].to(args.device)
            case_name = sampled_batch.get("case_name", [f"case_{index}.png"])[0]
            ori_image = sampled_batch.get("ori_image", image.detach().cpu())[0].detach().cpu().numpy()
            if ori_image.ndim == 3:
                ori_image = np.transpose(ori_image, (1, 2, 0))
            ori_image = np.clip(ori_image, 0, 255).astype(np.uint8)

            image_embeddings = trainer.sam_model.image_encoder(image)
            pred_unet, pred_vnet, pred_unet_soft, pred_vnet_soft, fusion_map = trainer.SGDL(image)
            _, boxes_embedding, _ = trainer.sam_model.super_prompt(image_embeddings)
            ra_output = trainer.response_audit_probe.probe(
                trainer.sam_model,
                image,
                image_embeddings,
                boxes_embedding,
                fusion_map,
                labeled_bs=0,
            )

            pred_label = logits_to_hard_label(ra_output.y0_logits).detach().cpu().numpy()[0]
            gt_label = label.detach().cpu().numpy()[0].astype(np.int64)
            pred_boundary = binary_boundary((pred_label > 0).astype(np.float32), radius=2)
            gt_boundary = binary_boundary((gt_label > 0).astype(np.float32), radius=2)
            boundary_region = to_numpy_map(ra_output.boundary_tube) > 0.5
            error_map = np.logical_xor(pred_boundary, gt_boundary).astype(np.float32)
            reliable_map = (1.0 - error_map).astype(np.float32)

            admission = to_numpy_map(ra_output.admission_map)
            pc = to_numpy_map(ra_output.pc_map)
            esr = to_numpy_map(ra_output.esr_map)
            ji = to_numpy_map(ra_output.ji_map)
            valid = to_numpy_map(ra_output.valid_map)
            prompt_dominated = to_numpy_map(ra_output.prompt_dominated_map)
            prior_locked = to_numpy_map(ra_output.prior_locked_map)
            evidence_sensitive = to_numpy_map(ra_output.evidence_sensitive_map)
            unidentifiable = to_numpy_map(ra_output.unidentifiable_map)
            entropy = to_numpy_map(normalized_entropy_map(torch.softmax(fusion_map, dim=1)))
            agreement = 1.0 - to_numpy_map(torch.abs(pred_unet_soft[:, 1:2] - pred_vnet_soft[:, 1:2]).clamp(0.0, 1.0))

            gray = cv2.cvtColor(ori_image, cv2.COLOR_BGR2GRAY) if ori_image.ndim == 3 else ori_image
            sobel_x = cv2.Sobel(gray.astype(np.float32) / 255.0, cv2.CV_32F, 1, 0, ksize=3)
            sobel_y = cv2.Sobel(gray.astype(np.float32) / 255.0, cv2.CV_32F, 0, 1, ksize=3)
            edge = np.sqrt(sobel_x * sobel_x + sobel_y * sobel_y)

            case_stem = Path(case_name).stem
            for name, value in [
                ("admission", admission),
                ("pc", pc),
                ("esr", esr),
                ("ji", ji),
                ("valid", valid),
                ("prompt_dominated", prompt_dominated),
                ("prior_locked", prior_locked),
                ("evidence_sensitive", evidence_sensitive),
                ("unidentifiable", unidentifiable),
                ("boundary_error", error_map),
            ]:
                save_heatmap(heatmap_dir / f"{case_stem}_{name}.png", value, ori_image)
            orig_curve, atten_curve = curve_means(ra_output)
            save_curve(curve_dir / f"{case_stem}_response_curve.png", trainer.response_audit_probe.delta_levels, orig_curve, atten_curve)

            mask = boundary_region
            reliable_flat = reliable_map[mask].reshape(-1)
            error_flat = error_map[mask].reshape(-1)
            scores = {
                "admission": admission[mask].reshape(-1),
                "pc_neg": (-pc)[mask].reshape(-1),
                "esr": esr[mask].reshape(-1),
                "entropy_neg": (-entropy)[mask].reshape(-1),
                "agreement": agreement[mask].reshape(-1),
                "edge": edge[mask].reshape(-1),
            }
            for key, value in scores.items():
                pooled[key].extend(value.tolist())
            pooled["reliable"].extend(reliable_flat.tolist())
            pooled["error"].extend(error_flat.tolist())

            row = {
                "index": index,
                "case_name": case_name,
                "boundary_pixels": int(mask.sum()),
                "boundary_error_rate": safe_mean_on_mask(error_map, mask),
                "admission_mean": safe_mean_on_mask(admission, mask),
                "pc_mean": safe_mean_on_mask(pc, mask),
                "esr_mean": safe_mean_on_mask(esr, mask),
                "ji_mean": safe_mean_on_mask(ji, mask),
                "valid_mean": safe_mean_on_mask(valid, mask),
                "prompt_dominated_ratio": safe_mean_on_mask(prompt_dominated, mask),
                "prior_locked_ratio": safe_mean_on_mask(prior_locked, mask),
                "evidence_sensitive_ratio": safe_mean_on_mask(evidence_sensitive, mask),
                "unidentifiable_ratio": safe_mean_on_mask(unidentifiable, mask),
                "entropy_mean": safe_mean_on_mask(entropy, mask),
                "agreement_mean": safe_mean_on_mask(agreement, mask),
                "edge_mean": safe_mean_on_mask(edge, mask),
                "admission_reliable_auroc": binary_auc(reliable_flat, scores["admission"]),
                "admission_reliable_aupr": binary_aupr(reliable_flat, scores["admission"]),
                "esr_reliable_auroc": binary_auc(reliable_flat, scores["esr"]),
                "pc_reliable_auroc": binary_auc(reliable_flat, scores["pc_neg"]),
            }
            rows.append(row)
            logging.info(
                "case=%s boundary_error=%.6f admission=%.6f pc=%.6f esr=%.6f valid=%.6f",
                case_name,
                row["boundary_error_rate"],
                row["admission_mean"],
                row["pc_mean"],
                row["esr_mean"],
                row["valid_mean"],
            )

    reliable = np.asarray(pooled["reliable"], dtype=np.int32)
    error = np.asarray(pooled["error"], dtype=np.float32)
    summary = {
        "split": args.split,
        "num_cases": len(rows),
        "log_path": str(log_path.resolve()),
        "save_dir": str(save_dir.resolve()),
        "boundary_error_rate": float(np.nanmean(error)) if len(error) else float("nan"),
    }
    for key in ["admission", "pc_neg", "esr", "entropy_neg", "agreement", "edge"]:
        score = np.asarray(pooled[key], dtype=np.float32)
        summary[f"{key}_reliable_auroc"] = binary_auc(reliable, score)
        summary[f"{key}_reliable_aupr"] = binary_aupr(reliable, score)
        summary[f"{key}_spearman_with_error"] = spearman(score, error)

    csv_path = save_dir / "case_diagnosis.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as csv_file:
        fieldnames = list(rows[0].keys()) if rows else ["index", "case_name"]
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    summary_path = save_dir / "summary.json"
    summary_path.write_text(
        json.dumps({"summary": summary, "cases": rows}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    logging.info("Diagnosis summary: %s", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

