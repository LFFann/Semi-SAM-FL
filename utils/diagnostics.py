import math

import numpy as np
import torch
import torch.nn.functional as F

from utils.utils import multiclass_segmentation_metrics


def compute_graph_stats(adjacency):
    """Return anti-collapse diagnostics for [N,N] or [B,N,N] adjacency."""
    if adjacency is None:
        return {
            "graph_adjacency_norm": 0.0,
            "A_diag_mean": 0.0,
            "A_offdiag_mean": 0.0,
            "A_offdiag_std": 0.0,
            "A_entropy": 0.0,
            "A_dynamic_delta_to_identity": 0.0,
        }
    A = adjacency.detach()
    if A.dim() == 2:
        A = A.unsqueeze(0)
    n = A.shape[-1]
    eye = torch.eye(n, device=A.device, dtype=A.dtype).unsqueeze(0)
    diag_mask = eye.bool()
    offdiag_mask = ~diag_mask
    entropy = -(A.clamp_min(1e-8) * A.clamp_min(1e-8).log()).sum(dim=-1).mean()
    return {
        "graph_adjacency_norm": float(A.norm().cpu()),
        "A_diag_mean": float(A.masked_select(diag_mask).mean().cpu()),
        "A_offdiag_mean": float(A.masked_select(offdiag_mask).mean().cpu()),
        "A_offdiag_std": float(A.masked_select(offdiag_mask).std(unbiased=False).cpu()),
        "A_entropy": float(entropy.cpu()),
        "A_dynamic_delta_to_identity": float((A - eye).norm().cpu()),
    }


def compute_deploy_delta(student_logits, deploy_logits, residual_logits, r_delta_hat, lambda_res, high_thresh=0.7, low_thresh=0.3):
    prob_student = torch.softmax(student_logits.detach(), dim=1)
    prob_deploy = torch.softmax(deploy_logits.detach(), dim=1)
    gate = r_delta_hat.detach().clamp(0.0, 1.0)
    gated_residual = lambda_res * gate * residual_logits.detach()
    return {
        "lambda_res_current": float(lambda_res),
        "deploy_student_delta": float((prob_deploy - prob_student).abs().mean().cpu()),
        "residual_logits_abs_mean": float(residual_logits.detach().abs().mean().cpu()),
        "gated_residual_abs_mean": float(gated_residual.abs().mean().cpu()),
        "low_reliability_ratio": float((gate < low_thresh).float().mean().cpu()),
        "high_reliability_ratio": float((gate > high_thresh).float().mean().cpu()),
    }


def _patch_slices(height, width, patch_size):
    for y0 in range(0, height, patch_size):
        y1 = min(y0 + patch_size, height)
        for x0 in range(0, width, patch_size):
            x1 = min(x0 + patch_size, width)
            yield y0, y1, x0, x1


def _patch_macro_dice(pred_patch, gt_patch, num_classes):
    scores = []
    for class_idx in range(1, num_classes):
        pred = pred_patch == class_idx
        gt = gt_patch == class_idx
        pred_sum = pred.sum()
        gt_sum = gt.sum()
        if pred_sum == 0 and gt_sum == 0:
            scores.append(1.0)
        elif pred_sum == 0 or gt_sum == 0:
            scores.append(0.0)
        else:
            scores.append(float(2.0 * (pred & gt).sum() / float(pred_sum + gt_sum)))
    return float(np.mean(scores)) if scores else 0.0


def compute_oracle_fusion(student_logits, corrected_logits, gt, r_delta_hat=None, patch_size=16, num_classes=None):
    """Patch oracle diagnostic only. GT is never used to form deploy logits."""
    if num_classes is None:
        num_classes = student_logits.shape[1]
    student_pred = torch.argmax(student_logits.detach(), dim=1).cpu().numpy()
    corrected_pred = torch.argmax(corrected_logits.detach(), dim=1).cpu().numpy()
    gt_np = gt.detach().long().cpu().numpy()
    oracle_pred = student_pred.copy()
    improvements = []
    r_scores = []
    better = worse = equal = 0

    if r_delta_hat is not None:
        r_np = F.interpolate(
            r_delta_hat.detach(), size=student_logits.shape[-2:], mode="bilinear", align_corners=False
        ).squeeze(1).cpu().numpy()
    else:
        r_np = None

    bsz, height, width = student_pred.shape
    for b in range(bsz):
        for y0, y1, x0, x1 in _patch_slices(height, width, patch_size):
            s_patch = student_pred[b, y0:y1, x0:x1]
            c_patch = corrected_pred[b, y0:y1, x0:x1]
            g_patch = gt_np[b, y0:y1, x0:x1]
            s_dice = _patch_macro_dice(s_patch, g_patch, num_classes)
            c_dice = _patch_macro_dice(c_patch, g_patch, num_classes)
            delta = c_dice - s_dice
            improvements.append(delta)
            if r_np is not None:
                r_scores.append(float(r_np[b, y0:y1, x0:x1].mean()))
            if delta > 1e-8:
                better += 1
                oracle_pred[b, y0:y1, x0:x1] = c_patch
            elif delta < -1e-8:
                worse += 1
            else:
                equal += 1

    oracle_tensor = torch.from_numpy(oracle_pred).to(student_logits.device)
    oracle_onehot = F.one_hot(oracle_tensor.long(), num_classes=num_classes).permute(0, 3, 1, 2).float()
    student_soft = F.one_hot(torch.from_numpy(student_pred).to(student_logits.device).long(), num_classes=num_classes)
    student_soft = student_soft.permute(0, 3, 1, 2).float()
    oracle_metrics = multiclass_segmentation_metrics(gt, oracle_onehot, num_classes)
    student_metrics = multiclass_segmentation_metrics(gt, student_soft, num_classes)
    total = max(better + worse + equal, 1)

    corr = 0.0
    if r_scores and len(r_scores) > 1 and np.std(r_scores) > 1e-8 and np.std(improvements) > 1e-8:
        corr = float(np.corrcoef(np.asarray(r_scores), np.asarray(improvements))[0, 1])
        if math.isnan(corr):
            corr = 0.0

    return {
        "oracle_fusion_avg_dice": float(oracle_metrics["avg_dice"]),
        "oracle_gain_over_student": float(oracle_metrics["avg_dice"] - student_metrics["avg_dice"]),
        "corrected_better_patch_ratio": float(better / total),
        "corrected_worse_patch_ratio": float(worse / total),
        "corrected_equal_patch_ratio": float(equal / total),
        "graph_or_residual_better_region_ratio": float(better / total),
        "graph_or_residual_worse_region_ratio": float(worse / total),
        "graph_or_residual_neutral_region_ratio": float(equal / total),
        "R_delta_improvement_corr": corr,
    }
