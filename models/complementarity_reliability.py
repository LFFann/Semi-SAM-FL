import torch
import torch.nn as nn
import torch.nn.functional as F


def entropy_map(prob):
    entropy = -torch.sum(prob * torch.log(prob.clamp_min(1e-6)), dim=1, keepdim=True)
    denom = torch.log(prob.new_tensor(float(prob.shape[1]))).clamp_min(1e-6)
    return (entropy / denom).clamp(0.0, 1.0)


def probability_margin(prob):
    top2 = torch.topk(prob, k=min(2, prob.shape[1]), dim=1).values
    if top2.shape[1] == 1:
        return top2[:, :1]
    return (top2[:, :1] - top2[:, 1:2]).clamp(0.0, 1.0)


def _sobel_map(x):
    if x.dim() == 3:
        x = x.unsqueeze(1)
    kernel_x = x.new_tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]).view(1, 1, 3, 3)
    kernel_y = x.new_tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]]).view(1, 1, 3, 3)
    gx = F.conv2d(x.float(), kernel_x, padding=1)
    gy = F.conv2d(x.float(), kernel_y, padding=1)
    edge = torch.sqrt(gx.square() + gy.square() + 1e-8)
    scale = edge.flatten(1).amax(dim=1).view(-1, 1, 1, 1).clamp_min(1e-6)
    return (edge / scale).clamp(0.0, 1.0)


def build_reliability_inputs(student_prob, residual_logits, teacher_prob=None, sam_prob=None, features=None):
    """Build pixel-level complementarity features.

    Output channels contain entropy, margin, teacher confidence, SAM/student
    disagreement, boundary uncertainty, residual magnitude, and optional
    low-channel student features.
    """
    maps = [entropy_map(student_prob), probability_margin(student_prob)]
    if teacher_prob is not None:
        maps.append(teacher_prob.max(dim=1, keepdim=True).values)
    else:
        maps.append(student_prob.max(dim=1, keepdim=True).values.detach())

    if sam_prob is not None:
        if sam_prob.shape[1] == 1 and student_prob.shape[1] > 1:
            sam_fg = sam_prob
            student_fg = student_prob[:, 1:, :, :].sum(dim=1, keepdim=True)
            disagreement = (sam_fg - student_fg).abs()
            boundary_diff = (_sobel_map(sam_fg) - _sobel_map(student_fg)).abs()
        else:
            disagreement = (sam_prob - student_prob).abs().mean(dim=1, keepdim=True)
            boundary_diff = (_sobel_map(sam_prob[:, 1:, :, :].sum(dim=1, keepdim=True)) - _sobel_map(
                student_prob[:, 1:, :, :].sum(dim=1, keepdim=True)
            )).abs()
    else:
        disagreement = entropy_map(student_prob)
        student_fg = student_prob[:, 1:, :, :].sum(dim=1, keepdim=True) if student_prob.shape[1] > 1 else student_prob[:, :1]
        boundary_diff = _sobel_map(student_fg)
    maps.extend([disagreement.clamp(0.0, 1.0), boundary_diff.clamp(0.0, 1.0)])

    residual_prob = torch.softmax(residual_logits, dim=1)
    residual_mag = residual_logits.detach().abs().mean(dim=1, keepdim=True)
    residual_mag = residual_mag / residual_mag.flatten(1).amax(dim=1).view(-1, 1, 1, 1).clamp_min(1e-6)
    graph_student_diff = (residual_prob - student_prob).abs().mean(dim=1, keepdim=True)
    maps.extend([residual_mag.clamp(0.0, 1.0), graph_student_diff.clamp(0.0, 1.0)])

    base = torch.cat(maps, dim=1)
    if features is not None:
        feat = F.interpolate(features, size=student_prob.shape[-2:], mode="bilinear", align_corners=False)
        feat = feat[:, : min(8, feat.shape[1])]
        base = torch.cat([base, feat], dim=1)
    return base


class ComplementarityReliabilityHead(nn.Module):
    """Predict whether residual correction is useful relative to student."""

    def __init__(self, in_channels, hidden=32):
        super().__init__()
        self.head = nn.Sequential(
            nn.Conv2d(in_channels, hidden, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, hidden, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, 1, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, reliability_inputs):
        return self.head(reliability_inputs).clamp(0.0, 1.0)


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
            scores.append(pred_patch.new_tensor(1.0, dtype=torch.float32))
        elif pred_sum == 0 or gt_sum == 0:
            scores.append(pred_patch.new_tensor(0.0, dtype=torch.float32))
        else:
            scores.append(2.0 * (pred & gt).float().sum() / (pred_sum.float() + gt_sum.float()).clamp_min(1.0))
    return torch.stack(scores).mean() if scores else pred_patch.new_tensor(0.0, dtype=torch.float32)


@torch.no_grad()
def compute_delta_utility_target(
    student_logits,
    corrected_logits,
    gt_mask=None,
    teacher_logits=None,
    patch_size=16,
    utility_margin=0.005,
    teacher_conf_thresh=0.8,
    ignore_neutral=True,
):
    """Patch-level target mapped back to pixels.

    Returns target [B,1,H,W], valid [B,1,H,W], and improvement [B,1,Hp,Wp].
    """
    bsz, num_classes, height, width = student_logits.shape
    student_pred = torch.argmax(student_logits.detach(), dim=1)
    corrected_pred = torch.argmax(corrected_logits.detach(), dim=1)
    target = student_logits.new_zeros(bsz, 1, height, width)
    valid = student_logits.new_zeros(bsz, 1, height, width)
    improvements = []

    if gt_mask is not None:
        ref = gt_mask.long()
        mode = "dice"
    elif teacher_logits is not None:
        teacher_prob = torch.softmax(teacher_logits.detach(), dim=1)
        teacher_conf, ref = teacher_prob.max(dim=1)
        mode = "teacher"
    else:
        return target, valid, student_logits.new_zeros(bsz, 1, 1, 1)

    for b in range(bsz):
        row_values = []
        for y0, y1, x0, x1 in _patch_slices(height, width, patch_size):
            if mode == "dice":
                s_score = _patch_macro_dice(student_pred[b, y0:y1, x0:x1], ref[b, y0:y1, x0:x1], num_classes)
                c_score = _patch_macro_dice(corrected_pred[b, y0:y1, x0:x1], ref[b, y0:y1, x0:x1], num_classes)
                patch_valid = True
            else:
                conf_patch = teacher_conf[b, y0:y1, x0:x1]
                patch_valid = bool((conf_patch > teacher_conf_thresh).float().mean().item() > 0.5)
                teacher_patch = ref[b, y0:y1, x0:x1]
                s_score = (student_pred[b, y0:y1, x0:x1] == teacher_patch).float().mean()
                c_score = (corrected_pred[b, y0:y1, x0:x1] == teacher_patch).float().mean()
            delta = c_score - s_score
            row_values.append(delta)
            if not patch_valid:
                continue
            if delta > utility_margin:
                target[b, :, y0:y1, x0:x1] = 1.0
                valid[b, :, y0:y1, x0:x1] = 1.0
            elif delta < -utility_margin or not ignore_neutral:
                target[b, :, y0:y1, x0:x1] = 0.0
                valid[b, :, y0:y1, x0:x1] = 1.0
        improvements.append(torch.stack(row_values) if row_values else student_logits.new_zeros(1))
    max_len = max(item.numel() for item in improvements)
    padded = []
    for item in improvements:
        if item.numel() < max_len:
            item = F.pad(item, (0, max_len - item.numel()))
        padded.append(item)
    improvement = torch.stack(padded).view(bsz, 1, -1, 1)
    return target, valid, improvement


def compute_reliability_losses(r_hat, target, valid, rank_margin=0.1, use_rank=True):
    valid_sum = valid.sum()
    if valid_sum.item() <= 0:
        zero = r_hat.sum() * 0.0
        return zero, zero
    pred = r_hat.clamp(1e-6, 1.0 - 1e-6)
    bce_map = F.binary_cross_entropy(pred, target, reduction="none")
    bce = (bce_map * valid).sum() / valid_sum.clamp_min(1.0)
    if not use_rank:
        return bce, r_hat.sum() * 0.0
    pos = pred[(target > 0.5) & (valid > 0.5)]
    neg = pred[(target <= 0.5) & (valid > 0.5)]
    if pos.numel() == 0 or neg.numel() == 0:
        return bce, r_hat.sum() * 0.0
    rank = F.relu(rank_margin - (pos.mean() - neg.mean()))
    return bce, rank
