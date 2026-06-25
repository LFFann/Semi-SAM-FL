import torch
import torch.nn.functional as F


def sigmoid_rampup(current, rampup_length):
    if rampup_length <= 0:
        return 1.0
    current = torch.clamp(torch.tensor(float(current)), 0.0, float(rampup_length))
    phase = 1.0 - current / float(rampup_length)
    return float(torch.exp(-5.0 * phase * phase))


def masked_ce_dice_loss(logits, pseudo_label, mask, num_classes):
    """CE + Dice over confident pixels only.

    Args:
        logits: [B, C, H, W]
        pseudo_label: [B, H, W]
        mask: [B, 1, H, W] or [B, H, W], 1 means valid
    """
    if logits.shape[0] == 0:
        return logits.sum() * 0.0
    if mask.dim() == 3:
        mask = mask.unsqueeze(1)
    mask = mask.float()
    valid = mask.sum()
    if valid.item() <= 0:
        return logits.sum() * 0.0

    ce_map = F.cross_entropy(logits, pseudo_label.long(), reduction="none").unsqueeze(1)
    ce_loss = (ce_map * mask).sum() / valid.clamp_min(1.0)

    prob = torch.softmax(logits, dim=1)
    target = F.one_hot(pseudo_label.long(), num_classes=num_classes).permute(0, 3, 1, 2).float()
    mask_c = mask.expand_as(prob)
    intersect = (prob * target * mask_c).sum(dim=(0, 2, 3))
    denom = ((prob.square() + target.square()) * mask_c).sum(dim=(0, 2, 3))
    dice_loss = 1.0 - (2.0 * intersect + 1e-6) / (denom + 1e-6)
    return ce_loss + dice_loss.mean()


def weak_strong_consistency_loss(student_logits, teacher_logits, conf_thresh, num_classes):
    """Teacher pseudo labels supervise student logits on high-confidence pixels."""
    if student_logits.shape[0] == 0 or teacher_logits is None:
        zero = student_logits.sum() * 0.0
        return zero, {
            "teacher_conf_mean": 0.0,
            "pseudo_mask_ratio": 0.0,
        }
    with torch.no_grad():
        teacher_prob = torch.softmax(teacher_logits, dim=1)
        pseudo_conf, pseudo_label = teacher_prob.max(dim=1)
        mask = (pseudo_conf > conf_thresh).float().unsqueeze(1)
    loss = masked_ce_dice_loss(student_logits, pseudo_label, mask, num_classes)
    return loss, {
        "teacher_conf_mean": float(pseudo_conf.mean().detach().cpu()),
        "pseudo_mask_ratio": float(mask.mean().detach().cpu()),
    }


def view_consistency_loss(logits_a, logits_b):
    if logits_a.shape[0] == 0:
        return logits_a.sum() * 0.0
    prob_a = torch.softmax(logits_a, dim=1)
    prob_b = torch.softmax(logits_b.detach(), dim=1)
    return F.mse_loss(prob_a, prob_b)
