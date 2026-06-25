import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.diagnostics import compute_graph_stats


def _sobel_map(x):
    if x.dim() == 3:
        x = x.unsqueeze(1)
    kernel_x = x.new_tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]).view(1, 1, 3, 3)
    kernel_y = x.new_tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]]).view(1, 1, 3, 3)
    gx = F.conv2d(x.float(), kernel_x, padding=1)
    gy = F.conv2d(x.float(), kernel_y, padding=1)
    edge = torch.sqrt(gx.square() + gy.square() + 1e-8)
    max_val = edge.flatten(1).amax(dim=1).view(-1, 1, 1, 1).clamp_min(1e-6)
    return (edge / max_val).clamp(0.0, 1.0)


def _resize_map(x, size):
    if x is None:
        return None
    return F.interpolate(x, size=size, mode="bilinear", align_corners=False)


def build_region_masks(student_prob, teacher_prob=None, uncertainty=None, disagreement=None, feature_size=None, detach=True):
    """Build five region masks [B, N, Hf, Wf] for prototype pooling."""
    if feature_size is None:
        feature_size = student_prob.shape[-2:]
    fg = student_prob[:, 1:, :, :].sum(dim=1, keepdim=True) if student_prob.shape[1] > 1 else student_prob[:, :1]
    boundary = _sobel_map(fg)
    if uncertainty is None:
        entropy = -torch.sum(student_prob * torch.log(student_prob.clamp_min(1e-6)), dim=1, keepdim=True)
        uncertainty = entropy / torch.log(student_prob.new_tensor(float(student_prob.shape[1]))).clamp_min(1e-6)
    if disagreement is None:
        disagreement = uncertainty
    if teacher_prob is not None:
        teacher_conf = teacher_prob.max(dim=1, keepdim=True).values
    else:
        teacher_conf = student_prob.max(dim=1, keepdim=True).values

    masks = [
        fg.clamp(0.0, 1.0),
        boundary.clamp(0.0, 1.0),
        uncertainty.clamp(0.0, 1.0),
        disagreement.clamp(0.0, 1.0),
        teacher_conf.clamp(0.0, 1.0),
    ]
    masks = [_resize_map(mask, feature_size) for mask in masks]
    masks = torch.cat(masks, dim=1)
    if detach:
        masks = masks.detach()
    return masks


class RegionBoundaryPrototypeGraph(nn.Module):
    """Region/boundary prototype graph that outputs residual correction logits.

    Inputs:
        features: [B, C, Hf, Wf]
        student_prob: [B, K, H, W]
    Outputs:
        residual_logits: [B, K, H, W]
        adjacency: [B, N, N]
        diagnostics: scalar logging fields
    """

    def __init__(
        self,
        num_classes,
        feature_dim,
        num_prototypes=5,
        temperature=0.5,
        residual_init_zero=True,
        detach_region_mask=True,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.feature_dim = feature_dim
        self.num_prototypes = num_prototypes
        self.temperature = temperature
        self.detach_region_mask = detach_region_mask
        self.prototype_proj = nn.Linear(feature_dim, feature_dim, bias=False)
        self.residual_head = nn.Sequential(
            nn.Conv2d(feature_dim * 2, feature_dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(feature_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(feature_dim, num_classes, kernel_size=1),
        )
        if residual_init_zero:
            final = self.residual_head[-1]
            nn.init.zeros_(final.weight)
            if final.bias is not None:
                nn.init.zeros_(final.bias)
        self.last_adjacency = None

    def _weighted_pool(self, features, masks):
        denom = masks.flatten(2).sum(dim=-1).clamp_min(1e-6)
        prototypes = torch.einsum("bchw,bnhw->bnc", features, masks) / denom.unsqueeze(-1)
        return prototypes

    def compute_dynamic_adjacency(self, prototypes):
        proto = F.normalize(self.prototype_proj(prototypes), dim=-1)
        sim = torch.einsum("bnc,bmc->bnm", proto, proto) / max(float(self.temperature), 1e-6)
        adjacency = torch.softmax(sim, dim=-1)
        self.last_adjacency = adjacency.detach()
        return adjacency

    def forward(self, features, student_prob, teacher_prob=None, uncertainty=None, disagreement=None, out_size=None):
        masks = build_region_masks(
            student_prob=student_prob,
            teacher_prob=teacher_prob,
            uncertainty=uncertainty,
            disagreement=disagreement,
            feature_size=features.shape[-2:],
            detach=self.detach_region_mask,
        )
        if masks.shape[1] > self.num_prototypes:
            masks = masks[:, : self.num_prototypes]
        prototypes = self._weighted_pool(features, masks)
        adjacency = self.compute_dynamic_adjacency(prototypes)
        messages = torch.bmm(adjacency, prototypes)
        context = torch.einsum("bnc,bnhw->bchw", messages, masks)
        context = context / masks.sum(dim=1, keepdim=True).clamp_min(1e-6)
        residual_logits = self.residual_head(torch.cat([features, context], dim=1))
        if out_size is not None and residual_logits.shape[-2:] != out_size:
            residual_logits = F.interpolate(residual_logits, size=out_size, mode="bilinear", align_corners=False)
        logs = compute_graph_stats(adjacency)
        logs["residual_logits_abs_mean"] = float(residual_logits.detach().abs().mean().cpu())
        return residual_logits, adjacency, logs
