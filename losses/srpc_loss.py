import torch
import torch.nn as nn
import torch.nn.functional as F


class SRPCLoss(nn.Module):
    """Structure-reliability weighted posterior pseudo-label learning."""

    def __init__(self, graph_lambda=0.5, eps=1e-6):
        super().__init__()
        self.graph_lambda = graph_lambda
        self.eps = eps
        self.last_weight_mean = None

    def forward(self, student_logits, teacher_logits, graph_logits, R_b, R_u):
        student_log_prob = F.log_softmax(student_logits, dim=1)
        teacher_prob = F.softmax(teacher_logits.detach(), dim=1).clamp_min(self.eps)
        graph_prob = F.softmax(graph_logits.detach(), dim=1).clamp_min(self.eps)

        pseudo_logits = teacher_prob.log() + self.graph_lambda * graph_prob.log()
        pseudo_prob = F.softmax(pseudo_logits, dim=1).detach()

        weight = (R_b * (1.0 - R_u)).detach().clamp(0.0, 1.0)
        if weight.shape[-2:] != student_logits.shape[-2:]:
            weight = F.interpolate(weight, size=student_logits.shape[-2:], mode="bilinear", align_corners=False)

        ce_map = -(pseudo_prob * student_log_prob).sum(dim=1, keepdim=True)
        loss = (weight * ce_map).sum() / weight.sum().clamp_min(self.eps)
        self.last_weight_mean = weight.mean().detach()
        return loss, pseudo_prob, weight

