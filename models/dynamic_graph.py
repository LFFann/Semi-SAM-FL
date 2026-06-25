import torch
import torch.nn as nn
import torch.nn.functional as F


class DynamicPrototypeGraph(nn.Module):
    """Dynamic class-prototype graph guided by SAM structural consistency."""

    def __init__(self, num_classes, feature_dim, alpha=0.5, momentum=0.9, affinity_size=16, scale=12.0):
        super().__init__()
        self.num_classes = num_classes
        self.feature_dim = feature_dim
        self.alpha = alpha
        self.momentum = momentum
        self.affinity_size = affinity_size
        self.scale = scale
        self.propagation = nn.Linear(feature_dim, feature_dim, bias=False)
        self.register_buffer("prototypes", torch.zeros(num_classes, feature_dim))
        self.register_buffer("prototype_seen", torch.zeros(num_classes))
        self.register_buffer("R_prior", torch.eye(num_classes))
        self.last_adjacency = None

    @torch.no_grad()
    def update_prototypes(self, features, labels, weights=None):
        labels = labels.long()
        if labels.shape[-2:] != features.shape[-2:]:
            labels = F.interpolate(labels.unsqueeze(1).float(), size=features.shape[-2:], mode="nearest").squeeze(1).long()
        if weights is not None and weights.shape[-2:] != features.shape[-2:]:
            weights = F.interpolate(weights, size=features.shape[-2:], mode="bilinear", align_corners=False)
        flat_features = features.permute(0, 2, 3, 1).reshape(-1, self.feature_dim)
        flat_labels = labels.reshape(-1)
        flat_weights = None if weights is None else weights.reshape(-1).clamp_min(0.0)

        for class_idx in range(self.num_classes):
            mask = flat_labels == class_idx
            if not torch.any(mask):
                continue
            selected = flat_features[mask]
            if flat_weights is not None:
                selected_w = flat_weights[mask].unsqueeze(1)
                denom = selected_w.sum().clamp_min(1e-6)
                mu = (selected * selected_w).sum(dim=0) / denom
            else:
                mu = selected.mean(dim=0)
            if self.prototype_seen[class_idx] > 0:
                self.prototypes[class_idx] = self.momentum * self.prototypes[class_idx] + (1.0 - self.momentum) * mu
            else:
                self.prototypes[class_idx] = mu
                self.prototype_seen[class_idx] = 1.0

    @torch.no_grad()
    def build_adjacency(self, labels, R_a):
        labels_small = F.interpolate(labels.unsqueeze(1).float(), size=(self.affinity_size, self.affinity_size),
                                     mode="nearest").squeeze(1).long()
        labels_flat = labels_small.reshape(labels_small.shape[0], -1)
        adjacency = R_a.new_zeros(self.num_classes, self.num_classes)
        counts = R_a.new_zeros(self.num_classes, self.num_classes)
        for b in range(labels_flat.shape[0]):
            label_b = labels_flat[b]
            affinity_b = R_a[b]
            for i in range(self.num_classes):
                row_mask = label_b == i
                if not torch.any(row_mask):
                    continue
                for j in range(self.num_classes):
                    col_mask = label_b == j
                    if not torch.any(col_mask):
                        continue
                    adjacency[i, j] += affinity_b[row_mask][:, col_mask].mean()
                    counts[i, j] += 1.0
        adjacency = adjacency / counts.clamp_min(1.0)
        adjacency = torch.where(counts > 0, adjacency, self.R_prior.to(R_a.device))
        adjacency = self.alpha * self.R_prior.to(R_a.device) + (1.0 - self.alpha) * adjacency
        adjacency = adjacency / adjacency.sum(dim=1, keepdim=True).clamp_min(1e-6)
        self.last_adjacency = adjacency.detach()
        return adjacency

    def propagated_prototypes(self, adjacency):
        source = self.propagation(self.prototypes)
        return self.prototypes + adjacency @ source

    def forward(self, features, adjacency=None):
        if adjacency is None:
            adjacency = self.R_prior.to(features.device)
        mu = self.propagated_prototypes(adjacency)
        mu = F.normalize(mu, dim=1)
        feat = F.normalize(features, dim=1)
        logits = torch.einsum("bdhw,cd->bchw", feat, mu) * self.scale
        return logits

