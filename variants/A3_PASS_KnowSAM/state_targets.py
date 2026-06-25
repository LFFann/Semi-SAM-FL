import torch
import torch.nn.functional as F


class StateTargetGenerator:
    """Build mask-derived acoustic-anatomical state targets.

    The targets are derived from existing segmentation masks and require no
    extra annotation. They are intentionally compact: a geometry vector and a
    low-resolution dense state map.
    """

    def __init__(self, state_size=64, eps=1e-6):
        self.state_size = state_size
        self.eps = eps

    def __call__(self, label):
        if label.dim() == 4:
            label = label.squeeze(1)
        mask = (label > 0.5).float()
        geometry = self._geometry(mask)
        dense = self._dense_state(mask)
        return {"geometry": geometry, "dense": dense}

    def _geometry(self, mask):
        device = mask.device
        b, h, w = mask.shape
        y_coords = torch.linspace(0.0, 1.0, h, device=device).view(1, h, 1)
        x_coords = torch.linspace(0.0, 1.0, w, device=device).view(1, 1, w)

        area_pixels = mask.sum(dim=(1, 2)).clamp_min(self.eps)
        area_ratio = mask.mean(dim=(1, 2))
        cx = (mask * x_coords).sum(dim=(1, 2)) / area_pixels
        cy = (mask * y_coords).sum(dim=(1, 2)) / area_pixels
        var_x = (mask * (x_coords - cx.view(b, 1, 1)) ** 2).sum(dim=(1, 2)) / area_pixels
        var_y = (mask * (y_coords - cy.view(b, 1, 1)) ** 2).sum(dim=(1, 2)) / area_pixels
        std_x = torch.sqrt(var_x.clamp_min(0.0) + self.eps)
        std_y = torch.sqrt(var_y.clamp_min(0.0) + self.eps)
        elongation = torch.maximum(std_x, std_y) / torch.minimum(std_x, std_y).clamp_min(1e-3)
        elongation = (elongation / 8.0).clamp(0.0, 1.0)

        has_fg = (mask.sum(dim=(1, 2)) > 0).float()
        geometry = torch.stack([area_ratio, cx, cy, std_x, std_y, elongation], dim=1)
        default = torch.tensor([0.0, 0.5, 0.5, 0.0, 0.0, 0.0], device=device).view(1, 6)
        return geometry * has_fg.view(b, 1) + default * (1.0 - has_fg).view(b, 1)

    def _dense_state(self, mask):
        mask_4d = mask.unsqueeze(1)
        low_mask = F.interpolate(mask_4d, size=(self.state_size, self.state_size), mode="nearest")
        dilated = F.max_pool2d(mask_4d, kernel_size=3, stride=1, padding=1)
        eroded = -F.max_pool2d(-mask_4d, kernel_size=3, stride=1, padding=1)
        boundary = (dilated - eroded).clamp(0.0, 1.0)
        low_boundary = F.interpolate(boundary, size=(self.state_size, self.state_size), mode="nearest")
        return torch.cat([low_mask, low_boundary], dim=1)


def perturb_ultrasound_tensor(image, noise_std=0.03, gain_range=0.12):
    """Small tensor-level acoustic perturbation for state consistency."""
    if image.shape[0] == 0:
        return image
    b = image.shape[0]
    device = image.device
    gain = 1.0 + (torch.rand(b, 1, 1, 1, device=device) * 2.0 - 1.0) * gain_range
    bias = (torch.rand(b, 1, 1, 1, device=device) * 2.0 - 1.0) * gain_range
    noise = torch.randn_like(image) * noise_std
    return (image * gain + bias + noise).clamp(0.0, 1.0)
