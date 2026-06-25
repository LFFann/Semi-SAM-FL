import math
from dataclasses import dataclass
from typing import Dict, List, Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F


@dataclass
class PERDOutput:
    kd_weight: torch.Tensor
    trust_map: torch.Tensor
    boundary_tube: torch.Tensor
    pc_map: torch.Tensor
    ed_map: torch.Tensor
    ji_map: torch.Tensor
    valid_map: torch.Tensor
    saturated_map: torch.Tensor
    stats: Dict[str, float]
    y0_logits: torch.Tensor
    orig_curve_logits: torch.Tensor
    atten_curve_logits: torch.Tensor


def parse_delta_levels(raw_value, delta_pixels=2) -> List[int]:
    if isinstance(raw_value, str):
        parts = [item.strip() for item in raw_value.split(",") if item.strip()]
        if parts:
            return [int(float(item)) for item in parts]
    if isinstance(raw_value, (list, tuple)):
        return [int(float(item)) for item in raw_value]
    d = int(delta_pixels)
    return [-2 * d, -d, 0, d, 2 * d]


def normalized_entropy_map(prob: torch.Tensor) -> torch.Tensor:
    entropy = -torch.sum(prob * torch.log(prob.clamp_min(1e-6)), dim=1, keepdim=True)
    return entropy / math.log(float(prob.shape[1]))


def foreground_boundary_tube(mask: torch.Tensor, radius: int) -> torch.Tensor:
    if mask.dim() == 3:
        mask = mask.unsqueeze(1)
    mask = mask.float()
    kernel = 2 * int(radius) + 1
    dilated = F.max_pool2d(mask, kernel_size=kernel, stride=1, padding=radius)
    eroded = 1.0 - F.max_pool2d(1.0 - mask, kernel_size=kernel, stride=1, padding=radius)
    return (dilated - eroded).clamp(0.0, 1.0)


def interior_band(mask: torch.Tensor, radius: int) -> torch.Tensor:
    if mask.dim() == 3:
        mask = mask.unsqueeze(1)
    mask = mask.float()
    kernel = 2 * int(radius) + 1
    eroded = 1.0 - F.max_pool2d(1.0 - mask, kernel_size=kernel, stride=1, padding=radius)
    return eroded.clamp(0.0, 1.0)


def signed_distance_transform(mask: torch.Tensor) -> torch.Tensor:
    if mask.dim() == 4:
        mask = mask[:, 0]
    device = mask.device
    mask_np = mask.detach().float().cpu().numpy()
    sdf_list = []
    for item in mask_np:
        binary = (item > 0.5).astype(np.uint8)
        if binary.max() == 0:
            sdf = -cv2.distanceTransform(1 - binary, cv2.DIST_L2, 5)
        elif binary.min() == 1:
            sdf = cv2.distanceTransform(binary, cv2.DIST_L2, 5)
        else:
            dist_in = cv2.distanceTransform(binary, cv2.DIST_L2, 5)
            dist_out = cv2.distanceTransform(1 - binary, cv2.DIST_L2, 5)
            sdf = dist_in - dist_out
        sdf_list.append(sdf.astype(np.float32))
    return torch.from_numpy(np.stack(sdf_list, axis=0)).to(device=device)


def mask_prompt_logits_from_sdf(sdf: torch.Tensor, delta: float, inside_logit: float) -> torch.Tensor:
    prompt_mask = (sdf >= float(delta)).float().unsqueeze(1)
    return prompt_mask * inside_logit + (1.0 - prompt_mask) * (-inside_logit)


def logits_to_hard_label(logits: torch.Tensor) -> torch.Tensor:
    if logits.shape[1] == 1:
        return (torch.sigmoid(logits[:, 0]) > 0.5).long()
    return torch.argmax(logits, dim=1).long()


def class_sdf_stack(logits: torch.Tensor, num_classes: int) -> torch.Tensor:
    labels = logits_to_hard_label(logits)
    sdf_by_class = []
    for class_idx in range(num_classes):
        sdf_by_class.append(signed_distance_transform((labels == class_idx).float()))
    return torch.stack(sdf_by_class, dim=1)


def safe_minmax_image(x: torch.Tensor) -> torch.Tensor:
    return x.clamp(0.0, 1.0)


class PERDProbe:
    def __init__(self, args):
        self.args = args
        self.num_classes = int(args.num_classes)
        self.delta_levels = parse_delta_levels(
            getattr(args, "perd_delta_levels", "-4,-2,0,2,4"),
            getattr(args, "perd_delta_pixels", 2),
        )
        if 0 not in self.delta_levels:
            self.delta_levels = sorted(self.delta_levels + [0])
        self.zero_delta_index = self.delta_levels.index(0)
        self.tube_radius = int(getattr(args, "perd_tube_radius", 5))
        self.atten_kernel = self._odd_kernel(int(getattr(args, "perd_atten_kernel", 9)))
        self.atten_strength = float(getattr(args, "perd_atten_strength", 0.25))
        self.beta_ed = float(getattr(args, "perd_beta_ed", 6.0))
        self.beta_pc = float(getattr(args, "perd_beta_pc", 2.0))
        self.beta_j = float(getattr(args, "perd_beta_j", 1.0))
        self.ed_mid = float(getattr(args, "perd_ed_mid", 0.05))
        self.min_area_change = float(getattr(args, "perd_min_area_change", 0.005))
        self.max_area_change = float(getattr(args, "perd_max_area_change", 0.40))
        self.saturation_iou = float(getattr(args, "perd_saturation_iou", 0.995))
        self.prompt_logit = float(getattr(args, "perd_prompt_logit", 6.0))
        self.attenuation_mode = getattr(args, "perd_attenuation_mode", "boundary")
        self.disable_ed = bool(int(getattr(args, "perd_disable_ed", 0)))
        self.disable_pc = bool(int(getattr(args, "perd_disable_pc", 0)))
        self.no_attenuation = bool(int(getattr(args, "perd_no_attenuation", 0)))
        self.baseline = getattr(args, "perd_baseline", "perd")

    @staticmethod
    def _odd_kernel(value: int) -> int:
        value = max(value, 3)
        return value if value % 2 == 1 else value + 1

    def build_prompt_family(self, fusion_logits: torch.Tensor) -> List[torch.Tensor]:
        labels = logits_to_hard_label(fusion_logits.detach())
        prompts = []
        for delta in self.delta_levels:
            class_prompts = []
            for class_idx in range(self.num_classes):
                sdf = signed_distance_transform((labels == class_idx).float())
                class_prompts.append(mask_prompt_logits_from_sdf(sdf, delta, self.prompt_logit))
            prompts.append(torch.cat(class_prompts, dim=1))
        return prompts

    def run_sam_with_prompts(
        self,
        sam_model,
        image_embeddings: torch.Tensor,
        boxes_embedding: torch.Tensor,
        prompt_logits: torch.Tensor,
    ) -> torch.Tensor:
        b, _, h, w = prompt_logits.shape
        low_res_masks_all = torch.empty(
            (b, 0, int(h / 4), int(w / 4)),
            device=prompt_logits.device,
        )
        for class_idx in range(self.num_classes):
            sparse_embeddings, dense_embeddings = sam_model.prompt_encoder(
                points=None,
                boxes=boxes_embedding[class_idx],
                masks=F.interpolate(
                    prompt_logits[:, class_idx:class_idx + 1],
                    size=(64, 64),
                    mode="bilinear",
                    align_corners=False,
                ),
            )
            low_res_masks, _ = sam_model.mask_decoder(
                image_embeddings=image_embeddings,
                image_pe=sam_model.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse_embeddings,
                dense_prompt_embeddings=dense_embeddings,
                multimask_output=self.args.multimask,
            )
            low_res_masks_all = torch.cat((low_res_masks_all, low_res_masks), dim=1)
        return F.interpolate(low_res_masks_all, size=(h, w), mode="bilinear", align_corners=False)

    def run_curve(
        self,
        sam_model,
        image_embeddings: torch.Tensor,
        boxes_embedding: torch.Tensor,
        prompts: List[torch.Tensor],
    ) -> torch.Tensor:
        logits = []
        for prompt_logits in prompts:
            logits.append(self.run_sam_with_prompts(sam_model, image_embeddings, boxes_embedding, prompt_logits))
        return torch.stack(logits, dim=0)

    def attenuation_mask(self, boundary_tube: torch.Tensor, y0_logits: torch.Tensor) -> torch.Tensor:
        if self.no_attenuation or self.attenuation_mode == "none":
            return torch.zeros_like(boundary_tube)
        if self.attenuation_mode == "boundary":
            return boundary_tube
        labels = logits_to_hard_label(y0_logits)
        fg_mask = (labels > 0).float().unsqueeze(1)
        if self.attenuation_mode == "interior":
            return interior_band(fg_mask, self.tube_radius).clamp(0.0, 1.0)
        if self.attenuation_mode == "random":
            shift_h = max(int(y0_logits.shape[-2] // 4), 1)
            shift_w = max(int(y0_logits.shape[-1] // 5), 1)
            return torch.roll(boundary_tube, shifts=(shift_h, shift_w), dims=(-2, -1))
        return boundary_tube

    def attenuate_image(self, image: torch.Tensor, boundary_tube: torch.Tensor, y0_logits: torch.Tensor) -> torch.Tensor:
        mask = self.attenuation_mask(boundary_tube, y0_logits)
        if mask.max() <= 0:
            return image
        pad = self.atten_kernel // 2
        smooth = F.avg_pool2d(image, kernel_size=self.atten_kernel, stride=1, padding=pad)
        local_mean = F.avg_pool2d(image * mask, kernel_size=self.atten_kernel, stride=1, padding=pad)
        norm = F.avg_pool2d(mask, kernel_size=self.atten_kernel, stride=1, padding=pad).clamp_min(1e-6)
        local_mean = local_mean / norm
        attenuated = 0.5 * smooth + 0.5 * local_mean
        blend = (mask * self.atten_strength).clamp(0.0, 1.0)
        return safe_minmax_image(image * (1.0 - blend) + attenuated * blend)

    def response_maps(self, curve_logits: torch.Tensor) -> torch.Tensor:
        base_sdf = class_sdf_stack(curve_logits[self.zero_delta_index], self.num_classes)
        responses = []
        scale = max(max(abs(float(delta)) for delta in self.delta_levels), 1.0)
        for logits in curve_logits:
            sdf = class_sdf_stack(logits, self.num_classes)
            responses.append((sdf - base_sdf) / scale)
        return torch.stack(responses, dim=0)

    def prompt_area_validity(self, prompts: List[torch.Tensor]) -> torch.Tensor:
        base = (prompts[self.zero_delta_index] > 0).float()
        valid_parts = []
        for idx, prompt in enumerate(prompts):
            if idx == self.zero_delta_index:
                continue
            area_change = torch.abs((prompt > 0).float().mean(dim=(2, 3), keepdim=True) - base.mean(dim=(2, 3), keepdim=True))
            valid_parts.append(((area_change >= self.min_area_change) & (area_change <= self.max_area_change)).float())
        if not valid_parts:
            return torch.ones_like(base[:, :1])
        valid = torch.stack(valid_parts, dim=0).max(dim=0).values
        return valid.mean(dim=1, keepdim=True)

    def saturation_map(self, curve_logits: torch.Tensor) -> torch.Tensor:
        base_label = logits_to_hard_label(curve_logits[self.zero_delta_index])
        saturated_parts = []
        for idx in range(curve_logits.shape[0]):
            if idx == self.zero_delta_index:
                continue
            label = logits_to_hard_label(curve_logits[idx])
            ious = []
            for class_idx in range(self.num_classes):
                base = base_label == class_idx
                pred = label == class_idx
                inter = (base & pred).float().sum(dim=(1, 2))
                union = (base | pred).float().sum(dim=(1, 2)).clamp_min(1.0)
                ious.append((inter / union).view(-1, 1, 1, 1))
            saturated_parts.append(torch.stack(ious, dim=1).mean(dim=1))
        if not saturated_parts:
            return torch.zeros(base_label.shape[0], 1, base_label.shape[1], base_label.shape[2], device=curve_logits.device)
        sat_sample = (torch.stack(saturated_parts, dim=0).min(dim=0).values >= self.saturation_iou).float()
        return sat_sample.expand(-1, -1, base_label.shape[1], base_label.shape[2])

    def compute_trust(
        self,
        prompts: List[torch.Tensor],
        orig_curve_logits: torch.Tensor,
        atten_curve_logits: torch.Tensor,
        boundary_tube: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        orig_resp = self.response_maps(orig_curve_logits)
        atten_resp = self.response_maps(atten_curve_logits)
        non_bg = slice(1, None) if self.num_classes > 1 else slice(0, 1)

        deltas = torch.tensor(self.delta_levels, device=orig_curve_logits.device, dtype=torch.float32).view(-1, 1, 1, 1, 1)
        delta_abs = deltas.abs().clamp_min(1.0)
        pc_class = (orig_resp.abs() / delta_abs).mean(dim=0)[:, non_bg].mean(dim=1, keepdim=True)
        ed_class = (orig_resp - atten_resp).abs().mean(dim=0)[:, non_bg].mean(dim=1, keepdim=True)
        if orig_resp.shape[0] > 1:
            ji_class = (orig_resp[1:] - orig_resp[:-1]).abs().mean(dim=0)[:, non_bg].mean(dim=1, keepdim=True)
        else:
            ji_class = torch.zeros_like(pc_class)

        prompt_valid = self.prompt_area_validity(prompts)
        saturated = self.saturation_map(orig_curve_logits)
        valid = (prompt_valid > 0).float() * (1.0 - saturated)
        valid = valid.clamp(0.0, 1.0)

        pc_term = torch.ones_like(pc_class) if self.disable_pc else torch.exp(-self.beta_pc * pc_class)
        ed_term = torch.ones_like(ed_class) if self.disable_ed else torch.sigmoid(self.beta_ed * (ed_class - self.ed_mid))
        ji_term = torch.exp(-self.beta_j * ji_class)
        trust = (pc_term * ed_term * ji_term * valid).clamp(0.0, 1.0)
        trust = trust * boundary_tube
        return trust, pc_class, ed_class, ji_class, valid, saturated

    def build_kd_weight(self, trust: torch.Tensor, boundary_tube: torch.Tensor, batch_size: int, labeled_bs: int) -> torch.Tensor:
        weight = torch.ones_like(trust)
        if batch_size <= labeled_bs:
            return weight
        unlabeled = slice(labeled_bs, batch_size)
        boundary = boundary_tube[unlabeled]
        if self.baseline == "prompt_ensemble":
            weight[unlabeled] = 1.0
            return weight
        # Interior keeps ordinary KD. Boundary is gated by PERD trust.
        weight[unlabeled] = (1.0 - boundary) + boundary * trust[unlabeled]
        return weight.clamp_min(0.0).detach()

    def probe(
        self,
        sam_model,
        image: torch.Tensor,
        image_embeddings: torch.Tensor,
        boxes_embedding: torch.Tensor,
        fusion_logits: torch.Tensor,
        labeled_bs: int,
    ) -> PERDOutput:
        prompts = self.build_prompt_family(fusion_logits)
        with torch.no_grad():
            orig_curve_logits = self.run_curve(sam_model, image_embeddings.detach(), boxes_embedding, prompts)
            y0_logits = orig_curve_logits[self.zero_delta_index]
            y0_label = logits_to_hard_label(y0_logits)
            fg_mask = (y0_label > 0).float().unsqueeze(1)
            boundary_tube = foreground_boundary_tube(fg_mask, self.tube_radius)
            atten_image = self.attenuate_image(image.detach(), boundary_tube, y0_logits)
            atten_embeddings = sam_model.image_encoder(atten_image)
            atten_curve_logits = self.run_curve(sam_model, atten_embeddings, boxes_embedding, prompts)
            trust, pc_map, ed_map, ji_map, valid_map, saturated_map = self.compute_trust(
                prompts,
                orig_curve_logits,
                atten_curve_logits,
                boundary_tube,
            )
            kd_weight = self.build_kd_weight(trust, boundary_tube, image.shape[0], labeled_bs)

        stats = {
            "perd_trust_mean": float(trust.mean().item()),
            "perd_pc_mean": float((pc_map * boundary_tube).sum().item() / boundary_tube.sum().clamp_min(1.0).item()),
            "perd_ed_mean": float((ed_map * boundary_tube).sum().item() / boundary_tube.sum().clamp_min(1.0).item()),
            "perd_ji_mean": float((ji_map * boundary_tube).sum().item() / boundary_tube.sum().clamp_min(1.0).item()),
            "perd_valid_ratio": float(valid_map.mean().item()),
            "perd_boundary_ratio": float(boundary_tube.mean().item()),
            "perd_saturated_ratio": float(saturated_map.mean().item()),
        }
        return PERDOutput(
            kd_weight=kd_weight,
            trust_map=trust.detach(),
            boundary_tube=boundary_tube.detach(),
            pc_map=pc_map.detach(),
            ed_map=ed_map.detach(),
            ji_map=ji_map.detach(),
            valid_map=valid_map.detach(),
            saturated_map=saturated_map.detach(),
            stats=stats,
            y0_logits=y0_logits.detach(),
            orig_curve_logits=orig_curve_logits.detach(),
            atten_curve_logits=atten_curve_logits.detach(),
        )
