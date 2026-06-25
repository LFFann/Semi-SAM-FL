import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "..", ".."))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)
if REPO_ROOT not in sys.path:
    sys.path.insert(1, REPO_ROOT)
from Model.sam.build_sam import sam_model_registry
import torch.optim as optim
from utils.losses_a3_rcp import dice_loss, loss_diff1, loss_diff2, KDLoss, DiceLoss
from utils.losses_a3_rcp import WeightedKDLoss, BoundaryLoss, soft_area_prior_loss
import logging
from utils.utils import dice_coef

import numpy as np

from Model.model import KnowSAM
from prediction_ACDC import test_single_volume

ce_loss = torch.nn.CrossEntropyLoss()

GPUdevice = torch.device('cuda', 0)
pos_weight = torch.ones([1]).cuda(device=GPUdevice)*2
criterion_G = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)


class Trainer(nn.Module):
    def __init__(self, args):
        super(Trainer, self).__init__()
        self.args = args
        self.criterion_mse = nn.MSELoss()
        self.KDLoss = KDLoss(T=10)
        self.weighted_kd_loss = WeightedKDLoss(T=10)
        self.boundary_loss = BoundaryLoss(args.num_classes)
        self.dice_loss = DiceLoss(args.num_classes)
        # 修改时间：2026-04-23
        # 修改功能：三项 A3 半监督创新的超参数，使用 getattr 保持旧实验命令兼容。
        self.uckd_alpha = getattr(args, "uckd_alpha", 2.0)
        self.uckd_min_weight = getattr(args, "uckd_min_weight", 0.15)
        self.qapl_min_weight = getattr(args, "qapl_min_weight", 0.20)
        self.rcp_alpha = getattr(args, "rcp_alpha", 2.0)
        self.rcp_min_weight = getattr(args, "rcp_min_weight", 0.10)
        self.rcp_sharpen = getattr(args, "rcp_sharpen", 1.5)
        self.sap_boundary_weight = getattr(args, "sap_boundary_weight", 0.10)
        self.sap_shape_weight = getattr(args, "sap_shape_weight", 0.05)
        self.sap_area_lower = getattr(args, "sap_area_lower", 0.001)
        self.sap_area_upper = getattr(args, "sap_area_upper", 0.08)

        self.sam_model = sam_model_registry[args.model_type](args).to(args.device).train()
        self.SGDL = KnowSAM(args).cuda().train()

        self.optimizer_sam = optim.Adam(self.sam_model.parameters(), lr=args.lr)
        self.optimizer_SGDL = torch.optim.SGD(self.SGDL.parameters(), lr=args.UNet_lr, momentum=0.9,
                                              weight_decay=0.0001)

        self.best_performance_sam = 0.0
        self.best_performance_SGDL = 0.0

        for n, value in self.sam_model.named_parameters():
            if "Adapter" in n:
                value.requires_grad = True
            elif "super_prompt" in n:
                value.requires_grad = True
            else:
                value.requires_grad = False

    def sigmoid_rampup(self, current, rampup_length):
        """Exponential rampup from https://arxiv.org/abs/1610.02242"""
        if rampup_length == 0:
            return 1.0
        else:
            current = np.clip(current, 0.0, rampup_length)
            phase = 1.0 - current / rampup_length
            return float(np.exp(-5.0 * phase * phase))

    def entropy_loss(self, p, C=2):
        # p N*C*W*H*D
        y1 = -1 * torch.sum(p * torch.log(p + 1e-6), dim=1) / \
             torch.tensor(np.log(C)).cuda()
        ent = torch.mean(y1)
        return ent

    def get_entropy_map(self, p):
        ent_map = -1 * torch.sum(p * torch.log(p + 1e-6), dim=1, keepdim=True)
        return ent_map

    def get_current_consistency_weight(self, epoch):
        # Consistency ramp-up from https://arxiv.org/abs/1610.02242
        return self.args.consistency * self.sigmoid_rampup(epoch, self.args.consistency_rampup)

    def normalized_entropy_map(self, prob):
        # 修改时间：2026-04-23
        # 修改功能：将多类熵归一化到 0-1，作为像素可靠性估计的基础。
        entropy = -1 * torch.sum(prob * torch.log(prob + 1e-6), dim=1, keepdim=True)
        return entropy / torch.log(torch.tensor(float(prob.shape[1]), device=prob.device))

    def build_reliability_map(self, pred_sam_soft, fusion_map_soft, pred_UNet_soft, pred_VNet_soft):
        # 修改时间：2026-04-23
        # 修改功能：U-CKD 像素级可靠性图，融合 SAM 熵、融合头熵和双分支分歧。
        sam_entropy = self.normalized_entropy_map(pred_sam_soft)
        fusion_entropy = self.normalized_entropy_map(fusion_map_soft)
        branch_gap = torch.abs(pred_UNet_soft[:, 1:2] - pred_VNet_soft[:, 1:2])
        uncertainty = (sam_entropy + fusion_entropy + branch_gap) / 3.0
        reliability = torch.exp(-self.uckd_alpha * uncertainty)
        return reliability.clamp(min=self.uckd_min_weight, max=1.0).detach()

    def build_a3_consensus_prompt(self, pred_UNet_soft, pred_VNet_soft, fusion_map_soft):
        # 修改时间：2026-04-23
        # 修改功能：RCP 框架优化，将 KnowSAM 原始 raw fusion prompt 改为可靠性校准的三分支共识提示。
        # 设计动机：SAM 的提示质量决定后续教师质量。低可靠区域不再强行提示为某类，而是退回均匀先验。
        consensus_prob = (pred_UNet_soft + pred_VNet_soft + fusion_map_soft) / 3.0
        consensus_prob = consensus_prob.clamp(1e-6, 1.0)
        sharpened = torch.pow(consensus_prob, self.rcp_sharpen)
        consensus_prob = sharpened / sharpened.sum(dim=1, keepdim=True).clamp_min(1e-6)

        consensus_entropy = self.normalized_entropy_map(consensus_prob)
        branch_gap = torch.abs(pred_UNet_soft[:, 1:2] - pred_VNet_soft[:, 1:2])
        prompt_uncertainty = (consensus_entropy + branch_gap) / 2.0
        prompt_reliability = torch.exp(-self.rcp_alpha * prompt_uncertainty)
        prompt_reliability = prompt_reliability.clamp(min=self.rcp_min_weight, max=1.0)

        uniform_prior = torch.full_like(consensus_prob, 1.0 / self.args.num_classes)
        calibrated_prob = prompt_reliability * consensus_prob + (1.0 - prompt_reliability) * uniform_prior
        calibrated_prob = calibrated_prob.clamp(1e-6, 1.0 - 1e-6)
        prompt_logits = torch.log(calibrated_prob / (1.0 - calibrated_prob))
        return prompt_logits.detach(), prompt_reliability.detach()

    def compute_a3_quality_weight(self, reliability_map, teacher_prob, prompt_reliability=None):
        # 修改时间：2026-04-23
        # 修改功能：QAPL 样本级伪标签质量评分，低质量无标签样本不会和高质量样本等权训练。
        reliability_score = reliability_map.mean(dim=(1, 2, 3))
        if prompt_reliability is not None:
            reliability_score = 0.5 * reliability_score + 0.5 * prompt_reliability.mean(dim=(1, 2, 3))
        fg_ratio = teacher_prob[:, 1].mean(dim=(1, 2))
        lower_gap = F.relu(self.sap_area_lower - fg_ratio)
        upper_gap = F.relu(fg_ratio - self.sap_area_upper)
        shape_score = torch.exp(-25.0 * (lower_gap + upper_gap))
        quality = reliability_score * shape_score
        return quality.clamp(min=self.qapl_min_weight, max=1.0).detach()

    def weighted_segmentation_loss(self, logits, prob, target, sample_weight):
        # 修改时间：2026-04-23
        # 修改功能：QAPL 的样本加权 CE+Dice，用于混合伪标签监督。
        if logits.shape[0] == 0:
            return logits.sum() * 0.0
        ce_per_sample = F.cross_entropy(logits, target.long(), reduction='none').mean(dim=(1, 2))
        target_onehot = F.one_hot(target.long(), num_classes=self.args.num_classes).permute(0, 3, 1, 2).float()
        target_onehot = target_onehot.to(prob.device)
        intersect = torch.sum(prob * target_onehot, dim=(2, 3))
        y_sum = torch.sum(target_onehot * target_onehot, dim=(2, 3))
        z_sum = torch.sum(prob * prob, dim=(2, 3))
        dice_per_class = 1.0 - (2 * intersect + 1e-6) / (y_sum + z_sum + 1e-6)
        dice_per_sample = dice_per_class.mean(dim=1)
        weight = sample_weight.to(logits.device).float().clamp_min(1e-6)
        return ((ce_per_sample + dice_per_sample) * weight).sum() / weight.sum()

    def mix_up(self, fusion_map_soft, volume_batch, pseudo_label, labeled_label, consistency_weight, patch_size=4,
               top_k=5, unlabeled_quality_weight=None):
        labeled_volume_batch = volume_batch[:self.args.labeled_bs]
        unlabeled_volume_batch = volume_batch[self.args.labeled_bs:]
        unlabel_pseudo_label = torch.argmax(pseudo_label.clone(), dim=1)

        paired_bs = min(
            labeled_volume_batch.shape[0],
            unlabeled_volume_batch.shape[0],
            labeled_label.shape[0],
            unlabel_pseudo_label.shape[0],
        )
        if paired_bs <= 0:
            zero_loss = fusion_map_soft.sum() * 0.0
            return zero_loss, zero_loss, zero_loss

        labeled_volume_batch = labeled_volume_batch[:paired_bs]
        unlabeled_volume_batch = unlabeled_volume_batch[:paired_bs]
        labeled_label = labeled_label[:paired_bs]
        unlabel_pseudo_label = unlabel_pseudo_label[:paired_bs]
        if unlabeled_quality_weight is None:
            mix_quality_weight = torch.ones(paired_bs, device=volume_batch.device)
        else:
            mix_quality_weight = unlabeled_quality_weight[:paired_bs].to(volume_batch.device)

        entropy_unlab = self.get_entropy_map(fusion_map_soft[self.args.labeled_bs:self.args.labeled_bs + paired_bs])
        entropy_lab = self.get_entropy_map(fusion_map_soft[:paired_bs])
        pooling = nn.AdaptiveAvgPool2d((patch_size, patch_size))
        entropy_unlab = pooling(entropy_unlab).view(paired_bs, -1)
        entropy_lab = pooling(entropy_lab).view(paired_bs, -1)

        # _, min_indices_flat = torch.topk(entropy_unlab, top_k, largest=False)
        top_k = min(top_k, patch_size * patch_size)
        _, min_indices_flat = torch.topk(entropy_unlab, top_k, largest=True)
        min_indices_2d = torch.stack([min_indices_flat // patch_size, min_indices_flat % patch_size], dim=-1)
        # _, min_indices_flat_lab = torch.topk(entropy_lab, top_k, largest=False)
        _, min_indices_flat_lab = torch.topk(entropy_lab, top_k, largest=True)
        min_indices_2d_lab = torch.stack([min_indices_flat_lab // patch_size, min_indices_flat_lab % patch_size],
                                         dim=-1)

        device = volume_batch.device
        unlabeled_volume_batch_mix = torch.zeros_like(unlabeled_volume_batch, device=device)
        unlabel_pseudo_label_mix = torch.zeros_like(unlabel_pseudo_label, device=device)
        labeled_volume_batch_mix = torch.zeros_like(labeled_volume_batch, device=device)
        labeled_pseudo_label_mix = torch.zeros_like(labeled_label, device=device)

        patch_h = int(self.args.image_size / patch_size)
        for b in range(paired_bs):
            index = min_indices_2d[b]
            img_mask = torch.zeros((self.args.image_size, self.args.image_size), device=device)
            index_lab = min_indices_2d_lab[b]
            img_mask_lab = torch.zeros((self.args.image_size, self.args.image_size), device=device)
            for n in index:
                img_mask[n[0] * patch_h: (n[0] + 1) * patch_h, n[1] * patch_h: (n[1] + 1) * patch_h] = 1
            for n in index_lab:
                img_mask_lab[n[0] * patch_h: (n[0] + 1) * patch_h, n[1] * patch_h: (n[1] + 1) * patch_h] = 1

            unlabeled_volume_batch_mix[b] = labeled_volume_batch[b] * img_mask + unlabeled_volume_batch[b] * (1 - img_mask)
            unlabel_pseudo_label_mix[b] = labeled_label[b] * img_mask + unlabel_pseudo_label[b] * (1 - img_mask)

            labeled_volume_batch_mix[b] = unlabeled_volume_batch[b] * img_mask_lab + labeled_volume_batch[b] * (1 - img_mask_lab)
            labeled_pseudo_label_mix[b] = unlabel_pseudo_label[b] * img_mask_lab + labeled_label[b] * (1 - img_mask_lab)

        volume_batch_mix = torch.cat([labeled_volume_batch_mix, unlabeled_volume_batch_mix], dim=0)
        label_batch_mix = torch.cat([labeled_pseudo_label_mix, unlabel_pseudo_label_mix], dim=0)

        pred_UNet_mix, pred_VNet_mix, pred_UNet_soft_mix, pred_VNet_soft_mix, fusion_map_mix = self.SGDL(volume_batch_mix)

        pseudo_label_mix = torch.argmax(fusion_map_mix, dim=1)
        mixed_labeled_bs = paired_bs

        fusion_map_soft_mix = torch.softmax(fusion_map_mix, dim=1)
        UNet_sup_mixed_loss = ce_loss(pred_UNet_mix, label_batch_mix.long()) + self.dice_loss(pred_UNet_soft_mix, label_batch_mix)
        UNet_enp_mixed_loss = self.entropy_loss(pred_UNet_soft_mix, C=2)
        UNet_cons_mixed_loss = loss_diff1(pred_UNet_soft_mix, pred_VNet_soft_mix.clone().detach())
        UNet_unsup_mixed_loss = self.weighted_segmentation_loss(
            pred_UNet_mix[mixed_labeled_bs:],
            pred_UNet_soft_mix[mixed_labeled_bs:],
            pseudo_label_mix[mixed_labeled_bs:],
            mix_quality_weight,
        )

        VNet_sup_mixed_loss = ce_loss(pred_VNet_mix, label_batch_mix.long()) + self.dice_loss(pred_VNet_soft_mix, label_batch_mix)
        VNet_enp_mixed_loss = self.entropy_loss(pred_VNet_soft_mix, C=2)
        VNet_cons_mixed_loss = loss_diff2(pred_VNet_soft_mix, pred_UNet_soft_mix.clone().detach())
        VNet_unsup_mixed_loss = self.weighted_segmentation_loss(
            pred_VNet_mix[mixed_labeled_bs:],
            pred_VNet_soft_mix[mixed_labeled_bs:],
            pseudo_label_mix[mixed_labeled_bs:],
            mix_quality_weight,
        )

        fusion_mixed_loss = ce_loss(fusion_map_mix, label_batch_mix.long()) + self.dice_loss(fusion_map_soft_mix, label_batch_mix)

        UNet_mixed_loss = UNet_sup_mixed_loss + 0.9 * UNet_enp_mixed_loss + consistency_weight * (UNet_cons_mixed_loss + UNet_unsup_mixed_loss)
        VNet_mixed_loss = VNet_sup_mixed_loss + 0.9 * VNet_enp_mixed_loss + consistency_weight * (VNet_cons_mixed_loss + VNet_unsup_mixed_loss)

        return UNet_mixed_loss, VNet_mixed_loss, fusion_mixed_loss

    def train(self, volume_batch, label_batch, iter_num):
        image_embeddings = self.sam_model.image_encoder(volume_batch)
        pred_UNet, pred_VNet, pred_UNet_soft, pred_VNet_soft, fusion_map = self.SGDL(volume_batch)

        fusion_map_soft = torch.softmax(fusion_map, dim=1)
        prompt_logits, prompt_reliability = self.build_a3_consensus_prompt(
            pred_UNet_soft,
            pred_VNet_soft,
            fusion_map_soft,
        )
        points_embedding, boxes_embedding, mask_embedding = self.sam_model.super_prompt(image_embeddings)
        low_res_masks_all = torch.empty((self.args.batch_size, 0, int(self.args.image_size/4), int(self.args.image_size/4)), device=self.args.device)

        for i in range(self.args.num_classes):
            sparse_embeddings, dense_embeddings = self.sam_model.prompt_encoder(
                # points=points_embedding[i].unsqueeze(0),
                points=None,
                boxes=boxes_embedding[i],
                # boxes=None,
                masks=F.interpolate(prompt_logits[:, i, ...].unsqueeze(1), size=(64, 64), mode='bilinear')
                # masks=None,
            )

            low_res_masks, iou_predictions = self.sam_model.mask_decoder(
                image_embeddings=image_embeddings,
                image_pe=self.sam_model.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse_embeddings,
                dense_prompt_embeddings=dense_embeddings,
                multimask_output=self.args.multimask,
            )

            low_res_masks_all = torch.cat((low_res_masks_all, low_res_masks), dim=1)

        pred_sam = F.interpolate(low_res_masks_all, size=(self.args.image_size, self.args.image_size), mode="bilinear", align_corners=False)
        pred_sam_soft = torch.softmax(pred_sam, dim=1)
        reliability_map = self.build_reliability_map(pred_sam_soft, fusion_map_soft, pred_UNet_soft, pred_VNet_soft)
        a3_quality_weight = self.compute_a3_quality_weight(reliability_map, pred_sam_soft, prompt_reliability)

        fusion_loss = ce_loss(fusion_map[:self.args.labeled_bs], label_batch[:self.args.labeled_bs].long()) + self.dice_loss(fusion_map_soft[:self.args.labeled_bs], label_batch[:self.args.labeled_bs])

        UNet_sup_loss = ce_loss(pred_UNet[:self.args.labeled_bs], label_batch[:self.args.labeled_bs].long()) + self.dice_loss(pred_UNet_soft[:self.args.labeled_bs], label_batch[:self.args.labeled_bs])
        UNet_cons_loss = loss_diff1(pred_UNet_soft, pred_VNet_soft.clone().detach())
        UNet_enp_loss = self.entropy_loss(pred_UNet_soft, C=2)
        UNet_kd_loss = self.weighted_kd_loss(pred_UNet, pred_sam.clone().detach(), reliability_map, a3_quality_weight)

        VNet_sup_loss = ce_loss(pred_VNet[:self.args.labeled_bs], label_batch[:self.args.labeled_bs].long()) + self.dice_loss(pred_VNet_soft[:self.args.labeled_bs], label_batch[:self.args.labeled_bs])
        VNet_cons_loss = loss_diff2(pred_VNet_soft, pred_UNet_soft.clone().detach())
        VNet_enp_loss = self.entropy_loss(pred_VNet_soft, C=2)
        VNet_kd_loss = self.weighted_kd_loss(pred_VNet, pred_sam.clone().detach(), reliability_map, a3_quality_weight)

        sam_sup_loss = ce_loss(pred_sam[:self.args.labeled_bs], label_batch[:self.args.labeled_bs].long()) + self.dice_loss(pred_sam_soft[:self.args.labeled_bs], label_batch[:self.args.labeled_bs])
        sgdl_boundary_loss = (
            self.boundary_loss(fusion_map_soft[:self.args.labeled_bs], label_batch[:self.args.labeled_bs]) +
            self.boundary_loss(pred_UNet_soft[:self.args.labeled_bs], label_batch[:self.args.labeled_bs]) +
            self.boundary_loss(pred_VNet_soft[:self.args.labeled_bs], label_batch[:self.args.labeled_bs])
        ) / 3.0
        sam_boundary_loss = self.boundary_loss(pred_sam_soft[:self.args.labeled_bs], label_batch[:self.args.labeled_bs])
        sgdl_shape_loss = (
            soft_area_prior_loss(fusion_map_soft, self.sap_area_lower, self.sap_area_upper) +
            soft_area_prior_loss(pred_UNet_soft, self.sap_area_lower, self.sap_area_upper) +
            soft_area_prior_loss(pred_VNet_soft, self.sap_area_lower, self.sap_area_upper)
        ) / 3.0
        sam_shape_loss = soft_area_prior_loss(pred_sam_soft, self.sap_area_lower, self.sap_area_upper)
        boundary_loss = (sgdl_boundary_loss + sam_boundary_loss.detach()) / 2.0
        shape_loss = (sgdl_shape_loss + sam_shape_loss.detach()) / 2.0
        structure_loss = self.sap_boundary_weight * sgdl_boundary_loss + self.sap_shape_weight * sgdl_shape_loss
        sam_structure_loss = self.sap_boundary_weight * sam_boundary_loss + self.sap_shape_weight * sam_shape_loss

        rampup_denom = max(int(self.args.max_iterations / self.args.consistency_rampup), 1)
        consistency_weight = self.get_current_consistency_weight(iter_num // rampup_denom) * 10

        UNet_loss = UNet_sup_loss + UNet_kd_loss + 0.9 * UNet_enp_loss + consistency_weight * UNet_cons_loss
        VNet_loss = VNet_sup_loss + VNet_kd_loss + 0.9 * VNet_enp_loss + consistency_weight * VNet_cons_loss

        if iter_num > self.args.mixed_iterations:
            UNet_sup_mixed_loss, VNet_sup_mixed_loss, fusion_mixed_loss = self.mix_up(
                fusion_map_soft,
                volume_batch,
                pred_sam_soft[self.args.labeled_bs:],
                label_batch[:self.args.labeled_bs],
                consistency_weight,
                unlabeled_quality_weight=a3_quality_weight[self.args.labeled_bs:],
            )
            SGDL_loss = (UNet_loss + UNet_sup_mixed_loss + VNet_loss + VNet_sup_mixed_loss) / 2 + fusion_loss + fusion_mixed_loss + structure_loss
        else:
            SGDL_loss = (UNet_loss + VNet_loss) / 2 + fusion_loss + structure_loss

        sam_loss = sam_sup_loss + sam_structure_loss

        self.optimizer_sam.zero_grad()
        self.optimizer_SGDL.zero_grad()

        sam_loss.backward()
        SGDL_loss.backward()

        self.optimizer_sam.step()
        self.optimizer_SGDL.step()

        lr_ = self.args.lr * (1.0 - iter_num / self.args.max_iterations)
        UNet_lr_ = self.args.UNet_lr * (1.0 - iter_num / self.args.max_iterations)

        for param_group in self.optimizer_sam.param_groups:
            param_group['lr'] = lr_
        for param_group in self.optimizer_SGDL.param_groups:
            param_group['lr'] = UNet_lr_

        logging.info('iteration %d : '
                     '  sam_loss : %f'
                     '  sam_lr_ : %10f'
                     
                     '  SGDL_loss : %f'
                     '  UNet_VNet_loss : %f'
                     '  fusion_loss : %f'
                     '  prompt_weight : %f'
                     '  uckd_weight : %f'
                     '  qapl_quality : %f'
                     '  boundary_loss : %f'
                     '  shape_loss : %f'
                     '  UNet_lr_ : %10f'

                     % (iter_num, sam_loss.item(), lr_,
                        SGDL_loss.item(), (UNet_loss + VNet_loss) / 2, fusion_loss,
                        prompt_reliability.mean().item(), reliability_map.mean().item(), a3_quality_weight.mean().item(),
                        boundary_loss.item(), shape_loss.item(), UNet_lr_,
                        ))
        return {
            "iteration": int(iter_num),
            "sam_loss": float(sam_loss.item()),
            "sam_lr": float(lr_),
            "sgdl_loss": float(SGDL_loss.item()),
            "unet_vnet_loss": float(((UNet_loss + VNet_loss) / 2).item()),
            "fusion_loss": float(fusion_loss.item()),
            "prompt_weight_mean": float(prompt_reliability.mean().item()),
            "uckd_weight_mean": float(reliability_map.mean().item()),
            "qapl_quality_mean": float(a3_quality_weight.mean().item()),
            "boundary_loss": float(boundary_loss.item()),
            "shape_loss": float(shape_loss.item()),
            "unet_lr": float(UNet_lr_),
        }

    def val(self, val_loader, snapshot_path, iter_num):
        self.sam_model.eval()
        self.SGDL.eval()

        avg_dice_sam = 0.0
        avg_dice_SGDL = 0.0
        avg_dice_unet = 0.0
        avg_dice_vnet = 0.0
        avg_sam_loss = 0.0
        avg_sgdl_loss = 0.0
        avg_unet_loss = 0.0
        avg_vnet_loss = 0.0
        avg_fusion_loss = 0.0

        for i_batch, sampled_batch in enumerate(val_loader):
            val_image, val_label = sampled_batch["image"].cuda(), sampled_batch["label"].cuda()
            image_embeddings = self.sam_model.image_encoder(val_image)
            pred_UNet, pred_VNet, pred_UNet_soft, pred_VNet_soft, fusion_map = self.SGDL(val_image)
            fusion_map_soft = torch.softmax(fusion_map, dim=1)
            prompt_logits, _ = self.build_a3_consensus_prompt(
                pred_UNet_soft,
                pred_VNet_soft,
                fusion_map_soft,
            )

            points_embedding, boxes_embedding, mask_embedding = self.sam_model.super_prompt(image_embeddings)

            low_res_masks_all = torch.empty(
                (1, 0, int(self.args.image_size / 4), int(self.args.image_size / 4)),
                device=self.args.device)
            with torch.no_grad():
                for i in range(self.args.num_classes):
                    sparse_embeddings, dense_embeddings = self.sam_model.prompt_encoder(
                        points=None,
                        boxes=boxes_embedding[i],
                        masks=F.interpolate(prompt_logits[:, i, ...].unsqueeze(1), size=(64, 64), mode='bilinear')
                    )
                    low_res_masks, iou_predictions = self.sam_model.mask_decoder(
                        image_embeddings=image_embeddings,
                        image_pe=self.sam_model.prompt_encoder.get_dense_pe(),
                        sparse_prompt_embeddings=sparse_embeddings,
                        dense_prompt_embeddings=dense_embeddings,
                        multimask_output=self.args.multimask,
                    )
                    low_res_masks_all = torch.cat((low_res_masks_all, low_res_masks), dim=1)
            pred_sam = F.interpolate(low_res_masks_all, size=(self.args.image_size, self.args.image_size))
            pred_sam_soft = torch.softmax(pred_sam, dim=1)
            sam_val_loss = ce_loss(pred_sam, val_label.long()) + self.dice_loss(pred_sam_soft, val_label)
            dice_sam = dice_coef(val_label, pred_sam_soft, thr=0.5)
            avg_dice_sam += dice_sam
            avg_sam_loss += sam_val_loss.item()

            fusion_val_loss = ce_loss(fusion_map, val_label.long()) + self.dice_loss(fusion_map_soft, val_label)
            dice_SGDL = dice_coef(val_label, fusion_map_soft, thr=0.5)
            avg_dice_SGDL += dice_SGDL
            avg_fusion_loss += fusion_val_loss.item()
            avg_sgdl_loss += fusion_val_loss.item()

            unet_val_loss = ce_loss(pred_UNet, val_label.long()) + self.dice_loss(pred_UNet_soft, val_label)
            dice_unet = dice_coef(val_label, pred_UNet_soft, thr=0.5)
            avg_dice_unet += dice_unet
            avg_unet_loss += unet_val_loss.item()
            vnet_val_loss = ce_loss(pred_VNet, val_label.long()) + self.dice_loss(pred_VNet_soft, val_label)
            dice_vnet = dice_coef(val_label, pred_VNet_soft, thr=0.5)
            avg_dice_vnet += dice_vnet
            avg_vnet_loss += vnet_val_loss.item()

        avg_dice_sam = avg_dice_sam / len(val_loader)
        avg_dice_SGDL = avg_dice_SGDL / len(val_loader)
        avg_dice_unet = avg_dice_unet / len(val_loader)
        avg_dice_vnet = avg_dice_vnet / len(val_loader)
        avg_sam_loss = avg_sam_loss / len(val_loader)
        avg_sgdl_loss = avg_sgdl_loss / len(val_loader)
        avg_unet_loss = avg_unet_loss / len(val_loader)
        avg_vnet_loss = avg_vnet_loss / len(val_loader)
        avg_fusion_loss = avg_fusion_loss / len(val_loader)

        logging.info('iteration %d : '
                     '  sam_mean_dice : %f '
                     '  SGDL_mean_dice : %f '
                     '  unet_mean_dice : %f '
                     '  vnet_mean_dice : %f '
                     '  sam_val_loss : %f '
                     '  sgdl_val_loss : %f '
                     '  unet_val_loss : %f '
                     '  vnet_val_loss : %f '
                     '  fusion_val_loss : %f '
                    % (iter_num, avg_dice_sam, avg_dice_SGDL, avg_dice_unet, avg_dice_vnet,
                       avg_sam_loss, avg_sgdl_loss, avg_unet_loss, avg_vnet_loss, avg_fusion_loss))

        if avg_dice_sam > self.best_performance_sam:
            self.best_performance_sam = avg_dice_sam
            save_best_sam = os.path.join(snapshot_path, 'sam_best_model.pth')
            torch.save(self.sam_model.state_dict(), save_best_sam)

        if avg_dice_SGDL > self.best_performance_SGDL:
            self.best_performance_SGDL = avg_dice_SGDL
            save_best_SGDL = os.path.join(snapshot_path, 'SGDL_best_model.pth')
            # save_best_SGDL = os.path.join(snapshot_path, 'SGDL_best_model_' + str(iter_num) + '.pth')
            torch.save(self.SGDL.state_dict(), save_best_SGDL)
        self.sam_model.train()
        self.SGDL.train()
        return {
            "iteration": int(iter_num),
            "sam_mean_dice": float(avg_dice_sam),
            "sgdl_mean_dice": float(avg_dice_SGDL),
            "unet_mean_dice": float(avg_dice_unet),
            "vnet_mean_dice": float(avg_dice_vnet),
            "sam_val_loss": float(avg_sam_loss),
            "sgdl_val_loss": float(avg_sgdl_loss),
            "unet_val_loss": float(avg_unet_loss),
            "vnet_val_loss": float(avg_vnet_loss),
            "fusion_val_loss": float(avg_fusion_loss),
        }

    def val_ACDC(self, val_loader, snapshot_path, iter_num):
        self.sam_model.eval()
        self.SGDL.eval()

        avg_dice_sam = 0.0
        avg_dice_SGDL = 0.0

        sam_info = np.array([0, 0, 0]).astype("float32")
        for i_batch, sampled_batch in enumerate(val_loader):
            val_image, val_label = sampled_batch["image"].cuda(), sampled_batch["label"].cuda()
            metric_list = test_single_volume(self.args, val_image, val_label, self.sam_model, self.SGDL)
            metric_list = np.array(metric_list).astype("float32")

            sam_info += metric_list[:, 0]

            metric_list = np.mean(metric_list, axis=0)
            avg_dice_sam += metric_list[0]
            avg_dice_SGDL += metric_list[1]

        avg_dice_sam = avg_dice_sam / len(val_loader)
        avg_dice_SGDL = avg_dice_SGDL / len(val_loader)

        sam_info = sam_info / len(val_loader)

        logging.info('iteration %d : '
                     '  sam_mean_dice : %f '
                     '  SGDL_mean_dice : %f '
                     '  sam_info : \n%s '
                     % (iter_num, avg_dice_sam, avg_dice_SGDL, str(sam_info)))

        if avg_dice_sam > self.best_performance_sam:
            self.best_performance_sam = avg_dice_sam
            save_best_sam = os.path.join(snapshot_path, 'sam_best_model.pth')
            torch.save(self.sam_model.state_dict(), save_best_sam)
        if avg_dice_SGDL > self.best_performance_SGDL:
            self.best_performance_SGDL = avg_dice_SGDL
            save_best_SGDL = os.path.join(snapshot_path, 'SGDL_best_model.pth')
            # save_best_SGDL = os.path.join(snapshot_path, 'SGDL_iter_' + str(iter_num) + ".pth")
            torch.save(self.SGDL.state_dict(), save_best_SGDL)

        self.sam_model.train()

        self.SGDL.train()
