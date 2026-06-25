import logging
import os
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "..", ".."))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)
if REPO_ROOT not in sys.path:
    sys.path.insert(1, REPO_ROOT)

from Model.model import KnowSAM
from Model.sam.build_sam import sam_model_registry
from perd_modules import PERDProbe
from prediction_ACDC import test_single_volume
from utils.losses_perd import DiceLoss, KDLoss, WeightedKDLoss, loss_diff1, loss_diff2
from utils.utils import dice_coef, multiclass_segmentation_metrics


ce_loss = torch.nn.CrossEntropyLoss()


class Trainer(nn.Module):
    def __init__(self, args):
        super(Trainer, self).__init__()
        self.args = args
        self.criterion_mse = nn.MSELoss()
        self.KDLoss = KDLoss(T=10)
        self.weighted_kd_loss = WeightedKDLoss(T=10)
        self.dice_loss = DiceLoss(args.num_classes)
        self.perd_enabled = bool(int(getattr(args, "perd_enabled", 1)))
        self.perd_probe = PERDProbe(args)

        self.sam_model = sam_model_registry[args.model_type](args).to(args.device).train()
        self.SGDL = KnowSAM(args).cuda().train()

        self.optimizer_sam = optim.Adam(self.sam_model.parameters(), lr=args.lr)
        self.optimizer_SGDL = torch.optim.SGD(
            self.SGDL.parameters(),
            lr=args.UNet_lr,
            momentum=0.9,
            weight_decay=0.0001,
        )

        self.best_performance_sam = 0.0
        self.best_performance_SGDL = 0.0

        for name, value in self.sam_model.named_parameters():
            if "Adapter" in name or "super_prompt" in name:
                value.requires_grad = True
            else:
                value.requires_grad = False

    def sigmoid_rampup(self, current, rampup_length):
        if rampup_length == 0:
            return 1.0
        current = np.clip(current, 0.0, rampup_length)
        phase = 1.0 - current / rampup_length
        return float(np.exp(-5.0 * phase * phase))

    def entropy_loss(self, p, C=2):
        y1 = -1 * torch.sum(p * torch.log(p + 1e-6), dim=1) / torch.tensor(np.log(C)).cuda()
        return torch.mean(y1)

    def get_entropy_map(self, p):
        return -1 * torch.sum(p * torch.log(p + 1e-6), dim=1, keepdim=True)

    def get_current_consistency_weight(self, epoch):
        return self.args.consistency * self.sigmoid_rampup(epoch, self.args.consistency_rampup)

    def forward_sam_from_logits(self, image_embeddings, prompt_logits):
        _, boxes_embedding, _ = self.sam_model.super_prompt(image_embeddings)
        b, _, h, w = prompt_logits.shape
        low_res_masks_all = torch.empty((b, 0, int(h / 4), int(w / 4)), device=self.args.device)
        for class_idx in range(self.args.num_classes):
            sparse_embeddings, dense_embeddings = self.sam_model.prompt_encoder(
                points=None,
                boxes=boxes_embedding[class_idx],
                masks=F.interpolate(
                    prompt_logits[:, class_idx:class_idx + 1].clone().detach(),
                    size=(64, 64),
                    mode="bilinear",
                    align_corners=False,
                ),
            )
            low_res_masks, _ = self.sam_model.mask_decoder(
                image_embeddings=image_embeddings,
                image_pe=self.sam_model.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse_embeddings,
                dense_prompt_embeddings=dense_embeddings,
                multimask_output=self.args.multimask,
            )
            low_res_masks_all = torch.cat((low_res_masks_all, low_res_masks), dim=1)
        return F.interpolate(low_res_masks_all, size=(self.args.image_size, self.args.image_size), mode="bilinear", align_corners=False)

    def mix_up(
        self,
        fusion_map_soft,
        volume_batch,
        pseudo_label,
        labeled_label,
        consistency_weight,
        patch_size=4,
        top_k=5,
    ):
        unlabel_pseudo_label = torch.argmax(pseudo_label.clone(), dim=1)
        paired_bs = min(volume_batch[:self.args.labeled_bs].shape[0], volume_batch[self.args.labeled_bs:].shape[0])
        if paired_bs <= 0:
            zero_loss = fusion_map_soft.sum() * 0.0
            return zero_loss, zero_loss, zero_loss

        entropy_unlab = self.get_entropy_map(fusion_map_soft[self.args.labeled_bs:self.args.labeled_bs + paired_bs])
        entropy_lab = self.get_entropy_map(fusion_map_soft[:paired_bs])
        pooling = nn.AdaptiveAvgPool2d((patch_size, patch_size))
        entropy_unlab = pooling(entropy_unlab).view(paired_bs, -1)
        entropy_lab = pooling(entropy_lab).view(paired_bs, -1)

        top_k = min(top_k, patch_size * patch_size)
        _, min_indices_flat = torch.topk(entropy_unlab, top_k, largest=True)
        min_indices_2d = torch.stack([min_indices_flat // patch_size, min_indices_flat % patch_size], dim=-1)
        _, min_indices_flat_lab = torch.topk(entropy_lab, top_k, largest=True)
        min_indices_2d_lab = torch.stack([min_indices_flat_lab // patch_size, min_indices_flat_lab % patch_size], dim=-1)

        labeled_volume_batch = volume_batch[:paired_bs]
        unlabeled_volume_batch = volume_batch[self.args.labeled_bs:self.args.labeled_bs + paired_bs]
        labeled_label = labeled_label[:paired_bs]
        unlabel_pseudo_label = unlabel_pseudo_label[:paired_bs]

        device = volume_batch.device
        unlabeled_volume_batch_mix = torch.zeros_like(unlabeled_volume_batch, device=device)
        unlabel_pseudo_label_mix = torch.zeros_like(unlabel_pseudo_label, device=device)
        labeled_volume_batch_mix = torch.zeros_like(labeled_volume_batch, device=device)
        labeled_pseudo_label_mix = torch.zeros_like(labeled_label, device=device)

        patch_h = int(self.args.image_size / patch_size)
        for b in range(paired_bs):
            img_mask = torch.zeros((self.args.image_size, self.args.image_size), device=device)
            img_mask_lab = torch.zeros((self.args.image_size, self.args.image_size), device=device)
            for n in min_indices_2d[b]:
                img_mask[n[0] * patch_h:(n[0] + 1) * patch_h, n[1] * patch_h:(n[1] + 1) * patch_h] = 1
            for n in min_indices_2d_lab[b]:
                img_mask_lab[n[0] * patch_h:(n[0] + 1) * patch_h, n[1] * patch_h:(n[1] + 1) * patch_h] = 1

            unlabeled_volume_batch_mix[b] = labeled_volume_batch[b] * img_mask + unlabeled_volume_batch[b] * (1 - img_mask)
            unlabel_pseudo_label_mix[b] = labeled_label[b] * img_mask + unlabel_pseudo_label[b] * (1 - img_mask)
            labeled_volume_batch_mix[b] = unlabeled_volume_batch[b] * img_mask_lab + labeled_volume_batch[b] * (1 - img_mask_lab)
            labeled_pseudo_label_mix[b] = unlabel_pseudo_label[b] * img_mask_lab + labeled_label[b] * (1 - img_mask_lab)

        volume_batch_mix = torch.cat([labeled_volume_batch_mix, unlabeled_volume_batch_mix], dim=0)
        label_batch_mix = torch.cat([labeled_pseudo_label_mix, unlabel_pseudo_label_mix], dim=0)
        pred_UNet_mix, pred_VNet_mix, pred_UNet_soft_mix, pred_VNet_soft_mix, fusion_map_mix = self.SGDL(volume_batch_mix)
        pseudo_label_mix = torch.argmax(fusion_map_mix, dim=1)
        fusion_map_soft_mix = torch.softmax(fusion_map_mix, dim=1)

        UNet_sup_mixed_loss = ce_loss(pred_UNet_mix, label_batch_mix.long()) + self.dice_loss(pred_UNet_soft_mix, label_batch_mix)
        UNet_enp_mixed_loss = self.entropy_loss(pred_UNet_soft_mix, C=self.args.num_classes)
        UNet_cons_mixed_loss = loss_diff1(pred_UNet_soft_mix, pred_VNet_soft_mix.clone().detach())
        UNet_unsup_mixed_loss = ce_loss(pred_UNet_mix[paired_bs:], pseudo_label_mix[paired_bs:].long()) + self.dice_loss(pred_UNet_soft_mix[paired_bs:], pseudo_label_mix[paired_bs:])

        VNet_sup_mixed_loss = ce_loss(pred_VNet_mix, label_batch_mix.long()) + self.dice_loss(pred_VNet_soft_mix, label_batch_mix)
        VNet_enp_mixed_loss = self.entropy_loss(pred_VNet_soft_mix, C=self.args.num_classes)
        VNet_cons_mixed_loss = loss_diff2(pred_VNet_soft_mix, pred_UNet_soft_mix.clone().detach())
        VNet_unsup_mixed_loss = ce_loss(pred_VNet_mix[paired_bs:], pseudo_label_mix[paired_bs:].long()) + self.dice_loss(pred_VNet_soft_mix[paired_bs:], pseudo_label_mix[paired_bs:])

        fusion_mixed_loss = ce_loss(fusion_map_mix, label_batch_mix.long()) + self.dice_loss(fusion_map_soft_mix, label_batch_mix)
        UNet_mixed_loss = UNet_sup_mixed_loss + 0.9 * UNet_enp_mixed_loss + consistency_weight * (UNet_cons_mixed_loss + UNet_unsup_mixed_loss)
        VNet_mixed_loss = VNet_sup_mixed_loss + 0.9 * VNet_enp_mixed_loss + consistency_weight * (VNet_cons_mixed_loss + VNet_unsup_mixed_loss)
        return UNet_mixed_loss, VNet_mixed_loss, fusion_mixed_loss

    def train(self, volume_batch, label_batch, iter_num):
        image_embeddings = self.sam_model.image_encoder(volume_batch)
        pred_UNet, pred_VNet, pred_UNet_soft, pred_VNet_soft, fusion_map = self.SGDL(volume_batch)
        fusion_map_soft = torch.softmax(fusion_map, dim=1)

        pred_sam = self.forward_sam_from_logits(image_embeddings, fusion_map)
        pred_sam_soft = torch.softmax(pred_sam, dim=1)

        perd_stats = {
            "perd_trust_mean": 1.0,
            "perd_pc_mean": 0.0,
            "perd_ed_mean": 0.0,
            "perd_ji_mean": 0.0,
            "perd_valid_ratio": 1.0,
            "perd_boundary_ratio": 0.0,
            "perd_saturated_ratio": 0.0,
        }
        kd_teacher = pred_sam.clone().detach()
        kd_weight = torch.ones((volume_batch.shape[0], 1, self.args.image_size, self.args.image_size), device=volume_batch.device)

        if self.perd_enabled:
            _, boxes_embedding, _ = self.sam_model.super_prompt(image_embeddings)
            perd_output = self.perd_probe.probe(
                self.sam_model,
                volume_batch,
                image_embeddings,
                boxes_embedding,
                fusion_map,
                self.args.labeled_bs,
            )
            perd_stats.update(perd_output.stats)
            kd_weight = perd_output.kd_weight
            if getattr(self.args, "perd_baseline", "perd") == "prompt_ensemble":
                kd_teacher = perd_output.orig_curve_logits.mean(dim=0).detach()
            else:
                kd_teacher = perd_output.y0_logits.detach()

        fusion_loss = ce_loss(fusion_map[:self.args.labeled_bs], label_batch[:self.args.labeled_bs].long()) + self.dice_loss(
            fusion_map_soft[:self.args.labeled_bs],
            label_batch[:self.args.labeled_bs],
        )

        UNet_sup_loss = ce_loss(pred_UNet[:self.args.labeled_bs], label_batch[:self.args.labeled_bs].long()) + self.dice_loss(
            pred_UNet_soft[:self.args.labeled_bs],
            label_batch[:self.args.labeled_bs],
        )
        UNet_cons_loss = loss_diff1(pred_UNet_soft, pred_VNet_soft.clone().detach())
        UNet_enp_loss = self.entropy_loss(pred_UNet_soft, C=self.args.num_classes)
        UNet_kd_loss = self.weighted_kd_loss(pred_UNet, kd_teacher, kd_weight)

        VNet_sup_loss = ce_loss(pred_VNet[:self.args.labeled_bs], label_batch[:self.args.labeled_bs].long()) + self.dice_loss(
            pred_VNet_soft[:self.args.labeled_bs],
            label_batch[:self.args.labeled_bs],
        )
        VNet_cons_loss = loss_diff2(pred_VNet_soft, pred_UNet_soft.clone().detach())
        VNet_enp_loss = self.entropy_loss(pred_VNet_soft, C=self.args.num_classes)
        VNet_kd_loss = self.weighted_kd_loss(pred_VNet, kd_teacher, kd_weight)

        sam_sup_loss = ce_loss(pred_sam[:self.args.labeled_bs], label_batch[:self.args.labeled_bs].long()) + self.dice_loss(
            pred_sam_soft[:self.args.labeled_bs],
            label_batch[:self.args.labeled_bs],
        )

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
            )
            SGDL_loss = (UNet_loss + UNet_sup_mixed_loss + VNet_loss + VNet_sup_mixed_loss) / 2 + fusion_loss + fusion_mixed_loss
        else:
            SGDL_loss = (UNet_loss + VNet_loss) / 2 + fusion_loss

        sam_loss = sam_sup_loss
        self.optimizer_sam.zero_grad()
        self.optimizer_SGDL.zero_grad()
        sam_loss.backward()
        SGDL_loss.backward()
        self.optimizer_sam.step()
        self.optimizer_SGDL.step()

        lr_ = self.args.lr * (1.0 - iter_num / self.args.max_iterations)
        UNet_lr_ = self.args.UNet_lr * (1.0 - iter_num / self.args.max_iterations)
        for param_group in self.optimizer_sam.param_groups:
            param_group["lr"] = lr_
        for param_group in self.optimizer_SGDL.param_groups:
            param_group["lr"] = UNet_lr_

        logging.info(
            "iteration %d : sam_loss : %f sam_lr_ : %10f SGDL_loss : %f UNet_VNet_loss : %f "
            "fusion_loss : %f UNet_lr_ : %10f perd_trust_mean : %f perd_pc_mean : %f "
            "perd_ed_mean : %f perd_ji_mean : %f perd_valid_ratio : %f perd_boundary_ratio : %f "
            "perd_saturated_ratio : %f",
            iter_num,
            sam_loss.item(),
            lr_,
            SGDL_loss.item(),
            ((UNet_loss + VNet_loss) / 2).item(),
            fusion_loss.item(),
            UNet_lr_,
            perd_stats["perd_trust_mean"],
            perd_stats["perd_pc_mean"],
            perd_stats["perd_ed_mean"],
            perd_stats["perd_ji_mean"],
            perd_stats["perd_valid_ratio"],
            perd_stats["perd_boundary_ratio"],
            perd_stats["perd_saturated_ratio"],
        )
        return {
            "iteration": int(iter_num),
            "sam_loss": float(sam_loss.item()),
            "sam_lr": float(lr_),
            "sgdl_loss": float(SGDL_loss.item()),
            "unet_vnet_loss": float(((UNet_loss + VNet_loss) / 2).item()),
            "fusion_loss": float(fusion_loss.item()),
            "unet_lr": float(UNet_lr_),
            **perd_stats,
        }

    def val(self, val_loader, snapshot_path, iter_num):
        self.sam_model.eval()
        self.SGDL.eval()
        avg_dice_sam = 0.0
        avg_dice_SGDL = 0.0
        avg_dice_unet = 0.0
        avg_dice_vnet = 0.0
        multiclass_records = {"sam": [], "SGDL": [], "unet": [], "vnet": []}

        for sampled_batch in val_loader:
            val_image, val_label = sampled_batch["image"].cuda(), sampled_batch["label"].cuda()
            with torch.no_grad():
                image_embeddings = self.sam_model.image_encoder(val_image)
                pred_UNet, pred_VNet, pred_UNet_soft, pred_VNet_soft, fusion_map = self.SGDL(val_image)
                pred_sam = self.forward_sam_from_logits(image_embeddings, fusion_map)
                pred_sam_soft = torch.softmax(pred_sam, dim=1)
                fusion_map_soft = torch.softmax(fusion_map, dim=1)

            if self.args.num_classes > 2:
                sam_metrics = multiclass_segmentation_metrics(val_label, pred_sam_soft, self.args.num_classes)
                sgdl_metrics = multiclass_segmentation_metrics(val_label, fusion_map_soft, self.args.num_classes)
                unet_metrics = multiclass_segmentation_metrics(val_label, pred_UNet_soft, self.args.num_classes)
                vnet_metrics = multiclass_segmentation_metrics(val_label, pred_VNet_soft, self.args.num_classes)
                avg_dice_sam += sam_metrics["avg_dice"]
                avg_dice_SGDL += sgdl_metrics["avg_dice"]
                avg_dice_unet += unet_metrics["avg_dice"]
                avg_dice_vnet += vnet_metrics["avg_dice"]
                multiclass_records["sam"].append(sam_metrics)
                multiclass_records["SGDL"].append(sgdl_metrics)
                multiclass_records["unet"].append(unet_metrics)
                multiclass_records["vnet"].append(vnet_metrics)
            else:
                avg_dice_sam += dice_coef(val_label, pred_sam_soft, thr=0.5)
                avg_dice_SGDL += dice_coef(val_label, fusion_map_soft, thr=0.5)
                avg_dice_unet += dice_coef(val_label, pred_UNet_soft, thr=0.5)
                avg_dice_vnet += dice_coef(val_label, pred_VNet_soft, thr=0.5)

        avg_dice_sam /= len(val_loader)
        avg_dice_SGDL /= len(val_loader)
        avg_dice_unet /= len(val_loader)
        avg_dice_vnet /= len(val_loader)
        logging.info(
            "iteration %d : sam_mean_dice : %f SGDL_mean_dice : %f unet_mean_dice : %f vnet_mean_dice : %f",
            iter_num,
            avg_dice_sam,
            avg_dice_SGDL,
            avg_dice_unet,
            avg_dice_vnet,
        )

        if avg_dice_sam > self.best_performance_sam:
            self.best_performance_sam = avg_dice_sam
            torch.save(self.sam_model.state_dict(), os.path.join(snapshot_path, "sam_best_model.pth"))
        if avg_dice_SGDL > self.best_performance_SGDL:
            self.best_performance_SGDL = avg_dice_SGDL
            torch.save(self.SGDL.state_dict(), os.path.join(snapshot_path, "SGDL_best_model.pth"))
        self.sam_model.train()
        self.SGDL.train()
        return {
            "iteration": int(iter_num),
            "sam_mean_dice": float(avg_dice_sam),
            "sgdl_mean_dice": float(avg_dice_SGDL),
            "unet_mean_dice": float(avg_dice_unet),
            "vnet_mean_dice": float(avg_dice_vnet),
        }

    def val_ACDC(self, val_loader, snapshot_path, iter_num):
        self.sam_model.eval()
        self.SGDL.eval()
        avg_dice_sam = 0.0
        avg_dice_SGDL = 0.0
        sam_info = np.array([0, 0, 0]).astype("float32")
        for sampled_batch in val_loader:
            val_image, val_label = sampled_batch["image"].cuda(), sampled_batch["label"].cuda()
            metric_list = test_single_volume(self.args, val_image, val_label, self.sam_model, self.SGDL)
            metric_list = np.array(metric_list).astype("float32")
            sam_info += metric_list[:, 0]
            metric_list = np.mean(metric_list, axis=0)
            avg_dice_sam += metric_list[0]
            avg_dice_SGDL += metric_list[1]
        avg_dice_sam /= len(val_loader)
        avg_dice_SGDL /= len(val_loader)
        sam_info /= len(val_loader)
        logging.info(
            "iteration %d : sam_mean_dice : %f SGDL_mean_dice : %f sam_info : \n%s ",
            iter_num,
            avg_dice_sam,
            avg_dice_SGDL,
            str(sam_info),
        )
        if avg_dice_sam > self.best_performance_sam:
            self.best_performance_sam = avg_dice_sam
            torch.save(self.sam_model.state_dict(), os.path.join(snapshot_path, "sam_best_model.pth"))
        if avg_dice_SGDL > self.best_performance_SGDL:
            self.best_performance_SGDL = avg_dice_SGDL
            torch.save(self.SGDL.state_dict(), os.path.join(snapshot_path, "SGDL_best_model.pth"))
        self.sam_model.train()
        self.SGDL.train()
