import logging
import os
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "..", ".."))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)
if REPO_ROOT not in sys.path:
    sys.path.insert(1, REPO_ROOT)

from state_modules import A3PASSNet
from state_targets import StateTargetGenerator, perturb_ultrasound_tensor
from utils.losses_a3_pass import DiceLoss, BoundaryLoss, loss_diff1, loss_diff2, soft_area_prior_loss
from utils.utils import dice_coef, multiclass_segmentation_metrics

ce_loss = torch.nn.CrossEntropyLoss()


class Trainer(nn.Module):
    """A3-PASS trainer with three core mechanisms.

    AASP: acoustic-anatomical state posterior.
    SCMD: state-conditioned mask decoding.
    PGUL: posterior-guided unlabeled learning.
    """

    def __init__(self, args):
        super().__init__()
        self.args = args
        self.PASS = A3PASSNet(args).to(args.device).train()
        self.dice_loss = DiceLoss(args.num_classes)
        self.boundary_loss = BoundaryLoss(args.num_classes)
        self.state_target_generator = StateTargetGenerator(
            state_size=getattr(args, "pass_state_size", 64)
        )

        self.pass_state_weight = getattr(args, "pass_state_weight", 0.20)
        self.pass_decode_weight = getattr(args, "pass_decode_weight", 1.00)
        self.pass_state_consistency_weight = getattr(args, "pass_state_consistency_weight", 0.20)
        self.pass_pseudo_weight = getattr(args, "pass_pseudo_weight", 0.35)
        self.pass_decode_consistency_weight = getattr(args, "pass_decode_consistency_weight", 0.10)
        self.pass_reliability_alpha = getattr(args, "pass_reliability_alpha", 4.0)
        self.pass_min_reliability = getattr(args, "pass_min_reliability", 0.05)
        self.sap_boundary_weight = getattr(args, "sap_boundary_weight", 0.05)
        self.sap_shape_weight = getattr(args, "sap_shape_weight", 0.03)
        self.sap_area_lower = getattr(args, "sap_area_lower", 0.001)
        self.sap_area_upper = getattr(args, "sap_area_upper", 0.08)

        sgdl_params = list(self.PASS.SGDL.parameters())
        state_params = [
            p for n, p in self.PASS.named_parameters()
            if not n.startswith("SGDL.")
        ]
        self.optimizer = torch.optim.SGD(
            [
                {"params": sgdl_params, "lr": args.UNet_lr},
                {"params": state_params, "lr": getattr(args, "pass_state_lr", 1e-3)},
            ],
            momentum=0.9,
            weight_decay=1e-4,
        )

        self.best_performance_PASS = 0.0
        self.best_performance_SGDL = 0.0

    def sigmoid_rampup(self, current, rampup_length):
        if rampup_length == 0:
            return 1.0
        current = np.clip(current, 0.0, rampup_length)
        phase = 1.0 - current / rampup_length
        return float(np.exp(-5.0 * phase * phase))

    def get_current_consistency_weight(self, epoch):
        return self.args.consistency * self.sigmoid_rampup(epoch, self.args.consistency_rampup)

    def entropy_loss(self, p, C=2):
        y1 = -1 * torch.sum(p * torch.log(p + 1e-6), dim=1) / torch.tensor(np.log(C), device=p.device)
        return torch.mean(y1)

    def normalized_entropy(self, prob):
        entropy = -torch.sum(prob * torch.log(prob + 1e-6), dim=1)
        return entropy / torch.log(torch.tensor(float(prob.shape[1]), device=prob.device))

    def state_supervised_loss(self, state, targets, sample_weight=None):
        geometry_loss = F.smooth_l1_loss(state["geometry"], targets["geometry"], reduction="none").mean(dim=1)
        dense_loss = F.l1_loss(state["dense"], targets["dense"], reduction="none").mean(dim=(1, 2, 3))
        loss = geometry_loss + dense_loss
        if sample_weight is not None:
            weight = sample_weight.detach().to(loss.device).float()
            return (loss * weight).sum() / weight.sum().clamp_min(1e-6)
        return loss.mean()

    def state_consistency_loss(self, state_a, state_b, sample_weight=None):
        geometry_loss = F.smooth_l1_loss(
            state_a["geometry"],
            state_b["geometry"].detach(),
            reduction="none",
        ).mean(dim=1)
        dense_loss = F.l1_loss(
            state_a["dense"],
            state_b["dense"].detach(),
            reduction="none",
        ).mean(dim=(1, 2, 3))
        latent_loss = F.mse_loss(
            F.normalize(state_a["latent"], dim=1),
            F.normalize(state_b["latent"].detach(), dim=1),
            reduction="none",
        ).mean(dim=1)
        loss = geometry_loss + dense_loss + latent_loss
        if sample_weight is not None:
            weight = sample_weight.detach().to(loss.device).float()
            return (loss * weight).sum() / weight.sum().clamp_min(1e-6)
        return loss.mean()

    def posterior_reliability(self, out, out_perturbed=None):
        prob = out["pass_soft"]
        pixel_entropy = self.normalized_entropy(prob).mean(dim=(1, 2))
        pass_fg = out["pass_soft"][:, 1:].sum(dim=1)
        fusion_fg = out["fusion_soft"][:, 1:].sum(dim=1)
        decode_gap = torch.abs(pass_fg - fusion_fg).mean(dim=(1, 2))
        uncertainty = 0.5 * pixel_entropy + 0.5 * decode_gap
        if out_perturbed is not None:
            geometry_gap = torch.abs(out["state"]["geometry"] - out_perturbed["state"]["geometry"]).mean(dim=1)
            dense_gap = torch.abs(out["state"]["dense"] - out_perturbed["state"]["dense"]).mean(dim=(1, 2, 3))
            uncertainty = (uncertainty + geometry_gap + dense_gap) / 3.0
        reliability = torch.exp(-self.pass_reliability_alpha * uncertainty)
        return reliability.clamp(min=self.pass_min_reliability, max=1.0).detach()

    def weighted_segmentation_loss(self, logits, prob, pseudo_label, sample_weight):
        if logits.shape[0] == 0:
            return logits.sum() * 0.0
        ce_per_sample = F.cross_entropy(logits, pseudo_label.long(), reduction="none").mean(dim=(1, 2))
        target_onehot = F.one_hot(pseudo_label.long(), num_classes=self.args.num_classes).permute(0, 3, 1, 2).float()
        target_onehot = target_onehot.to(prob.device)
        intersect = torch.sum(prob * target_onehot, dim=(2, 3))
        y_sum = torch.sum(target_onehot * target_onehot, dim=(2, 3))
        z_sum = torch.sum(prob * prob, dim=(2, 3))
        dice_per_class = 1.0 - (2 * intersect + 1e-6) / (y_sum + z_sum + 1e-6)
        dice_per_sample = dice_per_class.mean(dim=1)
        weight = sample_weight.detach().to(logits.device).float().clamp_min(1e-6)
        return ((ce_per_sample + dice_per_sample) * weight).sum() / weight.sum()

    def train(self, volume_batch, label_batch, iter_num):
        out = self.PASS(volume_batch)
        labeled_bs = self.args.labeled_bs
        labeled_labels = label_batch[:labeled_bs].long()

        targets = self.state_target_generator(label_batch[:labeled_bs])
        supervised_pass_loss = ce_loss(out["pass_logits"][:labeled_bs], labeled_labels)
        supervised_pass_loss = supervised_pass_loss + self.dice_loss(out["pass_soft"][:labeled_bs], label_batch[:labeled_bs])

        fusion_loss = ce_loss(out["fusion_logits"][:labeled_bs], labeled_labels)
        fusion_loss = fusion_loss + self.dice_loss(out["fusion_soft"][:labeled_bs], label_batch[:labeled_bs])
        unet_loss = ce_loss(out["pred_unet"][:labeled_bs], labeled_labels)
        unet_loss = unet_loss + self.dice_loss(out["pred_unet_soft"][:labeled_bs], label_batch[:labeled_bs])
        vnet_loss = ce_loss(out["pred_vnet"][:labeled_bs], labeled_labels)
        vnet_loss = vnet_loss + self.dice_loss(out["pred_vnet_soft"][:labeled_bs], label_batch[:labeled_bs])

        labeled_state = {
            "geometry": out["state"]["geometry"][:labeled_bs],
            "dense": out["state"]["dense"][:labeled_bs],
            "latent": out["state"]["latent"][:labeled_bs],
        }
        state_sup_loss = self.state_supervised_loss(labeled_state, targets)

        boundary_loss = self.boundary_loss(out["pass_soft"][:labeled_bs], label_batch[:labeled_bs])
        shape_loss = soft_area_prior_loss(out["pass_soft"], self.sap_area_lower, self.sap_area_upper)

        consistency_loss = loss_diff1(out["pred_unet_soft"], out["pred_vnet_soft"].detach())
        consistency_loss = consistency_loss + loss_diff2(out["pred_vnet_soft"], out["pred_unet_soft"].detach())
        entropy_regularizer = self.entropy_loss(out["pred_unet_soft"], C=self.args.num_classes)
        entropy_regularizer = entropy_regularizer + self.entropy_loss(out["pred_vnet_soft"], C=self.args.num_classes)

        rampup_denom = max(int(self.args.max_iterations / self.args.consistency_rampup), 1)
        consistency_weight = self.get_current_consistency_weight(iter_num // rampup_denom)
        unlabeled_loss = volume_batch.sum() * 0.0
        state_cons_loss = volume_batch.sum() * 0.0
        pseudo_loss = volume_batch.sum() * 0.0
        decode_cons_loss = volume_batch.sum() * 0.0
        reliability_mean = torch.tensor(0.0, device=volume_batch.device)

        if volume_batch.shape[0] > labeled_bs:
            unlabeled_volume = volume_batch[labeled_bs:]
            perturbed_volume = perturb_ultrasound_tensor(
                unlabeled_volume,
                noise_std=getattr(self.args, "pass_noise_std", 0.03),
                gain_range=getattr(self.args, "pass_gain_range", 0.12),
            )
            out_unlabeled = {
                "pass_logits": out["pass_logits"][labeled_bs:],
                "pass_soft": out["pass_soft"][labeled_bs:],
                "fusion_soft": out["fusion_soft"][labeled_bs:],
                "state": {
                    "geometry": out["state"]["geometry"][labeled_bs:],
                    "dense": out["state"]["dense"][labeled_bs:],
                    "latent": out["state"]["latent"][labeled_bs:],
                },
            }
            out_perturbed = self.PASS(perturbed_volume)
            reliability = self.posterior_reliability(out_unlabeled, out_perturbed)
            reliability_mean = reliability.mean()
            state_cons_loss = self.state_consistency_loss(out_perturbed["state"], out_unlabeled["state"], reliability)
            pseudo_label = torch.argmax(out_unlabeled["pass_soft"].detach(), dim=1)
            pseudo_loss = self.weighted_segmentation_loss(
                out_unlabeled["pass_logits"],
                out_unlabeled["pass_soft"],
                pseudo_label,
                reliability,
            )
            decode_cons_map = torch.abs(out_unlabeled["pass_soft"] - out_unlabeled["fusion_soft"].detach()).mean(dim=(1, 2, 3))
            decode_cons_loss = (decode_cons_map * reliability).sum() / reliability.sum().clamp_min(1e-6)
            unlabeled_loss = (
                self.pass_state_consistency_weight * state_cons_loss
                + self.pass_pseudo_weight * pseudo_loss
                + self.pass_decode_consistency_weight * decode_cons_loss
            )

        total_loss = (
            self.pass_decode_weight * supervised_pass_loss
            + fusion_loss
            + 0.5 * (unet_loss + vnet_loss)
            + self.pass_state_weight * state_sup_loss
            + consistency_weight * consistency_loss
            + 0.1 * entropy_regularizer
            + consistency_weight * unlabeled_loss
            + self.sap_boundary_weight * boundary_loss
            + self.sap_shape_weight * shape_loss
        )

        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()

        sgdl_lr = self.args.UNet_lr * (1.0 - iter_num / self.args.max_iterations)
        state_lr = getattr(self.args, "pass_state_lr", 1e-3) * (1.0 - iter_num / self.args.max_iterations)
        self.optimizer.param_groups[0]["lr"] = sgdl_lr
        self.optimizer.param_groups[1]["lr"] = state_lr

        logging.info(
            "iteration %d : pass_loss=%f supervised=%f state=%f unlabeled=%f "
            "state_cons=%f pseudo=%f reliability=%f fusion=%f sgdl_lr=%10f state_lr=%10f",
            iter_num,
            total_loss.item(),
            supervised_pass_loss.item(),
            state_sup_loss.item(),
            unlabeled_loss.item(),
            state_cons_loss.item(),
            pseudo_loss.item(),
            reliability_mean.item(),
            fusion_loss.item(),
            sgdl_lr,
            state_lr,
        )
        return {
            "iteration": int(iter_num),
            "pass_loss": float(total_loss.item()),
            "supervised_pass_loss": float(supervised_pass_loss.item()),
            "state_supervised_loss": float(state_sup_loss.item()),
            "unlabeled_loss": float(unlabeled_loss.item()),
            "state_consistency_loss": float(state_cons_loss.item()),
            "pseudo_loss": float(pseudo_loss.item()),
            "decode_consistency_loss": float(decode_cons_loss.item()),
            "posterior_reliability_mean": float(reliability_mean.item()),
            "fusion_loss": float(fusion_loss.item()),
            "boundary_loss": float(boundary_loss.item()),
            "shape_loss": float(shape_loss.item()),
            "sgdl_lr": float(sgdl_lr),
            "state_lr": float(state_lr),
        }

    def val(self, val_loader, snapshot_path, iter_num):
        self.PASS.eval()
        avg_dice_pass = 0.0
        avg_dice_sgdl = 0.0
        avg_dice_unet = 0.0
        avg_dice_vnet = 0.0
        avg_pass_loss = 0.0
        avg_fusion_loss = 0.0
        multiclass_records = {
            "pass": [],
            "sgdl": [],
            "unet": [],
            "vnet": [],
        }

        with torch.no_grad():
            for sampled_batch in val_loader:
                val_image = sampled_batch["image"].to(self.args.device)
                val_label = sampled_batch["label"].to(self.args.device)
                out = self.PASS(val_image)
                pass_loss = ce_loss(out["pass_logits"], val_label.long()) + self.dice_loss(out["pass_soft"], val_label)
                fusion_loss = ce_loss(out["fusion_logits"], val_label.long()) + self.dice_loss(out["fusion_soft"], val_label)
                if self.args.num_classes > 2:
                    pass_metrics = multiclass_segmentation_metrics(val_label, out["pass_soft"], self.args.num_classes)
                    sgdl_metrics = multiclass_segmentation_metrics(val_label, out["fusion_soft"], self.args.num_classes)
                    unet_metrics = multiclass_segmentation_metrics(val_label, out["pred_unet_soft"], self.args.num_classes)
                    vnet_metrics = multiclass_segmentation_metrics(val_label, out["pred_vnet_soft"], self.args.num_classes)
                    avg_dice_pass += pass_metrics["avg_dice"]
                    avg_dice_sgdl += sgdl_metrics["avg_dice"]
                    avg_dice_unet += unet_metrics["avg_dice"]
                    avg_dice_vnet += vnet_metrics["avg_dice"]
                    multiclass_records["pass"].append(pass_metrics)
                    multiclass_records["sgdl"].append(sgdl_metrics)
                    multiclass_records["unet"].append(unet_metrics)
                    multiclass_records["vnet"].append(vnet_metrics)
                else:
                    avg_dice_pass += dice_coef(val_label, out["pass_soft"], thr=0.5)
                    avg_dice_sgdl += dice_coef(val_label, out["fusion_soft"], thr=0.5)
                    avg_dice_unet += dice_coef(val_label, out["pred_unet_soft"], thr=0.5)
                    avg_dice_vnet += dice_coef(val_label, out["pred_vnet_soft"], thr=0.5)
                avg_pass_loss += pass_loss.item()
                avg_fusion_loss += fusion_loss.item()

        n = max(len(val_loader), 1)
        avg_dice_pass /= n
        avg_dice_sgdl /= n
        avg_dice_unet /= n
        avg_dice_vnet /= n
        avg_pass_loss /= n
        avg_fusion_loss /= n

        logging.info(
            "iteration %d : pass_mean_dice=%f sgdl_mean_dice=%f unet_mean_dice=%f vnet_mean_dice=%f pass_val_loss=%f fusion_val_loss=%f",
            iter_num,
            avg_dice_pass,
            avg_dice_sgdl,
            avg_dice_unet,
            avg_dice_vnet,
            avg_pass_loss,
            avg_fusion_loss,
        )
        if self.args.num_classes > 2:
            for model_name, records in multiclass_records.items():
                if not records:
                    continue
                avg_record = {
                    key: float(np.nanmean([record[key] for record in records]))
                    for key in records[0].keys()
                }
                class_parts = []
                for class_idx in range(1, self.args.num_classes):
                    class_parts.append(
                        "class_%d_dice=%.6f class_%d_iou=%.6f class_%d_hd95=%.6f" % (
                            class_idx, avg_record[f"class_{class_idx}_dice"],
                            class_idx, avg_record[f"class_{class_idx}_iou"],
                            class_idx, avg_record[f"class_{class_idx}_hd95"],
                        )
                    )
                logging.info(
                    "iteration %d : %s_multiclass_val avg_dice=%.6f avg_iou=%.6f avg_hd95=%.6f %s",
                    iter_num,
                    model_name,
                    avg_record["avg_dice"],
                    avg_record["avg_iou"],
                    avg_record["avg_hd95"],
                    " ".join(class_parts),
                )

        if avg_dice_pass > self.best_performance_PASS:
            self.best_performance_PASS = avg_dice_pass
            torch.save(self.PASS.state_dict(), os.path.join(snapshot_path, "PASS_best_model.pth"))

        if avg_dice_sgdl > self.best_performance_SGDL:
            self.best_performance_SGDL = avg_dice_sgdl
            torch.save(self.PASS.SGDL.state_dict(), os.path.join(snapshot_path, "SGDL_best_model.pth"))

        self.PASS.train()
        return {
            "iteration": int(iter_num),
            "pass_mean_dice": float(avg_dice_pass),
            "sgdl_mean_dice": float(avg_dice_sgdl),
            "unet_mean_dice": float(avg_dice_unet),
            "vnet_mean_dice": float(avg_dice_vnet),
            "pass_val_loss": float(avg_pass_loss),
            "fusion_val_loss": float(avg_fusion_loss),
        }

    def val_ACDC(self, val_loader, snapshot_path, iter_num):
        return self.val(val_loader, snapshot_path, iter_num)
