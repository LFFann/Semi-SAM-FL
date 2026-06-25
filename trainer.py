import copy
import logging
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from Model.model import KnowSAM
from Model.sam.build_sam import sam_model_registry
from losses.srpc_loss import SRPCLoss
from models.complementarity_reliability import (
    ComplementarityReliabilityHead,
    build_reliability_inputs,
    compute_delta_utility_target,
    compute_reliability_losses,
    entropy_map,
)
from models.dynamic_graph import DynamicPrototypeGraph
from models.region_boundary_graph import RegionBoundaryPrototypeGraph
from models.reliability_head import ReliabilityHead
from models.ssrf import build_structure_prompts, compute_structure_reliability
from utils.diagnostics import compute_deploy_delta, compute_graph_stats, compute_oracle_fusion
from utils.ema import initialize_ema_model, update_ema
from utils.losses import DiceLoss
from utils.ssl_losses import masked_ce_dice_loss, sigmoid_rampup, view_consistency_loss, weak_strong_consistency_loss
from utils.utils import dice_coef, multiclass_segmentation_metrics


class Trainer(nn.Module):
    def __init__(self, args):
        super(Trainer, self).__init__()
        self.args = args
        self.dice_loss = DiceLoss(args.num_classes)
        self.class_weights = self._build_class_weights(args.class_weights)
        self.ce_loss = torch.nn.CrossEntropyLoss(weight=self.class_weights)
        self.srpc_loss = SRPCLoss(graph_lambda=args.srg_graph_lambda)

        self.use_train_only_sam = getattr(args, "use_train_only_sam", True)
        self.train_stage = getattr(args, "train_stage", "joint")
        self.use_ca_srg = getattr(args, "use_ca_srg_sampp", False)
        self.graph_type = getattr(args, "graph_type", "legacy")
        self.reliability_type = getattr(args, "reliability_type", "old_R")
        self.deploy_fusion_type = getattr(args, "deploy_fusion_type", "weighted_sum")

        # Adapter-SAM semantic branch: training-only soft teacher. It is not used
        # by the Lite deploy path, and it never produces hard pseudo labels.
        self.sam_model = None
        self.sam_trainable_params = []
        if self.use_train_only_sam and getattr(args, "use_adapter_sam_semantic", True):
            self.sam_model = sam_model_registry[args.model_type](args).to(args.device).train()
            for name, param in self.sam_model.named_parameters():
                trainable = "Adapter" in name or "super_prompt" in name
                param.requires_grad_(trainable)
                if trainable:
                    self.sam_trainable_params.append(param)
        elif self.use_train_only_sam:
            self.sam_model = sam_model_registry[args.model_type](args).to(args.device).eval()
            for param in self.sam_model.parameters():
                param.requires_grad_(False)

        # Frozen-SAM structure branch: training-only SSRF teacher for R_b/R_a/R_u.
        # This branch is never optimized and its outputs are not semantic labels.
        self.sam_struct = None
        if self.use_train_only_sam and getattr(args, "use_frozen_sam_struct", True):
            self.sam_struct = sam_model_registry[args.model_type](args).to(args.device).eval()
            for param in self.sam_struct.parameters():
                param.requires_grad_(False)

        self.SGDL = KnowSAM(args).to(args.device).train()
        if getattr(args, "sgdl_init_checkpoint", ""):
            checkpoint = torch.load(args.sgdl_init_checkpoint, map_location=args.device)
            self.SGDL.load_state_dict(checkpoint, strict=True)
            logging.info("Loaded SGDL init checkpoint: %s", args.sgdl_init_checkpoint)
        self.teacher = initialize_ema_model(self.SGDL).to(args.device)

        if self.graph_type == "region_boundary_prototype":
            self.graph = RegionBoundaryPrototypeGraph(
                num_classes=args.num_classes,
                feature_dim=32,
                num_prototypes=getattr(args, "graph_num_prototypes", 5),
                temperature=getattr(args, "graph_adj_temperature", 0.5),
                residual_init_zero=getattr(args, "graph_residual_init_zero", True),
                detach_region_mask=getattr(args, "graph_detach_region_mask", True),
            ).to(args.device)
        else:
            self.graph = DynamicPrototypeGraph(
                num_classes=args.num_classes,
                feature_dim=32,
                alpha=args.srg_graph_alpha,
                affinity_size=args.srg_affinity_size,
            ).to(args.device)
        if getattr(args, "graph_init_checkpoint", ""):
            checkpoint = torch.load(args.graph_init_checkpoint, map_location=args.device)
            self.graph.load_state_dict(checkpoint, strict=True)
            logging.info("Loaded DPG init checkpoint: %s", args.graph_init_checkpoint)

        if self.reliability_type == "complementarity":
            self.reliability_head = ComplementarityReliabilityHead(in_channels=15).to(args.device)
        else:
            self.reliability_head = ReliabilityHead(in_channels=32).to(args.device)

        if self.use_ca_srg and self.train_stage == "residual" and getattr(args, "freeze_student_in_residual", True):
            for param in self.SGDL.parameters():
                param.requires_grad_(False)

        self.optimizer_student = optim.SGD(
            self.SGDL.parameters(),
            lr=args.UNet_lr,
            momentum=0.9,
            weight_decay=0.0001,
        )
        self.optimizer_graph = optim.SGD(self.graph.parameters(), lr=args.UNet_lr, momentum=0.9, weight_decay=0.0001)
        self.optimizer_reliability = optim.SGD(
            self.reliability_head.parameters(), lr=args.UNet_lr, momentum=0.9, weight_decay=0.0001
        )
        self.optimizer_SGDL = self.optimizer_student
        self.optimizer_sam = None
        if self.sam_trainable_params:
            self.optimizer_sam = optim.Adam(self.sam_trainable_params, lr=args.lr)
            logging.info("Adapter-SAM trainable params: %d", sum(p.numel() for p in self.sam_trainable_params))
        else:
            logging.info("Adapter-SAM semantic branch disabled or no trainable Adapter/super_prompt params found.")

        self.best_performance_SGDL = -1.0
        self.best_performance_final = -1.0
        self.best_performance_sam = -1.0
        self.best_oracle_diagnostic_value = -1.0

    def _build_class_weights(self, class_weights):
        if not class_weights:
            return None
        values = [float(item.strip()) for item in class_weights.split(",") if item.strip()]
        if len(values) != self.args.num_classes:
            raise ValueError("class_weights must contain exactly num_classes values.")
        weights = torch.tensor(values, dtype=torch.float32, device=self.args.device)
        logging.info("Using class weights: %s", values)
        return weights

    def _dice_supervised_loss(self, probs, labels):
        if not getattr(self.args, "exclude_bg_dice", False):
            weights = None if self.class_weights is None else self.class_weights.detach().cpu().tolist()
            return self.dice_loss(probs, labels, weight=weights)

        target = F.one_hot(labels.long(), num_classes=self.args.num_classes).permute(0, 3, 1, 2).float()
        losses = []
        for class_idx in range(1, self.args.num_classes):
            score = probs[:, class_idx]
            gt = target[:, class_idx]
            intersect = torch.sum(score * gt)
            denom = torch.sum(score.square()) + torch.sum(gt.square())
            dice = 1.0 - (2.0 * intersect + 1e-10) / (denom + 1e-10)
            class_weight = 1.0 if self.class_weights is None else self.class_weights[class_idx]
            losses.append(dice * class_weight)
        return torch.stack(losses).sum() / sum(
            [1.0 if self.class_weights is None else float(self.class_weights[i].detach().cpu()) for i in range(1, self.args.num_classes)]
        )

    @torch.no_grad()
    def _update_teacher(self):
        decay = self.args.srg_ema_decay
        for teacher_param, student_param in zip(self.teacher.parameters(), self.SGDL.parameters()):
            teacher_param.data.mul_(decay).add_(student_param.data, alpha=1.0 - decay)

    def _supervised_loss(self, pred_unet, pred_vnet, pred_unet_soft, pred_vnet_soft, fusion_map, label_batch):
        labeled_bs = self.args.labeled_bs
        labels = label_batch[:labeled_bs].long()
        fusion_soft = torch.softmax(fusion_map[:labeled_bs], dim=1)
        fusion_loss = self.ce_loss(fusion_map[:labeled_bs], labels) + self._dice_supervised_loss(fusion_soft, labels)
        unet_loss = self.ce_loss(pred_unet[:labeled_bs], labels) + self._dice_supervised_loss(pred_unet_soft[:labeled_bs], labels)
        vnet_loss = self.ce_loss(pred_vnet[:labeled_bs], labels) + self._dice_supervised_loss(pred_vnet_soft[:labeled_bs], labels)
        return fusion_loss + 0.5 * (unet_loss + vnet_loss), fusion_loss, unet_loss, vnet_loss

    def _edge_map(self, x):
        if x.dim() == 3:
            x = x.unsqueeze(1)
        kernel_x = x.new_tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]).view(1, 1, 3, 3)
        kernel_y = x.new_tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]]).view(1, 1, 3, 3)
        gx = F.conv2d(x.float(), kernel_x, padding=1)
        gy = F.conv2d(x.float(), kernel_y, padding=1)
        return torch.sqrt(gx.square() + gy.square() + 1e-8)

    def _boundary_loss(self, fusion_map, label_batch):
        labeled_bs = self.args.labeled_bs
        prob = torch.softmax(fusion_map[:labeled_bs], dim=1)
        if self.args.num_classes > 1:
            pred_fg = prob[:, 1:, :, :].sum(dim=1, keepdim=True)
        else:
            pred_fg = prob[:, :1, :, :]
        label_fg = (label_batch[:labeled_bs] > 0).float().unsqueeze(1)
        return F.l1_loss(self._edge_map(pred_fg), self._edge_map(label_fg))

    def _sam_semantic_logits(self, volume_batch, student_logits):
        """Reuse KnowSAM's Adapter/super_prompt SAM path as semantic auxiliary logits."""
        if self.sam_model is None or not getattr(self.args, "use_adapter_sam_semantic", True):
            return student_logits.new_zeros(student_logits.shape)

        image_embeddings = self.sam_model.image_encoder(volume_batch)
        _, boxes_embedding, _ = self.sam_model.super_prompt(image_embeddings)
        low_res_size = int(self.args.image_size / 4)
        low_res_masks_all = torch.empty(
            (volume_batch.shape[0], 0, low_res_size, low_res_size),
            device=volume_batch.device,
            dtype=student_logits.dtype,
        )
        prompt_mask_size = self.sam_model.prompt_encoder.mask_input_size
        for class_idx in range(self.args.num_classes):
            mask_prompt = F.interpolate(
                student_logits[:, class_idx, ...].unsqueeze(1).detach(),
                size=prompt_mask_size,
                mode="bilinear",
                align_corners=False,
            )
            sparse_embeddings, dense_embeddings = self.sam_model.prompt_encoder(
                points=None,
                boxes=boxes_embedding[class_idx],
                masks=mask_prompt,
            )
            low_res_masks, _ = self.sam_model.mask_decoder(
                image_embeddings=image_embeddings,
                image_pe=self.sam_model.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse_embeddings,
                dense_prompt_embeddings=dense_embeddings,
                multimask_output=self.args.multimask,
            )
            if low_res_masks.shape[1] > 1:
                low_res_masks = low_res_masks[:, :1]
            low_res_masks_all = torch.cat((low_res_masks_all, low_res_masks), dim=1)
        return F.interpolate(
            low_res_masks_all,
            size=student_logits.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )

    def _structure_gate(self, R_b, R_u, target_size):
        R = R_b * (1.0 - R_u)
        R = torch.nan_to_num(R, nan=0.0, posinf=1.0, neginf=0.0).clamp(0.0, 1.0)
        if R.shape[-2:] != target_size:
            R = F.interpolate(R, size=target_size, mode="bilinear", align_corners=False)
        return R

    def _deploy_logits(self, student_logits, graph_logits, R_hat):
        """SAM-free Lite deploy branch used by validation and prediction."""
        if not getattr(self.args, "use_dpg_deploy", True):
            return student_logits
        if graph_logits.shape[-2:] != student_logits.shape[-2:]:
            graph_logits = F.interpolate(graph_logits, size=student_logits.shape[-2:], mode="bilinear", align_corners=False)
        if getattr(self.args, "use_reliability_head", True):
            gate = R_hat
        else:
            gate = student_logits.new_ones(student_logits.shape[0], 1, *student_logits.shape[-2:])
        return student_logits + self.args.alpha_graph * gate * graph_logits

    def _lambda_res(self, iter_num):
        max_value = getattr(self.args, "deploy_lambda_res_max", 0.3)
        ramp_iters = getattr(self.args, "deploy_lambda_res_rampup_iters", 1000)
        return max_value * sigmoid_rampup(iter_num, ramp_iters)

    def _ca_deploy_logits(self, student_logits, residual_logits, R_delta_hat, iter_num):
        """Gated residual deploy: student + lambda_res * R_delta_hat * residual."""
        if self.train_stage == "student":
            return student_logits, student_logits.new_tensor(0.0), 0.0
        if getattr(self.args, "deploy_fusion_type", "gated_residual") == "weighted_sum":
            gate = R_delta_hat if getattr(self.args, "use_reliability_head", True) else 1.0
            logits = student_logits + self.args.alpha_graph * gate * residual_logits
            return logits, residual_logits, self.args.alpha_graph
        if getattr(self.args, "deploy_clamp_residual", True):
            clamp_value = getattr(self.args, "deploy_residual_clamp_value", 3.0)
            residual_logits = residual_logits.clamp(-clamp_value, clamp_value)
        lambda_res = self._lambda_res(iter_num)
        if getattr(self.args, "use_reliability_head", True) and self.reliability_type != "none":
            gate = R_delta_hat
        else:
            gate = student_logits.new_ones(student_logits.shape[0], 1, *student_logits.shape[-2:])
        return student_logits + lambda_res * gate * residual_logits, residual_logits, lambda_res

    def _safe_loss(self, deploy_logits, student_logits, R_delta_hat):
        if not getattr(self.args, "deploy_use_safe_loss", True):
            return deploy_logits.sum() * 0.0
        low_mask = (R_delta_hat.detach() < getattr(self.args, "deploy_safe_thresh", 0.3)).float()
        if low_mask.sum().item() <= 0:
            return deploy_logits.sum() * 0.0
        diff = (torch.softmax(deploy_logits, dim=1) - torch.softmax(student_logits.detach(), dim=1)).abs().mean(dim=1, keepdim=True)
        return (diff * low_mask).sum() / low_mask.sum().clamp_min(1.0)

    def _residual_sparse_loss(self, residual_logits, R_delta_hat):
        if not getattr(self.args, "deploy_use_residual_sparse_loss", True):
            return residual_logits.sum() * 0.0
        return (R_delta_hat.detach() * residual_logits).abs().mean()

    def _graph_anti_collapse_loss(self, adjacency):
        if not getattr(self.args, "graph_use_anti_collapse_loss", True) or adjacency is None:
            return self.SGDL.UNet.out_conv.weight.sum() * 0.0
        A = adjacency if adjacency.dim() == 3 else adjacency.unsqueeze(0)
        entropy = -(A.clamp_min(1e-8) * A.clamp_min(1e-8).log()).sum(dim=-1).mean()
        min_entropy = getattr(self.args, "graph_min_entropy", 0.3)
        # Only penalize collapsed low-entropy adjacency; high entropy is logged but not maximized.
        return F.relu(A.new_tensor(min_entropy) - entropy)

    def _ca_graph_forward(self, features, student_prob, teacher_prob=None, uncertainty=None, disagreement=None):
        if self.train_stage == "student" or self.graph_type == "none":
            residual = features.new_zeros(features.shape[0], self.args.num_classes, *student_prob.shape[-2:])
            adjacency = None
            logs = compute_graph_stats(None)
            logs["residual_logits_abs_mean"] = 0.0
            return residual, adjacency, logs
        if self.graph_type == "region_boundary_prototype":
            return self.graph(
                features=features,
                student_prob=student_prob,
                teacher_prob=teacher_prob,
                uncertainty=uncertainty,
                disagreement=disagreement,
                out_size=student_prob.shape[-2:],
            )
        adjacency = self.graph.last_adjacency if self.graph.last_adjacency is not None else self.graph.R_prior.to(features.device)
        residual = self.graph(features, adjacency.to(features.device))
        if residual.shape[-2:] != student_prob.shape[-2:]:
            residual = F.interpolate(residual, size=student_prob.shape[-2:], mode="bilinear", align_corners=False)
        logs = compute_graph_stats(adjacency)
        logs["residual_logits_abs_mean"] = float(residual.detach().abs().mean().cpu())
        return residual, adjacency, logs

    def _ca_reliability_forward(self, student_prob, residual_logits, teacher_prob=None, features=None):
        if self.train_stage == "student":
            return student_prob.new_zeros(student_prob.shape[0], 1, *student_prob.shape[-2:])
        if self.reliability_type == "none":
            return student_prob.new_ones(student_prob.shape[0], 1, *student_prob.shape[-2:])
        if self.reliability_type == "complementarity":
            inputs = build_reliability_inputs(
                student_prob=student_prob,
                residual_logits=residual_logits,
                teacher_prob=teacher_prob,
                sam_prob=None,
                features=features,
            )
            return self.reliability_head(inputs)
        return self.reliability_head(features, out_size=student_prob.shape[-2:])

    def _reliability_weighted_kl(self, logits, target_prob, reliability):
        reliability = reliability.detach().clamp(0.0, 1.0)
        log_prob = F.log_softmax(logits, dim=1)
        kl_map = F.kl_div(log_prob, target_prob.detach(), reduction="none").sum(dim=1, keepdim=True)
        return (reliability * kl_map).sum() / reliability.sum().clamp_min(1e-6)

    def _reliability_distillation_loss(self, R_hat, R):
        target = R.detach().clamp(0.0, 1.0)
        pred = R_hat.clamp(1e-6, 1.0 - 1e-6)
        return F.l1_loss(pred, target) + F.binary_cross_entropy(pred, target)

    def _fuse_logits(self, student_logits, graph_logits=None, sam_logits=None, R=None):
        """Reliability-gated fusion branch: Z_s + alpha R Z_g + beta R Z_a."""
        if not getattr(self.args, "use_fusion", True):
            return student_logits
        if R is None or not getattr(self.args, "use_reliability_gate", True):
            R = student_logits.new_ones(student_logits.shape[0], 1, *student_logits.shape[-2:])
        final_logits = student_logits
        if getattr(self.args, "use_dpg_logits_fusion", True) and graph_logits is not None:
            if graph_logits.shape[-2:] != student_logits.shape[-2:]:
                graph_logits = F.interpolate(graph_logits, size=student_logits.shape[-2:], mode="bilinear", align_corners=False)
            final_logits = final_logits + self.args.alpha_graph * R * graph_logits
        if getattr(self.args, "use_adapter_sam_semantic", True) and sam_logits is not None:
            if sam_logits.shape[-2:] != student_logits.shape[-2:]:
                sam_logits = F.interpolate(sam_logits, size=student_logits.shape[-2:], mode="bilinear", align_corners=False)
            final_logits = final_logits + self.args.beta_sam * R * sam_logits
        return final_logits

    def _soft_supervised_loss(self, logits, labels):
        probs = torch.softmax(logits, dim=1)
        return self.ce_loss(logits, labels.long()) + self._dice_supervised_loss(probs, labels.long())

    def _srpc_pp_loss(self, student_logits, teacher_logits, graph_logits, sam_logits, R):
        eps = 1e-6
        with torch.no_grad():
            teacher_prob = F.softmax(teacher_logits, dim=1).clamp_min(eps)
            pseudo_log_prob = teacher_prob.log()
            if getattr(self.args, "use_dpg_logits_fusion", True) and graph_logits is not None:
                graph_prob = F.softmax(graph_logits.detach(), dim=1).clamp_min(eps)
                pseudo_log_prob = pseudo_log_prob + self.args.lambda_pg * graph_prob.log()
            if getattr(self.args, "use_adapter_sam_semantic", True) and sam_logits is not None:
                sam_prob = F.softmax(sam_logits.detach(), dim=1).clamp_min(eps)
                pseudo_log_prob = pseudo_log_prob + self.args.lambda_pa * sam_prob.log()
            pseudo_prob = F.softmax(pseudo_log_prob, dim=1).detach()
            weight = R.detach().clamp(0.0, 1.0)
        ce_map = -(pseudo_prob * F.log_softmax(student_logits, dim=1)).sum(dim=1, keepdim=True)
        return (weight * ce_map).sum() / weight.sum().clamp_min(eps), pseudo_prob, weight

    def _tensor_mean_std(self, tensor):
        tensor = tensor.detach()
        return float(tensor.mean().cpu()), float(tensor.std(unbiased=False).cpu())

    def _train_ca(self, volume_batch, label_batch, iter_num):
        labeled_bs = self.args.labeled_bs
        self.SGDL.train()
        if self.train_stage == "residual" and getattr(self.args, "freeze_student_in_residual", True):
            self.SGDL.eval()
        self.graph.train()
        self.reliability_head.train()

        pred_unet, pred_vnet, pred_unet_soft, pred_vnet_soft, fusion_map, features = self.SGDL(
            volume_batch, return_features=True
        )
        student_logits = fusion_map
        student_prob = torch.softmax(student_logits, dim=1)
        loss_sup_student, fusion_loss, unet_loss, vnet_loss = self._supervised_loss(
            pred_unet, pred_vnet, pred_unet_soft, pred_vnet_soft, fusion_map, label_batch
        )

        teacher_logits_all = None
        teacher_prob_all = None
        if getattr(self.args, "ssl_use_ema_teacher", True):
            with torch.no_grad():
                _, _, _, _, teacher_logits_all, _ = self.teacher(volume_batch, return_features=True)
                teacher_prob_all = torch.softmax(teacher_logits_all, dim=1)

        uncertainty = entropy_map(student_prob)
        if teacher_prob_all is not None:
            disagreement = (teacher_prob_all.detach() - student_prob.detach()).abs().mean(dim=1, keepdim=True)
        else:
            disagreement = uncertainty.detach()

        if self.graph_type == "legacy" and hasattr(self.graph, "update_prototypes"):
            with torch.no_grad():
                self.graph.update_prototypes(features[:labeled_bs].detach(), label_batch[:labeled_bs].detach())

        residual_logits, adjacency, graph_logs = self._ca_graph_forward(
            features=features,
            student_prob=student_prob,
            teacher_prob=teacher_prob_all,
            uncertainty=uncertainty,
            disagreement=disagreement,
        )
        R_delta_hat = self._ca_reliability_forward(
            student_prob=student_prob,
            residual_logits=residual_logits,
            teacher_prob=teacher_prob_all,
            features=features,
        )
        deploy_logits, residual_logits, lambda_res = self._ca_deploy_logits(
            student_logits, residual_logits, R_delta_hat, iter_num
        )
        deploy_prob = torch.softmax(deploy_logits, dim=1)

        if self.train_stage == "student":
            loss_sup_deploy = student_logits.sum() * 0.0
        else:
            loss_sup_deploy = self._soft_supervised_loss(deploy_logits[:labeled_bs], label_batch[:labeled_bs])

        loss_unsup_student = student_logits.sum() * 0.0
        loss_unsup_deploy = student_logits.sum() * 0.0
        ssl_logs = {"teacher_conf_mean": 0.0, "pseudo_mask_ratio": 0.0}
        if volume_batch.shape[0] > labeled_bs and teacher_logits_all is not None and getattr(self.args, "ssl_use_weak_strong", True):
            loss_unsup_student, ssl_logs = weak_strong_consistency_loss(
                student_logits[labeled_bs:],
                teacher_logits_all[labeled_bs:],
                getattr(self.args, "ssl_conf_thresh", 0.75),
                self.args.num_classes,
            )
            if self.train_stage != "student":
                loss_unsup_deploy, _ = weak_strong_consistency_loss(
                    deploy_logits[labeled_bs:],
                    teacher_logits_all[labeled_bs:],
                    getattr(self.args, "ssl_conf_thresh", 0.75),
                    self.args.num_classes,
                )

        loss_view_cons = student_logits.sum() * 0.0
        if getattr(self.args, "ssl_use_view_consistency", True) and volume_batch.shape[0] > labeled_bs:
            _, _, _, _, view_logits, _ = self.SGDL(volume_batch[labeled_bs:], return_features=True)
            loss_view_cons = view_consistency_loss(view_logits, student_logits[labeled_bs:])

        loss_copy_paste = student_logits.sum() * 0.0
        copy_paste_used_ratio = 0.0
        if (
            getattr(self.args, "ssl_use_copy_paste", False)
            and volume_batch.shape[0] > labeled_bs
            and teacher_logits_all is not None
            and np.random.rand() < getattr(self.args, "ssl_copy_paste_prob", 0.5)
        ):
            pair_count = min(labeled_bs, volume_batch.shape[0] - labeled_bs)
            src_image = volume_batch[:pair_count]
            src_label = label_batch[:pair_count].long()
            dst_image = volume_batch[labeled_bs:labeled_bs + pair_count]
            teacher_prob_cp = teacher_prob_all[labeled_bs:labeled_bs + pair_count].detach()
            teacher_conf, teacher_label = teacher_prob_cp.max(dim=1)
            paste_mask = (src_label > 0).float().unsqueeze(1)
            valid_mask = torch.maximum(
                paste_mask,
                (teacher_conf > getattr(self.args, "ssl_copy_paste_min_conf", 0.8)).float().unsqueeze(1),
            )
            if valid_mask.sum().item() > 0:
                mixed_image = src_image * paste_mask + dst_image * (1.0 - paste_mask)
                mixed_label = torch.where(paste_mask.squeeze(1) > 0.5, src_label, teacher_label)
                _, _, _, _, mixed_logits, _ = self.SGDL(mixed_image, return_features=True)
                loss_copy_paste = masked_ce_dice_loss(mixed_logits, mixed_label, valid_mask, self.args.num_classes)
                copy_paste_used_ratio = 1.0

        loss_R_delta_bce = student_logits.sum() * 0.0
        loss_R_delta_rank = student_logits.sum() * 0.0
        R_delta_target = torch.zeros_like(R_delta_hat)
        R_delta_valid = torch.zeros_like(R_delta_hat)
        if self.train_stage != "student" and self.reliability_type == "complementarity":
            target_l, valid_l, _ = compute_delta_utility_target(
                student_logits=student_logits[:labeled_bs].detach(),
                corrected_logits=deploy_logits[:labeled_bs].detach(),
                gt_mask=label_batch[:labeled_bs].detach(),
                patch_size=getattr(self.args, "reliability_patch_size", 16),
                utility_margin=getattr(self.args, "reliability_utility_margin", 0.005),
                ignore_neutral=getattr(self.args, "reliability_ignore_neutral", True),
            )
            R_delta_target[:labeled_bs] = target_l
            R_delta_valid[:labeled_bs] = valid_l
            if (
                volume_batch.shape[0] > labeled_bs
                and teacher_logits_all is not None
                and getattr(self.args, "reliability_use_unlabeled_target", True)
            ):
                target_u, valid_u, _ = compute_delta_utility_target(
                    student_logits=student_logits[labeled_bs:].detach(),
                    corrected_logits=deploy_logits[labeled_bs:].detach(),
                    teacher_logits=teacher_logits_all[labeled_bs:].detach(),
                    patch_size=getattr(self.args, "reliability_patch_size", 16),
                    utility_margin=getattr(self.args, "reliability_utility_margin", 0.005),
                    teacher_conf_thresh=getattr(self.args, "reliability_teacher_conf_thresh", 0.8),
                    ignore_neutral=getattr(self.args, "reliability_ignore_neutral", True),
                )
                R_delta_target[labeled_bs:] = target_u
                R_delta_valid[labeled_bs:] = valid_u
            loss_R_delta_bce, loss_R_delta_rank = compute_reliability_losses(
                R_delta_hat,
                R_delta_target,
                R_delta_valid,
                rank_margin=getattr(self.args, "reliability_rank_margin", 0.1),
                use_rank=getattr(self.args, "reliability_use_rank_loss", True),
            )

        loss_safe = self._safe_loss(deploy_logits, student_logits, R_delta_hat) if self.train_stage != "student" else student_logits.sum() * 0.0
        loss_residual_sparse = self._residual_sparse_loss(residual_logits, R_delta_hat) if self.train_stage != "student" else student_logits.sum() * 0.0
        loss_graph_anti = self._graph_anti_collapse_loss(adjacency) if self.train_stage != "student" else student_logits.sum() * 0.0

        unsup_ramp = sigmoid_rampup(iter_num, getattr(self.args, "ssl_unsup_rampup_iters", 1000))
        view_ramp = sigmoid_rampup(iter_num, getattr(self.args, "ssl_view_cons_rampup_iters", 1000))
        if self.train_stage == "student":
            loss_total = (
                getattr(self.args, "loss_w_sup_student", 1.0) * loss_sup_student
                + getattr(self.args, "loss_w_unsup", 1.0) * unsup_ramp * loss_unsup_student
                + getattr(self.args, "loss_w_view_cons", 0.1) * view_ramp * loss_view_cons
                + getattr(self.args, "loss_w_unsup", 1.0) * unsup_ramp * loss_copy_paste
            )
        else:
            loss_total = (
                getattr(self.args, "loss_w_sup_student", 1.0) * loss_sup_student
                + getattr(self.args, "loss_w_sup_deploy", 1.0) * loss_sup_deploy
                + getattr(self.args, "loss_w_unsup", 1.0) * unsup_ramp * (loss_unsup_student + loss_unsup_deploy)
                + getattr(self.args, "loss_w_view_cons", 0.1) * view_ramp * loss_view_cons
                + getattr(self.args, "loss_w_unsup", 1.0) * unsup_ramp * loss_copy_paste
                + getattr(self.args, "loss_w_r_delta_bce", 1.0) * loss_R_delta_bce
                + getattr(self.args, "loss_w_r_delta_rank", 0.1) * loss_R_delta_rank
                + getattr(self.args, "loss_w_safe", 0.05) * loss_safe
                + getattr(self.args, "loss_w_residual_sparse", 0.001) * loss_residual_sparse
                + getattr(self.args, "loss_w_graph_anti_collapse", 0.01) * loss_graph_anti
            )
        if not torch.isfinite(loss_total):
            raise FloatingPointError("Non-finite CA_SRG_SAMPP loss: %s" % loss_total.item())

        self.optimizer_student.zero_grad()
        self.optimizer_graph.zero_grad()
        self.optimizer_reliability.zero_grad()
        loss_total.backward()
        if not (self.train_stage == "residual" and getattr(self.args, "freeze_student_in_residual", True)):
            torch.nn.utils.clip_grad_norm_(self.SGDL.parameters(), max_norm=12.0)
            self.optimizer_student.step()
        torch.nn.utils.clip_grad_norm_(self.graph.parameters(), max_norm=12.0)
        torch.nn.utils.clip_grad_norm_(self.reliability_head.parameters(), max_norm=6.0)
        self.optimizer_graph.step()
        self.optimizer_reliability.step()

        if getattr(self.args, "ssl_use_ema_teacher", True):
            update_after = getattr(self.args, "ssl_ema_update_after", 0)
            update_every = max(1, getattr(self.args, "ssl_ema_update_every", 1))
            if iter_num >= update_after and iter_num % update_every == 0:
                update_ema(self.teacher, self.SGDL, decay=getattr(self.args, "ssl_ema_decay", self.args.srg_ema_decay))

        lr_ = self.args.UNet_lr * (1.0 - iter_num / self.args.max_iterations)
        student_lr = lr_
        if self.train_stage == "residual":
            student_lr = lr_ * getattr(self.args, "student_lr_scale_in_residual", 0.1)
        for param_group in self.optimizer_student.param_groups:
            param_group["lr"] = student_lr
        for optimizer in (self.optimizer_graph, self.optimizer_reliability):
            for param_group in optimizer.param_groups:
                param_group["lr"] = lr_

        deploy_logs = compute_deploy_delta(
            student_logits, deploy_logits, residual_logits, R_delta_hat, lambda_res,
            high_thresh=0.7,
            low_thresh=getattr(self.args, "deploy_safe_thresh", 0.3),
        )
        target_valid_ratio = float(R_delta_valid.detach().mean().cpu())
        target_pos_ratio = float((R_delta_target.detach() * R_delta_valid.detach()).sum().cpu() / R_delta_valid.detach().sum().clamp_min(1.0).cpu())
        R_delta_hat_mean = float(R_delta_hat.detach().mean().cpu())
        R_delta_hat_std = float(R_delta_hat.detach().std(unbiased=False).cpu())
        graph_logs = {key: float(value) for key, value in graph_logs.items()}

        logging.info(
            "iteration %d : loss_total=%f loss_sup_student=%f loss_sup_deploy=%f loss_unsup=%f "
            "loss_view_cons=%f loss_copy_paste=%f loss_R_delta_bce=%f loss_R_delta_rank=%f loss_safe=%f "
            "loss_residual_sparse=%f loss_graph_anti_collapse=%f teacher_conf_mean=%f pseudo_mask_ratio=%f "
            "copy_paste_used_ratio=%f R_delta_hat_mean=%f R_delta_hat_std=%f R_delta_target_pos_ratio=%f "
            "R_delta_target_valid_ratio=%f graph_adjacency_norm=%f A_diag_mean=%f A_offdiag_mean=%f "
            "A_offdiag_std=%f A_entropy=%f A_dynamic_delta_to_identity=%f residual_logits_abs_mean=%f "
            "lambda_res_current=%f deploy_student_delta=%f gated_residual_abs_mean=%f "
            "low_reliability_ratio=%f high_reliability_ratio=%f lr=%10f student_lr=%10f",
            iter_num,
            loss_total.item(),
            loss_sup_student.item(),
            loss_sup_deploy.item(),
            (loss_unsup_student + loss_unsup_deploy).item(),
            loss_view_cons.item(),
            loss_copy_paste.item(),
            loss_R_delta_bce.item(),
            loss_R_delta_rank.item(),
            loss_safe.item(),
            loss_residual_sparse.item(),
            loss_graph_anti.item(),
            ssl_logs["teacher_conf_mean"],
            ssl_logs["pseudo_mask_ratio"],
            copy_paste_used_ratio,
            R_delta_hat_mean,
            R_delta_hat_std,
            target_pos_ratio,
            target_valid_ratio,
            graph_logs["graph_adjacency_norm"],
            graph_logs["A_diag_mean"],
            graph_logs["A_offdiag_mean"],
            graph_logs["A_offdiag_std"],
            graph_logs["A_entropy"],
            graph_logs["A_dynamic_delta_to_identity"],
            graph_logs["residual_logits_abs_mean"],
            deploy_logs["lambda_res_current"],
            deploy_logs["deploy_student_delta"],
            deploy_logs["gated_residual_abs_mean"],
            deploy_logs["low_reliability_ratio"],
            deploy_logs["high_reliability_ratio"],
            lr_,
            student_lr,
        )
        return {
            "loss_total": float(loss_total.detach().cpu()),
            "sup_loss": float(loss_sup_student.detach().cpu()),
            "loss_sup_student": float(loss_sup_student.detach().cpu()),
            "loss_student_sup": float(loss_sup_student.detach().cpu()),
            "loss_sup_deploy": float(loss_sup_deploy.detach().cpu()),
            "loss_deploy_sup": float(loss_sup_deploy.detach().cpu()),
            "loss_sam_sup": 0.0,
            "loss_graph_sup": 0.0,
            "loss_unsup": float((loss_unsup_student + loss_unsup_deploy).detach().cpu()),
            "loss_u": float((loss_unsup_student + loss_unsup_deploy).detach().cpu()),
            "loss_view_cons": float(loss_view_cons.detach().cpu()),
            "loss_copy_paste": float(loss_copy_paste.detach().cpu()),
            "loss_R_delta_bce": float(loss_R_delta_bce.detach().cpu()),
            "loss_R_delta_rank": float(loss_R_delta_rank.detach().cpu()),
            "loss_R": float(loss_R_delta_bce.detach().cpu()),
            "loss_safe": float(loss_safe.detach().cpu()),
            "loss_residual_sparse": float(loss_residual_sparse.detach().cpu()),
            "loss_graph_anti_collapse": float(loss_graph_anti.detach().cpu()),
            "teacher_conf_mean": ssl_logs["teacher_conf_mean"],
            "pseudo_mask_ratio": ssl_logs["pseudo_mask_ratio"],
            "copy_paste_used_ratio": copy_paste_used_ratio,
            "R_delta_hat_mean": R_delta_hat_mean,
            "R_delta_hat_std": R_delta_hat_std,
            "R_delta_target_pos_ratio": target_pos_ratio,
            "R_delta_target_valid_ratio": target_valid_ratio,
            "R_hat_mean": R_delta_hat_mean,
            "R_mean": target_pos_ratio,
            "pseudo_weight_mean": target_valid_ratio,
            **graph_logs,
            **deploy_logs,
        }

    def train(self, volume_batch, label_batch, iter_num):
        if self.use_ca_srg:
            return self._train_ca(volume_batch, label_batch, iter_num)

        labeled_bs = self.args.labeled_bs
        self.SGDL.train()
        self.graph.train()
        self.reliability_head.train()
        if self.sam_model is not None and getattr(self.args, "use_adapter_sam_semantic", True):
            self.sam_model.train()
        elif self.sam_model is not None:
            self.sam_model.eval()
        if self.sam_struct is not None:
            self.sam_struct.eval()

        pred_unet, pred_vnet, pred_unet_soft, pred_vnet_soft, fusion_map, features = self.SGDL(
            volume_batch, return_features=True
        )
        student_logits = fusion_map
        fusion_soft = torch.softmax(fusion_map, dim=1)
        student_sup_loss, fusion_loss, unet_loss, vnet_loss = self._supervised_loss(
            pred_unet, pred_vnet, pred_unet_soft, pred_vnet_soft, fusion_map, label_batch
        )

        # Supervised prototypes are available before the unlabeled graph posterior is formed.
        with torch.no_grad():
            self.graph.update_prototypes(features[:labeled_bs].detach(), label_batch[:labeled_bs].detach())

        image_u = volume_batch[labeled_bs:]
        if image_u.shape[0] > 0:
            post_update_labels = None
            post_update_weights = None
            with torch.no_grad():
                _, _, _, _, teacher_logits_u, _ = self.teacher(image_u, return_features=True)
                teacher_pseudo_label = torch.softmax(teacher_logits_u, dim=1).argmax(dim=1)

                if self.sam_struct is not None:
                    prompts = build_structure_prompts(
                        image_u,
                        student_prob=fusion_soft[labeled_bs:].detach(),
                        num_prompts=self.args.srg_prompt_count,
                    )
                    R_b, R_a, R_u = compute_structure_reliability(
                        image_u,
                        {
                            "sam_model": self.sam_struct,
                            "items": prompts,
                            "affinity_size": self.args.srg_affinity_size,
                        },
                    )
                    R_ij = self.graph.build_adjacency(teacher_pseudo_label, R_a)
                else:
                    R_b = image_u.new_ones(image_u.shape[0], 1, *student_logits.shape[-2:])
                    R_u = image_u.new_zeros(image_u.shape[0], 1, *student_logits.shape[-2:])
                    R_ij = self.graph.R_prior.to(image_u.device)

            graph_logits_all = self.graph(features, R_ij)
            graph_logits_u = graph_logits_all[labeled_bs:]
            R_u_gate = self._structure_gate(R_b, R_u, student_logits.shape[-2:])
            R_labeled_gate = student_logits.new_ones(labeled_bs, 1, *student_logits.shape[-2:])
            R_all = torch.cat((R_labeled_gate, R_u_gate), dim=0)
            R_hat_all = self.reliability_head(features, out_size=student_logits.shape[-2:])
            deploy_logits = self._deploy_logits(student_logits, graph_logits_all, R_hat_all)
            sam_logits = self._sam_semantic_logits(volume_batch, student_logits)

            with torch.no_grad():
                assist_logits = teacher_logits_u
                if getattr(self.args, "use_dpg_deploy", True):
                    assist_logits = assist_logits + self.args.alpha_graph * R_u_gate * graph_logits_u.detach()
                if self.sam_model is not None and getattr(self.args, "use_adapter_sam_semantic", True):
                    assist_logits = assist_logits + self.args.beta_sam * R_u_gate * sam_logits[labeled_bs:].detach()
                pseudo_prob = torch.softmax(assist_logits, dim=1).detach()
            loss_srpc = self._reliability_weighted_kl(deploy_logits[labeled_bs:], pseudo_prob, R_u_gate)
            loss_R = (
                self._reliability_distillation_loss(R_hat_all[labeled_bs:], R_u_gate)
                if getattr(self.args, "use_reliability_head", True)
                else student_logits.new_tensor(0.0)
            )
            loss_srpc_fusion = loss_srpc
            loss_srpc_unet = student_logits.new_tensor(0.0)
            loss_srpc_vnet = student_logits.new_tensor(0.0)
            branch_unet = self.args.lambda_branch if self.args.lambda_branch_unet < 0 else self.args.lambda_branch_unet
            branch_vnet = self.args.lambda_branch if self.args.lambda_branch_vnet < 0 else self.args.lambda_branch_vnet
            pseudo_weight = R_u_gate

            with torch.no_grad():
                post_update_labels = torch.cat((label_batch[:labeled_bs].long(), pseudo_prob.argmax(dim=1)), dim=0)
                post_update_weights = torch.cat(
                    (
                        torch.ones(labeled_bs, 1, volume_batch.shape[-2], volume_batch.shape[-1], device=volume_batch.device),
                        pseudo_weight,
                    ),
                    dim=0,
                )
        else:
            post_update_labels = None
            post_update_weights = None
            loss_srpc = fusion_map.new_tensor(0.0)
            loss_srpc_fusion = fusion_map.new_tensor(0.0)
            loss_srpc_unet = fusion_map.new_tensor(0.0)
            loss_srpc_vnet = fusion_map.new_tensor(0.0)
            branch_unet = 0.0
            branch_vnet = 0.0
            graph_logits_all = self.graph(features)
            R_all = student_logits.new_ones(volume_batch.shape[0], 1, *student_logits.shape[-2:])
            R_hat_all = self.reliability_head(features, out_size=student_logits.shape[-2:])
            deploy_logits = self._deploy_logits(student_logits, graph_logits_all, R_hat_all)
            sam_logits = self._sam_semantic_logits(volume_batch, student_logits)
            R_b = fusion_map.new_ones(1, 1, 1, 1)
            R_u = fusion_map.new_zeros(1, 1, 1, 1)
            R_u_gate = fusion_map.new_ones(1, 1, 1, 1)
            pseudo_weight = fusion_map.new_ones(1, 1, 1, 1)
            R_ij = self.graph.R_prior.to(fusion_map.device)
            loss_R = fusion_map.new_tensor(0.0)

        if self.sam_model is not None and getattr(self.args, "use_adapter_sam_semantic", True):
            loss_sam_sup = self._soft_supervised_loss(sam_logits[:labeled_bs], label_batch[:labeled_bs])
        else:
            loss_sam_sup = student_logits.new_tensor(0.0)
        loss_deploy_sup = self._soft_supervised_loss(deploy_logits[:labeled_bs], label_batch[:labeled_bs])
        if getattr(self.args, "use_dpg_deploy", True):
            loss_graph_sup = self._soft_supervised_loss(graph_logits_all[:labeled_bs], label_batch[:labeled_bs])
            loss_graph_reg = (R_ij - torch.eye(self.args.num_classes, device=R_ij.device, dtype=R_ij.dtype)).square().mean()
        else:
            loss_graph_sup = student_logits.new_tensor(0.0)
            loss_graph_reg = student_logits.new_tensor(0.0)
        loss_struct = self._boundary_loss(deploy_logits, label_batch)

        if getattr(self.args, "sgdl_init_checkpoint", ""):
            lambda_u_scale = 1.0
        else:
            warmup = max(1, int(getattr(self.args, "lambda_u_warmup", 1)))
            lambda_u_scale = min(1.0, float(iter_num + 1) / float(warmup))
        lambda_u_eff = self.args.lambda_u * lambda_u_scale
        loss_total = (
            student_sup_loss
            + self.args.lambda_deploy * loss_deploy_sup
            + self.args.lambda_sam * loss_sam_sup
            + self.args.lambda_graph * loss_graph_sup
            + self.args.lambda_R * loss_R
            + lambda_u_eff * loss_srpc
            + self.args.lambda_struct * loss_struct
            + self.args.lambda_graph_reg * loss_graph_reg
        )
        if not torch.isfinite(loss_total):
            raise FloatingPointError(
                "Non-finite SRG-SAM++-Lite loss: total=%s student=%s deploy=%s sam=%s graph=%s R=%s u=%s struct=%s"
                % (
                    loss_total.item(),
                    student_sup_loss.item(),
                    loss_deploy_sup.item(),
                    loss_sam_sup.item(),
                    loss_graph_sup.item(),
                    loss_R.item(),
                    loss_srpc.item(),
                    loss_struct.item(),
                )
            )

        self.optimizer_student.zero_grad()
        self.optimizer_graph.zero_grad()
        self.optimizer_reliability.zero_grad()
        if self.optimizer_sam is not None:
            self.optimizer_sam.zero_grad()
        loss_total.backward()
        torch.nn.utils.clip_grad_norm_(self.SGDL.parameters(), max_norm=12.0)
        torch.nn.utils.clip_grad_norm_(self.graph.parameters(), max_norm=12.0)
        torch.nn.utils.clip_grad_norm_(self.reliability_head.parameters(), max_norm=6.0)
        if self.sam_trainable_params:
            torch.nn.utils.clip_grad_norm_(self.sam_trainable_params, max_norm=6.0)
        self.optimizer_student.step()
        self.optimizer_graph.step()
        self.optimizer_reliability.step()
        if self.optimizer_sam is not None:
            self.optimizer_sam.step()
        if post_update_labels is not None:
            with torch.no_grad():
                self.graph.update_prototypes(features.detach(), post_update_labels.detach(), post_update_weights.detach())
        self._update_teacher()

        lr_ = self.args.UNet_lr * (1.0 - iter_num / self.args.max_iterations)
        sam_lr_ = self.args.lr * (1.0 - iter_num / self.args.max_iterations)
        for optimizer in (self.optimizer_student, self.optimizer_graph, self.optimizer_reliability):
            for param_group in optimizer.param_groups:
                param_group["lr"] = lr_
        if self.optimizer_sam is not None:
            for param_group in self.optimizer_sam.param_groups:
                param_group["lr"] = sam_lr_

        r_mean = float(R_all.mean().detach().cpu())
        r_std = float(R_all.std(unbiased=False).detach().cpu())
        rhat_mean = float(R_hat_all.mean().detach().cpu())
        rhat_std = float(R_hat_all.std(unbiased=False).detach().cpu())
        rb_mean = float(R_b.mean().detach().cpu())
        ru_mean = float(R_u.mean().detach().cpu())
        weight_mean = float(pseudo_weight.mean().detach().cpu())
        adjacency_norm = float(R_ij.norm().detach().cpu())
        student_mean, student_std = self._tensor_mean_std(student_logits)
        graph_mean, graph_std = self._tensor_mean_std(graph_logits_all)
        sam_mean, sam_std = self._tensor_mean_std(sam_logits)
        deploy_mean, deploy_std = self._tensor_mean_std(deploy_logits)
        logging.info("R mean/std: %f %f", r_mean, r_std)
        logging.info("R_hat mean/std: %f %f", rhat_mean, rhat_std)
        logging.info("R_b mean: %f", rb_mean)
        logging.info("R_u mean: %f", ru_mean)
        logging.info("student logits mean/std: %f %f", student_mean, student_std)
        logging.info("graph logits mean/std: %f %f", graph_mean, graph_std)
        logging.info("sam logits mean/std: %f %f", sam_mean, sam_std)
        logging.info("deploy logits mean/std: %f %f", deploy_mean, deploy_std)
        logging.info("loss_student_sup: %f", student_sup_loss.item())
        logging.info("loss_deploy_sup: %f", loss_deploy_sup.item())
        logging.info("loss_sam_sup: %f", loss_sam_sup.item())
        logging.info("loss_graph_sup: %f", loss_graph_sup.item())
        logging.info("loss_R: %f", loss_R.item())
        logging.info("loss_u: %f", loss_srpc.item())
        logging.info("loss_struct: %f", loss_struct.item())
        logging.info("pseudo weight mean: %f", weight_mean)
        logging.info("graph adjacency norm: %f", adjacency_norm)

        logging.info(
            "iteration %d : loss_total=%f loss_student_sup=%f loss_deploy_sup=%f loss_sam_sup=%f "
            "loss_graph_sup=%f loss_R=%f loss_u=%f loss_struct=%f loss_graph_reg=%f "
            "srpc_fusion=%f srpc_unet=%f srpc_vnet=%f branch_unet=%f branch_vnet=%f lambda_u_eff=%f "
            "fusion_loss=%f unet_loss=%f vnet_loss=%f lr=%10f sam_lr=%10f "
            "R_mean=%f R_std=%f R_hat_mean=%f R_hat_std=%f R_b_mean=%f R_u_mean=%f pseudo_weight_mean=%f graph_adjacency_norm=%f "
            "student_logits_mean=%f student_logits_std=%f graph_logits_mean=%f graph_logits_std=%f "
            "sam_logits_mean=%f sam_logits_std=%f deploy_logits_mean=%f deploy_logits_std=%f",
            iter_num,
            loss_total.item(),
            student_sup_loss.item(),
            loss_deploy_sup.item(),
            loss_sam_sup.item(),
            loss_graph_sup.item(),
            loss_R.item(),
            loss_srpc.item(),
            loss_struct.item(),
            loss_graph_reg.item(),
            loss_srpc_fusion.item(),
            loss_srpc_unet.item(),
            loss_srpc_vnet.item(),
            branch_unet,
            branch_vnet,
            lambda_u_eff,
            fusion_loss.item(),
            unet_loss.item(),
            vnet_loss.item(),
            lr_,
            sam_lr_,
            r_mean,
            r_std,
            rhat_mean,
            rhat_std,
            rb_mean,
            ru_mean,
            weight_mean,
            adjacency_norm,
            student_mean,
            student_std,
            graph_mean,
            graph_std,
            sam_mean,
            sam_std,
            deploy_mean,
            deploy_std,
        )
        return {
            "loss_total": float(loss_total.detach().cpu()),
            "sup_loss": float(student_sup_loss.detach().cpu()),
            "loss_student_sup": float(student_sup_loss.detach().cpu()),
            "loss_sam_sup": float(loss_sam_sup.detach().cpu()),
            "loss_deploy_sup": float(loss_deploy_sup.detach().cpu()),
            "loss_fuse_sup": float(loss_deploy_sup.detach().cpu()),
            "srpc_loss": float(loss_srpc.detach().cpu()),
            "loss_u_srpc": float(loss_srpc.detach().cpu()),
            "loss_u": float(loss_srpc.detach().cpu()),
            "loss_R": float(loss_R.detach().cpu()),
            "loss_graph_sup": float(loss_graph_sup.detach().cpu()),
            "srpc_fusion": float(loss_srpc_fusion.detach().cpu()),
            "srpc_unet": float(loss_srpc_unet.detach().cpu()),
            "srpc_vnet": float(loss_srpc_vnet.detach().cpu()),
            "branch_unet": float(branch_unet),
            "branch_vnet": float(branch_vnet),
            "lambda_u_eff": float(lambda_u_eff),
            "graph_loss": float(loss_graph_sup.detach().cpu()),
            "loss_struct": float(loss_struct.detach().cpu()),
            "boundary_loss": float(loss_struct.detach().cpu()),
            "R_mean": r_mean,
            "R_std": r_std,
            "R_hat_mean": rhat_mean,
            "R_hat_std": rhat_std,
            "R_b_mean": rb_mean,
            "R_u_mean": ru_mean,
            "pseudo_weight_mean": weight_mean,
            "graph_adjacency_norm": adjacency_norm,
            "deploy_logits_mean": deploy_mean,
            "deploy_logits_std": deploy_std,
            "final_logits_mean": deploy_mean,
            "final_logits_std": deploy_std,
        }

    @torch.no_grad()
    def _val_ca(self, val_loader, snapshot_path, iter_num):
        self.SGDL.eval()
        self.graph.eval()
        self.reliability_head.eval()
        self.teacher.eval()

        avg_dice_deploy = 0.0
        avg_dice_student = 0.0
        avg_dice_corrected = 0.0
        avg_dice_unet = 0.0
        avg_dice_vnet = 0.0
        multiclass_records = {"deploy": [], "student": [], "corrected": [], "unet": [], "vnet": []}
        oracle_records = []
        graph_stat_records = []
        deploy_delta_records = []
        r_corr_records = []

        for sampled_batch in val_loader:
            val_image = sampled_batch["image"].to(self.args.device)
            val_label = sampled_batch["label"].to(self.args.device)
            pred_unet, pred_vnet, pred_unet_soft, pred_vnet_soft, fusion_map, features = self.SGDL(
                val_image, return_features=True
            )
            student_logits = fusion_map
            student_prob = torch.softmax(student_logits, dim=1)
            if getattr(self.args, "ssl_use_ema_teacher", True):
                _, _, _, _, teacher_logits, _ = self.teacher(val_image, return_features=True)
                teacher_prob = torch.softmax(teacher_logits, dim=1)
            else:
                teacher_prob = None
            uncertainty = entropy_map(student_prob)
            disagreement = (teacher_prob - student_prob).abs().mean(dim=1, keepdim=True) if teacher_prob is not None else uncertainty
            residual_logits, adjacency, graph_logs = self._ca_graph_forward(
                features=features,
                student_prob=student_prob,
                teacher_prob=teacher_prob,
                uncertainty=uncertainty,
                disagreement=disagreement,
            )
            R_delta_hat = self._ca_reliability_forward(
                student_prob=student_prob,
                residual_logits=residual_logits,
                teacher_prob=teacher_prob,
                features=features,
            )
            deploy_logits, residual_logits, lambda_res = self._ca_deploy_logits(
                student_logits, residual_logits, R_delta_hat, iter_num
            )
            corrected_logits = student_logits + lambda_res * residual_logits

            deploy_soft = torch.softmax(deploy_logits, dim=1)
            student_soft = torch.softmax(student_logits, dim=1)
            corrected_soft = torch.softmax(corrected_logits, dim=1)

            if self.args.num_classes > 2:
                deploy_metrics = multiclass_segmentation_metrics(val_label, deploy_soft, self.args.num_classes)
                student_metrics = multiclass_segmentation_metrics(val_label, student_soft, self.args.num_classes)
                corrected_metrics = multiclass_segmentation_metrics(val_label, corrected_soft, self.args.num_classes)
                unet_metrics = multiclass_segmentation_metrics(val_label, pred_unet_soft, self.args.num_classes)
                vnet_metrics = multiclass_segmentation_metrics(val_label, pred_vnet_soft, self.args.num_classes)
                avg_dice_deploy += deploy_metrics["avg_dice"]
                avg_dice_student += student_metrics["avg_dice"]
                avg_dice_corrected += corrected_metrics["avg_dice"]
                avg_dice_unet += unet_metrics["avg_dice"]
                avg_dice_vnet += vnet_metrics["avg_dice"]
                multiclass_records["deploy"].append(deploy_metrics)
                multiclass_records["student"].append(student_metrics)
                multiclass_records["corrected"].append(corrected_metrics)
                multiclass_records["unet"].append(unet_metrics)
                multiclass_records["vnet"].append(vnet_metrics)
            else:
                avg_dice_deploy += dice_coef(val_label, deploy_soft, thr=0.5)
                avg_dice_student += dice_coef(val_label, student_soft, thr=0.5)
                avg_dice_corrected += dice_coef(val_label, corrected_soft, thr=0.5)
                avg_dice_unet += dice_coef(val_label, pred_unet_soft, thr=0.5)
                avg_dice_vnet += dice_coef(val_label, pred_vnet_soft, thr=0.5)

            if getattr(self.args, "diagnostic_use_oracle_fusion", True):
                oracle_records.append(
                    compute_oracle_fusion(
                        student_logits=student_logits,
                        corrected_logits=corrected_logits,
                        gt=val_label,
                        r_delta_hat=R_delta_hat,
                        patch_size=getattr(self.args, "diagnostic_oracle_patch_size", getattr(self.args, "reliability_patch_size", 16)),
                        num_classes=self.args.num_classes,
                    )
                )
            graph_stat_records.append(graph_logs)
            deploy_delta_records.append(
                compute_deploy_delta(
                    student_logits, deploy_logits, residual_logits, R_delta_hat, lambda_res,
                    low_thresh=getattr(self.args, "deploy_safe_thresh", 0.3),
                )
            )

        n = max(len(val_loader), 1)
        avg_dice_deploy /= n
        avg_dice_student /= n
        avg_dice_corrected /= n
        avg_dice_unet /= n
        avg_dice_vnet /= n

        logging.info(
            "iteration %d : deploy_mean_dice : %f student_mean_dice : %f corrected_mean_dice : %f "
            "unet_mean_dice : %f vnet_mean_dice : %f",
            iter_num,
            avg_dice_deploy,
            avg_dice_student,
            avg_dice_corrected,
            avg_dice_unet,
            avg_dice_vnet,
        )
        val_summary = {
            "iteration": int(iter_num),
            "deploy_mean_dice": float(avg_dice_deploy),
            "student_mean_dice": float(avg_dice_student),
            "corrected_mean_dice": float(avg_dice_corrected),
            "unet_mean_dice": float(avg_dice_unet),
            "vnet_mean_dice": float(avg_dice_vnet),
        }
        if self.args.num_classes > 2:
            for model_name, records in multiclass_records.items():
                if not records:
                    continue
                avg_record = {}
                for key in records[0].keys():
                    values = np.asarray([record[key] for record in records], dtype=np.float32)
                    finite_values = values[np.isfinite(values)]
                    avg_record[key] = float(finite_values.mean()) if finite_values.size > 0 else 0.0
                class_parts = []
                for class_idx in range(1, self.args.num_classes):
                    class_dice = avg_record[f"class_{class_idx}_dice"]
                    class_iou = avg_record[f"class_{class_idx}_iou"]
                    class_hd95 = avg_record[f"class_{class_idx}_hd95"]
                    class_parts.append(
                        "class_%d_dice=%.6f class_%d_iou=%.6f class_%d_hd95=%.6f"
                        % (
                            class_idx,
                            class_dice,
                            class_idx,
                            class_iou,
                            class_idx,
                            class_hd95,
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
                for key, value in avg_record.items():
                    val_summary[f"{model_name}_{key}"] = value

        def _avg_record(records, key):
            return float(np.nanmean([record.get(key, 0.0) for record in records])) if records else 0.0

        diag_keys = [
            "oracle_fusion_avg_dice",
            "oracle_gain_over_student",
            "corrected_better_patch_ratio",
            "corrected_worse_patch_ratio",
            "corrected_equal_patch_ratio",
            "graph_or_residual_better_region_ratio",
            "graph_or_residual_worse_region_ratio",
            "graph_or_residual_neutral_region_ratio",
            "R_delta_improvement_corr",
        ]
        for key in diag_keys:
            val_summary[key] = _avg_record(oracle_records, key)
        graph_keys = [
            "graph_adjacency_norm",
            "A_diag_mean",
            "A_offdiag_mean",
            "A_offdiag_std",
            "A_entropy",
            "A_dynamic_delta_to_identity",
            "residual_logits_abs_mean",
        ]
        for key in graph_keys:
            val_summary[key] = _avg_record(graph_stat_records, key)
        deploy_keys = [
            "lambda_res_current",
            "deploy_student_delta",
            "gated_residual_abs_mean",
            "low_reliability_ratio",
            "high_reliability_ratio",
        ]
        for key in deploy_keys:
            val_summary[key] = _avg_record(deploy_delta_records, key)

        logging.info(
            "iteration %d : oracle_fusion_avg_dice=%.6f oracle_gain_over_student=%.6f "
            "corrected_better_patch_ratio=%.6f corrected_worse_patch_ratio=%.6f "
            "corrected_equal_patch_ratio=%.6f R_delta_improvement_corr=%.6f "
            "graph_adjacency_norm=%.6f A_diag_mean=%.6f A_offdiag_mean=%.6f A_offdiag_std=%.6f "
            "A_entropy=%.6f A_dynamic_delta_to_identity=%.6f deploy_student_delta=%.6f",
            iter_num,
            val_summary["oracle_fusion_avg_dice"],
            val_summary["oracle_gain_over_student"],
            val_summary["corrected_better_patch_ratio"],
            val_summary["corrected_worse_patch_ratio"],
            val_summary["corrected_equal_patch_ratio"],
            val_summary["R_delta_improvement_corr"],
            val_summary["graph_adjacency_norm"],
            val_summary["A_diag_mean"],
            val_summary["A_offdiag_mean"],
            val_summary["A_offdiag_std"],
            val_summary["A_entropy"],
            val_summary["A_dynamic_delta_to_identity"],
            val_summary["deploy_student_delta"],
        )

        if avg_dice_deploy > self.best_performance_final:
            self.best_performance_final = avg_dice_deploy
            torch.save(self.SGDL.state_dict(), os.path.join(snapshot_path, "CA_SRG_SAMPP_deploy_SGDL_best_model.pth"))
            torch.save(self.graph.state_dict(), os.path.join(snapshot_path, "CA_SRG_SAMPP_deploy_graph_best_model.pth"))
            torch.save(
                self.reliability_head.state_dict(),
                os.path.join(snapshot_path, "CA_SRG_SAMPP_deploy_reliability_best_model.pth"),
            )
            torch.save(self.SGDL.state_dict(), os.path.join(snapshot_path, "best_deploy_model.pth"))

        if avg_dice_student > self.best_performance_SGDL:
            self.best_performance_SGDL = avg_dice_student
            torch.save(self.SGDL.state_dict(), os.path.join(snapshot_path, "best_student_model.pth"))
            torch.save(self.SGDL.state_dict(), os.path.join(snapshot_path, "SGDL_best_model.pth"))

        if val_summary["oracle_fusion_avg_dice"] > self.best_oracle_diagnostic_value:
            self.best_oracle_diagnostic_value = val_summary["oracle_fusion_avg_dice"]
            logging.info("iteration %d : best_oracle_diagnostic_value=%.6f", iter_num, self.best_oracle_diagnostic_value)

        self.SGDL.train()
        self.graph.train()
        self.reliability_head.train()
        return val_summary

    @torch.no_grad()
    def val(self, val_loader, snapshot_path, iter_num):
        if self.use_ca_srg:
            return self._val_ca(val_loader, snapshot_path, iter_num)

        self.SGDL.eval()
        self.graph.eval()
        self.reliability_head.eval()
        if self.sam_model is not None:
            self.sam_model.eval()
        if self.sam_struct is not None:
            self.sam_struct.eval()

        avg_dice_deploy = 0.0
        avg_dice_student = 0.0
        avg_dice_graph = 0.0
        avg_dice_full = 0.0
        avg_dice_unet = 0.0
        avg_dice_vnet = 0.0
        multiclass_records = {"deploy": [], "student": [], "graph": [], "unet": [], "vnet": []}
        eval_full = getattr(self.args, "eval_full_sam_assisted", False)
        if eval_full:
            multiclass_records["full_sam_assisted"] = []

        for sampled_batch in val_loader:
            val_image, val_label = sampled_batch["image"].to(self.args.device), sampled_batch["label"].to(self.args.device)
            pred_unet, pred_vnet, pred_unet_soft, pred_vnet_soft, fusion_map, features = self.SGDL(
                val_image, return_features=True
            )
            graph_adj = self.graph.last_adjacency if self.graph.last_adjacency is not None else self.graph.R_prior.to(val_image.device)
            graph_logits = self.graph(features, graph_adj.to(val_image.device))
            R_hat = self.reliability_head(features, out_size=fusion_map.shape[-2:])
            deploy_logits = self._deploy_logits(fusion_map, graph_logits, R_hat)

            deploy_soft = torch.softmax(deploy_logits, dim=1)
            student_soft = torch.softmax(fusion_map, dim=1)
            graph_soft = torch.softmax(graph_logits, dim=1)
            full_soft = None
            if eval_full and self.sam_model is not None:
                sam_logits = self._sam_semantic_logits(val_image, fusion_map)
                if getattr(self.args, "eval_with_ssrf", False) and self.sam_struct is not None:
                    prompts = build_structure_prompts(
                        val_image,
                        student_prob=student_soft.detach(),
                        num_prompts=self.args.srg_prompt_count,
                    )
                    R_b_val, _, R_u_val = compute_structure_reliability(
                        val_image,
                        {
                            "sam_model": self.sam_struct,
                            "items": prompts,
                            "affinity_size": self.args.srg_affinity_size,
                        },
                    )
                    R_val = self._structure_gate(R_b_val, R_u_val, fusion_map.shape[-2:])
                else:
                    R_val = fusion_map.new_ones(val_image.shape[0], 1, *fusion_map.shape[-2:])
                full_logits = self._fuse_logits(fusion_map, graph_logits, sam_logits, R_val)
                full_soft = torch.softmax(full_logits, dim=1)

            if self.args.num_classes > 2:
                deploy_metrics = multiclass_segmentation_metrics(val_label, deploy_soft, self.args.num_classes)
                student_metrics = multiclass_segmentation_metrics(val_label, student_soft, self.args.num_classes)
                graph_metrics = multiclass_segmentation_metrics(val_label, graph_soft, self.args.num_classes)
                unet_metrics = multiclass_segmentation_metrics(val_label, pred_unet_soft, self.args.num_classes)
                vnet_metrics = multiclass_segmentation_metrics(val_label, pred_vnet_soft, self.args.num_classes)
                dice_deploy = deploy_metrics["avg_dice"]
                dice_student = student_metrics["avg_dice"]
                dice_graph = graph_metrics["avg_dice"]
                dice_unet = unet_metrics["avg_dice"]
                dice_vnet = vnet_metrics["avg_dice"]
                multiclass_records["deploy"].append(deploy_metrics)
                multiclass_records["student"].append(student_metrics)
                multiclass_records["graph"].append(graph_metrics)
                multiclass_records["unet"].append(unet_metrics)
                multiclass_records["vnet"].append(vnet_metrics)
                if full_soft is not None:
                    full_metrics = multiclass_segmentation_metrics(val_label, full_soft, self.args.num_classes)
                    dice_full = full_metrics["avg_dice"]
                    multiclass_records["full_sam_assisted"].append(full_metrics)
                else:
                    dice_full = 0.0
            else:
                dice_deploy = dice_coef(val_label, deploy_soft, thr=0.5)
                dice_student = dice_coef(val_label, student_soft, thr=0.5)
                dice_graph = dice_coef(val_label, graph_soft, thr=0.5)
                dice_full = dice_coef(val_label, full_soft, thr=0.5) if full_soft is not None else 0.0
                dice_unet = dice_coef(val_label, pred_unet_soft, thr=0.5)
                dice_vnet = dice_coef(val_label, pred_vnet_soft, thr=0.5)
            avg_dice_deploy += dice_deploy
            avg_dice_student += dice_student
            avg_dice_graph += dice_graph
            avg_dice_full += dice_full
            avg_dice_unet += dice_unet
            avg_dice_vnet += dice_vnet

        avg_dice_deploy = avg_dice_deploy / len(val_loader)
        avg_dice_student = avg_dice_student / len(val_loader)
        avg_dice_graph = avg_dice_graph / len(val_loader)
        avg_dice_full = avg_dice_full / len(val_loader)
        avg_dice_unet = avg_dice_unet / len(val_loader)
        avg_dice_vnet = avg_dice_vnet / len(val_loader)

        logging.info(
            "iteration %d : deploy_mean_dice : %f student_mean_dice : %f graph_mean_dice : %f "
            "full_sam_assisted_mean_dice : %f unet_mean_dice : %f vnet_mean_dice : %f",
            iter_num,
            avg_dice_deploy,
            avg_dice_student,
            avg_dice_graph,
            avg_dice_full,
            avg_dice_unet,
            avg_dice_vnet,
        )
        if self.args.num_classes > 2:
            for model_name, records in multiclass_records.items():
                if not records:
                    continue
                avg_record = {key: float(np.nanmean([record[key] for record in records])) for key in records[0].keys()}
                class_parts = []
                for class_idx in range(1, self.args.num_classes):
                    class_parts.append(
                        "class_%d_dice=%.6f class_%d_iou=%.6f class_%d_hd95=%.6f"
                        % (
                            class_idx,
                            avg_record[f"class_{class_idx}_dice"],
                            class_idx,
                            avg_record[f"class_{class_idx}_iou"],
                            class_idx,
                            avg_record[f"class_{class_idx}_hd95"],
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

        if avg_dice_deploy > self.best_performance_final:
            self.best_performance_final = avg_dice_deploy
            torch.save(self.SGDL.state_dict(), os.path.join(snapshot_path, "SRG_SAM_Lite_deploy_SGDL_best_model.pth"))
            torch.save(self.graph.state_dict(), os.path.join(snapshot_path, "SRG_SAM_Lite_deploy_DPG_best_model.pth"))
            torch.save(
                self.reliability_head.state_dict(),
                os.path.join(snapshot_path, "SRG_SAM_Lite_deploy_reliability_best_model.pth"),
            )

        if avg_dice_student > self.best_performance_SGDL:
            self.best_performance_SGDL = avg_dice_student
            torch.save(self.SGDL.state_dict(), os.path.join(snapshot_path, "SGDL_best_model.pth"))
            torch.save(self.graph.state_dict(), os.path.join(snapshot_path, "DPG_best_model.pth"))
        if eval_full and avg_dice_full > self.best_performance_sam and self.sam_model is not None:
            self.best_performance_sam = avg_dice_full
            torch.save(self.sam_model.state_dict(), os.path.join(snapshot_path, "sam_adapter_best_model.pth"))

        self.SGDL.train()
        self.graph.train()
        self.reliability_head.train()
        if self.sam_model is not None and getattr(self.args, "use_adapter_sam_semantic", True):
            self.sam_model.train()

    def val_ACDC(self, val_loader, snapshot_path, iter_num):
        self.val(val_loader, snapshot_path, iter_num)
