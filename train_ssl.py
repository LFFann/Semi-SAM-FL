import argparse
import logging
import os
import random
import shutil
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataloader.TwoStreamBatchSampler import TwoStreamBatchSampler
from dataloader.dataset import build_Dataset
from dataloader.transforms import build_weak_strong_transforms
from trainer import Trainer
from utils.utils import patients_to_slices


def str2bool(value):
    if isinstance(value, bool):
        return value
    value = value.lower()
    if value in ("yes", "true", "t", "1", "y"):
        return True
    if value in ("no", "false", "f", "0", "n"):
        return False
    raise argparse.ArgumentTypeError("Boolean value expected.")


parser = argparse.ArgumentParser()
parser.add_argument("--data_path", type=str, default="./SampleData", help="dataset root")
parser.add_argument("--dataset", type=str, default="/260513_data_multiclass", help="dataset folder under data_path")
parser.add_argument("--labeled_num", type=int, default=1, help="fallback labeled split preset")
parser.add_argument("--num_classes", type=int, default=3, help="number of segmentation classes")
parser.add_argument("--in_channels", type=int, default=3, help="input image channels")

parser.add_argument("-UNet_lr", type=float, default=0.003, help="student and DPG learning rate")
parser.add_argument("-lr", type=float, default=1e-4, help="Adapter-SAM learning rate")
parser.add_argument("-VNet_lr", type=float, default=0.01, help="kept for CLI compatibility")
parser.add_argument("--image_size", type=int, default=256, help="input size")
parser.add_argument("--batch_size", type=int, default=24, help="batch size")
parser.add_argument("--labeled_bs", type=int, default=12, help="labeled samples per batch")
parser.add_argument("--num_workers", type=int, default=2, help="train dataloader workers")
parser.add_argument("--val_num_workers", type=int, default=1, help="validation dataloader workers")
parser.add_argument("--seed", type=int, default=2024, help="random seed")
parser.add_argument("--max_iterations", type=int, default=50000, help="maximum iterations")
parser.add_argument("--mixed_iterations", type=int, default=0, help="kept for CLI compatibility; unused by SRG-SAM")
parser.add_argument("--val_interval", type=int, default=200, help="validation interval")
parser.add_argument("--snapshot_path", type=str, default="", help="output path")
parser.add_argument("--n_fold", type=int, default=1, help="number of folds")

parser.add_argument("--mod", type=str, default="sam_adpt", help="SAM image encoder mode")
parser.add_argument("--model_type", type=str, default="vit_b", help="SAM type")
parser.add_argument("-thd", type=bool, default=False, help="kept for SAM compatibility")
parser.add_argument("--point_nums", type=int, default=5, help="kept for SAM compatibility")
parser.add_argument("--box_nums", type=int, default=1, help="kept for SAM compatibility")
parser.add_argument("--multimask", type=bool, default=False, help="SAM multimask output")
parser.add_argument("--encoder_adapter", type=bool, default=True, help="kept for SAM compatibility")
parser.add_argument("--sam_checkpoint", type=str, default="./sam_vit_b_01ec64.pth", help="SAM checkpoint")
parser.add_argument("--device", type=str, default="cuda", help="training device")
parser.add_argument("--sgdl_init_checkpoint", type=str, default="", help="optional SGDL checkpoint for fine-tuning")
parser.add_argument("--graph_init_checkpoint", type=str, default="", help="optional DPG checkpoint for fine-tuning")

parser.add_argument("--lambda_u", type=float, default=1.0, help="SRPC unsupervised loss weight")
parser.add_argument("--lambda_branch", type=float, default=0.3, help="SRPC weight for UNet/VNet auxiliary branches")
parser.add_argument("--lambda_branch_unet", type=float, default=-1.0, help="override SRPC weight for UNet branch")
parser.add_argument("--lambda_branch_vnet", type=float, default=-1.0, help="override SRPC weight for VNet branch")
parser.add_argument("--lambda_u_warmup", type=int, default=500, help="iterations to ramp SRPC from zero when training from scratch")
parser.add_argument("--lambda_g", type=float, default=0.1, help="backward-compatible alias for --lambda_graph")
parser.add_argument("--lambda_b", type=float, default=0.05, help="backward-compatible alias for --lambda_struct")
parser.add_argument("--lambda_sam", type=float, default=0.5, help="Adapter-SAM supervised loss weight")
parser.add_argument("--lambda_fuse", type=float, default=1.0, help="final-fusion supervised loss weight")
parser.add_argument("--lambda_deploy", type=float, default=1.0, help="deploy branch supervised loss weight")
parser.add_argument("--lambda_graph", type=float, default=None, help="DPG graph supervised loss weight")
parser.add_argument("--lambda_struct", type=float, default=None, help="structure/boundary loss weight")
parser.add_argument("--lambda_R", type=float, default=0.2, help="Reliability Head distillation loss weight")
parser.add_argument("--lambda_graph_reg", type=float, default=0.01, help="DPG adjacency regularization loss weight")
parser.add_argument("--lambda_aff", type=float, default=0.1, help="reserved affinity structure loss weight")
parser.add_argument("--lambda_pg", type=float, default=0.5, help="graph posterior exponent in SRG-SAM++ SRPC")
parser.add_argument("--lambda_pa", type=float, default=0.5, help="Adapter-SAM posterior exponent in SRG-SAM++ SRPC")
parser.add_argument("--class_weights", type=str, default="", help="comma-separated CE/Dice class weights, e.g. 0.2,1.6,1.2")
parser.add_argument("--exclude_bg_dice", action="store_true", help="exclude background from supervised Dice loss")
parser.add_argument("--srg_graph_lambda", type=float, default=0.5, help="graph posterior weight in SRPC")
parser.add_argument("--srg_graph_alpha", type=float, default=0.5, help="prior/SAM dynamic adjacency blend")
parser.add_argument("--srg_prompt_count", type=int, default=4, help="number of SSRF prompts per unlabeled image")
parser.add_argument("--srg_affinity_size", type=int, default=16, help="side length for compact R_a affinity")
parser.add_argument("--srg_ema_decay", type=float, default=0.99, help="EMA teacher decay")
parser.add_argument("--use_srg_sam_pp", type=str2bool, nargs="?", const=True, default=True, help="enable SRG-SAM++ flow")
parser.add_argument("--use_srg_sam_lite", type=str2bool, nargs="?", const=True, default=True, help="enable SAM-free Lite deploy branch")
parser.add_argument("--use_train_only_sam", type=str2bool, nargs="?", const=True, default=True, help="allow SAM only during training")
parser.add_argument("--inference_without_sam", type=str2bool, nargs="?", const=True, default=True, help="validation/prediction default to no-SAM deploy path")
parser.add_argument("--use_frozen_sam_struct", type=str2bool, nargs="?", const=True, default=True, help="enable frozen SAM SSRF branch")
parser.add_argument("--use_adapter_sam_semantic", type=str2bool, nargs="?", const=True, default=True, help="enable Adapter-SAM semantic logits branch")
parser.add_argument("--use_dpg_logits_fusion", type=str2bool, nargs="?", const=True, default=True, help="fuse DPG graph logits into final logits")
parser.add_argument("--use_dpg_deploy", type=str2bool, nargs="?", const=True, default=True, help="use DPG logits in Lite deploy branch")
parser.add_argument("--use_reliability_head", type=str2bool, nargs="?", const=True, default=True, help="use R_hat in Lite deploy branch")
parser.add_argument("--use_fusion", type=str2bool, nargs="?", const=True, default=True, help="use reliability-gated final fusion; false means student-only")
parser.add_argument("--use_reliability_gate", type=str2bool, nargs="?", const=True, default=True, help="gate graph/SAM logits with SSRF reliability")
parser.add_argument("--alpha_graph", type=float, default=0.3, help="DPG logits fusion scale")
parser.add_argument("--beta_sam", type=float, default=0.3, help="Adapter-SAM logits fusion scale")
parser.add_argument("--eval_with_ssrf", type=str2bool, nargs="?", const=True, default=False, help="compute SSRF gate during validation")
parser.add_argument("--eval_full_sam_assisted", type=str2bool, nargs="?", const=True, default=False, help="optional SAM-assisted upper-bound validation")

# Complementarity-aware SRG-SAM++ controls. Defaults preserve the existing SRG-SAM++-Lite path.
parser.add_argument("--use_ca_srg_sampp", type=str2bool, nargs="?", const=True, default=False, help="enable CA_SRG_SAMPP path")
parser.add_argument("--train_stage", type=str, default="joint", choices=["student", "residual", "joint"], help="TRAIN.STAGE")
parser.add_argument("--freeze_student_in_residual", type=str2bool, nargs="?", const=True, default=True)
parser.add_argument("--student_lr_scale_in_residual", type=float, default=0.1)

parser.add_argument("--ssl_use_ema_teacher", type=str2bool, nargs="?", const=True, default=True)
parser.add_argument("--ssl_ema_decay", type=float, default=0.99)
parser.add_argument("--ssl_ema_update_after", type=int, default=0)
parser.add_argument("--ssl_ema_update_every", type=int, default=1)
parser.add_argument("--ssl_use_weak_strong", type=str2bool, nargs="?", const=True, default=True)
parser.add_argument("--ssl_conf_thresh", type=float, default=0.75)
parser.add_argument("--ssl_unsup_weight", type=float, default=1.0)
parser.add_argument("--ssl_unsup_rampup_iters", type=int, default=1000)
parser.add_argument("--ssl_use_view_consistency", type=str2bool, nargs="?", const=True, default=True)
parser.add_argument("--ssl_view_cons_weight", type=float, default=0.1)
parser.add_argument("--ssl_view_cons_rampup_iters", type=int, default=1000)
parser.add_argument("--ssl_use_copy_paste", type=str2bool, nargs="?", const=True, default=False)
parser.add_argument("--ssl_copy_paste_prob", type=float, default=0.5)
parser.add_argument("--ssl_copy_paste_min_conf", type=float, default=0.8)

parser.add_argument("--reliability_type", type=str, default="old_R", choices=["old_R", "complementarity", "none"])
parser.add_argument("--reliability_target_type", type=str, default="old_R", choices=["old_R", "delta_utility"])
parser.add_argument("--reliability_patch_size", type=int, default=16)
parser.add_argument("--reliability_utility_margin", type=float, default=0.005)
parser.add_argument("--reliability_use_unlabeled_target", type=str2bool, nargs="?", const=True, default=True)
parser.add_argument("--reliability_teacher_conf_thresh", type=float, default=0.8)
parser.add_argument("--reliability_ignore_neutral", type=str2bool, nargs="?", const=True, default=True)
parser.add_argument("--reliability_use_rank_loss", type=str2bool, nargs="?", const=True, default=True)
parser.add_argument("--reliability_rank_margin", type=float, default=0.1)
parser.add_argument("--reliability_bce_weight", type=float, default=1.0)
parser.add_argument("--reliability_rank_weight", type=float, default=0.1)
parser.add_argument("--reliability_old_R_loss_weight", type=float, default=0.0)

parser.add_argument("--graph_type", type=str, default="legacy", choices=["legacy", "class", "region_boundary_prototype", "none"])
parser.add_argument("--graph_num_prototypes", type=int, default=5)
parser.add_argument("--graph_detach_region_mask", type=str2bool, nargs="?", const=True, default=True)
parser.add_argument("--graph_use_dynamic_adj", type=str2bool, nargs="?", const=True, default=True)
parser.add_argument("--graph_use_prior_adj", type=str2bool, nargs="?", const=True, default=False)
parser.add_argument("--graph_prior_weight", type=float, default=0.1)
parser.add_argument("--graph_adj_temperature", type=float, default=0.5)
parser.add_argument("--graph_residual_init_zero", type=str2bool, nargs="?", const=True, default=True)
parser.add_argument("--graph_use_anti_collapse_loss", type=str2bool, nargs="?", const=True, default=True)
parser.add_argument("--graph_min_entropy", type=float, default=0.3)
parser.add_argument("--graph_anti_collapse_weight", type=float, default=0.01)

parser.add_argument("--deploy_fusion_type", type=str, default="weighted_sum", choices=["weighted_sum", "gated_residual"])
parser.add_argument("--deploy_lambda_res_max", type=float, default=0.3)
parser.add_argument("--deploy_lambda_res_rampup_iters", type=int, default=1000)
parser.add_argument("--deploy_clamp_residual", type=str2bool, nargs="?", const=True, default=True)
parser.add_argument("--deploy_residual_clamp_value", type=float, default=3.0)
parser.add_argument("--deploy_use_safe_loss", type=str2bool, nargs="?", const=True, default=True)
parser.add_argument("--deploy_safe_thresh", type=float, default=0.3)
parser.add_argument("--deploy_safe_loss_weight", type=float, default=0.05)
parser.add_argument("--deploy_use_residual_sparse_loss", type=str2bool, nargs="?", const=True, default=True)
parser.add_argument("--deploy_sparse_weight", type=float, default=0.001)

parser.add_argument("--diagnostic_use_oracle_fusion", type=str2bool, nargs="?", const=True, default=True)
parser.add_argument("--diagnostic_oracle_patch_size", type=int, default=16)
parser.add_argument("--diagnostic_log_graph_stats", type=str2bool, nargs="?", const=True, default=True)
parser.add_argument("--diagnostic_log_reliability_corr", type=str2bool, nargs="?", const=True, default=True)
parser.add_argument("--diagnostic_log_deploy_delta", type=str2bool, nargs="?", const=True, default=True)

parser.add_argument("--loss_w_sup_student", type=float, default=1.0)
parser.add_argument("--loss_w_sup_deploy", type=float, default=1.0)
parser.add_argument("--loss_w_unsup", type=float, default=1.0)
parser.add_argument("--loss_w_view_cons", type=float, default=0.1)
parser.add_argument("--loss_w_r_delta_bce", type=float, default=1.0)
parser.add_argument("--loss_w_r_delta_rank", type=float, default=0.1)
parser.add_argument("--loss_w_safe", type=float, default=0.05)
parser.add_argument("--loss_w_residual_sparse", type=float, default=0.001)
parser.add_argument("--loss_w_graph_anti_collapse", type=float, default=0.01)

# Deprecated KnowSAM SSL knobs are accepted so old scripts do not fail, but are not used.
parser.add_argument("--consistency", type=float, default=0.0, help=argparse.SUPPRESS)
parser.add_argument("--consistency_rampup", type=float, default=0.0, help=argparse.SUPPRESS)

args = parser.parse_args()
if args.lambda_graph is None:
    args.lambda_graph = args.lambda_g
if args.lambda_struct is None:
    args.lambda_struct = args.lambda_b
if args.ssl_unsup_weight != 1.0:
    args.loss_w_unsup = args.ssl_unsup_weight
if args.ssl_view_cons_weight != 0.1:
    args.loss_w_view_cons = args.ssl_view_cons_weight
if args.reliability_bce_weight != 1.0:
    args.loss_w_r_delta_bce = args.reliability_bce_weight
if args.reliability_rank_weight != 0.1:
    args.loss_w_r_delta_rank = args.reliability_rank_weight
if args.deploy_safe_loss_weight != 0.05:
    args.loss_w_safe = args.deploy_safe_loss_weight
if args.deploy_sparse_weight != 0.001:
    args.loss_w_residual_sparse = args.deploy_sparse_weight
if args.graph_anti_collapse_weight != 0.01:
    args.loss_w_graph_anti_collapse = args.graph_anti_collapse_weight


class TeeStream:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for stream in self.streams:
            stream.write(data)
            stream.flush()

    def flush(self):
        for stream in self.streams:
            stream.flush()


class ConsoleLogFilter(logging.Filter):
    def filter(self, record):
        message = record.getMessage()
        if "R mean/std:" in message:
            return False
        if "R_hat mean/std:" in message:
            return False
        if "R_b mean:" in message:
            return False
        if "R_u mean:" in message:
            return False
        if "student logits" in message:
            return False
        if "graph logits" in message:
            return False
        if "sam logits" in message:
            return False
        if "deploy logits" in message:
            return False
        if "final logits" in message:
            return False
        if "loss_student_sup:" in message:
            return False
        if "loss_deploy_sup:" in message:
            return False
        if "loss_sam_sup:" in message:
            return False
        if "loss_graph_sup:" in message:
            return False
        if "loss_R:" in message:
            return False
        if "loss_u:" in message:
            return False
        if "loss_fuse_sup:" in message:
            return False
        if "loss_u_srpc:" in message:
            return False
        if "loss_graph:" in message:
            return False
        if "loss_struct:" in message:
            return False
        if "pseudo weight mean:" in message:
            return False
        if "graph adjacency norm:" in message:
            return False
        if "loss_total=" in message:
            return False
        return True


def setup_run_logging(snapshot_path):
    os.makedirs(snapshot_path, exist_ok=True)
    log_path = os.path.join(snapshot_path, "log.txt")
    terminal_path = os.path.join(snapshot_path, "terminal_output.txt")

    terminal_file = open(terminal_path, "a", encoding="utf-8", buffering=1)
    sys.stdout = TeeStream(sys.__stdout__, terminal_file)
    sys.stderr = TeeStream(sys.__stderr__, terminal_file)

    root_logger = logging.getLogger()
    for handler in list(root_logger.handlers):
        root_logger.removeHandler(handler)
        handler.close()
    root_logger.setLevel(logging.INFO)

    formatter = logging.Formatter(
        "[%(asctime)s.%(msecs)03d] %(message)s",
        datefmt="%H:%M:%S",
    )
    file_handler = logging.FileHandler(log_path, mode="a", encoding="utf-8")
    file_handler.setFormatter(formatter)
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    stream_handler.addFilter(ConsoleLogFilter())
    root_logger.addHandler(file_handler)
    root_logger.addHandler(stream_handler)

    logging.info("Logging to %s", os.path.abspath(log_path))
    logging.info("Terminal output tee to %s", os.path.abspath(terminal_path))
    return log_path, terminal_path


def worker_init_fn(worker_id):
    random.seed(args.seed + worker_id)


def train(args, snapshot_path):
    trainer = Trainer(args)
    data_transforms = build_weak_strong_transforms(args)
    train_dataset = build_Dataset(
        args=args,
        data_dir=args.data_path + args.dataset,
        split="train_semi",
        transform=data_transforms,
    )
    val_dataset = build_Dataset(
        args=args,
        data_dir=args.data_path + args.dataset,
        split="val",
        transform=data_transforms["valid_test"],
    )

    total_slices = len(train_dataset)
    if hasattr(train_dataset, "sample_list_labeled") and hasattr(train_dataset, "sample_list_unlabeled"):
        labeled_slice = len(train_dataset.sample_list_labeled)
        unlabeled_slice = len(train_dataset.sample_list_unlabeled)
        logging.info("Using dataset split counts: %d labeled, %d unlabeled", labeled_slice, unlabeled_slice)
    else:
        labeled_slice = patients_to_slices(args.dataset, args.labeled_num)
        unlabeled_slice = total_slices - labeled_slice
        logging.info("Using preset split counts: %d labeled, %d unlabeled", labeled_slice, unlabeled_slice)

    labeled_idxs = list(range(0, labeled_slice))
    unlabeled_idxs = list(range(labeled_slice, total_slices))
    if not labeled_idxs or not unlabeled_idxs:
        raise ValueError("SRG-SAM requires both labeled and unlabeled samples in train_semi split.")
    if args.batch_size <= args.labeled_bs:
        raise ValueError("batch_size must be larger than labeled_bs.")

    batch_sampler = TwoStreamBatchSampler(
        labeled_idxs,
        unlabeled_idxs,
        args.batch_size,
        args.batch_size - args.labeled_bs,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_sampler=batch_sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        worker_init_fn=worker_init_fn,
    )
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=args.val_num_workers)

    logging.info("%d iterations per epoch", len(train_loader))
    max_epoch = args.max_iterations // len(train_loader) + 1
    iter_num = 0
    progress = tqdm(total=args.max_iterations, ncols=140, dynamic_ncols=True)
    for _ in range(max_epoch):
        for sampled_batch in train_loader:
            volume_batch = sampled_batch["image"].to(args.device)
            label_batch = sampled_batch["label"].to(args.device)
            train_stats = trainer.train(volume_batch, label_batch, iter_num)
            iter_num += 1
            progress.update(1)
            progress.set_postfix(
                {
                    "loss": f"{train_stats['loss_total']:.4f}",
                    "sup": f"{train_stats['sup_loss']:.4f}",
                    "dep": f"{train_stats['loss_deploy_sup']:.4f}",
                    "sam": f"{train_stats['loss_sam_sup']:.4f}",
                    "u": f"{train_stats['loss_u']:.4f}",
                    "Rloss": f"{train_stats['loss_R']:.4f}",
                    "R": f"{train_stats['R_mean']:.4f}",
                    "Rh": f"{train_stats['R_hat_mean']:.4f}",
                    "w": f"{train_stats['pseudo_weight_mean']:.4f}",
                    "adj": f"{train_stats['graph_adjacency_norm']:.4f}",
                }
            )
            if args.val_interval > 0 and iter_num % args.val_interval == 0:
                progress.write("")
                trainer.val(val_loader, snapshot_path, iter_num)
            if iter_num >= args.max_iterations:
                progress.close()
                return
    progress.close()


if __name__ == "__main__":
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)

    for fold in range(args.n_fold):
        if args.snapshot_path:
            snapshot_path = args.snapshot_path
            if args.n_fold > 1:
                snapshot_path = os.path.join(snapshot_path, "fold_" + str(fold))
        else:
            snapshot_path = os.path.join("./Results", "SRG-SAM", "fold_" + str(fold))

        os.makedirs(os.path.join(snapshot_path, "code"), exist_ok=True)
        for file_name in [
            "train_ssl.py",
            "trainer.py",
            "models/ssrf.py",
            "models/dynamic_graph.py",
            "models/reliability_head.py",
            "models/complementarity_reliability.py",
            "models/region_boundary_graph.py",
            "losses/srpc_loss.py",
            "Model/model.py",
            "utils/ema.py",
            "utils/ssl_losses.py",
            "utils/diagnostics.py",
        ]:
            dst = os.path.join(snapshot_path, "code", file_name)
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            shutil.copyfile(file_name, dst)

        setup_run_logging(snapshot_path)
        logging.info(str(args))
        train(args, snapshot_path)
