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

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "..", ".."))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)
if REPO_ROOT not in sys.path:
    sys.path.insert(1, REPO_ROOT)

from dataloader.TwoStreamBatchSampler import TwoStreamBatchSampler
from dataloader.dataset import build_Dataset
from dataloader.transforms import build_weak_strong_transforms
from trainer_ra import Trainer
from utils.training_monitor import save_training_artifacts
from utils.utils import patients_to_slices


parser = argparse.ArgumentParser()
parser.add_argument("--data_path", type=str, default="./SampleData")
parser.add_argument("--labeled_num", type=int, default=1)
parser.add_argument("--dataset", type=str, default="/260513_data_label1")
parser.add_argument("--num_classes", type=int, default=2)
parser.add_argument("--in_channels", type=int, default=3)
parser.add_argument("-lr", type=float, default=1e-4)
parser.add_argument("-UNet_lr", type=float, default=0.01)
parser.add_argument("-VNet_lr", type=float, default=0.01)
parser.add_argument("--image_size", type=int, default=256)
parser.add_argument("--point_nums", type=int, default=5)
parser.add_argument("--box_nums", type=int, default=1)
parser.add_argument("--mod", type=str, default="sam_adpt")
parser.add_argument("--model_type", type=str, default="vit_b")
parser.add_argument("-thd", type=bool, default=False)
parser.add_argument("--batch_size", type=int, default=24)
parser.add_argument("--labeled_bs", type=int, default=12)
parser.add_argument("--num_workers", type=int, default=2)
parser.add_argument("--val_num_workers", type=int, default=1)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--mixed_iterations", type=int, default=12000)
parser.add_argument("--max_iterations", type=int, default=50000)
parser.add_argument("--val_interval", type=int, default=200)
parser.add_argument("--snapshot_path", type=str, default="")
parser.add_argument("--n_fold", type=int, default=1)
parser.add_argument("--consistency", type=float, default=0.1)
parser.add_argument("--consistency_rampup", type=float, default=200.0)
parser.add_argument("--device", type=str, default="cuda")
parser.add_argument("--multimask", type=bool, default=False)
parser.add_argument("--encoder_adapter", type=bool, default=True)
parser.add_argument("--sam_checkpoint", type=str, default="./sam_vit_b_01ec64.pth")

parser.add_argument("--ra_enabled", type=int, default=1)
parser.add_argument("--ra_delta_pixels", type=int, default=2)
parser.add_argument("--ra_delta_levels", type=str, default="-4,-2,0,2,4")
parser.add_argument("--ra_tube_radius", type=int, default=5)
parser.add_argument("--ra_intervention_kernel", type=int, default=9)
parser.add_argument("--ra_intervention_strength", type=float, default=0.25)
parser.add_argument("--ra_beta_esr", type=float, default=6.0)
parser.add_argument("--ra_beta_prompt", type=float, default=2.0)
parser.add_argument("--ra_beta_jump", type=float, default=1.0)
parser.add_argument("--ra_esr_mid", type=float, default=0.05)
parser.add_argument("--ra_min_area_change", type=float, default=0.005)
parser.add_argument("--ra_max_area_change", type=float, default=0.40)
parser.add_argument("--ra_saturation_iou", type=float, default=0.995)
parser.add_argument("--ra_prompt_logit", type=float, default=6.0)
parser.add_argument("--ra_disable_esr", type=int, default=0)
parser.add_argument("--ra_disable_prompt", type=int, default=0)
parser.add_argument("--ra_no_intervention", type=int, default=0)
parser.add_argument("--ra_baseline", type=str, default="response_audit", choices=["response_audit", "prompt_ensemble"])
parser.add_argument("--ra_intervention_mode", type=str, default="boundary", choices=["boundary", "random", "interior", "none"])
parser.add_argument("--ra_prompt_high", type=float, default=0.20)
parser.add_argument("--ra_prompt_low", type=float, default=0.08)
parser.add_argument("--ra_esr_high", type=float, default=0.08)
parser.add_argument("--ra_esr_low", type=float, default=0.03)
parser.add_argument("--ra_jump_high", type=float, default=0.25)
parser.add_argument("--ra_min_inside_change", type=float, default=0.002)
parser.add_argument("--ra_max_inside_change", type=float, default=0.20)
parser.add_argument("--ra_max_outside_change", type=float, default=0.002)
parser.add_argument("--ra_enforce_intervention_validity", type=int, default=1)

args = parser.parse_args()


def worker_init_fn(worker_id):
    random.seed(args.seed + worker_id)


def train(args, snapshot_path):
    train_records = []
    val_records = []
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
        logging.info("Using preset split counts from labeled_num=%s: %d labeled, %d unlabeled", str(args.labeled_num), labeled_slice, unlabeled_slice)

    labeled_idxs = list(range(0, labeled_slice))
    unlabeled_idxs = list(range(labeled_slice, total_slices))
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
    stop_training = False

    for _ in tqdm(range(max_epoch), ncols=70):
        for sampled_batch in train_loader:
            volume_batch = sampled_batch["image"].cuda()
            label_batch = sampled_batch["label"].cuda()
            train_records.append(trainer.train(volume_batch, label_batch, iter_num))
            iter_num += 1
            if args.val_interval > 0 and iter_num > 0 and iter_num % args.val_interval == 0:
                if "ACDC" not in args.dataset:
                    val_record = trainer.val(val_loader, snapshot_path, iter_num)
                else:
                    val_record = trainer.val_ACDC(val_loader, snapshot_path, iter_num)
                if val_record is not None:
                    val_records.append(val_record)
                save_training_artifacts(snapshot_path, train_records, val_records, vars(args))
            if iter_num >= args.max_iterations:
                stop_training = True
                break
        if stop_training:
            break
    save_training_artifacts(snapshot_path, train_records, val_records, vars(args))


if __name__ == "__main__":
    for fold in range(args.n_fold):
        torch.autograd.set_detect_anomaly(True)
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)
        torch.backends.cudnn.benchmark = True

        if args.snapshot_path:
            snapshot_path = args.snapshot_path
            if args.n_fold > 1:
                snapshot_path = os.path.join(snapshot_path, "fold_" + str(fold))
        else:
            snapshot_path = "./Results/RA_SAM_SSL/fold_" + str(fold)

        os.makedirs(snapshot_path, exist_ok=True)
        code_dir = os.path.join(snapshot_path, "code")
        if os.path.exists(code_dir):
            shutil.rmtree(code_dir)
        os.makedirs(os.path.join(code_dir, "utils"), exist_ok=True)
        for filename in ["train_semi_SAM_ra.py", "trainer_ra.py", "ra_modules.py"]:
            shutil.copyfile(os.path.join(CURRENT_DIR, filename), os.path.join(code_dir, filename))
        shutil.copyfile(
            os.path.join(CURRENT_DIR, "utils", "losses_ra.py"),
            os.path.join(code_dir, "utils", "losses_ra.py"),
        )

        logging.basicConfig(
            filename=os.path.join(snapshot_path, "log.txt"),
            level=logging.INFO,
            format="[%(asctime)s.%(msecs)03d] %(message)s",
            datefmt="%H:%M:%S",
            force=True,
        )
        logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
        logging.info(str(args))
        train(args, snapshot_path)

