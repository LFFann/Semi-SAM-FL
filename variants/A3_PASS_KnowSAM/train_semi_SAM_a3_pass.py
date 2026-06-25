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
from trainer_a3_pass import Trainer
from utils.training_monitor import save_training_artifacts
from utils.utils import patients_to_slices


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="./SampleData")
    parser.add_argument("--dataset", type=str, default="/260513_data_label1")
    parser.add_argument("--labeled_num", type=int, default=1)
    parser.add_argument("--num_classes", type=int, default=2)
    parser.add_argument("--in_channels", type=int, default=3)
    parser.add_argument("-lr", type=float, default=1e-4, help="compatibility placeholder")
    parser.add_argument("-UNet_lr", type=float, default=0.0025)
    parser.add_argument("-VNet_lr", type=float, default=0.0025, help="compatibility placeholder")
    parser.add_argument("--image_size", type=int, default=256)
    parser.add_argument("--point_nums", type=int, default=5)
    parser.add_argument("--box_nums", type=int, default=1)
    parser.add_argument("--mod", type=str, default="sam_adpt")
    parser.add_argument("--model_type", type=str, default="vit_b")
    parser.add_argument("-thd", type=bool, default=False)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--labeled_bs", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--val_num_workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--mixed_iterations", type=int, default=12000, help="compatibility placeholder")
    parser.add_argument("--max_iterations", type=int, default=50000)
    parser.add_argument("--val_interval", type=int, default=200)
    parser.add_argument("--snapshot_path", type=str, default="")
    parser.add_argument("--n_fold", type=int, default=1)
    parser.add_argument("--consistency", type=float, default=0.1)
    parser.add_argument("--consistency_rampup", type=float, default=200.0)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--multimask", type=bool, default=False)
    parser.add_argument("--encoder_adapter", type=bool, default=True)
    parser.add_argument("--sam_checkpoint", type=str, default="./sam_vit_b_01ec64.pth", help="compatibility placeholder")

    parser.add_argument("--pass_state_size", type=int, default=64)
    parser.add_argument("--pass_state_dim", type=int, default=64)
    parser.add_argument("--pass_base_channels", type=int, default=32)
    parser.add_argument("--pass_state_lr", type=float, default=0.001)
    parser.add_argument("--pass_state_weight", type=float, default=0.20)
    parser.add_argument("--pass_decode_weight", type=float, default=1.00)
    parser.add_argument("--pass_state_consistency_weight", type=float, default=0.20)
    parser.add_argument("--pass_pseudo_weight", type=float, default=0.35)
    parser.add_argument("--pass_decode_consistency_weight", type=float, default=0.10)
    parser.add_argument("--pass_reliability_alpha", type=float, default=4.0)
    parser.add_argument("--pass_min_reliability", type=float, default=0.05)
    parser.add_argument("--pass_noise_std", type=float, default=0.03)
    parser.add_argument("--pass_gain_range", type=float, default=0.12)

    parser.add_argument("--sap_boundary_weight", type=float, default=0.05)
    parser.add_argument("--sap_shape_weight", type=float, default=0.03)
    parser.add_argument("--sap_area_lower", type=float, default=0.001)
    parser.add_argument("--sap_area_upper", type=float, default=0.08)
    return parser.parse_args()


args = parse_args()


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
        logging.info("Using preset split counts from labeled_num=%s: %d labeled, %d unlabeled",
                     str(args.labeled_num), labeled_slice, unlabeled_slice)

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
    iterator = tqdm(range(max_epoch), ncols=70)

    iter_num = 0
    stop_training = False
    for _ in iterator:
        for sampled_batch in train_loader:
            volume_batch = sampled_batch["image"].to(args.device)
            label_batch = sampled_batch["label"].to(args.device)
            train_record = trainer.train(volume_batch, label_batch, iter_num)
            train_records.append(train_record)
            iter_num += 1
            if args.val_interval > 0 and iter_num % args.val_interval == 0:
                val_record = trainer.val(val_loader, snapshot_path, iter_num)
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
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)
        torch.backends.cudnn.benchmark = True

        if args.snapshot_path:
            snapshot_path = os.path.join(args.snapshot_path, "fold_" + str(fold))
        else:
            snapshot_path = "./Results/A3_PASS_KnowSAM/fold_" + str(fold)

        os.makedirs(snapshot_path, exist_ok=True)
        code_path = os.path.join(snapshot_path, "code")
        if os.path.exists(code_path):
            shutil.rmtree(code_path)
        os.makedirs(os.path.join(code_path, "utils"), exist_ok=True)

        shutil.copyfile(os.path.join(CURRENT_DIR, "train_semi_SAM_a3_pass.py"), os.path.join(code_path, "train_semi_SAM_a3_pass.py"))
        shutil.copyfile(os.path.join(CURRENT_DIR, "trainer_a3_pass.py"), os.path.join(code_path, "trainer_a3_pass.py"))
        shutil.copyfile(os.path.join(CURRENT_DIR, "state_modules.py"), os.path.join(code_path, "state_modules.py"))
        shutil.copyfile(os.path.join(CURRENT_DIR, "state_targets.py"), os.path.join(code_path, "state_targets.py"))
        shutil.copyfile(os.path.join(CURRENT_DIR, "utils", "losses_a3_pass.py"), os.path.join(code_path, "utils", "losses_a3_pass.py"))

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
