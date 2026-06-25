import argparse
import numpy as np
import random
import torch
import os
import logging
import sys
from tqdm import tqdm
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "..", ".."))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)
if REPO_ROOT not in sys.path:
    sys.path.insert(1, REPO_ROOT)
from dataloader.dataset import build_Dataset
from torch.utils.data import DataLoader
from utils.utils import patients_to_slices
from utils.training_monitor import save_training_artifacts
from dataloader.transforms import build_transforms, build_weak_strong_transforms
from dataloader.TwoStreamBatchSampler import TwoStreamBatchSampler

from trainer_a3_rcp import Trainer


parser = argparse.ArgumentParser()
parser.add_argument('--data_path', type=str, default='./SampleData',
                    help='Name of Experiment')
parser.add_argument('--labeled_num', type=int, default=1,
                    help='Percentage of label quantity')

parser.add_argument('--dataset', type=str, default='/260513_data_label1',
                    help='Name of Experiment')
# parser.add_argument('--dataset', type=str, default='/ISIC_TrainDataset_10',
#                     help='Name of Experiment')
# parser.add_argument('--dataset', type=str, default='/thyroid_30',
#                     help='Name of Experiment')
# parser.add_argument('--dataset', type=str, default='/BrainMRI_30',
#                     help='Name of Experiment')

parser.add_argument('--num_classes', type=int,  default=2,
                    help='output channel of network')
parser.add_argument('--in_channels', type=int, default=3,
                    help='input channel of network')

parser.add_argument('-lr', type=float, default=1e-4, help='initial learning rate')
parser.add_argument('-UNet_lr', type=float, default=0.01, help='initial learning rate')
parser.add_argument('-VNet_lr', type=float, default=0.01, help='initial learning rate')
parser.add_argument('--image_size', type=int, default=256, help='image_size')
parser.add_argument('--point_nums', type=int, default=5, help='points number')
parser.add_argument('--box_nums', type=int, default=1, help='boxes number')
parser.add_argument('--mod', type=str, default='sam_adpt', help='mod type:seg,cls,val_ad')
parser.add_argument("--model_type", type=str, default="vit_b", help="sam model_type")
parser.add_argument('-thd', type=bool, default=False, help='3d or not')
parser.add_argument('--batch_size', type=int, default=24,
                    help='batch_size per gpu')
parser.add_argument('--labeled_bs', type=int, default=12,
                    help='labeled_batch_size per gpu')
parser.add_argument('--num_workers', type=int, default=2,
                    help='number of workers for train dataloader')
parser.add_argument('--val_num_workers', type=int, default=1,
                    help='number of workers for val dataloader')
parser.add_argument('--seed', type=int,  default=42,
                    help='random seed')

parser.add_argument('--mixed_iterations', type=int, default=12000,
                    help='maximum epoch number to train')
parser.add_argument('--max_iterations', type=int, default=50000,
                    help='maximum epoch number to train')
parser.add_argument('--val_interval', type=int, default=200,
                    help='validation interval in iterations')
parser.add_argument('--snapshot_path', type=str, default='',
                    help='custom snapshot path for outputs')

parser.add_argument('--n_fold', type=int, default=1,
                    help='maximum epoch number to train')
parser.add_argument('--consistency', type=float, default=0.1,
                    help='consistency')
parser.add_argument('--consistency_rampup', type=float,
                    default=200.0, help='consistency_rampup')
parser.add_argument('--device', type=str, default='cuda')
parser.add_argument("--multimask", type=bool, default=False, help="ouput multimask")
parser.add_argument("--encoder_adapter", type=bool, default=True, help="use adapter")
parser.add_argument("--sam_checkpoint", type=str, default="./sam_vit_b_01ec64.pth", help="sam checkpoint")
# 修改时间：2026-04-23
# 修改功能：A3 外侧裂半监督分割三项创新的可配置参数。
parser.add_argument('--uckd_alpha', type=float, default=2.0,
                    help='U-CKD uncertainty sensitivity for pixel-wise SAM distillation')
parser.add_argument('--uckd_min_weight', type=float, default=0.15,
                    help='minimum pixel reliability weight in U-CKD')
parser.add_argument('--qapl_min_weight', type=float, default=0.20,
                    help='minimum sample quality weight in QAPL pseudo-label learning')
parser.add_argument('--rcp_alpha', type=float, default=2.0,
                    help='RCP uncertainty sensitivity for consensus prompt calibration')
parser.add_argument('--rcp_min_weight', type=float, default=0.10,
                    help='minimum consensus prompt reliability in RCP')
parser.add_argument('--rcp_sharpen', type=float, default=1.5,
                    help='temperature-like sharpening factor for RCP consensus probability')
parser.add_argument('--sap_boundary_weight', type=float, default=0.10,
                    help='weight of SAP-BR boundary refinement loss')
parser.add_argument('--sap_shape_weight', type=float, default=0.05,
                    help='weight of SAP-BR soft area prior loss')
parser.add_argument('--sap_area_lower', type=float, default=0.001,
                    help='lower foreground area ratio for A3 soft anatomical prior')
parser.add_argument('--sap_area_upper', type=float, default=0.08,
                    help='upper foreground area ratio for A3 soft anatomical prior')

args = parser.parse_args()


def sigmoid_rampup(current, rampup_length):
    """Exponential rampup from https://arxiv.org/abs/1610.02242"""
    if rampup_length == 0:
        return 1.0
    else:
        current = np.clip(current, 0.0, rampup_length)
        phase = 1.0 - current / rampup_length
        return float(np.exp(-5.0 * phase * phase))


def worker_init_fn(worker_id):
    random.seed(args.seed + worker_id)


def get_current_consistency_weight(epoch):
    # Consistency ramp-up from https://arxiv.org/abs/1610.02242
    return args.consistency * sigmoid_rampup(epoch, args.consistency_rampup)


def train(args, snapshot_path):
    batch_size = args.batch_size
    max_iterations = args.max_iterations
    train_records = []
    val_records = []
    # model
    trainer = Trainer(args)
    # dataset
    data_transforms = build_weak_strong_transforms(args)
    train_dataset = build_Dataset(args=args, data_dir=args.data_path + args.dataset, split="train_semi",
                                  transform=data_transforms)
    val_dataset = build_Dataset(args=args, data_dir=args.data_path + args.dataset, split="val",
                                transform=data_transforms["valid_test"])

    # sampler
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
    batch_sampler = TwoStreamBatchSampler(labeled_idxs, unlabeled_idxs, batch_size, batch_size-args.labeled_bs)

    # dataloader
    train_loader = DataLoader(
        train_dataset,
        batch_sampler=batch_sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        worker_init_fn=worker_init_fn
    )
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=args.val_num_workers)
    logging.info("{} iterations per epoch".format(len(train_loader)))
    max_epoch = max_iterations // len(train_loader) + 1
    iterator = tqdm(range(max_epoch), ncols=70)

    # else:
    iter_num = 0
    stop_training = False
    for _ in iterator:
        for i_batch, sampled_batch in enumerate(train_loader):
            volume_batch, label_batch = sampled_batch['image'].cuda(), sampled_batch['label'].cuda()
            train_record = trainer.train(volume_batch, label_batch, iter_num)
            train_records.append(train_record)
            iter_num = iter_num + 1
            if args.val_interval > 0 and iter_num > 0 and iter_num % args.val_interval == 0:
                if "ACDC" not in args.dataset:
                    val_record = trainer.val(val_loader, snapshot_path, iter_num)
                else:
                    val_record = trainer.val_ACDC(val_loader, snapshot_path, iter_num)
                if val_record is not None:
                    val_records.append(val_record)
                save_training_artifacts(snapshot_path, train_records, val_records, vars(args))
            if iter_num >= max_iterations:
                stop_training = True
                break
        if stop_training:
            break
    save_training_artifacts(snapshot_path, train_records, val_records, vars(args))


if __name__ == '__main__':
    import shutil
    for fold in range(args.n_fold):
        torch.autograd.set_detect_anomaly(True)
        random.seed(2024)
        np.random.seed(2024)
        torch.manual_seed(2024)
        torch.cuda.manual_seed(2024)
        torch.backends.cudnn.benchmark = True

        if args.snapshot_path:
            snapshot_path = args.snapshot_path
            if args.n_fold > 1:
                snapshot_path = os.path.join(snapshot_path, "fold_" + str(fold))
        else:
            snapshot_path = "./Results/A3_RCP_KnowSAM/fold_" + str(fold)

        if not os.path.exists(snapshot_path):
            os.makedirs(snapshot_path)
        if os.path.exists(snapshot_path + '/code'):
            shutil.rmtree(snapshot_path + '/code')
        if not os.path.exists(snapshot_path + '/code'):
            os.makedirs(snapshot_path + '/code')

        shutil.copyfile(os.path.join(CURRENT_DIR, "train_semi_SAM_a3_rcp.py"), snapshot_path + "/code/train_semi_SAM_a3_rcp.py")
        shutil.copyfile(os.path.join(CURRENT_DIR, "trainer_a3_rcp.py"), snapshot_path + "/code/trainer_a3_rcp.py")
        os.makedirs(snapshot_path + "/code/utils", exist_ok=True)
        shutil.copyfile(os.path.join(CURRENT_DIR, "utils", "losses_a3_rcp.py"), snapshot_path + "/code/utils/losses_a3_rcp.py")

        logging.basicConfig(filename=snapshot_path+"/log.txt", level=logging.INFO,
                            format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S',
                            force=True)
        logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
        logging.info(str(args))
        train(args, snapshot_path)
