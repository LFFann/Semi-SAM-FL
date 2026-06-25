import cv2
import torch
import numpy as np
from medpy import metric
import torch.nn.functional as F
import matplotlib.pyplot as plt


def _hausdorff_distance_safe(true_mask, pred_mask):
    try:
        from hausdorff import hausdorff_distance
        return float(hausdorff_distance(true_mask, pred_mask) * 0.95)
    except Exception:
        try:
            return float(metric.binary.hd95(pred_mask.astype(bool), true_mask.astype(bool)))
        except Exception:
            return float("nan")


def _binary_class_metrics(true_mask, pred_mask):
    true_mask = true_mask.astype(bool)
    pred_mask = pred_mask.astype(bool)
    true_sum = true_mask.sum()
    pred_sum = pred_mask.sum()
    if true_sum == 0 and pred_sum == 0:
        return 1.0, 1.0, 0.0
    if true_sum == 0 or pred_sum == 0:
        return 0.0, 0.0, float("nan")
    return (
        float(metric.binary.dc(pred_mask, true_mask)),
        float(metric.binary.jc(pred_mask, true_mask)),
        _hausdorff_distance_safe(true_mask.astype(np.uint8), pred_mask.astype(np.uint8)),
    )


def multiclass_segmentation_metrics(y_true, y_pred, num_classes=None):
    if torch.is_tensor(y_true):
        y_true = y_true.detach().cpu().numpy()
    if torch.is_tensor(y_pred):
        if num_classes is None and y_pred.ndim >= 2:
            num_classes = int(y_pred.shape[1])
        y_pred = torch.argmax(y_pred, dim=1).detach().cpu().numpy()

    y_true = np.squeeze(y_true).astype(np.int64)
    y_pred = np.squeeze(y_pred).astype(np.int64)
    if num_classes is None:
        num_classes = int(max(np.max(y_true), np.max(y_pred)) + 1)

    result = {}
    rows = []
    for class_idx in range(1, num_classes):
        dice, iou, hd95 = _binary_class_metrics(y_true == class_idx, y_pred == class_idx)
        result[f"class_{class_idx}_dice"] = dice
        result[f"class_{class_idx}_iou"] = iou
        result[f"class_{class_idx}_hd95"] = hd95
        rows.append((dice, iou, hd95))

    if rows:
        metrics_array = np.array(rows, dtype=np.float32)
        result["avg_dice"] = float(np.nanmean(metrics_array[:, 0]))
        result["avg_iou"] = float(np.nanmean(metrics_array[:, 1]))
        result["avg_hd95"] = float(np.nanmean(metrics_array[:, 2]))
    else:
        result["avg_dice"] = 0.0
        result["avg_iou"] = 0.0
        result["avg_hd95"] = float("nan")
    return result




def eval(y_true, y_pred, thr=0.5, epsilon=0.001):
    if y_pred.shape[1] > 2:
        result = multiclass_segmentation_metrics(y_true, y_pred, num_classes=y_pred.shape[1])
        return result["avg_dice"], result["avg_iou"], result["avg_hd95"]
    elif y_pred.shape[1] == 1:
        y_true = y_true.to(torch.float32).squeeze(0).cpu().detach().numpy()
        y_pred = (y_pred > thr).to(torch.float32).squeeze(0).squeeze(0).cpu().detach().numpy()
    else:
        y_true = y_true.to(torch.float32).squeeze(0).cpu().detach().numpy()
        y_pred = (y_pred > thr).to(torch.float32).squeeze(0)[1].cpu().detach().numpy()
    single_class_res = []
    single_class_res.append(metric.binary.dc(y_pred, y_true))
    single_class_res.append(metric.binary.jc(y_pred, y_true))
    single_class_res.append(_hausdorff_distance_safe(y_true, y_pred))
    return single_class_res



def dice_coef(y_true, y_pred, thr=0.5, epsilon=0.001):
    if y_pred.shape[1] > 2:
        return multiclass_segmentation_metrics(y_true, y_pred, num_classes=y_pred.shape[1])["avg_dice"]
    elif y_pred.shape[1] > 1:
        y_true = y_true.to(torch.float32).squeeze(0).cpu().detach().numpy()
        y_pred = (y_pred > thr).to(torch.float32).squeeze(0)[1].cpu().detach().numpy()
    else:
        y_true = y_true.to(torch.float32).squeeze(0).squeeze(0).cpu().detach().numpy()
        y_pred = (y_pred > thr).to(torch.float32).squeeze(0).squeeze(0).cpu().detach().numpy()
    inter_map = y_true * y_pred
    inter = inter_map.sum()
    den = y_true.sum() + y_pred.sum()
    dice = ((2 * inter) / (den + epsilon)) if den > 0 else 0
    return dice


def evaluate_95hd(y_true, y_pred, thr=0.5):
    if y_true.shape[1] > 1:
        y_true = y_true.to(torch.float32).squeeze(0)[1].cpu().detach().numpy()
        y_pred = (y_pred > thr).to(torch.float32).squeeze(0)[1].cpu().detach().numpy()
    else:
        y_true = y_true.to(torch.float32).squeeze(0).squeeze(0).cpu().detach().numpy()
        y_pred = (y_pred > thr).to(torch.float32).squeeze(0).squeeze(0).cpu().detach().numpy()
    return _hausdorff_distance_safe(y_true, y_pred)


def calculate_iou(y_true, y_pred, thr=0.5):
    if y_true.shape[1] > 1:
        y_true = y_true.to(torch.float32).squeeze(0)[1].cpu().detach().numpy()
        y_pred = (y_pred > thr).to(torch.float32).squeeze(0)[1].cpu().detach().numpy()
    else:
        y_true = y_true.to(torch.float32).squeeze(0).squeeze(0).cpu().detach().numpy()
        y_pred = (y_pred > thr).to(torch.float32).squeeze(0).squeeze(0).cpu().detach().numpy()
    intersection = np.logical_and(y_true > 0, y_pred > 0)
    intersection = np.sum(intersection)
    union = np.logical_or(y_true > 0, y_pred > 0)
    union = np.sum(union)
    # 计算Jaccard指数（IoU）
    iou = intersection / union if union > 0 else 0
    return iou



def calculate_metric_percase(pred, gt):
    pred[pred > 0] = 1
    gt[gt > 0] = 1
    if pred.sum() > 0:
        dice = metric.binary.dc(pred, gt)
        hd95 = metric.binary.hd95(pred, gt)
        return dice, hd95
    else:
        return 0, 0


def test_single_2D_colorImage(image, label, net, classes):
    thr = 0.5
    net.eval()
    label = label.squeeze(0).cpu().detach().numpy()
    with torch.no_grad():
        out = net(image.cuda())
        out = torch.nn.Sigmoid()(out)
        out = (out > thr).to(torch.float32)
        out = torch.argmax(out, dim=1).squeeze(0)
        prediction = out.cpu().detach().numpy()
    metric_list = []

    for i in range(1, classes):
        metric_list.append(calculate_metric_percase(
            prediction == i, label == i))
    return metric_list


def patients_to_slices(dataset, patiens_num):
    ref_dict = {}
    if "ACDC" in dataset:
        ref_dict = {"3": 68, "7": 136,
                    "14": 256, "21": 396, "28": 512, "35": 664, "140": 1312}
    elif "tumor" in dataset:
        ref_dict = {"1": 15, "10": 145, "20": 290, "30": 435, }
    elif "260513_data" in dataset:
        ref_dict = {"1": 106}
    elif "ISIC" in dataset:
        ref_dict = {"10": 207, "30": 622, }
    elif "thyroid" in dataset:
        ref_dict = {"10": 613, "30": 1841, }
    elif "BrainMRI" in dataset:
        ref_dict = {"10": 103, "30": 310, }
    elif "MRI_Hippocampus_Seg" in dataset:
        ref_dict = {"10": 282, "30": 846, }
    else:
        print("Error")
    return ref_dict[str(patiens_num)]


def get_uncertainty_map(model, image_batch, num_classes, T=8, uncertainty_bs=2): # uncertainty_bs must be divisible by T
    _, _, w, h = image_batch.shape
    volume_batch_r = image_batch.repeat(uncertainty_bs, 1, 1, 1)
    stride = volume_batch_r.shape[0] // uncertainty_bs
    preds = torch.zeros([stride * T, num_classes, w, h]).cuda()  # init preds
    for i in range(T // uncertainty_bs):
        ema_inputs = volume_batch_r + torch.clamp(torch.randn_like(volume_batch_r) * 0.1, -0.2, 0.2)  # add noise
        with torch.no_grad():
            preds[uncertainty_bs * stride * i: uncertainty_bs * stride * (i + 1)] = model(ema_inputs)
    preds = F.softmax(preds, dim=1)
    preds = preds.reshape(T, stride, num_classes, w, h)
    preds = torch.mean(preds, dim=0)
    uncertainty = -1.0 * torch.sum(preds * torch.log(preds + 1e-6), dim=1, keepdim=True)
    return uncertainty


def get_no_noise_uncertainty_map(model, image_batch, num_classes, T=8): # uncertainty_bs must be divisible by T
    b, _, w, h = image_batch.shape
    preds = torch.zeros([T * b, num_classes, w, h]).cuda()  # init preds
    for i in range(T):
        ema_inputs = image_batch
        with torch.no_grad():
            preds[i * b: i * b + b] = model(ema_inputs)
    # preds = F.softmax(preds, dim=1)
    preds = torch.nn.Sigmoid()(preds)
    preds = preds.reshape(T, b, num_classes, w, h)
    preds = torch.mean(preds, dim=0)
    uncertainty = -1.0 * torch.sum(preds * torch.log(preds + 1e-6), dim=1, keepdim=True)
    return uncertainty


def generate_mask(img):
    batch_size, channel, img_x, img_y = img.shape[0], img.shape[1], img.shape[2], img.shape[3]
    loss_mask = torch.ones(batch_size, img_x, img_y).cuda()
    mask = torch.ones(img_x, img_y).cuda()
    patch_x, patch_y = int(img_x*2/3), int(img_y*2/3)
    w = np.random.randint(0, img_x - patch_x)
    h = np.random.randint(0, img_y - patch_y)
    mask[w:w+patch_x, h:h+patch_y] = 0
    loss_mask[:, w:w+patch_x, h:h+patch_y] = 0
    return mask.long(), loss_mask.long()

