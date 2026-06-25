import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn import functional as F
CE = torch.nn.BCELoss()
mse = torch.nn.MSELoss()

class KDLoss(nn.Module):
    """
    Distilling the Knowledge in a Neural Network
    https://arxiv.org/pdf/1503.02531.pdf
    """

    def __init__(self, T):
        super(KDLoss, self).__init__()
        self.T = T

    def forward(self, out_s, out_t):
        loss = (
            F.kl_div(F.log_softmax(out_s / self.T, dim=1),
                     F.softmax(out_t / self.T, dim=1), reduction="batchmean")  # , reduction="batchmean"
            * self.T
            * self.T
        )
        return loss


class WeightedKDLoss(nn.Module):
    """
    修改时间：2026-04-23
    修改功能：A3 超声半监督分割的“不确定性感知 SAM 蒸馏”。
    方法说明：原 KnowSAM 对所有像素使用相同 KL 蒸馏权重，容易把 SAM 在低置信边界处的错误知识传给
    UNet/VNet。这里支持像素级可靠性图和样本级质量权重，只强化高可信区域的教师监督。
    """

    def __init__(self, T=10):
        super(WeightedKDLoss, self).__init__()
        self.T = T

    def forward(self, student_logits, teacher_logits, pixel_weight=None, sample_weight=None):
        assert student_logits.size() == teacher_logits.size(), 'student & teacher shape do not match'
        log_student = F.log_softmax(student_logits / self.T, dim=1)
        prob_teacher = F.softmax(teacher_logits / self.T, dim=1)
        kd_map = F.kl_div(log_student, prob_teacher, reduction="none").sum(dim=1, keepdim=True)
        kd_map = kd_map * self.T * self.T

        weight = torch.ones_like(kd_map)
        if pixel_weight is not None:
            if pixel_weight.dim() == 3:
                pixel_weight = pixel_weight.unsqueeze(1)
            weight = weight * pixel_weight.detach().to(kd_map.device).float()
        if sample_weight is not None:
            weight = weight * sample_weight.detach().to(kd_map.device).float().view(-1, 1, 1, 1)

        return (kd_map * weight).sum() / weight.sum().clamp_min(1e-6)


class BoundaryLoss(nn.Module):
    """
    修改时间：2026-04-23
    修改功能：A3 外侧裂结构的边界约束。
    方法说明：外侧裂在 A3 切面中呈细长低占比结构，仅依赖区域 Dice 容易出现边界变厚或断裂。
    该损失用 Sobel 梯度近似概率边界和标签边界，增强细结构轮廓学习。
    """

    def __init__(self, n_classes=2, foreground_index=1):
        super(BoundaryLoss, self).__init__()
        self.n_classes = n_classes
        self.foreground_index = min(foreground_index, n_classes - 1)
        kernel_x = torch.tensor([[-1.0, 0.0, 1.0],
                                 [-2.0, 0.0, 2.0],
                                 [-1.0, 0.0, 1.0]]).view(1, 1, 3, 3)
        kernel_y = torch.tensor([[-1.0, -2.0, -1.0],
                                 [0.0, 0.0, 0.0],
                                 [1.0, 2.0, 1.0]]).view(1, 1, 3, 3)
        self.register_buffer("kernel_x", kernel_x)
        self.register_buffer("kernel_y", kernel_y)

    def _gradient_magnitude(self, x):
        grad_x = F.conv2d(x, self.kernel_x.to(x.device), padding=1)
        grad_y = F.conv2d(x, self.kernel_y.to(x.device), padding=1)
        return torch.sqrt(grad_x * grad_x + grad_y * grad_y + 1e-6)

    def forward(self, prob, target, sample_weight=None):
        if prob.shape[0] == 0:
            return prob.sum() * 0.0
        if target.dim() == 4:
            target = target.squeeze(1)
        if self.n_classes > 2:
            fg_prob = prob[:, 1:].sum(dim=1, keepdim=True)
            fg_target = (target > 0).float().unsqueeze(1).to(prob.device)
        else:
            fg_prob = prob[:, self.foreground_index:self.foreground_index + 1]
            fg_target = (target == self.foreground_index).float().unsqueeze(1).to(prob.device)
        prob_boundary = self._gradient_magnitude(fg_prob)
        target_boundary = self._gradient_magnitude(fg_target)
        loss_map = torch.abs(prob_boundary - target_boundary)
        if sample_weight is not None:
            loss_map = loss_map * sample_weight.detach().to(prob.device).float().view(-1, 1, 1, 1)
            return loss_map.sum() / sample_weight.detach().to(prob.device).float().sum().clamp_min(1e-6) / loss_map.shape[-1] / loss_map.shape[-2]
        return loss_map.mean()


def soft_area_prior_loss(prob, lower=0.001, upper=0.08, foreground_index=1, sample_weight=None):
    """
    修改时间：2026-04-23
    修改功能：A3 外侧裂软解剖尺度先验。
    方法说明：外侧裂区域占比通常较小，预测为空或膨胀成大块都会破坏临床可解释性。
    该项仅惩罚超出宽松面积区间的概率面积，不强行绑定固定形状，适合小样本半监督场景。
    """
    if prob.shape[0] == 0:
        return prob.sum() * 0.0
    if prob.shape[1] > 2:
        fg_ratio = prob[:, 1:].sum(dim=1).mean(dim=(1, 2))
    else:
        fg_ratio = prob[:, foreground_index].mean(dim=(1, 2))
    lower_penalty = F.relu(lower - fg_ratio)
    upper_penalty = F.relu(fg_ratio - upper)
    loss = lower_penalty * lower_penalty + upper_penalty * upper_penalty
    if sample_weight is not None:
        weight = sample_weight.detach().to(prob.device).float()
        return (loss * weight).sum() / weight.sum().clamp_min(1e-6)
    return loss.mean()


def loss_diff1(u_prediction_1, u_prediction_2):
    loss_a = 0.0

    for i in range(u_prediction_2.size(1)):
        loss_a = CE(u_prediction_1[:, i, ...].clamp(1e-8, 1 - 1e-7),
                                 Variable(u_prediction_2[:, i, ...].float(), requires_grad=False))

    loss_diff_avg = loss_a.mean()
    return loss_diff_avg


def loss_diff2(u_prediction_1, u_prediction_2):
    loss_b = 0.0

    for i in range(u_prediction_2.size(1)):
        loss_b = CE(u_prediction_2[:, i, ...].clamp(1e-8, 1 - 1e-7),
                                 Variable(u_prediction_1[:, i, ...], requires_grad=False))

    loss_diff_avg = loss_b.mean()
    return loss_diff_avg


class DiceLoss(nn.Module):
    def __init__(self, n_classes):
        super(DiceLoss, self).__init__()
        self.n_classes = n_classes

    def _one_hot_encoder(self, input_tensor):
        tensor_list = []
        for i in range(self.n_classes):
            temp_prob = input_tensor * i == i * torch.ones_like(input_tensor)
            tensor_list.append(temp_prob)
        output_tensor = torch.cat(tensor_list, dim=1)
        return output_tensor.float()

    def _dice_loss(self, score, target):
        target = target.float()
        smooth = 1e-10
        intersect = torch.sum(score * target)
        y_sum = torch.sum(target * target)
        z_sum = torch.sum(score * score)
        loss = (2 * intersect + smooth) / (z_sum + y_sum + smooth)
        loss = 1 - loss
        return loss

    def _dice_mask_loss(self, score, target, mask):
        target = target.float()
        mask = mask.float()
        smooth = 1e-10
        intersect = torch.sum(score * target * mask)
        y_sum = torch.sum(target * target * mask)
        z_sum = torch.sum(score * score * mask)
        loss = (2 * intersect + smooth) / (z_sum + y_sum + smooth)
        loss = 1 - loss
        return loss

    def forward(self, inputs, target, weight=None, softmax=False):
        if softmax:
            inputs = torch.softmax(inputs, dim=1)
        target = self._one_hot_encoder(target.unsqueeze(1))
        if weight is None:
            weight = [1] * self.n_classes
        assert inputs.size() == target.size(), 'predict & target shape do not match'
        class_wise_dice = []
        loss = 0.0
        for i in range(0, self.n_classes):
            dice = self._dice_loss(inputs[:, i], target[:, i])
            class_wise_dice.append(1.0 - dice.item())
            loss += dice * weight[i]
        return loss / self.n_classes


def dice_loss(pred, label, epsilon=1e-5):
    intersection = torch.sum(pred * label, dim=(2, 3))
    union = torch.sum(pred, dim=(2, 3)) + torch.sum(label, dim=(2, 3))
    dice_coefficient = (2.0 * intersection + epsilon) / (union + epsilon)
    dice_loss = 1.0 - dice_coefficient
    return dice_loss.mean()


def sigmoid_mse_loss_map(input_logits, target_logits):
    assert input_logits.size() == target_logits.size()
    input_softmax = torch.nn.Sigmoid()(input_logits)
    target_softmax = torch.nn.Sigmoid()(target_logits)
    mse_loss_map = (input_softmax-target_softmax)**2
    return mse_loss_map


def mse_loss(input1, input2):
    return torch.mean((input1 - input2) ** 2)


class DiceLoss(nn.Module):
    def __init__(self, n_classes):
        super(DiceLoss, self).__init__()
        self.n_classes = n_classes

    def _one_hot_encoder(self, input_tensor):
        tensor_list = []
        for i in range(self.n_classes):
            temp_prob = input_tensor == i * torch.ones_like(input_tensor)
            tensor_list.append(temp_prob)
        output_tensor = torch.cat(tensor_list, dim=1)
        return output_tensor.float()

    def _one_hot_mask_encoder(self, input_tensor):
        tensor_list = []
        for i in range(self.n_classes):
            temp_prob = input_tensor * i == i * torch.ones_like(input_tensor)
            tensor_list.append(temp_prob)
        output_tensor = torch.cat(tensor_list, dim=1)
        return output_tensor.float()

    def _dice_loss(self, score, target):
        target = target.float()
        smooth = 1e-10
        intersect = torch.sum(score * target)
        y_sum = torch.sum(target * target)
        z_sum = torch.sum(score * score)
        loss = (2 * intersect + smooth) / (z_sum + y_sum + smooth)
        loss = 1 - loss
        return loss

    def _dice_mask_loss(self, score, target, mask):
        target = target.float()
        mask = mask.float()
        smooth = 1e-10
        intersect = torch.sum(score * target * mask)
        y_sum = torch.sum(target * target * mask)
        z_sum = torch.sum(score * score * mask)
        loss = (2 * intersect + smooth) / (z_sum + y_sum + smooth)
        loss = 1 - loss
        return loss

    def forward(self, inputs, target, mask=None, weight=None, softmax=False):
        if softmax:
            inputs = torch.softmax(inputs, dim=1)
        target = self._one_hot_encoder(target.unsqueeze(1))
        if weight is None:
            weight = [1] * self.n_classes
        assert inputs.size() == target.size(), 'predict & target shape do not match'
        class_wise_dice = []
        loss = 0.0
        if mask is not None:
            mask = self._one_hot_mask_encoder(mask)
            for i in range(0, self.n_classes):
                dice = self._dice_mask_loss(inputs[:, i], target[:, i], mask[:, i])
                class_wise_dice.append(1.0 - dice.item())
                loss += dice * weight[i]
        else:
            for i in range(0, self.n_classes):
                dice = self._dice_loss(inputs[:, i], target[:, i])
                class_wise_dice.append(1.0 - dice.item())
                loss += dice * weight[i]
        return loss / self.n_classes
