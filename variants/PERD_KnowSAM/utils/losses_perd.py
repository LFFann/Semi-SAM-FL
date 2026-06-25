import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn import functional as F


CE = torch.nn.BCELoss()
mse = torch.nn.MSELoss()


class KDLoss(nn.Module):
    def __init__(self, T):
        super(KDLoss, self).__init__()
        self.T = T

    def forward(self, out_s, out_t):
        loss = (
            F.kl_div(
                F.log_softmax(out_s / self.T, dim=1),
                F.softmax(out_t / self.T, dim=1),
                reduction="batchmean",
            )
            * self.T
            * self.T
        )
        return loss


class WeightedKDLoss(nn.Module):
    def __init__(self, T=10):
        super(WeightedKDLoss, self).__init__()
        self.T = T

    def forward(self, student_logits, teacher_logits, pixel_weight=None, sample_weight=None):
        assert student_logits.size() == teacher_logits.size(), "student & teacher shape do not match"
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


def loss_diff1(u_prediction_1, u_prediction_2):
    loss_a = 0.0
    for i in range(u_prediction_2.size(1)):
        loss_a = CE(
            u_prediction_1[:, i, ...].clamp(1e-8, 1 - 1e-7),
            Variable(u_prediction_2[:, i, ...].float(), requires_grad=False),
        )
    return loss_a.mean()


def loss_diff2(u_prediction_1, u_prediction_2):
    loss_b = 0.0
    for i in range(u_prediction_2.size(1)):
        loss_b = CE(
            u_prediction_2[:, i, ...].clamp(1e-8, 1 - 1e-7),
            Variable(u_prediction_1[:, i, ...], requires_grad=False),
        )
    return loss_b.mean()


class DiceLoss(nn.Module):
    def __init__(self, n_classes):
        super(DiceLoss, self).__init__()
        self.n_classes = n_classes

    def _one_hot_encoder(self, input_tensor):
        tensor_list = []
        for i in range(self.n_classes):
            temp_prob = input_tensor == i * torch.ones_like(input_tensor)
            tensor_list.append(temp_prob)
        return torch.cat(tensor_list, dim=1).float()

    def _one_hot_mask_encoder(self, input_tensor):
        tensor_list = []
        for i in range(self.n_classes):
            temp_prob = input_tensor * i == i * torch.ones_like(input_tensor)
            tensor_list.append(temp_prob)
        return torch.cat(tensor_list, dim=1).float()

    def _dice_loss(self, score, target):
        target = target.float()
        smooth = 1e-10
        intersect = torch.sum(score * target)
        y_sum = torch.sum(target * target)
        z_sum = torch.sum(score * score)
        loss = (2 * intersect + smooth) / (z_sum + y_sum + smooth)
        return 1 - loss

    def _dice_mask_loss(self, score, target, mask):
        target = target.float()
        mask = mask.float()
        smooth = 1e-10
        intersect = torch.sum(score * target * mask)
        y_sum = torch.sum(target * target * mask)
        z_sum = torch.sum(score * score * mask)
        loss = (2 * intersect + smooth) / (z_sum + y_sum + smooth)
        return 1 - loss

    def forward(self, inputs, target, mask=None, weight=None, softmax=False):
        if softmax:
            inputs = torch.softmax(inputs, dim=1)
        target = self._one_hot_encoder(target.unsqueeze(1))
        if weight is None:
            weight = [1] * self.n_classes
        assert inputs.size() == target.size(), "predict & target shape do not match"
        loss = 0.0
        if mask is not None:
            mask = self._one_hot_mask_encoder(mask)
            for i in range(0, self.n_classes):
                loss += self._dice_mask_loss(inputs[:, i], target[:, i], mask[:, i]) * weight[i]
        else:
            for i in range(0, self.n_classes):
                loss += self._dice_loss(inputs[:, i], target[:, i]) * weight[i]
        return loss / self.n_classes


def dice_loss(pred, label, epsilon=1e-5):
    intersection = torch.sum(pred * label, dim=(2, 3))
    union = torch.sum(pred, dim=(2, 3)) + torch.sum(label, dim=(2, 3))
    dice_coefficient = (2.0 * intersection + epsilon) / (union + epsilon)
    return (1.0 - dice_coefficient).mean()


def sigmoid_mse_loss_map(input_logits, target_logits):
    assert input_logits.size() == target_logits.size()
    input_softmax = torch.nn.Sigmoid()(input_logits)
    target_softmax = torch.nn.Sigmoid()(target_logits)
    return (input_softmax - target_softmax) ** 2


def mse_loss(input1, input2):
    return torch.mean((input1 - input2) ** 2)
