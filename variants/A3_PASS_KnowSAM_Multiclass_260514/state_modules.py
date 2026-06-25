import torch
import torch.nn as nn
import torch.nn.functional as F

from Model.model import KnowSAM


class AcousticAnatomicalStateHead(nn.Module):
    """AASP: estimate compact state posterior from image and branch predictions."""

    def __init__(self, in_channels, num_classes, state_size=64, state_dim=64, base_channels=32):
        super().__init__()
        input_channels = in_channels + num_classes * 3
        self.state_size = state_size
        self.encoder = nn.Sequential(
            nn.Conv2d(input_channels, base_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels, base_channels, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels, base_channels * 2, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(base_channels * 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels * 2, base_channels * 4, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(base_channels * 4),
            nn.ReLU(inplace=True),
        )
        hidden = base_channels * 4
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.geometry_head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(hidden, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, 6),
            nn.Sigmoid(),
        )
        self.latent_head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(hidden, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, state_dim),
        )
        self.dense_head = nn.Sequential(
            nn.Conv2d(hidden, base_channels * 2, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels * 2, 2, 1),
            nn.Sigmoid(),
        )

    def forward(self, image, pred_unet_soft, pred_vnet_soft, fusion_soft):
        x = torch.cat([image, pred_unet_soft, pred_vnet_soft, fusion_soft], dim=1)
        feat = self.encoder(x)
        pooled = self.pool(feat)
        geometry = self.geometry_head(pooled)
        latent = self.latent_head(pooled)
        dense = self.dense_head(feat)
        dense = F.interpolate(dense, size=(self.state_size, self.state_size), mode="bilinear", align_corners=False)
        return {"geometry": geometry, "dense": dense, "latent": latent, "feature": feat}


class StateConditionedMaskDecoder(nn.Module):
    """SCMD: decode masks conditioned on acoustic-anatomical state."""

    def __init__(self, in_channels, num_classes, state_dim=64, base_channels=32):
        super().__init__()
        input_channels = in_channels + num_classes * 3
        self.stem = nn.Sequential(
            nn.Conv2d(input_channels, base_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels, base_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True),
        )
        self.film = nn.Linear(state_dim + 6, base_channels * 2)
        self.refine = nn.Sequential(
            nn.Conv2d(base_channels, base_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels, num_classes, 1),
        )
        self.residual_scale = nn.Parameter(torch.tensor(0.5))

    def forward(self, image, pred_unet_soft, pred_vnet_soft, fusion_soft, fusion_logits, state):
        x = torch.cat([image, pred_unet_soft, pred_vnet_soft, fusion_soft], dim=1)
        feat = self.stem(x)
        state_vector = torch.cat([state["geometry"], state["latent"]], dim=1)
        gamma_beta = self.film(state_vector).unsqueeze(-1).unsqueeze(-1)
        gamma, beta = torch.chunk(gamma_beta, 2, dim=1)
        feat = feat * (1.0 + torch.tanh(gamma)) + beta
        residual_logits = self.refine(feat)
        return fusion_logits + self.residual_scale * residual_logits


class A3PASSNet(nn.Module):
    """KnowSAM backbone with AASP and SCMD heads."""

    def __init__(self, args):
        super().__init__()
        state_size = getattr(args, "pass_state_size", 64)
        state_dim = getattr(args, "pass_state_dim", 64)
        base_channels = getattr(args, "pass_base_channels", 32)
        self.SGDL = KnowSAM(args)
        self.state_head = AcousticAnatomicalStateHead(
            args.in_channels,
            args.num_classes,
            state_size=state_size,
            state_dim=state_dim,
            base_channels=base_channels,
        )
        self.state_decoder = StateConditionedMaskDecoder(
            args.in_channels,
            args.num_classes,
            state_dim=state_dim,
            base_channels=base_channels,
        )

    def forward(self, image):
        pred_unet, pred_vnet, pred_unet_soft, pred_vnet_soft, fusion_logits = self.SGDL(image)
        fusion_soft = torch.softmax(fusion_logits, dim=1)
        state = self.state_head(image, pred_unet_soft, pred_vnet_soft, fusion_soft)
        pass_logits = self.state_decoder(
            image,
            pred_unet_soft,
            pred_vnet_soft,
            fusion_soft,
            fusion_logits,
            state,
        )
        pass_soft = torch.softmax(pass_logits, dim=1)
        return {
            "pred_unet": pred_unet,
            "pred_vnet": pred_vnet,
            "pred_unet_soft": pred_unet_soft,
            "pred_vnet_soft": pred_vnet_soft,
            "fusion_logits": fusion_logits,
            "fusion_soft": fusion_soft,
            "state": state,
            "pass_logits": pass_logits,
            "pass_soft": pass_soft,
        }
