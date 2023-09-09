import random
from copy import deepcopy
from typing import List, Optional, Union

import albumentations as albu
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from arti_mani.algorithms.visual_net.scripts.dataset import AddSaltPepperNoise
from segmentation_models_pytorch.base import (
    ClassificationHead,
    SegmentationHead,
    SegmentationModel,
)
from segmentation_models_pytorch.decoders.unet.decoder import UnetDecoder
from segmentation_models_pytorch.decoders.unet.model import Unet


class CustomUnet(Unet):
    def __init__(
        self,
        has_dropout: bool = True,
        encoder_name: str = "resnet34",
        encoder_depth: int = 5,
        encoder_weights: Optional[str] = "imagenet",
        decoder_use_batchnorm: bool = True,
        decoder_channels: List[int] = (256, 128, 64, 32, 16),
        decoder_attention_type: Optional[str] = None,
        in_channels: int = 3,
        classes: int = 1,
        activation: Optional[Union[str, callable]] = None,
        aux_params: Optional[dict] = None,
    ):
        super().__init__(
            encoder_name,
            encoder_depth,
            encoder_weights,
            decoder_use_batchnorm,
            decoder_channels,
            decoder_attention_type,
            in_channels,
            classes,
            activation,
            aux_params,
        )

        self.has_dropput = has_dropout
        self.dropout = nn.Dropout2d(p=0.5, inplace=False)

    def forward(self, x, activate_dropout=False):
        """Sequentially pass `x` trough model`s encoder, decoder and heads"""
        if not activate_dropout:
            has_dropout = self.has_dropput
            self.has_dropput = False

        self.check_input_shape(x)

        features = self.encoder(x)
        if self.has_dropput:
            features[-1] = self.dropout(features[-1])

        decoder_output = self.decoder(*features)
        if self.has_dropput:
            decoder_output = self.dropout(decoder_output)

        masks = self.segmentation_head(decoder_output)

        if self.classification_head is not None:
            labels = self.classification_head(features[-1])
            return masks, labels

        if not activate_dropout:
            self.has_dropput = has_dropout

        return masks

    @torch.no_grad()
    def predict(self, x):
        """Inference method. Switch model to `eval` mode, call `.forward(x)` with `torch.no_grad()`

        Args:
            x: 4D torch tensor with shape (batch_size, channels, height, width)

        Return:
            prediction: 4D torch tensor with shape (batch_size, classes, height, width)

        """
        if self.training:
            self.eval()

        x = self.forward(x, activate_dropout=False)

        return x


rgb_aug = albu.Compose(
    [
        # albu.GaussNoise(var_limit=0.001, p=1),
        albu.ColorJitter(brightness=0.01, contrast=0.01, saturation=0.01, hue=0.01, p=1)
    ]
)
depth_aug = albu.Compose(
    [
        # albu.GaussNoise(var_limit=0.001, p=1),
        AddSaltPepperNoise(density=0.001, p=1)
    ]
)


def test_flip(x: np.ndarray, flip_mode):
    # flip_mode in [0, 1, 2, 3]
    if flip_mode == 3:
        return x
    # 0: both 1: hori 2: vert
    if flip_mode == 0:
        return x[::-1, ::-1]
    if flip_mode == 1:
        return x[:, ::-1]
    if flip_mode == 2:
        return x[::-1]


def feature_exchange(f_rgb, f_d, p=0.2):
    exchange_f_rgb = f_rgb.clone()
    exchange_f_d = f_d.clone()
    C = f_rgb.shape[1]
    sC = int(C * p)
    for i in range(f_rgb.shape[0]):
        exchange_f_rgb[i, -sC:] = f_d[i, -sC:]
        exchange_f_d[i, -sC:] = f_rgb[i, -sC:]
    return exchange_f_rgb, exchange_f_d


class CustomMobileNetEncoder(nn.Module):
    def __init__(self, in_channels=4, depth=5, dropout_p=0.2, **kwargs):
        super().__init__(**kwargs)
        self._depth = depth
        self.in_channels = in_channels
        self.out_channels = (3, 16, 24, 32, 96)

        self.dropout = nn.Dropout2d(dropout_p, inplace=True)
        # copy encoders for rgb and depth
        self.features_rgb = torchvision.models.MobileNetV2()  # weight init inside
        self.features_depth = torchvision.models.MobileNetV2()
        del self.features_rgb.classifier, self.features_depth.classifier
        self.stages_rgb = [
            nn.Identity(),
            self.features_rgb.features[:2],
            self.features_rgb.features[2:4],
            self.features_rgb.features[4:7],
            self.features_rgb.features[7:14],
            self.features_rgb.features[14:],
        ]
        self.stages_d = [
            nn.Identity(),
            self.features_depth.features[:2],
            self.features_depth.features[2:4],
            self.features_depth.features[4:7],
            self.features_depth.features[7:14],
            self.features_depth.features[14:],
        ]
        # conv skip connection
        self.connections = nn.ModuleList([nn.Identity()])
        for i in range(1, depth + 1):
            self.connections.append(
                nn.Sequential(
                    nn.Conv2d(
                        self.out_channels[i],
                        self.out_channels[i],
                        kernel_size=3,
                        padding=1,
                    ),
                    nn.BatchNorm2d(self.out_channels[i]),
                    nn.ReLU(),
                )
            )

    def forward(self, x):
        features = []
        f_rgb, f_d = x[:, :3], x[:, 3:4]
        f_d = f_d.repeat(1, 3, 1, 1)
        for i in range(self._depth + 1):
            # print(f_rgb.shape, f_d.shape)
            if i == 0:
                f_rgb = self.stages_rgb[i](f_rgb)
                f_d = self.stages_d[i](f_d)
            else:
                f_rgb, f_d = feature_exchange(f_rgb, f_d)
                f_rgb = self.dropout(self.stages_rgb[i](f_rgb))
                f_d = self.dropout(self.stages_d[i](f_d))
            features.append(self.connections[i](f_rgb + f_d))

        return features


class SplitUnet(SegmentationModel):
    def __init__(
        self,
        in_channels: int = 4,
        dropout_p: float = 0.2,
        encoder_name: str = "resnet34",
        encoder_depth: int = 5,
        encoder_weights: Optional[str] = "imagenet",
        decoder_use_batchnorm: bool = True,
        decoder_channels: List[int] = (256, 128, 64, 32, 16),
        decoder_attention_type: Optional[str] = None,
        classes: int = 1,
        activation=None,
    ):
        super().__init__()

        self.encoder = CustomMobileNetEncoder(
            in_channels=in_channels, depth=encoder_depth
        )

        self.decoder = UnetDecoder(
            encoder_channels=self.encoder.out_channels,
            decoder_channels=decoder_channels,
            n_blocks=encoder_depth,
            use_batchnorm=decoder_use_batchnorm,
            center=True if encoder_name.startswith("vgg") else False,
            attention_type=decoder_attention_type,
        )

        self.segmentation_head = SegmentationHead(
            in_channels=decoder_channels[-1],
            out_channels=classes,
            activation=activation,
            kernel_size=3,
        )

        self.in_channels = in_channels

        self.dropout_p = dropout_p
        self.dropout = nn.Dropout2d(p=dropout_p)

        self.classification_head = None

        self.name = "u-{}".format(encoder_name)
        self.initialize()

    def forward(self, x, activate_dropout=False):
        """Sequentially pass `x` trough model`s encoder, decoder and heads."""
        # self.check_input_shape(x)
        if activate_dropout:
            cur_dropout_status = self.dropout.training
            self.dropout.train()
            self.encoder.dropout.train()

        # use split backbone to extract features
        features = self.encoder(x)

        decoder_output = self.decoder(*features)
        decoder_output = self.dropout(decoder_output)

        masks = self.segmentation_head(decoder_output)

        if activate_dropout:
            self.dropout.train(cur_dropout_status)
            self.encoder.dropout.train(cur_dropout_status)

        return masks

    def predict(self, x, T=10, dropout=False):
        # no aug
        if T == 0:
            return self.forward(x, activate_dropout=dropout)

        # predict with test-time augmentation
        device = x.device
        x = x.cpu().numpy().squeeze()
        x = x.transpose(1, 2, 0)

        masks = []
        for _ in range(T):
            # aug
            flip_mode = np.random.randint(0, 4)
            aug_x = test_flip(x, flip_mode)
            aug_x[..., :3] = rgb_aug(image=aug_x[..., :3])["image"]
            aug_x[..., 3:4] = depth_aug(image=aug_x[..., 3:4])["image"]
            # aug as np.array
            aug_x = aug_x.transpose(2, 0, 1)[None].copy()
            aug_x = torch.from_numpy(aug_x).to(device=device, dtype=torch.float32)
            # mean
            cur_seg = self.forward(aug_x, activate_dropout=dropout)
            cur_seg = cur_seg.detach().cpu().numpy().squeeze().transpose(1, 2, 0)
            cur_seg = test_flip(cur_seg, flip_mode)
            cur_seg = torch.from_numpy(cur_seg.copy().transpose(2, 0, 1))
            masks.append(cur_seg.to(device=device, dtype=torch.float32))
        masks = F.softmax(torch.stack(masks), 1)
        masks = masks.mean(0, keepdim=True)
        return masks


class CustomUnetNew(Unet):
    def __init__(
        self,
        dropout_p: float = 0.2,
        encoder_name: str = "resnet34",
        encoder_depth: int = 5,
        encoder_weights: Optional[str] = "imagenet",
        decoder_use_batchnorm: bool = True,
        decoder_channels: List[int] = (256, 128, 64, 32, 16),
        decoder_attention_type: Optional[str] = None,
        in_channels: int = 3,
        classes: int = 1,
    ):
        super().__init__(
            encoder_name,
            encoder_depth,
            encoder_weights,
            decoder_use_batchnorm,
            decoder_channels,
            decoder_attention_type,
            in_channels,
            classes,
        )

        self.dropout_p = dropout_p
        self.dropout = nn.Dropout2d(p=self.dropout_p)

    def forward(self, x, activate_dropout=False):
        """Sequentially pass `x` trough model`s encoder, decoder and heads."""
        # self.check_input_shape(x)
        if activate_dropout:
            cur_dropout_status = self.dropout.training
            self.dropout.train()

        features = self.encoder(x)
        features[-1] = self.dropout(features[-1])

        decoder_output = self.decoder(*features)
        decoder_output = self.dropout(decoder_output)

        masks = self.segmentation_head(decoder_output)

        if activate_dropout:
            self.dropout.train(cur_dropout_status)

        if self.classification_head is not None:
            labels = self.classification_head(features[-1])
            return masks, labels
        return masks

    def predict(self, x, T=10, dropout=False):
        # no aug
        if T == 0:
            return self.forward(x, activate_dropout=dropout)

        # predict with test-time augmentation
        device = x.device
        x = x.cpu().numpy().squeeze()
        x = x.transpose(1, 2, 0)

        masks = []
        for _ in range(T):
            # aug
            flip_mode = np.random.randint(0, 4)
            aug_x = test_flip(x, flip_mode)
            aug_x[..., :3] = rgb_aug(image=aug_x[..., :3])["image"]
            aug_x[..., 3:4] = depth_aug(image=aug_x[..., 3:4])["image"]
            # aug as np.array
            aug_x = aug_x.transpose(2, 0, 1)[None].copy()
            aug_x = torch.from_numpy(aug_x).to(device=device, dtype=torch.float32)
            # mean
            cur_seg = self.forward(aug_x, activate_dropout=dropout)
            cur_seg = cur_seg.detach().cpu().numpy().squeeze().transpose(1, 2, 0)
            cur_seg = test_flip(cur_seg, flip_mode)
            cur_seg = torch.from_numpy(cur_seg.copy().transpose(2, 0, 1))
            masks.append(cur_seg.to(device=device, dtype=torch.float32))
        masks = F.softmax(torch.stack(masks), 1)
        masks = masks.mean(0, keepdim=True)
        return masks


def weighted_label_smoothed_nll_loss(
    lprobs: torch.Tensor,
    target: torch.Tensor,
    weight: torch.Tensor,
    epsilon: float,
    ignore_index=None,
    reduction="mean",
    dim=-1,
) -> torch.Tensor:
    """NLL loss with label smoothing.

    References:
        https://github.com/pytorch/fairseq/blob/master/fairseq/criterions/label_smoothed_cross_entropy.py

    Args:
        lprobs (torch.Tensor): Log-probabilities of predictions (e.g after log_softmax)
    """
    if target.dim() == lprobs.dim() - 1:
        target = target.unsqueeze(dim)

    if ignore_index is not None:
        pad_mask = target.eq(ignore_index)
        target = target.masked_fill(pad_mask, 0)
        nll_loss = -lprobs.gather(dim=dim, index=target)
        smooth_loss = -lprobs.sum(dim=dim, keepdim=True)

        # nll_loss.masked_fill_(pad_mask, 0.0)
        # smooth_loss.masked_fill_(pad_mask, 0.0)
        nll_loss = nll_loss.masked_fill(pad_mask, 0.0)
        smooth_loss = smooth_loss.masked_fill(pad_mask, 0.0)
    else:
        nll_loss = -lprobs.gather(dim=dim, index=target)
        smooth_loss = -lprobs.sum(dim=dim, keepdim=True)

        nll_loss = nll_loss.squeeze(dim)
        smooth_loss = smooth_loss.squeeze(dim)

    # multiply weight
    # print(nll_loss.shape, weight.shape)
    nll_loss *= weight
    smooth_loss *= weight

    if reduction == "sum":
        nll_loss = nll_loss.sum()
        smooth_loss = smooth_loss.sum()
    if reduction == "mean":
        nll_loss = nll_loss.mean()
        smooth_loss = smooth_loss.mean()

    eps_i = epsilon / lprobs.size(dim)
    loss = (1.0 - epsilon) * nll_loss + eps_i * smooth_loss
    return loss


class WCELoss(nn.Module):
    __constants__ = ["reduction", "ignore_index", "smooth_factor"]

    def __init__(
        self,
        reduction: str = "mean",
        smooth_factor: Optional[float] = None,
        ignore_index: Optional[int] = -100,
        dim: int = 1,
    ):
        """Drop-in replacement for torch.nn.CrossEntropyLoss with
        label_smoothing.

        Args:
            smooth_factor: Factor to smooth target (e.g. if smooth_factor=0.1 then [1, 0, 0] -> [0.9, 0.05, 0.05])

        Shape
             - **y_pred** - torch.Tensor of shape (N, C, H, W)
             - **y_true** - torch.Tensor of shape (N, H, W)

        Reference
            https://github.com/BloodAxe/pytorch-toolbelt
        """
        super().__init__()
        self.smooth_factor = smooth_factor
        self.ignore_index = ignore_index
        self.reduction = reduction
        self.dim = dim

    def forward(
        self, y_pred: torch.Tensor, y_true: torch.Tensor, weight: torch.Tensor
    ) -> torch.Tensor:
        log_prob = F.log_softmax(y_pred, dim=self.dim)
        return weighted_label_smoothed_nll_loss(
            log_prob,
            y_true,
            weight,
            epsilon=self.smooth_factor,
            ignore_index=self.ignore_index,
            reduction=self.reduction,
            dim=self.dim,
        )
