import copy
from typing import List

import torch
import torch.nn as nn
from arti_mani.algorithms.visual_net.Networks.base_model import BaseModel
from arti_mani.algorithms.visual_net.Networks.utils import (
    act_layer,
    norm_layer1d,
    norm_layer2d,
)

LRELU_SLOPE = 0.02


class DenseBlock(nn.Module):
    def __init__(self, in_features, out_features, norm=None, activation=None):
        super(DenseBlock, self).__init__()
        self.linear = nn.Linear(in_features, out_features)

        if activation is None:
            nn.init.xavier_uniform_(
                self.linear.weight, gain=nn.init.calculate_gain("linear")
            )
            nn.init.zeros_(self.linear.bias)
        elif activation == "tanh":
            nn.init.xavier_uniform_(
                self.linear.weight, gain=nn.init.calculate_gain("tanh")
            )
            nn.init.zeros_(self.linear.bias)
        elif activation == "lrelu":
            nn.init.kaiming_uniform_(
                self.linear.weight, a=LRELU_SLOPE, nonlinearity="leaky_relu"
            )
            nn.init.zeros_(self.linear.bias)
        elif activation == "relu":
            nn.init.kaiming_uniform_(self.linear.weight, nonlinearity="relu")
            nn.init.zeros_(self.linear.bias)
        else:
            raise ValueError()

        self.activation = None
        self.norm = None
        if norm is not None:
            self.norm = norm_layer1d(norm, out_features)
        if activation is not None:
            self.activation = act_layer(activation)

    def forward(self, x):
        x = self.linear(x)
        x = self.norm(x) if self.norm is not None else x
        x = self.activation(x) if self.activation is not None else x
        return x


class Conv2DBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_sizes,
        strides,
        norm=None,
        activation=None,
        padding_mode="replicate",
    ):
        super(Conv2DBlock, self).__init__()
        padding = (
            kernel_sizes // 2
            if isinstance(kernel_sizes, int)
            else (kernel_sizes[0] // 2, kernel_sizes[1] // 2)
        )
        self.conv2d = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_sizes,
            strides,
            padding=padding,
            padding_mode=padding_mode,
        )

        if activation is None:
            nn.init.xavier_uniform_(
                self.conv2d.weight, gain=nn.init.calculate_gain("linear")
            )
            nn.init.zeros_(self.conv2d.bias)
        elif activation == "tanh":
            nn.init.xavier_uniform_(
                self.conv2d.weight, gain=nn.init.calculate_gain("tanh")
            )
            nn.init.zeros_(self.conv2d.bias)
        elif activation == "lrelu":
            nn.init.kaiming_uniform_(
                self.conv2d.weight, a=LRELU_SLOPE, nonlinearity="leaky_relu"
            )
            nn.init.zeros_(self.conv2d.bias)
        elif activation == "relu":
            nn.init.kaiming_uniform_(self.conv2d.weight, nonlinearity="relu")
            nn.init.zeros_(self.conv2d.bias)
        else:
            raise ValueError()

        self.activation = None
        self.norm = None
        if norm is not None:
            self.norm = norm_layer2d(norm, out_channels)
        if activation is not None:
            self.activation = act_layer(activation)

    def forward(self, x):
        x = self.conv2d(x)
        x = self.norm(x) if self.norm is not None else x
        x = self.activation(x) if self.activation is not None else x
        return x


class Conv2DUpsampleBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_sizes,
        strides,
        norm=None,
        activation=None,
    ):
        super(Conv2DUpsampleBlock, self).__init__()
        layer = [
            Conv2DBlock(in_channels, out_channels, kernel_sizes, 1, norm, activation)
        ]
        if strides > 1:
            layer.append(
                nn.Upsample(scale_factor=strides, mode="bilinear", align_corners=False)
            )
        convt_block = Conv2DBlock(
            out_channels, out_channels, kernel_sizes, 1, norm, activation
        )
        layer.append(convt_block)
        self.conv_up = nn.Sequential(*layer)

    def forward(self, x):
        return self.conv_up(x)


class SiameseNet(BaseModel):
    def __init__(
        self,
        input_channels: List[int],
        filters: List[int],
        kernel_sizes: List[int],
        strides: List[int],
        norm: str = None,
        activation: str = "relu",
    ):
        super(SiameseNet, self).__init__()
        self._input_channels = input_channels
        self._filters = filters
        self._kernel_sizes = kernel_sizes
        self._strides = strides
        self._norm = norm
        self._activation = activation
        self.output_channels = filters[-1]  # * len(input_channels)

        self._siamese_blocks = nn.ModuleList()
        for i, ch in enumerate(self._input_channels):
            blocks = []
            for i, (filt, ksize, stride) in enumerate(
                zip(self._filters, self._kernel_sizes, self._strides)
            ):
                conv_block = Conv2DBlock(
                    ch, filt, ksize, stride, self._norm, self._activation
                )
                blocks.append(conv_block)
                ch = filt
            self._siamese_blocks.append(nn.Sequential(*blocks))
        self._fuse = Conv2DBlock(
            self._filters[-1] * len(self._siamese_blocks),
            self._filters[-1],
            1,
            1,
            self._norm,
            self._activation,
        )

    def forward(self, x):
        if len(x) != len(self._siamese_blocks):
            raise ValueError(
                "Expected a list of tensors of size %d." % len(self._siamese_blocks)
            )
        self.streams = [stream(y) for y, stream in zip(x, self._siamese_blocks)]
        y = self._fuse(torch.cat(self.streams, 1))
        return y


class Attention2DUNet(BaseModel):
    def __init__(
        self,
        siamese_net: SiameseNet,
        filters: List[int],
        kernel_sizes: List[int],
        strides: List[int],
        norm: str = None,
        activation: str = "relu",
        output_channels: int = 1,
        skip_connections: bool = True,
    ):
        super(Attention2DUNet, self).__init__()
        self._siamese_net = copy.deepcopy(siamese_net)
        self._input_channels = self._siamese_net.output_channels
        self._filters = filters
        self._kernel_sizes = kernel_sizes
        self._strides = strides
        self._norm = norm
        self._activation = activation
        self._output_channels = output_channels
        self._skip_connections = skip_connections

        self._down = []
        ch = self._input_channels
        for filt, ksize, stride in zip(
            self._filters, self._kernel_sizes, self._strides
        ):
            conv_block = Conv2DBlock(
                ch,
                filt,
                ksize,
                stride,
                self._norm,
                self._activation,
                padding_mode="replicate",
            )
            ch = filt
            self._down.append(conv_block)
        self._down = nn.ModuleList(self._down)

        reverse_conv_data = list(zip(self._filters, self._kernel_sizes, self._strides))
        reverse_conv_data.reverse()

        self._up = []
        for i, (filt, ksize, stride) in enumerate(reverse_conv_data):
            if i > 0 and self._skip_connections:
                # ch += reverse_conv_data[-i-1][0]
                ch += reverse_conv_data[i][0]
            convt_block = Conv2DUpsampleBlock(
                ch, filt, ksize, stride, self._norm, self._activation
            )
            ch = filt
            self._up.append(convt_block)
        self._up = nn.ModuleList(self._up)

        self._final_conv = Conv2DBlock(
            ch, self._output_channels, 3, 1, padding_mode="replicate"
        )

    def forward(self, observations):
        x = self._siamese_net(observations)
        _, _, h, w = x.shape
        layers_for_skip = []
        for l in self._down:
            x = l(x)
            layers_for_skip.append(x)
        self.latent = x  # (N, 16, H/4, W/4)
        layers_for_skip.reverse()
        for i, l in enumerate(self._up):
            if i > 0 and self._skip_connections:
                # Skip connections. Skip the first up layer.
                x = torch.cat([layers_for_skip[i], x], 1)
            x = l(x)
        x = self._final_conv(x)  # (N, 2, H, W)
        return x, self.latent


class WeightGen(BaseModel):
    """
    Unet seg feature encoding maps to generate weights
    """

    def __init__(
        self,
        feat_chan: int = 16,
        weight_num: int = 5,
        norm: str = None,
        activation: str = None,
    ):
        super(WeightGen, self).__init__()

        self.maxpool = nn.AdaptiveMaxPool2d(1)
        self.flatten = nn.Flatten()
        self.fc1 = DenseBlock(
            in_features=feat_chan,
            out_features=feat_chan * 4,
            norm=norm,
            activation=activation,
        )
        self.fc2 = DenseBlock(
            in_features=feat_chan * 4,
            out_features=feat_chan * 4,
            norm=norm,
            activation=activation,
        )
        self.fc3 = DenseBlock(
            in_features=feat_chan * 4,
            out_features=weight_num,
            norm=norm,
            activation=activation,
        )

    def forward(self, x):
        feat_max = self.flatten(self.maxpool(x))
        feat1 = self.fc1(feat_max)
        feat2 = self.fc2(feat1)
        weight = self.fc3(feat2)
        return weight


class FeatAggNet(BaseModel):
    """
    Unet seg feature maps aggregation net
    """

    def __init__(
        self,
        in_chan: int = 5,
        out_chan: int = 1,
        norm: str = None,
        activation: str = None,
    ):
        super(FeatAggNet, self).__init__()

        self.conv1 = Conv2DBlock(
            in_channels=in_chan,
            out_channels=out_chan,
            kernel_sizes=1,
            strides=1,
            norm=norm,
            activation=activation,
        )

    def forward(self, x):
        feat_norm = self.conv1(x)
        attn_map = torch.sigmoid(feat_norm)
        return attn_map


class AttentionDepthUNet(BaseModel):
    def __init__(
        self,
        depth_feat_channels: int,
        filters: List[int],
        kernel_sizes: List[int],
        strides: List[int],
        norm: str = None,
        activation: str = "relu",
        output_channels: int = 1,
        skip_connections: bool = True,
    ):
        super(AttentionDepthUNet, self).__init__()
        self._depth_feat_channels = depth_feat_channels
        self._filters = filters
        self._kernel_sizes = kernel_sizes
        self._strides = strides
        self._norm = norm
        self._activation = activation
        self._output_channels = output_channels
        self._skip_connections = skip_connections

        self.depth_conv_block = Conv2DBlock(
            in_channels=1,
            out_channels=self._depth_feat_channels,
            kernel_sizes=5,
            strides=1,
            norm=None,
            activation=self._norm,
        )

        self._down = []
        ch = self._depth_feat_channels
        for filt, ksize, stride in zip(
            self._filters, self._kernel_sizes, self._strides
        ):
            conv_block = Conv2DBlock(
                ch,
                filt,
                ksize,
                stride,
                self._norm,
                self._activation,
                padding_mode="replicate",
            )
            ch = filt
            self._down.append(conv_block)
        self._down = nn.ModuleList(self._down)

        reverse_conv_data = list(zip(self._filters, self._kernel_sizes, self._strides))
        reverse_conv_data.reverse()

        self._up = []
        for i, (filt, ksize, stride) in enumerate(reverse_conv_data):
            if i > 0 and self._skip_connections:
                # ch += reverse_conv_data[-i-1][0]
                ch += reverse_conv_data[i][0]
            convt_block = Conv2DUpsampleBlock(
                ch, filt, ksize, stride, self._norm, self._activation
            )
            ch = filt
            self._up.append(convt_block)
        self._up = nn.ModuleList(self._up)

        self._final_conv = Conv2DBlock(
            ch, self._output_channels, 3, 1, padding_mode="replicate"
        )
        # self.weight = torch.nn.Parameter(
        #                     torch.ones(ch) / ch, requires_grad=True
        #                 )

    def forward(self, observations):
        x = self.depth_conv_block(observations)
        _, _, h, w = x.shape
        self.ups = []
        self.downs = []
        layers_for_skip = []
        for l in self._down:
            x = l(x)
            layers_for_skip.append(x)
            self.downs.append(x)
        self.latent = x  # (N, 16, H/4, W/4)
        layers_for_skip.reverse()
        for i, l in enumerate(self._up):
            if i > 0 and self._skip_connections:
                # Skip connections. Skip the first up layer.
                x = torch.cat([layers_for_skip[i], x], 1)
            x = l(x)
            self.ups.append(x)
        x = self._final_conv(x)  # (N, 2, H, W)
        # x = F.softmax(x, dim=1)  # (N, 16, H, W)
        # weight = F.softmax(self.weight, dim=0)
        # soft_weight_map = torch.einsum("ijkl,j->ikl", [x, weight]) # (N, H, W)
        return x


class UNetEncoder(BaseModel):
    def __init__(
        self,
        siamese_net: SiameseNet,
        filters: List[int],
        kernel_sizes: List[int],
        strides: List[int],
        norm: str = None,
        activation: str = "relu",
    ):
        super(UNetEncoder, self).__init__()
        self._siamese_net = copy.deepcopy(siamese_net)
        self._input_channels = self._siamese_net.output_channels
        self._filters = filters
        self._kernel_sizes = kernel_sizes
        self._strides = strides
        self._norm = norm
        self._activation = activation
        self._down = []
        ch = self._input_channels
        for filt, ksize, stride in zip(
            self._filters, self._kernel_sizes, self._strides
        ):
            conv_block = Conv2DBlock(
                ch,
                filt,
                ksize,
                stride,
                self._norm,
                self._activation,
                padding_mode="replicate",
            )
            ch = filt
            self._down.append(conv_block)
        self._down = nn.ModuleList(self._down)
        self.maxpool = nn.AdaptiveMaxPool2d(1)  # (N, C, 1, 1)
        self.flatten = nn.Flatten()

    def forward(self, observations):
        x = self._siamese_net(observations)
        _, _, h, w = x.shape
        self.downs = []
        for l in self._down:
            x = l(x)
            self.downs.append(x)
        x = self.maxpool(x)  # (N, 16, H/4, W/4) => (N, 16)
        max_feat = self.flatten(x)  # (N, 16)
        return max_feat


class GeneralCNN(BaseModel):
    def __init__(
        self,
        siamese_net: SiameseNet,
        filters: List[int],
        kernel_sizes: List[int],
        strides: List[int],
        norm: str = None,
        activation: str = "relu",
    ):
        super(GeneralCNN, self).__init__()
        self._siamese_net = copy.deepcopy(siamese_net)
        self._input_channels = self._siamese_net.output_channels
        self._filters = filters
        self._kernel_sizes = kernel_sizes
        self._strides = strides
        self._norm = norm
        self._activation = activation
        self._down = []
        blocks_num = len(self._filters)
        ch = self._input_channels
        for filt, ksize, stride in zip(
            self._filters, self._kernel_sizes, self._strides
        ):
            conv_block = Conv2DBlock(
                ch,
                filt,
                ksize,
                stride,
                self._norm,
                self._activation,
                padding_mode="replicate",
            )
            ch = filt
            self._down.append(conv_block)
        self._down = nn.ModuleList(self._down)
        self.flatten = nn.Flatten()
        cnnfeat_size = ch * 72 * 128 // 4**blocks_num
        self.final = nn.Sequential(
            DenseBlock(cnnfeat_size, 1024, activation=self._activation),
            DenseBlock(1024, 128, activation=self._activation),
        )

    def forward(self, observations):
        x = self._siamese_net(observations)
        _, _, h, w = x.shape
        self.downs = []
        for l in self._down:
            x = l(x)
            self.downs.append(x)
        x = self.flatten(x)  # (N, 16, H/4, W/4) => (N, -1)
        final_feat = self.final(x)  # (N, 128)
        return final_feat


if __name__ == "__main__":
    data = [torch.zeros([4, 3, 72, 128]), torch.zeros([4, 1, 72, 128])]
    siamese_net = SiameseNet(
        input_channels=[3, 1],
        filters=[8],
        kernel_sizes=[5],
        strides=[1],
        activation="lrelu",
        norm=None,
    )
    attention_net = Attention2DUNet(
        siamese_net=siamese_net,
        filters=[16, 32, 64],
        kernel_sizes=[5, 5, 5],
        strides=[2, 2, 2],
        output_channels=2,
        norm=None,
        activation="lrelu",
        skip_connections=True,
    )
    for param in attention_net.parameters():
        param.requires_grad = True
    y, latent = attention_net(data)
    print(f"Attention2DUNet output size: {y.shape}, {latent.shape}")

    seg_map = torch.randn(4, 5, 72, 128)
    featagg = FeatAggNet(
        in_chan=5,
        out_chan=1,
        norm=None,
        activation="relu",
    )
    y = featagg(seg_map)
    print(f"FeatAggNet output size: {y.shape}")

    unet_encoder = UNetEncoder(
        siamese_net=siamese_net,
        filters=[16, 16],
        kernel_sizes=[5, 5],
        strides=[2, 2],
        norm=None,
        activation="lrelu",
    )
    y = unet_encoder(data)
    print(f"UNetEncoder output size: {y.shape}")

    gene_cnn = GeneralCNN(
        siamese_net=siamese_net,
        filters=[16, 16],
        kernel_sizes=[5, 5],
        strides=[2, 2],
        norm=None,
        activation="lrelu",
    )
    y = gene_cnn(data)
    print(f"GeneralCNN output size: {y.shape}")

    attn_depth = AttentionDepthUNet(
        depth_feat_channels=8,
        filters=[16, 16],
        kernel_sizes=[5, 5],
        strides=[2, 2],
        output_channels=2,
        norm=None,
        activation="lrelu",
        skip_connections=True,
    )
    y = attn_depth(data[1])
    print(f"AttentionDepthUNet output size: {y.shape}")
