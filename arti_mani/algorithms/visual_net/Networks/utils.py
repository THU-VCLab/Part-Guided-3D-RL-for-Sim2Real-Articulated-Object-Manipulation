from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

LRELU_SLOPE = 0.02


def act_layer(act):
    if act == "relu":
        return nn.ReLU()
    elif act == "lrelu":
        return nn.LeakyReLU(LRELU_SLOPE)
    elif act == "elu":
        return nn.ELU()
    elif act == "tanh":
        return nn.Tanh()
    elif act == "prelu":
        return nn.PReLU()
    else:
        raise ValueError("%s not recognized." % act)


def norm_layer2d(norm, channels):
    if norm == "batch":
        return nn.BatchNorm2d(channels)
    elif norm == "instance":
        return nn.InstanceNorm2d(channels, affine=True)
    elif norm == "layer":
        return nn.GroupNorm(1, channels, affine=True)
    elif norm == "group":
        return nn.GroupNorm(4, channels, affine=True)
    else:
        raise ValueError("%s not recognized." % norm)


def norm_layer1d(norm, num_channels):
    if norm == "batch":
        return nn.BatchNorm1d(num_channels)
    elif norm == "instance":
        return nn.InstanceNorm1d(num_channels, affine=True)
    elif norm == "layer":
        return nn.LayerNorm(num_channels)
    else:
        raise ValueError("%s not recognized." % norm)


def create_meshgrid(
    x: torch.Tensor, normalized_coordinates: Optional[bool]
) -> (torch.Tensor, torch.Tensor):
    assert len(x.shape) == 4, x.shape
    _, _, height, width = x.shape
    _device, _dtype = x.device, x.dtype
    if normalized_coordinates:
        xs = torch.linspace(-1.0, 1.0, width, device=_device, dtype=_dtype)
        ys = torch.linspace(-1.0, 1.0, height, device=_device, dtype=_dtype)
    else:
        xs = torch.linspace(0, width - 1, width, device=_device, dtype=_dtype)
        ys = torch.linspace(0, height - 1, height, device=_device, dtype=_dtype)
    return torch.meshgrid(ys, xs)  # pos_y, pos_x


class SpatialSoftArgmax2d(nn.Module):
    r"""Creates a module that computes the Spatial Soft-Argmax 2D
    of a given input heatmap.

    Returns the index of the maximum 2d coordinates of the give map.
    The output order is x-coord and y-coord.

    Arguments:
        normalized_coordinates (Optional[bool]): wether to return the
          coordinates normalized in the range of [-1, 1]. Otherwise,
          it will return the coordinates in the range of the input shape.
          Default is True.

    Shape:
        - Input: :math:`(B, N, H, W)`
        - Output: :math:`(B, N, 2)`

    Examples::
        >>> input = torch.rand(1, 4, 2, 3)
        >>> m = tgm.losses.SpatialSoftArgmax2d()
        >>> coords = m(input)  # 1x4x2
        >>> x_coord, y_coord = torch.chunk(coords, dim=-1, chunks=2)
    """

    def __init__(self, normalized_coordinates: Optional[bool] = True) -> None:
        super(SpatialSoftArgmax2d, self).__init__()
        self.normalized_coordinates: Optional[bool] = normalized_coordinates
        self.eps: float = 1e-6

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if not torch.is_tensor(input):
            raise TypeError(
                "Input input type is not a torch.Tensor. Got {}".format(type(input))
            )
        if not len(input.shape) == 4:
            raise ValueError(
                "Invalid input shape, we expect BxCxHxW. Got: {}".format(input.shape)
            )
        # unpack shapes and create view from input tensor
        batch_size, channels, height, width = input.shape
        x: torch.Tensor = input.view(batch_size, channels, -1)

        # compute softmax with max substraction trick
        exp_x = torch.exp(x - torch.max(x, dim=-1, keepdim=True)[0])
        exp_x_sum = 1.0 / (exp_x.sum(dim=-1, keepdim=True) + self.eps)

        # create coordinates grid
        pos_y, pos_x = create_meshgrid(input, self.normalized_coordinates)
        pos_x = pos_x.reshape(-1)
        pos_y = pos_y.reshape(-1)

        # compute the expected coordinates
        expected_y: torch.Tensor = torch.sum(
            (pos_y * exp_x) * exp_x_sum, dim=-1, keepdim=True
        )
        expected_x: torch.Tensor = torch.sum(
            (pos_x * exp_x) * exp_x_sum, dim=-1, keepdim=True
        )
        output: torch.Tensor = torch.cat([expected_x, expected_y], dim=-1)
        return output.view(batch_size, channels, 2)  # BxNx2


######################
# functional interface
######################


def spatial_soft_argmax2d(
    input: torch.Tensor, normalized_coordinates: Optional[bool] = True
) -> torch.Tensor:
    r"""Function that computes the Spatial Soft-Argmax 2D of a given heatmap.

    See :class:`torchgeometry.contrib.SpatialSoftArgmax2d` for details.
    """
    return SpatialSoftArgmax2d(normalized_coordinates)(input)


class SpatialSoftmax(torch.nn.Module):
    def __init__(self, height, width, channel, temperature=None, data_format="NCHW"):
        super(SpatialSoftmax, self).__init__()
        self.data_format = data_format
        self.height = height
        self.width = width
        self.channel = channel

        if temperature:
            self.temperature = nn.Parameter(torch.ones(1) * temperature)
        else:
            self.temperature = 1.0

        pos_x, pos_y = np.meshgrid(
            np.linspace(-1.0, 1.0, self.height), np.linspace(-1.0, 1.0, self.width)
        )
        pos_x = torch.from_numpy(pos_x.reshape(self.height * self.width)).float()
        pos_y = torch.from_numpy(pos_y.reshape(self.height * self.width)).float()
        self.register_buffer("pos_x", pos_x)
        self.register_buffer("pos_y", pos_y)

    def forward(self, feature):
        # Output:
        #   (N, C*2) x_0 y_0 ...
        if self.data_format == "NHWC":
            feature = (
                feature.transpose(1, 3)
                .tranpose(2, 3)
                .view(-1, self.height * self.width)
            )
        else:
            feature = feature.view(-1, self.height * self.width)

        softmax_attention = F.softmax(feature / self.temperature, dim=-1)
        expected_x = torch.sum(self.pos_x * softmax_attention, dim=1, keepdim=True)
        expected_y = torch.sum(self.pos_y * softmax_attention, dim=1, keepdim=True)
        expected_xy = torch.cat([expected_x, expected_y], 1)
        feature_keypoints = expected_xy.view(-1, self.channel * 2)

        return feature_keypoints
