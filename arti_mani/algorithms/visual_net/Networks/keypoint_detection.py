import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


def _assert_no_grad(tensor):
    assert not tensor.requires_grad, (
        "nn criterions don't compute the gradient w.r.t. targets - please "
        "mark these tensors as not requiring gradients"
    )


def soft_argmax(heatmaps, joint_num, output_shape):
    """
    Args:
        heatmaps: (N, J*D, H, W)
        joint_num: J, num of keypoints
        output_shape: (D, H, W)

    Returns:

    """
    assert isinstance(heatmaps, torch.Tensor)
    d, h, w = output_shape
    heatmaps = heatmaps.reshape((-1, joint_num, d * h * w))
    heatmaps = F.softmax(heatmaps, 2)
    heatmaps = heatmaps.reshape((-1, joint_num, d, h, w))

    accu_x = heatmaps.sum(dim=(2, 3))  # (N, J, W)
    accu_y = heatmaps.sum(dim=(2, 4))  # (N, J, H)
    accu_z = heatmaps.sum(dim=(3, 4))  # (N, J, D)

    # accu_x = accu_x * torch.arange(1, w + 1).float().to(accu_x.device)
    # accu_y = accu_y * torch.arange(1, h + 1).float().to(accu_y.device)
    # accu_z = accu_z * torch.arange(1, d + 1).float().to(accu_z.device)
    accu_x = accu_x * (torch.arange(0, w) / w).float().to(
        accu_x.device
    )  # normalized the coord [0, 1)
    accu_y = accu_y * (torch.arange(0, h) / h).float().to(accu_y.device)
    accu_z = accu_z * (torch.arange(0, d) / d).float().to(accu_z.device)

    accu_x = accu_x.sum(dim=2, keepdim=True)  # (N, J, 1)
    accu_y = accu_y.sum(dim=2, keepdim=True)  # (N, J, 1)
    accu_z = accu_z.sum(dim=2, keepdim=True)  # (N, J, 1)

    coord_out = torch.cat((accu_x, accu_y, accu_z), dim=2)  # (N, J, 3)

    return coord_out


class JointLocationLoss(nn.Module):
    def __init__(self):
        super(JointLocationLoss, self).__init__()

    def forward(self, heatmap_out, gt_coord, out_shape, gt_vis):
        joint_num = gt_coord.shape[1]
        coord_out = soft_argmax(heatmap_out, joint_num, out_shape)  # (N, J, 3)

        ## set not visable coord_out == gt_coord to make it not useful
        not_vis_coord = gt_vis == 0  # (N, J)
        gt_coord[not_vis_coord] = coord_out[not_vis_coord].clone().detach()

        _assert_no_grad(gt_coord)  # (N, J, 3)
        _assert_no_grad(gt_vis)  # (N, J)

        ## l1 loss
        # loss = F.l1_loss(coord_out, gt_coord, reduction="mean")
        # loss = torch.abs(coord_out - gt_coord)  # (N, J, 3)
        # loss = torch.einsum('ijk,ij->ijk', torch.abs(coord_out - gt_coord), gt_vis)  # (N, J, 3)
        ## smooth l1 loss
        # loss = F.smooth_l1_loss(coord_out, gt_coord) * 10
        ## l2 loss
        loss = F.mse_loss(coord_out, gt_coord) * 100

        # loss = (loss[:, :, 0] + loss[:, :, 1] + loss[:, :, 2]) / 3.
        # loss = loss.mean()

        return loss


class IntegralHumanPoseModel(nn.Module):
    def __init__(
        self,
        num_keypoints=3,
        num_deconv_layers=3,
        depth_dim=64,
        has_dropout=True,
        mode="RGBD",
    ):
        super(IntegralHumanPoseModel, self).__init__()

        self.mode = mode
        init_in_ch = 32  # 32, 64
        # init RGBD input layers
        if mode == "RGBD":
            self.init_layer = nn.Sequential(
                nn.Conv2d(
                    4,
                    init_in_ch,
                    kernel_size=(3, 3),
                    stride=(2, 2),
                    padding=(1, 1),
                    bias=False,
                ),
                nn.BatchNorm2d(init_in_ch),  # 32, 64
                nn.ReLU(inplace=True),
                # nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
            )
        elif self.mode == "RGB":
            pass
        elif self.mode == "D":
            self.init_layer = nn.Sequential(
                nn.Conv2d(
                    1,
                    init_in_ch,
                    kernel_size=(3, 3),
                    stride=(2, 2),
                    padding=(1, 1),
                    bias=False,
                ),
                nn.BatchNorm2d(init_in_ch),  # 32, 64
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            )
        else:
            raise NotImplementedError(self.mode)

        # Feature extraction network
        # resnet = models.resnet34(pretrained=True)
        # resnet = models.resnet18(pretrained=True)
        # self.feature_extraction = nn.Sequential(*list(resnet.children())[4:-2])

        mobilenet = models.mobilenet_v2(pretrained=True)
        # self.feature_extraction = nn.Sequential(*list(mobilenet.features[1:-1]))
        # self.feature_extraction = nn.Sequential(*list(mobilenet.features[:-1]))
        if self.mode == "RGBD" or self.mode == "D":
            self.feature_extraction = nn.Sequential(*list(mobilenet.features[1:-1]))
            # self.feature_extraction = nn.Sequential(*list(mobilenet.features[1:8]))
        elif self.mode == "RGB":
            # self.feature_extraction = nn.Sequential(*list(mobilenet.features[:-1]))
            self.feature_extraction = nn.Sequential(*list(mobilenet.features[:8]))

        # self.middle_net = nn.Sequential(
        #     nn.Conv2d(64, 512, kernel_size=(1, 1), stride=(1, 1), bias=False),
        #     nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=512, bias=False),
        #     nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d(512, 512, kernel_size=(1, 1), stride=(1, 1), bias=False),
        #     nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
        # )

        self.has_dropout = has_dropout
        self.dropout = nn.Dropout2d(p=0.2, inplace=False)

        # Intermediate supervision layers
        # in_ch = 64
        in_ch = 320
        out_ch = 256
        deconv_layers = []
        for i in range(num_deconv_layers):
            deconv_layers.append(
                nn.ConvTranspose2d(
                    in_channels=in_ch,
                    out_channels=out_ch,
                    kernel_size=4,
                    stride=2,
                    padding=1,
                    output_padding=0,
                    bias=False,
                )
            )
            deconv_layers.append(nn.BatchNorm2d(out_ch))
            deconv_layers.append(nn.ReLU(inplace=True))
            in_ch = out_ch
        self.deconv_layers = nn.Sequential(*deconv_layers)

        # Output layers
        self.final_layer = nn.Conv2d(
            in_channels=in_ch,
            out_channels=num_keypoints * depth_dim,
            kernel_size=1,
            stride=1,
            padding=0,
        )

    def forward(self, x):
        # input layer
        if self.mode == "RGBD" or self.mode == "D":
            x = self.init_layer(x)

        # Feature extraction
        features = self.feature_extraction(x)
        # features = self.middle_net(features)
        if self.has_dropout:
            features = self.dropout(features)

        # Intermediate supervision
        deconv_result = self.deconv_layers(features)
        if self.has_dropout:
            deconv_result = self.dropout(deconv_result)

        # Output
        out = self.final_layer(deconv_result)

        return out

    def init_weights(self):
        for name, m in self.named_modules():
            if "feature_extraction" not in name:
                if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                    n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                    m.weight.data.normal_(0, math.sqrt(2.0 / n))
                elif isinstance(m, (nn.LayerNorm, nn.BatchNorm1d, nn.BatchNorm2d)):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.Linear):
                    nn.init.kaiming_normal_(m.weight)
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)


if __name__ == "__main__":
    estmodel = IntegralHumanPoseModel()
    estmodel.init_weights()
    print(estmodel)

    data = torch.randn((10, 4, 144, 256))

    out = estmodel(data)
    print(data.shape, out.shape)
    model_params = sum(p.numel() for p in estmodel.parameters() if p.requires_grad)
    print(f"* number of parameters: {model_params}")
