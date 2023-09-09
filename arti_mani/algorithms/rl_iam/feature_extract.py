import random
import time

import gym
import segmentation_models_pytorch as smp
import torch
import torch.nn.functional as F
import yaml
from arti_mani import VISUALMODEL_DIR
from arti_mani.algorithms.visual_net.Networks.conv2dnet import (
    Attention2DUNet,
    AttentionDepthUNet,
    SiameseNet,
    UNetEncoder,
)
from arti_mani.algorithms.visual_net.Networks.Custom_Unet import CustomUnet
from arti_mani.algorithms.visual_net.Networks.img_encoder import ImgEncoder
from arti_mani.algorithms.visual_net.Networks.pointnet import (
    PointNetDenseCls,
    PointNetfeat,
)
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class CustomStateExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.Space):
        # We do not know features-dim here before going over all the items,
        # so put something dummy for now. PyTorch requires calling
        # nn.Module.__init__ before adding modules
        super(CustomStateExtractor, self).__init__(observation_space, features_dim=1)

        self._features_dim = 222

    def forward(self, observations) -> torch.Tensor:
        qpos = observations["qpos"]  # (N, 9)
        ee_pos_base = observations["ee_pos_base"]  # (N, 3)
        bs = qpos.shape[0]
        ee_xyz = observations["eepad_pts"].view(bs, -1)  # (N, 40*3)
        target_xyz = observations["target_pcd_ee"].view(bs, -1)  # (N, 30*3)

        state = torch.cat([qpos, ee_pos_base, ee_xyz, target_xyz], dim=1)

        return state


class CustomProcessStateExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.Space, mode="robot_xyz"):
        # We do not know features-dim here before going over all the items,
        # so put something dummy for now. PyTorch requires calling
        # nn.Module.__init__ before adding modules
        super(CustomProcessStateExtractor, self).__init__(
            observation_space, features_dim=1
        )
        support_modes = ["robot_xyz", "cam_xyz", "cam_zvu"]
        assert mode in support_modes, f"only support modes: {support_modes}"
        print("training mode： ", mode)

        self.mode = mode
        self.base_cam_intrin = torch.tensor(
            [[45.5882, 0.0, 31.7436], [0.0, 45.547, 17.6573], [0.0, 0.0, 1.0]],
            requires_grad=True,
            device="cuda:0",
        )
        self.base_cam_extrin = torch.tensor(
            [
                [0.4034, -0.915, -0.0018, -0.6329],
                [-0.4556, -0.1991, -0.8676, 0.3805],
                [0.7936, 0.3508, -0.4972, 0.6297],
                [0.0, 0.0, 0.0, 1.0],
            ],
            requires_grad=True,
            device="cuda:0",
        )

        # Update the features dim manually
        self._features_dim = 15

    def forward(self, observations) -> torch.Tensor:
        if self.mode == "robot_xyz":
            return observations
        elif self.mode == "cam_xyz":
            point_target_robot = observations[:, -3:]
            point_target_cam = (
                torch.mm(
                    self.base_cam_extrin[:3, :3],
                    point_target_robot.transpose(0, 1).contiguous(),
                )
                .transpose(0, 1)
                .contiguous()
            )
            point_target_cam += self.base_cam_extrin[:3, 3]
            return torch.cat([observations[:, :-3], point_target_cam], dim=1)
        elif self.mode == "cam_zvu":
            point_target_robot = observations[:, -3:]
            point_target_cam = (
                torch.mm(
                    self.base_cam_extrin[:3, :3],
                    point_target_robot.transpose(0, 1).contiguous(),
                )
                .transpose(0, 1)
                .contiguous()
            )
            point_target_cam += self.base_cam_extrin[:3, 3]
            uvz = (
                torch.mm(
                    self.base_cam_intrin, point_target_cam.transpose(0, 1).contiguous()
                )
                .transpose(0, 1)
                .contiguous()
            )

            base_target_uvz = torch.zeros_like(uvz)
            base_target_uvz[:, 0] = torch.round(uvz[:, 0] / uvz[:, 2]) / 64.0
            base_target_uvz[:, 1] = torch.round(uvz[:, 1] / uvz[:, 2]) / 36.0
            base_target_uvz[:, 2] = (uvz[:, 2] - 0.5) / (2 - 0.5)
            base_target_uvz = base_target_uvz[:, [2, 1, 0]]

            # Return a (B, self._features_dim) PyTorch tensor, where B is batch dimension.
            return torch.cat([observations[:, :-3], base_target_uvz], dim=1)


class CustomDepthSegExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Dict, state_repeat_times=2):
        # We do not know features-dim here before going over all the items,
        # so put something dummy for now. PyTorch requires calling
        # nn.Module.__init__ before adding modules
        super(CustomDepthSegExtractor, self).__init__(observation_space, features_dim=1)
        self.state_repeat_times = state_repeat_times
        self.ego_cam_intrin_inv = torch.tensor(
            [
                [0.01090477, 0.0, -0.69031547],
                [0.0, 0.01089966, -0.40202298],
                [0.0, 0.0, 1.0],
            ],
            device="cuda:0",
        )
        self.ego_cam_extrin = torch.tensor(
            [
                [-0.0127, -0.0079, -0.9999, -0.2265],
                [-0.9999, 0.0038, 0.0127, 0.0253],
                [0.0037, 1.0, -0.0079, 0.0774],
                [0.0, 0.0, 0.0, 1.0],
            ],
            device="cuda:0",
        )
        # load the Unet model
        self.attn = AttentionDepthUNet(
            depth_feat_channels=8,
            filters=[16, 16],
            kernel_sizes=[5, 5],
            strides=[2, 2],
            output_channels=2,
            norm=None,
            activation="lrelu",
            skip_connections=True,
        ).to(torch.device("cuda:0"))
        self.attn.train()

        # Update the features dim manually
        self._features_dim = 15 * self.state_repeat_times

    def forward(self, observations) -> torch.Tensor:
        qpos = observations["qpos"]
        ee_pos_base = observations["ee_pos_base"]

        depth = observations["depth"]  # (N, 1, H, W)
        N, ch, h, w = depth.shape
        depth_norm = torch.clip(depth - torch.mean(depth), -1, 1)
        attnmap = F.softmax(self.attn(depth_norm), dim=1)[
            :, 0
        ]  # (N, 2, H, W)=>(N, H, W)

        ## argmax to get maxpoint ind
        max_indice = torch.argmax(attnmap.view(N, -1), dim=1)

        # zc = torch.stack([-depth.view(N, -1)[i, max_indice[i]] for i in range(N)])  # (N, sample_num), too much time
        zc = ((-depth.view(N, -1)).gather(1, max_indice.unsqueeze(1))).squeeze(1)
        v = torch.div(max_indice, w, rounding_mode="floor")  # (N, sample_num)
        u = max_indice - v * w  # (N, sample_num)
        uvz_sample = torch.stack((u * zc, v * zc, zc), dim=1).unsqueeze(
            2
        )  # (N, 3, sample_num)
        camxyz_sample = torch.matmul(
            self.ego_cam_intrin_inv, uvz_sample
        )  # (N, 3, pts_num)
        camxyz_sample[:, 0] = -camxyz_sample[:, 0]  # (N, 3, pts_num)
        eexyz_sample = (
            torch.matmul(
                camxyz_sample.transpose(1, 2).contiguous(),
                (self.ego_cam_extrin[:3, :3]).t(),
            )
            + self.ego_cam_extrin[:3, 3]
        )  # (N,sample_num,3)
        eepts_pred = eexyz_sample[:, 0]

        state = torch.cat([qpos, ee_pos_base, eepts_pred], dim=1).repeat(
            1, self.state_repeat_times
        )

        return state


class CustomRGBDSegExtractor(BaseFeaturesExtractor):
    def __init__(
        self, observation_space: gym.spaces.Dict, state_repeat_times=2, sample_num=50
    ):
        # We do not know features-dim here before going over all the items,
        # so put something dummy for now. PyTorch requires calling
        # nn.Module.__init__ before adding modules
        super(CustomRGBDSegExtractor, self).__init__(observation_space, features_dim=1)
        self.state_repeat_times = state_repeat_times
        self.sample_num = sample_num
        # self.ego_cam_intrin = torch.tensor(
        #     [[91.703, 0., 63.304],
        #      [0., 91.746, 36.884],
        #      [0., 0., 1.]],
        #     device="cuda:0",
        # )
        self.ego_cam_intrin_inv = torch.tensor(
            [
                [0.01090477, 0.0, -0.69031547],
                [0.0, 0.01089966, -0.40202298],
                [0.0, 0.0, 1.0],
            ],
            device="cuda:0",
        )
        self.ego_cam_extrin = torch.tensor(
            [
                [-0.0127, -0.0079, -0.9999, -0.2265],
                [-0.9999, 0.0038, 0.0127, 0.0253],
                [0.0037, 1.0, -0.0079, 0.0774],
                [0.0, 0.0, 0.0, 1.0],
            ],
            device="cuda:0",
        )
        # self.ego_cam_extrin_inv = torch.tensor(
        #     [[-0.01272954, -0.99992464, 0.00369915, 0.02212854],
        #      [-0.00785192, 0.00379982, 0.99992353, -0.07926868],
        #      [-0.99987629, 0.01267029, -0.00794717, -0.22617743],
        #      [0., 0., 0., 1.]],
        #     device="cuda:0",
        # )

        # load the Attention model
        siamese_net = SiameseNet(
            input_channels=[3, 1],
            filters=[8],
            kernel_sizes=[5],
            strides=[1],
            activation="lrelu",
            norm=None,
        )
        self.attn = Attention2DUNet(
            siamese_net=siamese_net,
            filters=[16, 16],
            kernel_sizes=[5, 5],
            strides=[2, 2],
            output_channels=5,
            norm=None,
            activation="lrelu",
            skip_connections=True,
        ).to(torch.device("cuda:0"))
        self.attn.train()

        # self.weight_gen = WeightGen(
        #     feat_chan=16,
        #     weight_num=5,
        #     norm="layer",
        #     activation="lrelu",
        # ).to(torch.device("cuda:0"))
        # self.weight_gen.train()
        # self.weight = None

        # self.feat_agg = FeatAggNet(
        #     in_chan=5,
        #     out_chan=1,
        #     norm=None,
        #     activation="relu",
        # ).to(torch.device("cuda:0"))
        # self.feat_agg.train()

        # self.pn = PointNetfeat(in_ch=3, mlp_specs=[64, 128, 1024], xyz_transform=False, feature_transform=False)
        # self.pn.train()
        # self.weight_centers = torch.nn.Parameter(torch.ones(5), requires_grad=True)

        # Update the features dim manually
        self._features_dim = 12 * self.state_repeat_times + self.sample_num * 3 * 5

    def forward(self, observations) -> torch.Tensor:
        qpos = observations["qpos"]
        ee_pos_base = observations["ee_pos_base"]
        # check_grasp = observations["check_grasp"]
        # cam_xyz = observations["cam_xyz"]
        # ee_xyz = observations["ee_xyz"]
        rgb = observations["rgb"]
        rgb = rgb / 255.0  # Normalization
        N, ch, h, w = rgb.shape
        depth = observations["depth"]  # (N, 1, H, W)
        depth_norm = torch.clip(depth - torch.mean(depth), -1, 1)

        seg_feat, latent_feat = self.attn(
            [rgb, depth_norm]
        )  # (N, 5, H, W), (N, 16, H//4, W//4)
        seg_map = F.softmax(seg_feat, dim=1)  # (N, 5, H, W)
        # seg_map = F.softmax(self.attn([rgb, depth_norm]), dim=1)  # (N, 5, H, W)
        # seg_map = self.feat_agg(seg_map)  # (N, 1, H, W)

        ## argmax to get maxpoint ind
        # max_indice = torch.argmax(seg_map.view(N, 5, -1), dim=2)  # (N, 5)
        ## soft_argmax to get maxpoint ind
        # max_indice = spatial_soft_argmax2d(attnmap, normalized_coordinates=True)  # (N, 1, H, W)->(N,1,2)
        ## multinomial sampling
        # weights = torch.clone(attnmap)
        # weights[weights.ge(0.5)] = 200  # handle_ratio: 72*128/42 ~ 220
        # weights[weights.lt(0.5)] = 1
        # indices = torch.multinomial(weights, self.pts_num, replacement=False)  # (N, pts_num)
        ## gumbel_softmax to sample sample_num pixel coords
        # indices = torch.nn.functional.gumbel_softmax(
        #     seg_map.view(N, -1).unsqueeze(1).repeat(1, self.sample_num, 1),
        #     tau=0.1,
        #     hard=True
        # )  # (N, sample_num, h*w)

        ## gumber_softmax to get sample_num mask indexes
        # mask_ind_onehot = torch.nn.functional.gumbel_softmax(
        #     self.weight_centers.unsqueeze(1).expand(-1, self.sample_num),
        #     tau=0.1,
        #     hard=True
        # )  # (5, sample_num)
        # print("===", mask_ind_onehot.requires_grad)
        # sample_masks = torch.einsum("ijkl,jm->imkl", seg_map, mask_ind_onehot)  # (N, sample_num, H, W)
        # indices = spatial_soft_argmax2d(sample_masks, normalized_coordinates=True)  # (N, sample_num, 2)
        # zc = F.grid_sample(-depth, indices.unsqueeze(1), align_corners=False).squeeze(1).squeeze(1)  # (N, sample_num)
        # v = (indices[:, :, 1] + 1) / 2 * (h - 1)  # (N, sample_num)
        # u = (indices[:, :, 0] + 1) / 2 * (w - 1)  # (N, sample_num)
        # uvz_sample = torch.stack((u * zc, v * zc, zc), dim=1)  # (N, 3, sample_num)
        # camxyz_sample = torch.matmul(self.ego_cam_intrin_inv, uvz_sample)  # (N, 3, sample_num)
        # camxyz_sample[:, 0] = -camxyz_sample[:, 0]  # (N, 3, sample_num)
        # eexyz_sample = torch.matmul(camxyz_sample.transpose(1, 2).contiguous(),
        #                       (self.ego_cam_extrin[:3, :3]).t()) \
        #         + self.ego_cam_extrin[:3, 3]  # (N, sample_num, 3)
        # eepts_pred = eexyz_sample.view(N, -1)  # (N, sample_num*3)

        ### topk to get topk ind
        _, indices = torch.topk(
            seg_map.view(N, 5, -1), self.sample_num, dim=2
        )  # (N, 5, sample_num)
        zc = (
            -depth.view(N, -1).unsqueeze(1).repeat(1, 5, 1).gather(2, indices)
        )  # (N, 5, sample_num)
        v = torch.div(indices, w, rounding_mode="floor")  # (N, 5, sample_num)
        u = indices - v * w  # (N, 5, sample_num)
        ## mean point from topk
        # uvz_sample = torch.stack((u * zc, v * zc, zc), dim=1).mean(3)  # (N, 3, 5)
        ## all k points
        uvz_sample = torch.stack((u * zc, v * zc, zc), dim=1).view(
            N, 3, -1
        )  # (N, 3, 5 * sample_num)
        camxyz_sample = torch.matmul(
            self.ego_cam_intrin_inv, uvz_sample
        )  # (N, 3, 5 * sample_num)
        camxyz_sample[:, 0] = -camxyz_sample[:, 0]  # (N, 3, 5 * sample_num)
        eexyz_sample = (
            torch.matmul(
                camxyz_sample.transpose(1, 2).contiguous(),
                (self.ego_cam_extrin[:3, :3]).t(),
            )
            + self.ego_cam_extrin[:3, 3]
        )  # (N, 5 * sample_num, 3)
        # weight = F.softmax(self.weight_centers, dim=0)  # (5,)
        ## mean point from topk
        # eepts_pred = torch.einsum("ijl,j->il", eexyz_sample, weight)  # (N, 3)
        ## weighted k points
        # eexyz_part_sample = eexyz_sample.view(N, 5, self.sample_num, 3)  # (N, 5, sample_num, 3)
        # eepts_pred = torch.einsum("ijkl,j->ikl", eexyz_part_sample, weight).view(N, -1)  # (N, sample_num * 3)
        ## all 5*k points
        eepts_pred = eexyz_sample.view(N, -1)  # (N, 5*sample_num*3)

        # zc = torch.stack([-depth.view(N, -1)[i, indices[i]] for i in range(N)])  # (N, sample_num)
        # zc = (-depth.view(N, -1)).gather(1, indices)  # (N, sample_num)
        # v = torch.div(indices, w, rounding_mode="floor")  # (N, sample_num)
        # u = (indices - v * w)  # (N, sample_num)
        # uvz_sample = torch.stack((u * zc, v * zc, zc), dim=1)  # (N, 3, sample_num)
        # camxyz_sample = torch.matmul(self.ego_cam_intrin_inv, uvz_sample)  # (N, 3, pts_num)
        # camxyz_sample[:, 0] = -camxyz_sample[:, 0]  # (N, 3, pts_num)
        # ee_xyz_sample = torch.matmul(camxyz_sample.transpose(1, 2).contiguous(),
        #                       (self.ego_cam_extrin[:3, :3]).t()) \
        #         + self.ego_cam_extrin[:3, 3]  # (N,sample_num,3)
        ## vote one handle_center point from the sample_num points
        # eepts_pred = ee_xyz_sample[:, 0]  # (N,3)

        # zc = torch.stack([-depth.view(N, -1)[i, max_indice[i]] for i in range(N)])  # (N, sample_num), too much time
        # zc = ((-depth.view(N, -1)).gather(1, max_indice))  # (N, 5)
        # v = torch.div(max_indice, w, rounding_mode="floor")  # (N, 5)
        # u = max_indice - v * w  # (N, 5)
        # uvz_sample = torch.stack((u * zc, v * zc, zc), dim=1)  # (N, 3, 5)
        # camxyz_sample = torch.matmul(self.ego_cam_intrin_inv, uvz_sample)  # (N, 3, 5)
        # camxyz_sample[:, 0] = -camxyz_sample[:, 0]  # (N, 3, 5)
        # eexyz_sample = torch.matmul(camxyz_sample.transpose(1, 2).contiguous(),
        #                             (self.ego_cam_extrin[:3, :3]).t()) \
        #                + self.ego_cam_extrin[:3, 3]  # (N, 5, 3)
        # eepts_pred = eexyz_sample.view(N, -1)  # (N, 5*3)

        # zc = F.grid_sample(-depth, max_indice.unsqueeze(1), align_corners=False).squeeze(1).squeeze(1)  # (N, sample_num)
        # v = (max_indice[:, :, 1] + 1) / 2 * (h - 1)  # (N, sample_num)
        # u = (max_indice[:, :, 0] + 1) / 2 * (w - 1)  # (N, sample_num)
        # uvz_sample = torch.stack((u * zc, v * zc, zc), dim=1)  # (N, 3, sample_num)

        # zc = torch.einsum("ij,ikj->ik", -depth.view(N, -1), indices)  # (N, sample_num)
        # u, v = torch.arange(0, w, dtype=depth.dtype, device=depth.device), \
        #        torch.arange(0, h, dtype=depth.dtype, device=depth.device)
        # grid_v, grid_u = torch.meshgrid(v, u, indexing="ij")
        # idx_v = torch.einsum("ijk,k->ij", indices, grid_v.reshape(-1))  # (N, sample_num)
        # idx_u = torch.einsum("ijk,k->ij", indices, grid_u.reshape(-1))  # (N, sample_num)
        # uvz_sample = torch.stack((idx_u * zc, idx_v * zc, zc), dim=1)  # (N, 3, sample_num)
        # camxyz_sample = torch.matmul(self.ego_cam_intrin_inv, uvz_sample)  # (N, 3, sample_num)
        # camxyz_sample[:, 0] = -camxyz_sample[:, 0]  # (N, 3, sample_num)
        # eexyz_sample = torch.matmul(camxyz_sample.transpose(1, 2).contiguous(),
        #                             (self.ego_cam_extrin[:3, :3]).t()) \
        #                + self.ego_cam_extrin[:3, 3]  # (N, sample_num, 3)
        # eepts_pred = eexyz_sample.view(N, -1)

        ## for weight_center
        # zc = ((-depth.view(N, -1)).gather(1, max_indice)).squeeze(1)  # (N, 5)
        # v = torch.div(max_indice, w, rounding_mode="floor")  # (N, 5)
        # u = max_indice - v * w  # (N, 5)
        # uvz_sample = torch.stack((u * zc, v * zc, zc), dim=1)  # (N, 3, 5)
        # camxyz_sample = torch.matmul(self.ego_cam_intrin_inv, uvz_sample)  # (N, 3, 5)
        # camxyz_sample[:, 0] = -camxyz_sample[:, 0]  # (N, 3, 5)
        # eexyz_sample = torch.matmul(camxyz_sample.transpose(1, 2).contiguous(),
        #                             (self.ego_cam_extrin[:3, :3]).t()) \
        #                + self.ego_cam_extrin[:3, 3]  # (N, 5, 3)
        # weight = F.softmax(self.weight_centers, dim=0)  # (5,)
        # eepts_pred = torch.einsum("ijk,j->ik", eexyz_sample, weight)  # (N, 3)
        # self.weight = F.softmax(self.weight_gen(latent_feat.detach()), dim=1)  # (N, 5)
        # eepts_pred = torch.einsum("ijk,ij->ik", eexyz_sample, self.weight)  # (N, 3)

        # ee_xyz = observations["ee_xyz"]
        # eexyz_sample = torch.stack([ee_xyz.view(N, 3, -1)[i, :, indices[i]] for i in range(N)])  # (N, 3, pts_num)
        # eergb_pred = torch.stack([rgb.view(N, 3, -1)[i, :, indices[i]] for i in range(N)])  # (N, 3, pts_num)
        # pn_feature, _, _, _ = self.pn(torch.cat((eexyz_pred, eergb_pred), dim=1))  # (N, 6, pts_num) -> (N, 128)
        # pn_feature, _, _, _ = self.pn(ee_xyz)  # (N, 3, pts_num) -> (N, 128)
        # eepts_pred = torch.stack([ee_xyz.view(N, 3, -1)[i, :, max_indice[i]] for i in range(N)])  # (N, 3)

        state = torch.cat([qpos, ee_pos_base], dim=1).repeat(1, self.state_repeat_times)

        return torch.cat([state, eepts_pred], dim=1)


class CustomRGBDPretrainSegExtractor(BaseFeaturesExtractor):
    def __init__(
        self, observation_space: gym.spaces.Dict, state_repeat_times=2, sample_num=50
    ):
        # We do not know features-dim here before going over all the items,
        # so put something dummy for now. PyTorch requires calling
        # nn.Module.__init__ before adding modules
        super(CustomRGBDPretrainSegExtractor, self).__init__(
            observation_space, features_dim=1
        )

        self.state_repeat_times = state_repeat_times
        self.sample_num = sample_num

        self.ego_cam_intrin_inv = torch.tensor(
            [
                [0.01090477, 0.0, -0.69031547],
                [0.0, 0.01089966, -0.40202298],
                [0.0, 0.0, 1.0],
            ],
            device="cuda:0",
        )
        self.ego_cam_extrin = torch.tensor(
            [
                [-0.0127, -0.0079, -0.9999, -0.2265],
                [-0.9999, 0.0038, 0.0127, 0.0253],
                [0.0037, 1.0, -0.0079, 0.0774],
                [0.0, 0.0, 0.0, 1.0],
            ],
            device="cuda:0",
        )

        # log_name = "pretrain_unet/20221212_152355_cellb005_allcabfaucetfilter_bs512_lr1e-3_bnrelu_2siam4updown163264128"
        # log_name = "pretrain_unet/20221214_210516_cellbs0.1_allcabfaucetfilter_bs512_0.9step20lr1e-3_bnrelu_2siam2updown1616"
        log_name = (
            "smp_model/20221227_000530_allcabfaucet_bs256_0.9step50lr1e-3_unet++3_effb1"
        )
        model_path = VISUALMODEL_DIR / f"{log_name}/best.pth"
        config_path = VISUALMODEL_DIR / f"{log_name}/config.yaml"
        with open(config_path, "r") as f:
            cfg = yaml.safe_load(f)
        smp_cfg = cfg["smp_config"]
        self.attn = smp.UnetPlusPlus(
            encoder_name=smp_cfg["encoder"],
            encoder_depth=smp_cfg["encoder_depth"],
            decoder_channels=smp_cfg["decoder_channels"],
            encoder_weights=smp_cfg["encoder_weights"],
            in_channels=smp_cfg["in_channels"],
            classes=cfg["num_classes"],
            activation=smp_cfg["activation"],
        )
        self.attn.load_state_dict(torch.load(model_path))
        self.attn.to(torch.device("cuda:0"))
        # siamese_cfg = cfg["segnet_config"]["siamese_config"]
        # unet_cfg = cfg["segnet_config"]["unet_config"]
        # siamese_net = SiameseNet(
        #     input_channels=siamese_cfg["input_channels"],
        #     filters=siamese_cfg["filters"],
        #     kernel_sizes=siamese_cfg["kernel_sizes"],
        #     strides=siamese_cfg["strides"],
        #     norm=siamese_cfg["norm"],
        #     activation=siamese_cfg["activation"]
        # )
        # self.attn = Attention2DUNet(
        #     siamese_net=siamese_net,
        #     filters=unet_cfg["filters"],
        #     kernel_sizes=unet_cfg["kernel_sizes"],
        #     strides=unet_cfg["strides"],
        #     output_channels=cfg["num_classes"],
        #     norm=unet_cfg["norm"],
        #     activation=unet_cfg["activation"],
        #     skip_connections=unet_cfg["skip_connections"]
        # ).to(torch.device("cuda:0"))
        # self.attn.load_w(model_path)
        self.attn.eval()
        # self.attn.requires_grad(False)
        self.num_classes = cfg["num_classes"]

        # Update the features dim manually
        self._features_dim = (
            12 * self.state_repeat_times + self.sample_num * 3 * self.num_classes
        )

    def forward(self, observations) -> torch.Tensor:
        qpos = observations["qpos"]
        ee_pos_base = observations["ee_pos_base"]
        rgb = observations["rgb"]
        rgb = rgb / 255.0  # (N, 3, H, W)
        N, ch, h, w = rgb.shape
        depth = observations["depth"]  # (N, 1, H, W)
        # depth_norm = torch.clip(depth - torch.mean(depth), -1, 1)
        depth_norm = depth / torch.max(depth)

        # seg_feat, latent_feat = self.attn([rgb, depth_norm])  # (N, C, H, W), (N, 16, H//4, W//4)
        # seg_map = F.softmax(seg_feat, dim=1)  # (N, C, H, W)
        seg_feat = self.attn.predict(
            torch.cat([rgb, depth_norm], dim=1)
        )  # (N, C, H, W)
        seg_map = F.softmax(seg_feat, dim=1)  # (N, C, H, W)

        # handle masks delete other handles
        # other_handle_seg = observations["other_handleseg"]
        # seg_map[:, 1][other_handle_seg == 1] = 0

        ## argmax to get maxpoint ind
        # max_indice = torch.argmax(seg_map.view(N, self.num_classes, -1), dim=2)  # (N, C)
        # zc = ((-depth.view(N, -1)).gather(1, max_indice))  # (N, C)
        # v = torch.div(max_indice, w, rounding_mode="floor")  # (N, C)
        # u = max_indice - v * w  # (N, C)
        # uvz_sample = torch.stack((u * zc, v * zc, zc), dim=1)  # (N, 3, C)
        # camxyz_sample = torch.matmul(self.ego_cam_intrin_inv, uvz_sample)  # (N, 3, C)
        # camxyz_sample[:, 0] = -camxyz_sample[:, 0]  # (N, 3, C)
        # eexyz_sample = torch.matmul(camxyz_sample.transpose(1, 2).contiguous(),
        #                             (self.ego_cam_extrin[:3, :3]).t()) \
        #                + self.ego_cam_extrin[:3, 3]  # (N, C, 3)
        # eepts_pred = eexyz_sample.view(N, -1)  # (N, C*3)

        ### topk to get topk ind
        _, indices = torch.topk(
            seg_map.view(N, self.num_classes, -1), self.sample_num, dim=2
        )  # (N, C, sample_num)
        zc = (
            -depth.view(N, -1)
            .unsqueeze(1)
            .repeat(1, self.num_classes, 1)
            .gather(2, indices)
        )  # (N, C, sample_num)
        v = torch.div(indices, w, rounding_mode="floor")  # (N, C, sample_num)
        u = indices - v * w  # (N, C, sample_num)
        ## all k points
        uvz_sample = torch.stack((u * zc, v * zc, zc), dim=1).view(
            N, 3, -1
        )  # (N, 3, C * sample_num)
        camxyz_sample = torch.matmul(
            self.ego_cam_intrin_inv, uvz_sample
        )  # (N, 3, C * sample_num)
        camxyz_sample[:, 0] = -camxyz_sample[:, 0]  # (N, 3, C * sample_num)
        eexyz_sample = (
            torch.matmul(
                camxyz_sample.transpose(1, 2).contiguous(),
                (self.ego_cam_extrin[:3, :3]).t(),
            )
            + self.ego_cam_extrin[:3, 3]
        )  # (N, C * sample_num, 3)
        ## all 5*k points
        eepts_pred = eexyz_sample.view(N, -1)  # (N, C*sample_num*3)

        state = torch.cat([qpos, ee_pos_base], dim=1).repeat(1, self.state_repeat_times)

        return torch.cat([state, eepts_pred], dim=1)


class CustomRGBDPredSegResNetExtractor(BaseFeaturesExtractor):
    def __init__(
        self,
        observation_space: gym.spaces.Dict,
        state_repeat_times=2,
        vismodel_path=None,
        device="cuda",
    ):
        # We do not know features-dim here before going over all the items,
        # so put something dummy for now. PyTorch requires calling
        # nn.Module.__init__ before adding modules
        super(CustomRGBDPredSegResNetExtractor, self).__init__(
            observation_space, features_dim=1
        )

        self.state_repeat_times = state_repeat_times
        self.device = device

        # vismodel_path = VISUALMODEL_DIR / f"smp_model/20230228_220446_384_DR_randombg_mixfaucet_aug_dropout0.5_stereo_bs16_focalloss_0.5step50lr0.001_RGBDunet-163264128_mobilenet_v2"
        model_path = vismodel_path / "best.pth"
        config_path = vismodel_path / "config.yaml"
        with open(config_path, "r") as f:
            cfg = yaml.safe_load(f)
        smp_cfg = cfg["smp_config"]
        if smp_cfg["mode"] == "RGBD":
            in_ch = 4
        elif smp_cfg["mode"] == "RGB":
            in_ch = 3
        elif smp_cfg["mode"] == "D":
            in_ch = 1
        else:
            raise NotImplementedError
        self.segmodel = CustomUnet(
            has_dropout=False,
            encoder_name=smp_cfg["encoder"],
            encoder_depth=smp_cfg["encoder_depth"],
            decoder_channels=smp_cfg["decoder_channels"],
            encoder_weights=smp_cfg["encoder_weights"],
            in_channels=in_ch,
            classes=cfg["num_classes"],
            activation=smp_cfg["activation"],
        )
        self.segmodel.load_state_dict(torch.load(model_path))
        self.segmodel.to(torch.device(self.device))
        self.segmodel.eval()

        self.num_classes = cfg["num_classes"]

        # self.resnet = ResNet(
        #     block=BasicBlock,
        #     num_blocks=[1, 1, 1, 1],
        #     in_dim=self.num_classes + 1,
        #     planes=16
        # )
        # self.resnet.to(torch.device(self.device))
        # self.resnet.train()
        self.encoder = ImgEncoder(
            in_ch=self.num_classes + 1, feature_dim=256, num_layers=4, num_filters=32
        )

        # Update the features dim manually
        self._features_dim = 12 * self.state_repeat_times + 256

    def forward(self, observations) -> torch.Tensor:
        qpos = observations["qpos"]
        ee_pos_base = observations["ee_pos_base"]
        rgb = observations["rgb"]
        rgb = rgb / 255.0  # (N, 3, H, W)
        depth = observations["depth"]  # (N, 1, H, W)

        with torch.no_grad():
            seg_map = self.segmodel.predict(
                torch.cat([rgb, depth], dim=1)
            )  # (N, C, H, W)

        # img_feat_norm = self.resnet(torch.cat([seg_map, depth], dim=1))  # (N, 256)
        # img_feat_norm = F.layer_norm(img_feat, img_feat.size()[1:])
        img_feat_norm = self.encoder.forward(
            torch.cat([seg_map, depth], dim=1)
        )  # (N, 256)

        state = torch.cat([qpos, ee_pos_base], dim=1).repeat(1, self.state_repeat_times)

        return torch.cat([state, img_feat_norm], dim=1)


class CustomDSegResNetExtractor(BaseFeaturesExtractor):
    def __init__(
        self,
        observation_space: gym.spaces.Dict,
        state_repeat_times=2,
        num_classes=6,
        device="cuda",
    ):
        # We do not know features-dim here before going over all the items,
        # so put something dummy for now. PyTorch requires calling
        # nn.Module.__init__ before adding modules
        super(CustomDSegResNetExtractor, self).__init__(
            observation_space, features_dim=1
        )

        self.state_repeat_times = state_repeat_times
        self.num_classes = num_classes
        self.device = device

        # self.resnet = models.resnet18(pretrained=False)
        # self.resnet.conv1 = torch.nn.Conv2d(
        #     2,
        #     64,
        #     kernel_size=(7, 7),
        #     stride=(2, 2),
        #     padding=(3, 3),
        #     bias=False,
        # )
        # self.resnet.fc = torch.nn.Linear(in_features=512, out_features=256, bias=True)
        # self.resnet.to(torch.device(self.device))
        # self.resnet.train()
        self.encoder = ImgEncoder(
            in_ch=self.num_classes + 1, feature_dim=256, num_layers=4, num_filters=32
        )

        # Update the features dim manually
        self._features_dim = 12 * self.state_repeat_times + 256

    def forward(self, observations) -> torch.Tensor:
        qpos = observations["qpos"]
        ee_pos_base = observations["ee_pos_base"]
        depth = observations["depth"]  # (N, H, W)
        seg = observations["seg"]  # (N, H, W)
        seg_map = (
            F.one_hot(seg.long(), self.num_classes)
            .permute(0, 3, 1, 2)
            .to(self.device, dtype=torch.float32)
        )  # (N, C, H, W)

        # img_feat = self.resnet(torch.cat([depth.unsqueeze(1), seg.unsqueeze(1)], dim=1))  # (N, 256)
        # img_feat_norm = F.layer_norm(img_feat, img_feat.size()[1:])
        img_feat_norm = self.encoder.forward(
            torch.cat([seg_map, depth], dim=1)
        )  # (N, 256)

        state = torch.cat([qpos, ee_pos_base], dim=1).repeat(1, self.state_repeat_times)

        return torch.cat([state, img_feat_norm], dim=1)


class CustomRGBDPretrainSegPointNetExtractor(BaseFeaturesExtractor):
    def __init__(
        self, observation_space: gym.spaces.Dict, state_repeat_times=2, sample_num=50
    ):
        # We do not know features-dim here before going over all the items,
        # so put something dummy for now. PyTorch requires calling
        # nn.Module.__init__ before adding modules
        super(CustomRGBDPretrainSegPointNetExtractor, self).__init__(
            observation_space, features_dim=1
        )

        self.state_repeat_times = state_repeat_times
        self.sample_num = sample_num

        self.ego_cam_intrin_inv = torch.tensor(
            [
                [0.01090477, 0.0, -0.69031547],
                [0.0, 0.01089966, -0.40202298],
                [0.0, 0.0, 1.0],
            ],
            device="cuda:0",
        )
        self.ego_cam_extrin = torch.tensor(
            [
                [-0.0127, -0.0079, -0.9999, -0.2265],
                [-0.9999, 0.0038, 0.0127, 0.0253],
                [0.0037, 1.0, -0.0079, 0.0774],
                [0.0, 0.0, 0.0, 1.0],
            ],
            device="cuda:0",
        )

        # log_name = "smp_model/20230110_103746_cab39-17_faucet11-3_newaug_bs256_0.5step50lr0.001_RGBDunet3-3264128_mobilenet_v2"
        log_name = "smp_model/20230228_014637_384_noDR_norandombg_aug_dropout0.2_stereo_bs16_focalloss_0.5step50lr0.001_RGBDunet-163264128_mobilenet_v2"
        model_path = VISUALMODEL_DIR / f"{log_name}/best.pth"
        config_path = VISUALMODEL_DIR / f"{log_name}/config.yaml"
        with open(config_path, "r") as f:
            cfg = yaml.safe_load(f)
        smp_cfg = cfg["smp_config"]
        if smp_cfg["mode"] == "RGBD":
            in_ch = 4
        elif smp_cfg["mode"] == "RGB":
            in_ch = 3
        elif smp_cfg["mode"] == "D":
            in_ch = 1
        else:
            raise NotImplementedError
        self.segmodel = CustomUnet(
            has_dropout=False,
            encoder_name=smp_cfg["encoder"],
            encoder_depth=smp_cfg["encoder_depth"],
            decoder_channels=smp_cfg["decoder_channels"],
            encoder_weights=smp_cfg["encoder_weights"],
            in_channels=in_ch,
            classes=cfg["num_classes"],
            activation=smp_cfg["activation"],
        )
        self.segmodel.load_state_dict(torch.load(model_path))
        self.segmodel.to("cuda:0")
        self.segmodel.eval()
        self.num_classes = cfg["num_classes"]

        self.pn = PointNetfeat(
            in_ch=3 + self.num_classes + 1,
            global_feat=True,
            mlp_specs=[64, 128, 256],
            xyz_transform=False,
            feature_transform=False,
        )
        self.pn.train()
        seg_info = torch.zeros(
            self.num_classes * self.sample_num + 40,
            self.num_classes + 1,
            device=torch.device("cuda:0"),
        )
        for nc in range(self.num_classes):
            seg_info[nc * self.sample_num : (nc + 1) * self.sample_num, nc] = 1
        seg_info[-40:, self.num_classes] = 1

        # shuffle random 5 points label
        # random_label = torch.eye(
        #     self.num_classes, device=torch.device("cuda:0")
        # )[torch.randint(0, self.num_classes, (5 * self.num_classes,))]
        # random_label = torch.zeros(
        #     self.num_classes * 5, self.num_classes + 1,
        #     device=torch.device("cuda:0")
        # )
        # indices = torch.randint(0, self.num_classes, (self.num_classes * 5, ), device=torch.device("cuda:0"))
        # random_label.scatter_(1, indices.unsqueeze(1), 1)
        # seg_info[
        #     np.random.choice(range(self.sample_num*self.num_classes), 5 * self.num_classes), :
        # ] = random_label

        self.seg_info = seg_info

        # Update the features dim manually
        self._features_dim = 12 * self.state_repeat_times + 256

    def forward(self, observations) -> torch.Tensor:
        qpos = observations["qpos"]
        ee_pos_base = observations["ee_pos_base"]
        eepad_pts = observations["eepad_pts"]  # (N, 40, 3)
        rgb = observations["rgb"]
        # rgb = rgb / 255.0  # (N, 3, H, W)
        N, ch, h, w = rgb.shape
        # depth = observations["depth"]  # (N, 1, H, W)
        depth = observations["depth"]  # (N, 1, H, W)

        with torch.no_grad():
            # t0 = time.time()
            seg_map = self.segmodel.predict(
                torch.cat([rgb, depth], dim=1)
            )  # (N, C, H, W)
            # t1 = time.time()
            # print("+", t1-t0)

            ### random sampling
            indices = []
            num_classes = seg_map.shape[1]
            seg_map_mc = torch.argmax(seg_map, dim=1).view(N, -1)  # (N, H, W)
            t2 = time.time()
            # print("++", t2 - t1)
            for num in range(N):
                part_inds = []
                for ind in range(num_classes):
                    one_indices = torch.where(seg_map_mc[num] == ind)[0]
                    ones_num = one_indices.shape[0]
                    if ones_num == 0:
                        cur_indices = torch.zeros(
                            self.sample_num, dtype=torch.int64, device="cuda:0"
                        )
                    elif ones_num < self.sample_num:
                        repeat_times = self.sample_num // ones_num
                        remain_num = self.sample_num - repeat_times * ones_num
                        cur_indices = torch.cat(
                            (one_indices.repeat(repeat_times), one_indices[:remain_num])
                        )
                    else:
                        cur_indices = one_indices[
                            random.sample(range(len(one_indices)), self.sample_num)
                        ]
                    part_inds.append(cur_indices[None])
                pts_index = torch.cat(part_inds, dim=0)  # (C, sample_num)
                indices.append(pts_index[None])
            indices = torch.cat(indices, dim=0)  # (N, C, sample_num)
            # t3 = time.time()
            # print("+++", t3-t2)

        ### topk to get topk ind
        # values, indices = torch.topk(
        #     seg_map.view(N, self.num_classes, -1), self.sample_num, dim=2
        # )  # (N, C, sample_num)
        zc = (
            -depth.view(N, -1)
            .unsqueeze(1)
            .repeat(1, self.num_classes, 1)
            .gather(2, indices)
        )  # (N, C, sample_num)
        v = torch.div(indices, w, rounding_mode="floor")  # (N, C, sample_num)
        u = indices - v * w  # (N, C, sample_num)
        ## all k points
        uvz_sample = torch.stack((u * zc, v * zc, zc), dim=1).view(
            N, 3, -1
        )  # (N, 3, C * sample_num)
        camxyz_sample = torch.matmul(
            self.ego_cam_intrin_inv, uvz_sample
        )  # (N, 3, C * sample_num)
        camxyz_sample[:, 0] = -camxyz_sample[:, 0]  # (N, 3, C * sample_num)
        eexyz_sample = (
            torch.matmul(
                camxyz_sample.transpose(1, 2).contiguous(),
                (self.ego_cam_extrin[:3, :3]).t(),
            )
            + self.ego_cam_extrin[:3, 3]
        )  # (N, C * sample_num, 3)

        eewpad_xyz_sample = torch.cat(
            (eexyz_sample, eepad_pts), dim=1
        )  # (N, C*sample_num+40, 3)

        ## concat class info
        eexyzC_sample = torch.cat(
            (eewpad_xyz_sample, self.seg_info.repeat(N, 1, 1)), dim=2
        )  # (N, C*sample_num+40, 3+C+1)
        ## concat class & confidence info
        # conf_scores = torch.cat(
        #     (values.view(N, -1), torch.ones((N, 40), device=torch.device("cuda:0"))), dim=1
        # ).unsqueeze(2)  # (N, C*sample_num+40, 1)
        # eexyzS_sample = torch.cat(
        #     (eewpad_xyz_sample, self.seg_info.repeat(N, 1, 1), conf_scores), dim=2
        # )

        ## PointNet to get 128 feat
        pn_feature, _, _ = self.pn(
            eexyzC_sample.transpose(2, 1).contiguous()
        )  # (N, 3+C+1, C*sample_num+40)
        # pn_feature, _, _ = self.pn(eexyzS_sample.transpose(2, 1).contiguous())  # (N, 3+C+1+1, C*sample_num+40)

        state = torch.cat([qpos, ee_pos_base], dim=1).repeat(1, self.state_repeat_times)

        return torch.cat([state, pn_feature], dim=1)


class CustomPretrainSegPNExtractor(BaseFeaturesExtractor):
    def __init__(
        self,
        observation_space: gym.spaces.Dict,
        num_classes=6,
        state_repeat_times=2,
        sample_num=50,
    ):
        # We do not know features-dim here before going over all the items,
        # so put something dummy for now. PyTorch requires calling
        # nn.Module.__init__ before adding modules
        super(CustomPretrainSegPNExtractor, self).__init__(
            observation_space, features_dim=1
        )

        self.num_classes = num_classes
        self.state_repeat_times = state_repeat_times
        self.sample_num = sample_num

        self.pn = PointNetfeat(
            in_ch=3 + self.num_classes + 1,
            global_feat=True,
            mlp_specs=[64, 128, 256],
            xyz_transform=False,
            feature_transform=False,
        )
        self.pn.train()
        seg_info = torch.zeros(
            self.num_classes * self.sample_num + 40,
            self.num_classes + 1,
            device=torch.device("cuda:0"),
        )
        for nc in range(self.num_classes):
            seg_info[nc * self.sample_num : (nc + 1) * self.sample_num, nc] = 1
        seg_info[-40:, self.num_classes] = 1

        # shuffle random 5 points label
        # random_label = torch.eye(
        #     self.num_classes, device=torch.device("cuda:0")
        # )[torch.randint(0, self.num_classes, (5 * self.num_classes,))]
        # random_label = torch.zeros(
        #     self.num_classes * 5, self.num_classes + 1,
        #     device=torch.device("cuda:0")
        # )
        # indices = torch.randint(0, self.num_classes, (self.num_classes * 5, ), device=torch.device("cuda:0"))
        # random_label.scatter_(1, indices.unsqueeze(1), 1)
        # seg_info[
        #     np.random.choice(range(self.sample_num*self.num_classes), 5 * self.num_classes), :
        # ] = random_label

        self.seg_info = seg_info

        # Update the features dim manually
        self._features_dim = 12 * self.state_repeat_times + 256

    def forward(self, observations) -> torch.Tensor:
        qpos = observations["qpos"]  # (N, 9)
        ee_pos_base = observations["ee_pos_base"]  # (N, 3)
        eepad_pts = observations["eepad_pts"]  # (N, 40, 3)
        eexyz_sample = observations["segsampled_pts"]  # (N, C*sample_num, 3)
        N = qpos.shape[0]

        eewpad_xyz_sample = torch.cat(
            (eexyz_sample, eepad_pts), dim=1
        )  # (N, C*sample_num+40, 3)

        ## concat class info
        eexyzC_sample = torch.cat(
            (eewpad_xyz_sample, self.seg_info.repeat(N, 1, 1)), dim=2
        )  # (N, C*sample_num+40, 3+C+1)
        ## concat class & confidence info
        # conf_scores = torch.cat(
        #     (values.view(N, -1), torch.ones((N, 40), device=torch.device("cuda:0"))), dim=1
        # ).unsqueeze(2)  # (N, C*sample_num+40, 1)
        # eexyzS_sample = torch.cat(
        #     (eewpad_xyz_sample, self.seg_info.repeat(N, 1, 1), conf_scores), dim=2
        # )

        ## PointNet to get 128 feat
        pn_feature, _, _ = self.pn(
            eexyzC_sample.transpose(2, 1).contiguous()
        )  # (N, 3+C+1, C*sample_num+40)
        # pn_feature, _, _ = self.pn(eexyzS_sample.transpose(2, 1).contiguous())  # (N, 3+C+1+1, C*sample_num+40)

        state = torch.cat([qpos, ee_pos_base], dim=1).repeat(1, self.state_repeat_times)

        return torch.cat([state, pn_feature], dim=1)


class CustomSegPNExtractor(BaseFeaturesExtractor):
    def __init__(
        self,
        observation_space: gym.spaces.Dict,
        num_classes=6,
        state_repeat_times=2,
        sample_num=50,
    ):
        # We do not know features-dim here before going over all the items,
        # so put something dummy for now. PyTorch requires calling
        # nn.Module.__init__ before adding modules
        super(CustomSegPNExtractor, self).__init__(observation_space, features_dim=1)

        self.num_classes = num_classes
        self.state_repeat_times = state_repeat_times
        self.sample_num = sample_num

        self.pn = PointNetfeat(
            in_ch=3 + self.num_classes,
            # in_ch=3 + self.num_classes+1,
            global_feat=True,
            mlp_specs=[64, 128, 256],
            xyz_transform=False,
            feature_transform=False,
        )
        self.pn.train()

        # Update the features dim manually
        self._features_dim = 12 * self.state_repeat_times + 256

    def forward(self, observations) -> torch.Tensor:
        qpos = observations["qpos"]  # (N, 9)
        ee_pos_base = observations["ee_pos_base"]  # (N, 3)
        eexyzc_sample = observations["segsampled_ptsC"]  # (N, SN+40, 3+C+1)

        ## PointNet to get 128 feat
        pn_feature, _, _ = self.pn(
            eexyzc_sample.transpose(2, 1).contiguous()
        )  # (N, 3+C+1, SN+40)
        # pn_feature, _, _ = self.pn(eexyzS_sample.transpose(2, 1).contiguous())  # (N, 3+C+1+1, SN+40)

        state = torch.cat([qpos, ee_pos_base], dim=1).repeat(1, self.state_repeat_times)

        return torch.cat([state, pn_feature], dim=1)


class CustomKptExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Dict, num_kpts=3):
        # We do not know features-dim here before going over all the items,
        # so put something dummy for now. PyTorch requires calling
        # nn.Module.__init__ before adding modules
        super(CustomKptExtractor, self).__init__(observation_space, features_dim=1)

        # self.ego_cam_intrin_inv = torch.tensor(
        #     [
        #         [0.01090477, 0.0, -0.69031547],
        #         [0.0, 0.01089966, -0.40202298],
        #         [0.0, 0.0, 1.0],
        #     ],
        #     device="cuda:0",
        # )
        # self.ego_cam_extrin = torch.tensor(
        #     [
        #         [-0.0127, -0.0079, -0.9999, -0.2265],
        #         [-0.9999, 0.0038, 0.0127, 0.0253],
        #         [0.0037, 1.0, -0.0079, 0.0774],
        #         [0.0, 0.0, 0.0, 1.0],
        #     ],
        #     device="cuda:0",
        # )

        self.num_kpts = num_kpts

        # Update the features dim manually
        self._features_dim = 12 + num_kpts * 3

    def forward(self, observations) -> torch.Tensor:
        qpos = observations["qpos"]  # (N, 9)
        ee_pos_base = observations["ee_pos_base"]  # (N, 3)
        N = qpos.shape[0]
        # uvz_pred = observations["uvz_pred"]  # (N, K, 3)
        eexyz_sample = observations["xyz_ee_pred"]  # (N, K, 3)

        ## all k points
        # camxyz_sample = torch.matmul(
        #     self.ego_cam_intrin_inv, uvz_pred.transpose(1, 2)
        # )  # (N, 3, K)
        # camxyz_sample[:, 0] = -camxyz_sample[:, 0]  # (N, 3, K)
        # eexyz_sample = (
        #         torch.matmul(
        #             camxyz_sample.transpose(1, 2).contiguous(),
        #             (self.ego_cam_extrin[:3, :3]).t(),
        #         )
        #         + self.ego_cam_extrin[:3, 3]
        # )  # (N, K, 3)
        ## all 5*k points
        eepts_pred = eexyz_sample.view(N, -1)  # (N, K*3)

        state = torch.cat([qpos, ee_pos_base, eepts_pred], dim=1)

        return state


class CustomKptGTExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Dict, num_kpts=3):
        # We do not know features-dim here before going over all the items,
        # so put something dummy for now. PyTorch requires calling
        # nn.Module.__init__ before adding modules
        super(CustomKptGTExtractor, self).__init__(observation_space, features_dim=1)

        # self.ego_cam_intrin_inv = torch.tensor(
        #     [
        #         [0.01090477, 0.0, -0.69031547],
        #         [0.0, 0.01089966, -0.40202298],
        #         [0.0, 0.0, 1.0],
        #     ],
        #     device="cuda:0",
        # )
        # self.ego_cam_extrin = torch.tensor(
        #     [
        #         [-0.0127, -0.0079, -0.9999, -0.2265],
        #         [-0.9999, 0.0038, 0.0127, 0.0253],
        #         [0.0037, 1.0, -0.0079, 0.0774],
        #         [0.0, 0.0, 0.0, 1.0],
        #     ],
        #     device="cuda:0",
        # )

        self.num_kpts = num_kpts

        # Update the features dim manually
        self._features_dim = 12 + num_kpts * 3

    def forward(self, observations) -> torch.Tensor:
        qpos = observations["qpos"]  # (N, 9)
        ee_pos_base = observations["ee_pos_base"]  # (N, 3)
        N = qpos.shape[0]
        # uvz_gt = observations["uvz"]  # (N, K, 3)

        ## all k points
        # camxyz_sample = torch.matmul(
        #     self.ego_cam_intrin_inv, uvz_gt.transpose(1, 2)
        # )  # (N, 3, K)
        # camxyz_sample[:, 0] = -camxyz_sample[:, 0]  # (N, 3, K)
        # eexyz_sample = (
        #     torch.matmul(
        #         camxyz_sample.transpose(1, 2).contiguous(),
        #         (self.ego_cam_extrin[:3, :3]).t(),
        #     )
        #     + self.ego_cam_extrin[:3, 3]
        # )  # (N, K, 3)
        # ## all 5*k points
        # eepts_pred = eexyz_sample.view(N, -1)  # (N, K*3)

        kpts_gt_noise = observations["kpts"]  # (N, K, 3)
        eepts_pred = kpts_gt_noise.view(N, -1)  # (N, K*3)

        state = torch.cat([qpos, ee_pos_base, eepts_pred], dim=1)

        return state


class CustomRGBDSegGTPointNetExtractor(BaseFeaturesExtractor):
    def __init__(
        self, observation_space: gym.spaces.Dict, state_repeat_times=2, sample_num=50
    ):
        # We do not know features-dim here before going over all the items,
        # so put something dummy for now. PyTorch requires calling
        # nn.Module.__init__ before adding modules
        super(CustomRGBDSegGTPointNetExtractor, self).__init__(
            observation_space, features_dim=1
        )

        self.state_repeat_times = state_repeat_times
        self.sample_num = sample_num

        self.ego_cam_intrin_inv = torch.tensor(
            [
                [0.01090477, 0.0, -0.69031547],
                [0.0, 0.01089966, -0.40202298],
                [0.0, 0.0, 1.0],
            ],
            device="cuda:0",
        )
        self.ego_cam_extrin = torch.tensor(
            [
                [-0.0127, -0.0079, -0.9999, -0.2265],
                [-0.9999, 0.0038, 0.0127, 0.0253],
                [0.0037, 1.0, -0.0079, 0.0774],
                [0.0, 0.0, 0.0, 1.0],
            ],
            device="cuda:0",
        )
        self.num_classes = 6

        self.pn = PointNetfeat(
            in_ch=3 + self.num_classes + 1,
            mlp_specs=[64, 128, 256],
            global_feat=True,
            xyz_transform=False,
            feature_transform=False,
        )
        self.pn.train()
        seg_info = torch.zeros(
            self.num_classes * self.sample_num + 40,
            self.num_classes + 1,
            device=torch.device("cuda:0"),
        )
        for nc in range(self.num_classes):
            seg_info[nc * self.sample_num : (nc + 1) * self.sample_num, nc] = 1
        seg_info[-40:, self.num_classes] = 1
        self.seg_info = seg_info

        # Update the features dim manually
        self._features_dim = 12 * self.state_repeat_times + 256

    def forward(self, observations) -> torch.Tensor:
        qpos = observations["qpos"]
        ee_pos_base = observations["ee_pos_base"]
        eepad_pts = observations["eepad_pts"]  # (N, 40, 3)
        rgb = observations["rgb"]
        rgb = rgb / 255.0  # (N, 3, H, W)
        N, ch, h, w = rgb.shape
        depth = observations["depth"]  # (N, 1, H, W)
        seg = observations["seg"]  # (N, H, W)

        seg_map = (
            F.one_hot(seg.long(), num_classes=self.num_classes)
            .permute(0, 3, 1, 2)
            .contiguous()
        )  # (N, C, H, W)
        seg_map_noise = (
            seg_map + torch.randn(seg_map.shape, device=seg_map.device) * 0.1
        )

        ### topk to get topk ind
        values, indices = torch.topk(
            seg_map_noise.view(N, self.num_classes, -1), self.sample_num, dim=2
        )  # (N, C, sample_num)
        zc = (
            -depth.view(N, -1)
            .unsqueeze(1)
            .repeat(1, self.num_classes, 1)
            .gather(2, indices)
        )  # (N, C, sample_num)
        v = torch.div(indices, w, rounding_mode="floor")  # (N, C, sample_num)
        u = indices - v * w  # (N, C, sample_num)
        ## all k points
        uvz_sample = torch.stack((u * zc, v * zc, zc), dim=1).view(
            N, 3, -1
        )  # (N, 3, C * sample_num)
        camxyz_sample = torch.matmul(
            self.ego_cam_intrin_inv, uvz_sample
        )  # (N, 3, C * sample_num)
        camxyz_sample[:, 0] = -camxyz_sample[:, 0]  # (N, 3, C * sample_num)
        eexyz_sample = (
            torch.matmul(
                camxyz_sample.transpose(1, 2).contiguous(),
                (self.ego_cam_extrin[:3, :3]).t(),
            )
            + self.ego_cam_extrin[:3, 3]
        )  # (N, C * sample_num, 3)

        eewpad_xyz_sample = torch.cat(
            (eexyz_sample, eepad_pts), dim=1
        )  # (N, C*sample_num+40, 3)

        ## concat class info
        eexyzC_sample = torch.cat(
            (eewpad_xyz_sample, self.seg_info.repeat(N, 1, 1)), dim=2
        )  # (N, C*sample_num+40, 3+C+1)

        ## PointNet to get 128 feat
        pn_feature, _, _ = self.pn(
            eexyzC_sample.transpose(2, 1)
        )  # (N, 3+C+1, C*sample_num+40)

        state = torch.cat([qpos, ee_pos_base], dim=1).repeat(1, self.state_repeat_times)

        return torch.cat([state, pn_feature], dim=1)


class CustomRGBDEncodExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Dict, state_repeat_times=2):
        # We do not know features-dim here before going over all the items,
        # so put something dummy for now. PyTorch requires calling
        # nn.Module.__init__ before adding modules
        super(CustomRGBDEncodExtractor, self).__init__(
            observation_space, features_dim=1
        )
        self.state_repeat_times = state_repeat_times
        # load the Attention model
        # siamese_net = SiameseNet(
        #     input_channels=[3, 1],
        #     filters=[8],
        #     kernel_sizes=[5],
        #     strides=[1],
        #     activation="lrelu",
        #     norm=None,
        # )
        # self.gene_cnn = GeneralCNN(
        #     siamese_net=siamese_net,
        #     filters=[16, 16],
        #     kernel_sizes=[5, 5],
        #     strides=[2, 2],
        #     norm=None,
        #     activation="lrelu",
        # )
        # self.gene_cnn.train()
        siamese_net = SiameseNet(
            input_channels=[3, 1],
            filters=[8],
            kernel_sizes=[5],
            strides=[1],
            activation="lrelu",
            norm=None,
        )
        self.unet_encod = UNetEncoder(
            siamese_net=siamese_net,
            filters=[16, 16],
            kernel_sizes=[5, 5],
            strides=[2, 2],
            norm=None,
            activation="lrelu",
        ).to(torch.device("cuda:0"))
        self.unet_encod.train()

        # Update the features dim manually
        self._features_dim = 12 * self.state_repeat_times + 16

    def forward(self, observations) -> torch.Tensor:
        qpos = observations["qpos"]
        ee_pos_base = observations["ee_pos_base"]
        rgb = observations["rgb"]
        rgb = rgb / 255.0  # Normalization

        depth = observations["depth"]  # (N, 1, H, W)
        depth_norm = torch.clip(depth - torch.mean(depth), -1, 1)
        encod_latent = self.unet_encod([rgb, depth_norm])  # (N, 16)
        # gene_latent = self.gene_cnn([rgb, depth_norm])  # (N, 128)

        # state = torch.cat([qpos, ee_pos_base, encod_latent], dim=1).repeat(1, self.state_repeat_times)
        state = torch.cat([qpos, ee_pos_base], dim=1).repeat(1, self.state_repeat_times)

        return torch.cat([state, encod_latent], dim=1)


class CustomPCSegExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Dict, state_repeat_times=1):
        # We do not know features-dim here before going over all the items,
        # so put something dummy for now. PyTorch requires calling
        # nn.Module.__init__ before adding modules
        super(CustomPCSegExtractor, self).__init__(observation_space, features_dim=1)
        self.state_repeat_times = state_repeat_times

        self.attn = PointNetDenseCls(in_ch=3, mlp_specs=[64, 128, 512], k=2).to(
            torch.device("cuda:0")
        )
        self.attn.train()

        # Update the features dim manually
        self._features_dim = 15 * self.state_repeat_times

    def forward(self, observations) -> torch.Tensor:
        world_xyz = observations["xyz"]
        bs, N, _ = world_xyz.shape
        # handle_seg = observations["handle_seg"]
        qpos = observations["qpos"]
        ee_pos_base = observations["ee_pos_base"]
        ee_posemat = observations["ee_posemat_world"]  # (bs,4,4)

        ## normalization
        # compute the lenght of max(X,Y,Z), scale to [-1,1]
        # m = torch.max(torch.sqrt(torch.sum(xyz ** 2, dim=1)))
        # xyz_norm = xyz / m
        # pn2_sigmap = self.pn2score_net(torch.cat((xyz_norm, rgb), dim=2))  # (bs, N, 6) => (bs, N, 2)
        # attnmap = torch.max(pn2_sigmap, dim=2)[0]  # (bs, N)

        # attnmap, _, _ = self.pn_seg(torch.cat((xyz_norm, rgb), dim=2).transpose(2, 1).contiguous())   # (bs, N, 1)
        # # max_indice = torch.argmax(attnmap[:, :, 0], dim=1)  # (bs, N) => (bs,1)
        # max_indice = torch.argmax(attnmap[:, :, 0], dim=1)  # (bs, N) => (bs,1)
        # max_ee_xyz = torch.stack([xyz[i, max_indice[i], :] for i in range(bs)])  # (N, 3)
        # pn_feature, _, _ = self.pn(torch.cat((xyz_norm, rgb, attnmap), dim=2).transpose(2, 1).contiguous())  # (bs, N, 7) => (bs, 128)

        # (bs, 128), (bs, N, 2)
        _, attnmap, _, _ = self.attn(world_xyz.transpose(2, 1).contiguous())
        max_indice = torch.argmax(attnmap[:, :, 0], dim=1)  # (bs, N) => (bs,1)
        max_world_xyz = torch.stack(
            [world_xyz[i, max_indice[i], :] for i in range(bs)]
        )  # (bs, 3)

        max_ee_xyz = (
            torch.einsum("ij,ikj->ik", max_world_xyz, ee_posemat[:, :3, :3])
            + ee_posemat[:, :3, 3]
        )  # (bs, 3)

        state = torch.cat([qpos, ee_pos_base, max_ee_xyz], dim=1).repeat(
            1, self.state_repeat_times
        )  # (bs, 15)

        return state


class CustomPCRGBSegGTExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Dict):
        # We do not know features-dim here before going over all the items,
        # so put something dummy for now. PyTorch requires calling
        # nn.Module.__init__ before adding modules
        super(CustomPCRGBSegGTExtractor, self).__init__(
            observation_space, features_dim=1
        )

        self.pn = PointNetfeat(
            in_ch=5,
            mlp_specs=[64, 128, 1024],
            xyz_transform=False,
            feature_transform=False,
        )
        self.pn.train()

        # Update the features dim manually
        self._features_dim = 128 + 12 * 10

    def forward(self, observations) -> torch.Tensor:
        xyz = observations["xyz"]
        # rgb = observations["rgb"]
        handle_seg = observations["handle_seg"]
        qpos = observations["qpos"]
        ee_pos_base = observations["ee_pos_base"]
        # check_grasp = observations["check_grasp"]

        ## normalization
        # rgb = rgb * 2 - 1
        # compute the lenght of max(X,Y,Z)
        # m = torch.max(torch.sqrt(torch.sum(xyz ** 2, dim=1)))
        # scale to [-1,1]
        # xyz_norm = xyz / m

        bs, N, _ = xyz.shape
        # (bs, 128), (bs, N, 1)
        pn_feature, _, _, _ = self.pn(
            torch.cat((xyz, handle_seg), dim=2).transpose(2, 1).contiguous()
        )
        # max_indice = torch.argmax(handle_seg[:, :, 0], dim=1)  # (bs, N) => (bs,1)
        # max_ee_xyz = torch.stack([xyz[i, max_indice[i], :] for i in range(bs)])  # (bs, 3)

        state = torch.cat([qpos, ee_pos_base], dim=1).repeat(1, 10)  # (bs, 120)

        return torch.cat([state, pn_feature], dim=1)


class CustomPCSegGTExtractor(BaseFeaturesExtractor):
    def __init__(
        self, observation_space: gym.spaces.Dict, state_repeat_times=2, pts_num=300
    ):
        # We do not know features-dim here before going over all the items,
        # so put something dummy for now. PyTorch requires calling
        # nn.Module.__init__ before adding modules
        super(CustomPCSegGTExtractor, self).__init__(observation_space, features_dim=1)
        self.state_repeat_times = state_repeat_times
        # self.pts_num = pts_num
        # self.pn = PointNetfeat(in_ch=3, mlp_specs=[64, 128, 1024], xyz_transform=False, feature_transform=False)
        # self.pn.train()
        self.weight_centers = torch.nn.Parameter(
            torch.ones(2) * 0.5, requires_grad=True
        )

        # Update the features dim manually
        self._features_dim = 15 * self.state_repeat_times

    def forward(self, observations) -> torch.Tensor:
        ee_xyz = observations["ee_xyz"]  ## (N, 3, H, W)
        # rgb = observations["rgb"]
        handle_seg = observations["handle_seg"]  ## (N, 1, H, W)
        qpos = observations["qpos"]
        ee_pos_base = observations["ee_pos_base"]
        # check_grasp = observations["check_grasp"]

        N, ch, h, w = ee_xyz.shape
        eepts = ee_xyz.view(N, 3, -1)
        handel_mask = handle_seg.view(N, -1).bool()
        background_mask = ~handel_mask
        ee_ptcenter = torch.zeros((N, 3), device="cuda:0")
        background_ptcenter = torch.zeros((N, 3), device="cuda:0")
        for i in range(N):
            if torch.any(handel_mask[i]):
                ee_ptcenter[i] = (eepts[i, :, handel_mask[i]]).mean(1)  # (3)
                # zc = torch.stack([-depth.view(N, -1)[i, max_indice[i]] for i in range(N)])  # (N, sample_num), too much time
                # eepts[i] = ((-depth.view(N, -1)).gather(1, max_indice.unsqueeze(1))).squeeze(1)
            # else:
            #     eepts.append((ee_xyz.view(N, 3, -1))[i, :, np.random.choice(np.arange(h * w))])
            background_ptcenter[i] = eepts[i, :, background_mask[i]].mean(1)
        weight = F.softmax(self.weight_centers, dim=0)
        eepts_pred = weight[0] * ee_ptcenter + weight[1] * background_ptcenter
        # weights = handle_seg.view(N, -1)
        ## argmax to get maxpoint ind
        # max_indice = torch.argmax(weights, dim=1)

        ## multinomial sampling
        # weights[weights.ge(0.5)] = 200  # handle_ratio: 72*128/42 ~ 220
        # weights[weights.lt(0.5)] = 1
        # indices = torch.multinomial(weights, self.pts_num, replacement=False)  # (N, pts_num)
        # eexyz_pred = torch.stack([ee_xyz.view(N, 3, -1)[i, :, indices[i]] for i in range(N)])  # (N, 3, pts_num)
        # eepts_pred = torch.stack([ee_xyz.view(N, 3, -1)[i, :, max_indice[i]] for i in range(N)])  # (N, 3)

        # pn_feature, _, _, _ = self.pn(eexyz_pred)
        # max_indice = torch.argmax(handle_seg[:, :, 0], dim=1)  # (bs, N) => (bs,1)
        # max_ee_xyz = torch.stack([xyz[i, max_indice[i], :] for i in range(bs)])  # (bs, 3)

        state = torch.cat([qpos, ee_pos_base, eepts_pred], dim=1).repeat(
            1, self.state_repeat_times
        )  # (bs, 120)

        return state
