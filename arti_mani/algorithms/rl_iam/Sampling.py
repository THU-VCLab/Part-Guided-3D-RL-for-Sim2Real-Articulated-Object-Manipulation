from collections import deque

import numpy as np
import torch
from arti_mani.utils.o3d_utils import pcd_uni_down_sample_with_crop
from scipy.spatial.distance import cdist


class UniDownsampler:
    def __init__(self, sample_num: int):
        self.sample_num = sample_num
        self.sample_indices = None

    def sampling(
        self,
        raw_xyz: np.ndarray,  # (N, 3), H*W=N
        episode_rng,  # RandomState from env
    ):
        ## crop the pts & uniform downsample to num pts
        sample_indices = pcd_uni_down_sample_with_crop(
            raw_xyz,
            self.sample_num,
            min_bound=np.array([-0.5, -1, 1e-3]),
            max_bound=np.array([2, 2, 2]),
        )
        indices_len = len(sample_indices)
        if indices_len < 1000:
            sample_indices.extend(
                episode_rng.choice(raw_xyz.shape[0], self.sample_num - indices_len)
            )
        self.sample_indices = sample_indices
        return sample_indices


class ScoreSampler:
    def __init__(self, sample_num: int):
        self.sample_num = sample_num
        self.sample_indices = None

    def sampling(
        self,
        seg_map: torch.Tensor,  # (C, H, W)
    ):
        num_classes = seg_map.shape[0]
        score_map = seg_map.view(num_classes, -1)  # (C, N)
        _, indices = torch.topk(score_map, self.sample_num, dim=1)  # (C, sample_num)
        pts_index = indices.detach().cpu().numpy().reshape(-1)  # (C*sample_num)
        self.sample_indices = pts_index
        return pts_index


def farthest_point_sample(xyz, npoint, episode_rng):
    N, C = xyz.shape
    centroids = np.zeros((npoint,), dtype=np.int_)
    distance = np.ones((N,)) * 1e10
    farthest = episode_rng.randint(0, N)

    for i in range(npoint):
        centroids[i] = farthest
        centroid = xyz[farthest, :]
        dist = np.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = np.argmax(distance, -1)
    return centroids


class FPSSampler:
    def __init__(self, sample_num: int):
        self.sample_num = sample_num
        self.sample_indices = None

    def sampling(
        self,
        raw_xyz: np.ndarray,
        num_classes: int,
        seg_mc_map: np.ndarray,  # (H*W)
        episode_rng,  # RandomState from env
    ):
        part_inds = []
        for ind in range(num_classes):
            one_indices = np.where(seg_mc_map == ind)[0]
            ones_num = one_indices.shape[0]
            if ones_num == 0:
                cur_indices = np.zeros(self.sample_num, dtype=np.int_)
            elif ones_num < self.sample_num:
                repeat_times = self.sample_num // ones_num
                remain_num = self.sample_num - repeat_times * ones_num
                cur_indices = np.concatenate(
                    (one_indices.repeat(repeat_times), one_indices[:remain_num])
                )
            else:
                part_xyz = raw_xyz[one_indices]
                rel_indices = farthest_point_sample(
                    part_xyz, self.sample_num, episode_rng
                )
                cur_indices = one_indices[rel_indices]
            part_inds.append(cur_indices)
        pts_index = np.concatenate(part_inds)  # (C*sample_num)
        self.sample_indices = pts_index
        return pts_index


class RandomSampler:
    def __init__(self, sample_num: int):
        self.sample_num = sample_num
        self.sample_indices = None

    def sampling(
        self,
        seg_map_mc: np.ndarray,  # (H*W)
        num_classes: int,
        episode_rng,  # RandomState from env
    ):
        part_inds = []
        for ind in range(num_classes):
            one_indices = np.where(seg_map_mc == ind)[0]
            ones_num = one_indices.shape[0]
            if ones_num == 0:
                cur_indices = np.zeros(self.sample_num, dtype=np.int_)
            elif ones_num < self.sample_num:
                repeat_times = self.sample_num // ones_num
                remain_num = self.sample_num - repeat_times * ones_num
                cur_indices = np.concatenate(
                    (one_indices.repeat(repeat_times), one_indices[:remain_num])
                )
            else:
                cur_indices = episode_rng.choice(
                    one_indices, self.sample_num, replace=False
                )
            part_inds.append(cur_indices)
        pts_index = np.concatenate(part_inds)  # (C*sample_num)
        self.sample_indices = pts_index
        return pts_index


class FCUWSampler:
    """Frame-Consistent Uncertainty-aware Weighted Sampler"""

    def __init__(self, frame_num: int, sample_num: int):
        self.partpts_q = deque(maxlen=frame_num)
        self.frame_num = frame_num
        self.sample_num = sample_num
        self.sample_indices = None

    def sampling(
        self,
        raw_xyz: np.ndarray,
        num_classes: int,
        seg_map_mc: np.ndarray,  # (H*W)
        uncertainty_map: np.ndarray,
        episode_rng,  # RandomState from env
    ):
        # (H*W)
        uncertainty_indices = uncertainty_map.reshape((-1,))
        weight_map = np.zeros((num_classes, seg_map_mc.shape[0]))
        pts_index = []
        for ind in range(num_classes):
            part_indices = np.where(seg_map_mc == ind)[0]
            part_pts_num = part_indices.shape[0]
            # (H*W, ) => (PN, ), (0, 1.7)
            if part_pts_num == 0:
                pts_index += np.zeros(self.sample_num, dtype=np.int_).tolist()
            elif part_pts_num < self.sample_num:
                repeat_times = self.sample_num // part_pts_num
                remain_num = self.sample_num - repeat_times * part_pts_num
                pts_index += np.concatenate(
                    (part_indices.repeat(repeat_times), part_indices[:remain_num])
                ).tolist()
            else:
                part_xyz = raw_xyz[part_indices]
                # (PN, 3), (L*30, 3) => (PN, L*30)
                part_confidence = np.exp(
                    -uncertainty_indices[part_indices]
                )  # w_uncertainty
                if len(self.partpts_q) != 0:
                    cache_partpts = (
                        np.array(self.partpts_q)
                        .transpose((1, 0, 2, 3))
                        .reshape((num_classes, -1, 3))
                    )
                    part_dist = cdist(part_xyz, cache_partpts[ind], metric="euclidean")
                    w_frameconsist = 2 ** (-40 * (part_dist.min(axis=1)))
                    part_confidence = w_frameconsist * part_confidence
                norm_w = part_confidence / np.sum(part_confidence)
                weight_map[ind, part_indices] = norm_w
                pts_index += episode_rng.choice(
                    part_indices, self.sample_num, replace=True, p=norm_w
                ).tolist()
        # (C*sample_num, 3)
        sample_xyz = raw_xyz[pts_index]
        self.partpts_q.append(
            sample_xyz.reshape(num_classes, self.sample_num, 3)
        )  # (C, sample_num, 3)
        self.sample_indices = pts_index
        return pts_index, weight_map
