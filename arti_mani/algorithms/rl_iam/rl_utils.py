import time
from typing import Optional, Tuple

import gym
import numpy as np
import torch
from arti_mani import ASSET_DIR
from arti_mani.utils.wrappers import NormalizeActionWrapper, RenderInfoWrapper
from stable_baselines3.common.vec_env import VecMonitor
from stable_baselines3.common.vec_env.base_vec_env import (
    VecEnv,
    VecEnvObs,
    VecEnvStepReturn,
    VecEnvWrapper,
)


class Custom_VecMonitor(VecMonitor):
    def __init__(
        self,
        venv: VecEnv,
        filename: Optional[str] = None,
        info_keywords: Tuple[str, ...] = (),
    ):
        super().__init__(
            venv,
            filename,
            info_keywords,
        )
        self.success_flag = np.zeros(self.num_envs, dtype=np.bool_)

    def reset(self) -> VecEnvObs:
        obs = self.venv.reset()
        self.episode_returns = np.zeros(self.num_envs, dtype=np.float32)
        self.episode_lengths = np.zeros(self.num_envs, dtype=np.int32)
        self.success_flag = np.zeros(self.num_envs, dtype=np.bool_)
        return obs

    def step_wait(self) -> VecEnvStepReturn:
        obs, rewards, dones, infos = self.venv.step_wait()
        for i in range(self.num_envs):
            if infos[i]["is_success"] == 1:
                arti_fail_rate = self.venv.unwrapped.get_attr("_arti_fail_rate")

        self.episode_returns += rewards
        self.episode_lengths += 1
        new_infos = list(infos[:])
        for i in range(len(dones)):
            if dones[i]:
                info = infos[i].copy()
                episode_return = self.episode_returns[i]
                episode_length = self.episode_lengths[i]
                episode_info = {
                    "r": episode_return,
                    "l": episode_length,
                    "t": round(time.time() - self.t_start, 6),
                }
                for key in self.info_keywords:
                    episode_info[key] = info[key]
                info["episode"] = episode_info
                self.episode_count += 1
                self.episode_returns[i] = 0
                self.episode_lengths[i] = 0
                if self.results_writer:
                    self.results_writer.write_row(episode_info)
                new_infos[i] = info
        return obs, rewards, dones, new_infos


def sb3_make_env(env_id, seed=0, obs_mode="state_depth", control_mode="pd_joint_vel"):
    """
    Utility function for custom env.

    :param env_id: (str) the environment ID
    :param seed: (int) the inital seed for RNG
    :param rank: (int) index of the subprocess
    """
    env = gym.make(
        env_id,
        articulation_config_path=ASSET_DIR
        / "partnet_mobility_configs/1999-cabinet-door-real-1.yml",
        obs_mode=obs_mode,  # "state", "state_dict", "rgbd", "pointcloud"
        control_mode=control_mode,  # "pd_joint_delta_pos", "pd_ee_delta_pose", "pd_ee_delta_pos"
        process_pc_mode="ms2_cam",  # "ms2_cam", "ms2_gt", "ms2_real", "ms2_realfull"
        gen_pc_mode="cam_depth",  ## "cam_depth", "sensor_depth"
        reward_mode="dense",  # "sparse", "dense"
    )
    env = NormalizeActionWrapper(env)
    env.seed(seed)
    return env


def sb3_make_evalenv(
    env_id, seed=0, obs_mode="state_depth", control_mode="pd_joint_vel"
):
    """
    Utility function for custom env.

    :param env_id: (str) the environment ID
    :param seed: (int) the inital seed for RNG
    :param rank: (int) index of the subprocess
    """
    env = gym.make(
        env_id,
        articulation_config_path=ASSET_DIR
        / "partnet_mobility_configs/1999-cabinet-door-real-1.yml",
        obs_mode=obs_mode,  # "state", "state_dict", "rgbd", "pointcloud"
        control_mode=control_mode,  # "pd_joint_delta_pos", "pd_ee_delta_pose", "pd_ee_delta_pos"
        process_pc_mode="ms2_cam",  # "ms2_cam", "ms2_gt", "ms2_real", "ms2_realfull"
        gen_pc_mode="cam_depth",  ## "cam_depth", "sensor_depth"
        reward_mode="dense",  # "sparse", "dense"
    )
    env = NormalizeActionWrapper(env)
    env = RenderInfoWrapper(env)
    env.seed(seed)
    return env


def sb3_make_env_multiinput(
    env_id, rank, seed=0, obs_mode="state_depth", control_mode="pd_joint_vel"
):
    """
    Utility function for multiprocessed env.

    :param env_id: (str) the environment ID
    :param rank: (int) index of the subprocess
    :param seed: (int) the inital seed for RNG
    :param obs_mode: (str) observation mode
    :param control_mode: (str) control mode
    """

    def _init():
        env = gym.make(
            env_id,
            articulation_ids=np.array(["1999-cabinet-door-real-1"]),
            articulation_config_path=ASSET_DIR / "partnet_mobility_configs/",
            obs_mode=obs_mode,  # "state", "state_dict", "rgbd", "pointcloud", "state_depth"
            control_mode=control_mode,  # "pd_joint_delta_pos", "pd_joint_vel", "pd_ee_delta_pose", "pd_ee_delta_pos"
            reward_mode="dense",  # "sparse", "dense"
        )
        env = NormalizeActionWrapper(env)
        # Important: use a different seed for each environment
        env.seed(seed + rank)
        return env

    return _init


def sb3_make_env_multicab(
    env_id,
    arti_ids,
    arti_config_path,
    obs_mode="state_depth",
    control_mode="pd_joint_vel",
    rank=0,
    seed=0,
):
    """
    Utility function for multiprocessed env.

    :param env_id: (str) the environment ID
    :param rank: (int) index of the subprocess
    :param seed: (int) the inital seed for RNG
    :param obs_mode: (str) observation mode
    :param control_mode: (str) control mode
    """

    def _init():
        env = gym.make(
            env_id,
            articulation_ids=arti_ids,
            articulation_config_path=arti_config_path,
            obs_mode=obs_mode,  # "state", "state_dict", "rgbd", "pointcloud", "state_depth"
            control_mode=control_mode,  # "pd_joint_delta_pos", "pd_joint_vel", "pd_ee_delta_pose", "pd_ee_delta_pos"
            reward_mode="dense",  # "sparse", "dense"
        )
        env = NormalizeActionWrapper(env)
        # Important: use a different seed for each environment
        env.seed(seed + rank)
        return env

    return _init


def sb3_make_env_multiarti(
    env_id,
    arti_ids,
    segmodel_path,
    sample_mode,
    frame_num,
    sample_num,
    other_handle_visible,
    obs_mode,
    control_mode,
    num_classes=6,
    device="",
    rank=0,
    seed=0,
):
    """
    Utility function for multiprocessed env.

    :param env_id: (str) the environment ID
    :param rank: (int) index of the subprocess
    :param seed: (int) the inital seed for RNG
    :param obs_mode: (str) observation mode
    :param control_mode: (str) control mode
    """

    def _init():
        env = gym.make(
            env_id,
            articulation_ids=arti_ids,
            segmodel_path=segmodel_path,
            sample_mode=sample_mode,
            frame_num=frame_num,
            sample_num=sample_num,
            other_handle_visible=other_handle_visible,
            num_classes=num_classes,
            obs_mode=obs_mode,  # "state", "state_dict", "rgbd", "pointcloud", "state_depth"
            control_mode=control_mode,  # "pd_joint_delta_pos", "pd_joint_vel", "pd_ee_delta_pose", "pd_ee_delta_pos"
            reward_mode="dense",  # "sparse", "dense"
            device=device,
        )
        env = NormalizeActionWrapper(env)
        # Important: use a different seed for each environment
        env.seed(seed + rank)
        return env

    return _init


def compute_zvupred(pred_vhp):
    N, Z, H, W = pred_vhp.shape
    indexes = torch.argmax(pred_vhp.view(N, -1), dim=1)
    zvu_pred = torch.zeros((pred_vhp.shape[0], 3), device=indexes.device)
    zvu_pred[:, 0] = torch.div(indexes, (H * W), rounding_mode="floor")
    zvu_pred[:, 1] = torch.div(
        indexes - (H * W) * zvu_pred[:, 0], W, rounding_mode="floor"
    )
    zvu_pred[:, 2] = indexes - (H * W) * zvu_pred[:, 0] - W * zvu_pred[:, 1]
    # zvu_pred = get_local_maxima_3d(vhmaps=pred_vhp, threshold=0.1, device=pred_vhp.device)
    # zvu_pred[:, 0] = zvu_pred[:, 0] / 39.0
    # zvu_pred[:, 1] = zvu_pred[:, 1] / 36.0
    # zvu_pred[:, 2] = zvu_pred[:, 2] / 64.0
    # zvu_pred = zvu_pred / torch.tensor((39.0, 36.0, 64.0), device=zvu_pred.device)
    # zvu_pred = zvu_pred / torch.tensor(hmap_size, device=zvu_pred.device)
    hmap_size = torch.tensor([32, 32, 32], device=indexes.device)
    xyz_min = torch.tensor([0.4, -0.7, 0.1], device=indexes.device)
    xyz_max = torch.tensor([1.2, 0.1, 0.9], device=indexes.device)
    zvu_pred = (zvu_pred + 0.5) / hmap_size * (
        xyz_max - xyz_min
    ) + xyz_min  # pred zvu in unit (m)

    return zvu_pred
