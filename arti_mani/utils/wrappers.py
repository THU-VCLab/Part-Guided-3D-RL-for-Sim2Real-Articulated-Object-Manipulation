from collections import OrderedDict
from copy import deepcopy
from typing import *

import cv2
import numpy as np
from arti_mani.utils.common import (
    clip_and_scale_action,
    convert_np_bool_to_float,
    normalize_action,
    normalize_action_space,
)
from arti_mani.utils.o3d_utils import pcd_voxel_down_sample_with_crop
from gym import ActionWrapper, ObservationWrapper, Wrapper, spaces


class NormalizeActionWrapper(ActionWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.action_space = normalize_action_space(self.env.action_space)

    def action(self, action):
        return clip_and_scale_action(
            action, self.env.action_space.low, self.env.action_space.high
        )

    def reverse_action(self, action):
        return normalize_action(
            action, self.env.action_space.low, self.env.action_space.high
        )


def put_texts_on_image(image: np.ndarray, lines: List[str]):
    assert image.dtype == np.uint8, image.dtype
    image = image.copy()

    font_size = 0.4
    font_thickness = 1
    font = cv2.FONT_HERSHEY_SIMPLEX

    y = 0
    for line in lines:
        textsize = cv2.getTextSize(line, font, font_size, font_thickness)[0]
        y += textsize[1] + 10
        x = 10
        cv2.putText(
            image,
            line,
            (x, y),
            font,
            font_size,
            (255, 0, 0),
            font_thickness,
            lineType=cv2.LINE_AA,
        )
    return image


def put_infos_on_image(image, info: Dict[str, np.ndarray], overlay=True):
    lines = [f"{k}: {v.round(3)}" for k, v in info.items()]
    return put_texts_on_image(image, lines)


class RenderInfoWrapper(Wrapper):
    def step(self, action):
        obs, rew, done, info = super().step(action)
        if "TimeLimit.truncated" in info.keys():
            info["TimeLimit.truncated"] = convert_np_bool_to_float(
                info["TimeLimit.truncated"]
            )
        self._info_for_render = info
        return obs, rew, done, info

    def reset(self, **kwargs):
        obs = super().reset(**kwargs)
        # self._info_for_render = self.env.get_info()
        self._info_for_render = {}
        return obs

    def render(self, mode="rgb_array", **kwargs):
        if mode == "rgb_array":
            img = super().render(mode=mode, **kwargs)
            img = (img * 255).astype(np.uint8)
            return put_infos_on_image(img, self._info_for_render, overlay=True)
        elif mode == "cameras":
            img = super().render(mode=mode, **kwargs)
            return put_infos_on_image(img, self._info_for_render, overlay=True)
        else:
            return super().render(mode=mode, **kwargs)


class ActionControlWrapper(ActionWrapper):
    def __init__(self, env, control_mode=None):
        super().__init__(env)
        if control_mode is None:
            control_mode = self.env.unwrapped.agent.control_mode
        else:
            self.env.unwrapped.agent.set_control_mode(control_mode)
        self._control_mode = control_mode
        self.action_space = self.env.action_space[self._control_mode]

    def reset(self, *args, **kwargs):
        ret = super().reset(*args, **kwargs)
        self.env.unwrapped.agent.set_control_mode(self._control_mode)
        return ret

    def action(self, action):
        return {"control_mode": self._control_mode, "action": action}

    @property
    def control_mode(self):
        return self._control_mode


class ObsWrapper(ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        if self.obs_mode in ["state", "state_dict", "state_depth"]:
            pass
        else:
            raise NotImplementedError()

    def observation(self, observation):
        from arti_mani.utils.common import flatten_state_dict

        if self.obs_mode == "state":
            return observation
        elif self.obs_mode == "state_dict":
            obs = observation
            obs.pop("task", None)
            obs.pop("articulation", None)
            return obs
        elif self.obs_mode == "state_depth":
            obs = observation
            ### process uvz
            base_target_uvz = obs.pop("base_target_uvz", None)
            base_target_uvz[:2] = base_target_uvz[:2] / 4
            base_target_uvz[2] = 39 * (base_target_uvz[2] - 0.5) / (2 - 0.5)
            base_target_uvz = np.round(base_target_uvz).astype(np.int_)
            obs["zvu"] = deepcopy(base_target_uvz[:, [2, 1, 0]])  # (3, )
            ### process depth
            depth = obs.pop("base_depth", None)
            # depth = depth.transpose(2,0,1)[:, ::4, ::4]  # (1, H, W)
            depth = depth.transpose(2, 0, 1).squeeze()[::4, ::4]
            depth = (depth - np.mean(depth)) / np.std(depth)
            obs["depth"] = deepcopy(depth)  # (H//4, W//4)
            return obs
        else:
            raise NotImplementedError()


class ManiObsWrapper(ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        if self.obs_mode in ["state", "state_dict"]:
            pass
        else:
            raise NotImplementedError()

    def observation(self, observation):
        from arti_mani.utils.common import flatten_state_dict

        if self.obs_mode == "state":
            return observation
        elif self.obs_mode == "state_dict":
            obs = observation
            obs.pop("task", None)
            obs.pop("articulation", None)
            return obs
        else:
            raise NotImplementedError()

    @property
    def _max_episode_steps(self):
        return self.env.unwrapped._max_episode_steps

    def render(self, mode="human", *args, **kwargs):
        if mode == "human":
            self.env.render(mode, *args, **kwargs)
            return

        if mode in ["rgb_array", "color_image"]:
            img = self.env.render(mode="rgb_array", *args, **kwargs)
        else:
            img = self.env.render(mode=mode, *args, **kwargs)
        if isinstance(img, dict):
            if "world" in img:
                img = img["world"]
            elif "main" in img:
                img = img["main"]
            else:
                print(img.keys())
                exit(0)
        if isinstance(img, dict):
            img = img["rgb"]
        if img.ndim == 4:
            assert img.shape[0] == 1
            img = img[0]
        if img.dtype in [np.float32, np.float64]:
            img = np.clip(img, a_min=0, a_max=1) * 255
        img = img[..., :3]
        img = img.astype(np.uint8)
        return img


class EgoDepthCutoff(ObservationWrapper):
    """For Egocentric camera realsense D415, depth should be cutoff due to real sensor range."""

    def __init__(self, env, depth_cut=False, cutlen=0.225):
        ### default cutlen: sensor range(0.45)-grasp_site_z(0.225)
        super().__init__(env)
        self.depth_cut = depth_cut
        self.cutlen = cutlen
        # self.grasp_z = self.env.grasp_site.pose.p[2]
        # self.gripper_pose = self.env.grasp_site.pose

    def observation(self, observation):
        obs = observation
        if (self.obs_mode == "depth" or self.obs_mode == "rgbd") and self.depth_cut:
            if self.env.grasp_site.pose.p[2] < self.cutlen:
                obs["depth"][..., 0] = np.zeros_like(obs["depth"][..., 0])
        return obs


class PointCloudPreprocessObsWrapper(ObservationWrapper):
    """Preprocess point cloud, crop and voxel downsample"""

    _num_pts: int
    _vox_size: float

    def __init__(
        self,
        env,
        num_pts=2048,
        vox_size=0.003,
        min_bound=np.array([-1, -1, 1e-3]),
        max_bound=np.array([1, 1, 1]),
    ):
        super().__init__(env)
        self._num_pts = num_pts
        self._vox_size = vox_size
        self._min_bound = min_bound
        self._max_bound = max_bound

    def observation(self, observation):
        if not self.obs_mode == "pointcloud":
            return observation

        obs = observation
        pointcloud = obs["pointcloud"]
        xyz = pointcloud["xyz"]
        sample_indices = pcd_voxel_down_sample_with_crop(
            xyz, self._vox_size, self._min_bound, self._max_bound
        )
        if len(sample_indices) >= self._num_pts:
            sample_indices = np.random.choice(
                sample_indices, self._num_pts, replace=False
            )
        else:
            ext_sample_indices = np.random.choice(
                sample_indices, (self._num_pts - len(sample_indices))
            )
            sample_indices.extend(ext_sample_indices)
        for k, v in pointcloud.items():
            pointcloud[k] = v[sample_indices]
        obs["pointcloud"] = pointcloud
        return obs
