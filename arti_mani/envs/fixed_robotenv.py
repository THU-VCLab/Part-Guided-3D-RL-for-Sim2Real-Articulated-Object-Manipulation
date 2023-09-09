from collections import OrderedDict
from typing import Dict, Tuple, Union

import numpy as np
import sapien.core as sapien
from arti_mani import AGENT_CONFIG_DIR, ASSET_DIR
from arti_mani.agents.camera import get_camera_rgb
from arti_mani.agents.fixed_xmate3_robotiq import FixedXmate3Robotiq
from arti_mani.agents.panda import Panda
from arti_mani.envs.base import BaseEnv
from arti_mani.utils.articulation_utils import (
    get_actor_state,
    get_articulation_state,
    set_actor_state,
    set_articulation_state,
)
from arti_mani.utils.geometry import angle_distance
from transforms3d.euler import euler2quat


class FixedXmate3RobotiqEnv(BaseEnv):
    SUPPORTED_OBS_MODES: Tuple[str] = ()
    SUPPORTED_REWARD_MODES: Tuple[str] = ()
    SUPPORTED_CONTROL_MODES: Tuple[str] = ()
    _agent: FixedXmate3Robotiq

    def __init__(
        self,
        obs_mode=None,
        reward_mode=None,
        control_mode=None,
        sim_freq=500,
        control_freq=20,
        device="",
    ):
        self._articulation = None
        self.step_in_ep = 0
        self._prev_actor_pose = sapien.Pose()
        self._cache_obs_state_dict: OrderedDict = (
            OrderedDict()
        )  # save obs state dict to save reward computation time
        self._cache_info = {}
        super().__init__(
            obs_mode, reward_mode, control_mode, sim_freq, control_freq, device
        )

    # -------------------------------------------------------------------------- #
    # Reset
    # -------------------------------------------------------------------------- #

    def _setup_physical_materials(self):
        self.add_physical_material("default", 1.0, 1.0, 0.0)

    def _setup_render_materials(self):
        self.add_render_material(
            "ground",
            color=[0.5, 0.5, 0.5, 1],
            metallic=1.0,
            roughness=0.7,
            specular=0.04,
        )
        self.add_render_material(
            "default",
            color=[0.8, 0.8, 0.8, 1],
            metallic=0,
            roughness=0.9,
            specular=0.0,
        )

    def _load_articulations(self):
        pass

    def _load_agent(self):
        self._agent = FixedXmate3Robotiq.from_config_file(
            AGENT_CONFIG_DIR / "fixed_xmate3_robotiq.yml",
            self._scene,
            self._control_freq,
        )

    def _initialize_articulations(self):
        pass

    def _initialize_agent(self):
        raise NotImplementedError

    def _load_table(self):
        loader = self._scene.create_actor_builder()
        loader.add_visual_from_file(
            str(ASSET_DIR / "descriptions/optical_table/visual/optical_table.dae"),
            # scale=[3, 3, 1]
        )
        loader.add_collision_from_file(
            str(ASSET_DIR / "descriptions/optical_table/visual/optical_table.dae"),
            # scale=[3, 3, 1]
        )
        self._table = loader.build_static(name="table")
        self._table.set_pose(sapien.Pose([0.0, 0.0, -0.04]))

    def reconfigure(self):
        self._prev_actor_pose = sapien.Pose()
        self._clear()

        self._setup_scene()
        self._setup_physical_materials()
        self._setup_render_materials()
        self._load_table()
        self._load_actors()
        self._load_agent()
        self._load_articulations()
        self._setup_camera()
        self._setup_lighting()

        if self._viewer is not None:
            self._setup_viewer()

        # cache actors and articulations for sim state
        self._actors = self.get_actors()
        self._articulations = self.get_articulations()
        # Cache initial simulation state
        # self._initial_sim_state = self.get_sim_state()

    def reset(self, seed=None, reconfigure=False):
        super().reset(seed, reconfigure)
        self.step_in_ep = 0
        # for actor in self._agent._robot.get_links():
        #     make_actor_visible(actor, False)
        self._cache_obs_state_dict.clear()
        self._cache_info.clear()
        return self.get_obs()

    # -------------------------------------------------------------------------- #
    # Step
    # -------------------------------------------------------------------------- #
    def check_success(self) -> bool:
        raise NotImplementedError

    def get_done(self):
        # return self.check_success()
        return False

    def get_info(self):
        return dict(is_success=self.check_success())

    def step(self, action: Union[None, np.ndarray, Dict]):
        self.step_in_ep += 1
        self.step_action(action)
        obs = self.get_obs()
        reward = self.get_reward(obs=obs, action=action)
        info = self.get_info()
        done = self.get_done()
        return obs, reward, done, info

    # ---------------------------------------------------------------------------- #
    # Visualization
    # ---------------------------------------------------------------------------- #
    def _setup_lighting(self):
        self._scene.set_ambient_light([0.3, 0.3, 0.3])
        self._scene.add_point_light([2, 2, 2], [1, 1, 1])
        self._scene.add_point_light([2, -2, 2], [1, 1, 1])
        self._scene.add_point_light([-2, 0, 2], [1, 1, 1])
        self._scene.add_directional_light([1, -1, -1], [0.3, 0.3, 0.3])
        self._scene.add_directional_light([0, 0, -1], [1, 1, 1])

    def _setup_viewer(self):
        super()._setup_viewer()
        ### front view
        # self._viewer.set_camera_xyz(1.0, 0.0, 1.2)
        # self._viewer.set_camera_rpy(0, -0.5, 3.14)
        ### right side view
        # self._viewer.set_camera_xyz(0, -1.0, 1.2)
        # self._viewer.set_camera_rpy(0, -0.5, -1.57)
        ### top view
        self._viewer.set_camera_xyz(0.0, 0.0, 1.2)
        self._viewer.set_camera_rpy(0, -3.14, 0)

    def _setup_camera(self):
        # self.render_camera = self._scene.add_camera("sideview", 512, 512, 1, 0.01, 10)
        # self.render_camera.set_local_pose(
        #     sapien.Pose([0, -1.0, 1.2], euler2quat(0, 0.5, 1.57))
        # )
        self.render_camera = self._scene.add_camera("topview", 512, 512, 1, 0.01, 10)
        self.render_camera.set_local_pose(
            sapien.Pose([-0.1, 0.0, 1.5], euler2quat(0, 1.57, 0))
        )

    def render(self, mode="human"):
        if mode == "rgb_array":
            self._scene.update_render()
            self.render_camera.take_picture()
            rgb = get_camera_rgb(self.render_camera)
            # rgb = self.render_camera.get_color_rgba()[..., :3]
            # rgb = np.clip(rgb * 255, 0, 255).astype(np.uint8)
            return rgb
        else:
            return super().render(mode=mode)

    # utilization
    def check_actor_static(self, actor, max_v=None, max_ang_v=None):
        if self.step_in_ep <= 1:
            flag_v = (max_v is None) or (np.linalg.norm(actor.get_velocity()) <= max_v)
            flag_ang_v = (max_ang_v is None) or (
                np.linalg.norm(actor.get_angular_velocity()) <= max_ang_v
            )
        else:
            pose = actor.get_pose()
            t = 1.0 / self._control_freq
            flag_v = (max_v is None) or (
                np.linalg.norm(pose.p - self._prev_actor_pose.p) <= max_v * t
            )
            flag_ang_v = (max_ang_v is None) or (
                angle_distance(self._prev_actor_pose, pose) <= max_ang_v * t
            )
        self._prev_actor_pose = actor.get_pose()
        return flag_v and flag_ang_v

    # -------------------------------------------------------------------------- #
    # Simulation state (required for MPC)
    # -------------------------------------------------------------------------- #
    def get_actors(self):
        return [x for x in self._scene.get_all_actors()]

    def get_articulations(self):
        return [self._agent._robot, self._articulation]

    def get_sim_state(self) -> np.ndarray:
        state = []
        # for actor in self._scene.get_all_actors():
        for actor in self._actors:
            state.append(get_actor_state(actor))
        # for articulation in self._scene.get_all_articulations():
        for articulation in self._articulations:
            state.append(get_articulation_state(articulation))
        return np.hstack(state)

    def set_sim_state(self, state: np.ndarray):
        KINEMANTIC_DIM = 13  # [pos, quat, lin_vel, ang_vel]
        start = 0
        for actor in self._actors:
            set_actor_state(actor, state[start : start + KINEMANTIC_DIM])
            start += KINEMANTIC_DIM
        for articulation in self._articulations:
            ndim = KINEMANTIC_DIM + 2 * articulation.dof
            set_articulation_state(articulation, state[start : start + ndim])
            start += ndim

    def get_state(self):
        return self.get_sim_state()

    def set_state(self, state: np.ndarray):
        return self.set_sim_state(state)


class PandaEnv(FixedXmate3RobotiqEnv):
    _agent: Panda

    def _load_agent(self):
        self._agent = Panda.from_config_file(
            AGENT_CONFIG_DIR / "panda.yml", self._scene, self._control_freq
        )


class FixedXmate3RobotiqSensorEnv(FixedXmate3RobotiqEnv):
    def _load_agent(self):
        self._agent = FixedXmate3Robotiq.from_config_file(
            AGENT_CONFIG_DIR / "fixed_xmate3_robotiq_sensors.yml",
            self._scene,
            self._control_freq,
        )


class FixedXmate3RobotiqSensorLowResEnv(FixedXmate3RobotiqEnv):
    def _load_agent(self):
        self._agent = FixedXmate3Robotiq.from_config_file(
            AGENT_CONFIG_DIR / "fixed_xmate3_robotiq_sensors_low_res.yml",
            self._scene,
            self._control_freq,
        )


class FixedXmate3RobotiqLowResEnv(FixedXmate3RobotiqEnv):
    def _load_agent(self):
        self._agent = FixedXmate3Robotiq.from_config_file(
            AGENT_CONFIG_DIR / "fixed_xmate3_robotiq_low_res.yml",
            self._scene,
            self._control_freq,
        )
