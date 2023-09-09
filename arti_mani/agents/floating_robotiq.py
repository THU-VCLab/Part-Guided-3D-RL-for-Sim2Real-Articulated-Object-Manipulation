import copy
from collections import OrderedDict
from typing import Dict

import numpy as np
import sapien.core as sapien
import yaml
from arti_mani import DESCRIPTION_DIR
from arti_mani.agents.base_agent import (
    ActiveLightSensor,
    AgentConfig,
    MountedActiveLightSensorConfig,
    MountedCameraConfig,
    create_mounted_camera,
    create_mounted_sensor,
    parse_urdf_config,
)
from arti_mani.agents.camera import get_camera_images
from arti_mani.agents.controllers.combined_controller import CombinedController
from arti_mani.utils.common import compute_angle_between
from arti_mani.utils.geometry import transform_points
from arti_mani.utils.sapien_utils import (
    check_joint_stuck,
    get_actor_by_name,
    get_entity_by_name,
    get_pairwise_contact_impulse,
)
from gym import spaces
from sapien.core import Pose


class FloatingRobotiq:
    _config: AgentConfig
    _scene: sapien.Scene
    _robot: sapien.Articulation
    _cameras: Dict[str, sapien.CameraEntity]
    _sensors: Dict[str, ActiveLightSensor]

    def __init__(self, config: AgentConfig, scene: sapien.Scene, control_freq: int):
        self._config = copy.deepcopy(config)
        self._scene = scene
        self._control_freq = control_freq
        self._initialize_robot()
        self._initialize_controllers()
        self._initialize_cameras()
        self._initialize_sensors()

        self.finger1_link: sapien.LinkBase = get_actor_by_name(
            self._robot.get_links(), "left_inner_finger_pad"
        )
        self.finger2_link: sapien.LinkBase = get_actor_by_name(
            self._robot.get_links(), "right_inner_finger_pad"
        )
        self.fingers_visid = []
        for visual_body in self.finger1_link.get_visual_bodies():
            self.fingers_visid.append(visual_body.get_visual_id())
        for visual_body in self.finger2_link.get_visual_bodies():
            self.fingers_visid.append(visual_body.get_visual_id())
        self.finger_size = (0.03, 0.07, 0.0075)  # values from URDF
        self.grasp_site: sapien.Link = get_entity_by_name(
            self._robot.get_links(), "grasp_convenient_link"
        )

    def _initialize_robot(self):
        loader = self._scene.create_urdf_loader()
        loader.fix_root_link = self._config.fix_root_link

        urdf_file = DESCRIPTION_DIR / self._config.urdf_file
        urdf_config = parse_urdf_config(self._config.urdf_config, self._scene)
        self._robot = loader.load(str(urdf_file), urdf_config)
        self._robot.set_name(self._config.name)
        self._control_mode = self._config.default_control_mode

    def _initialize_cameras(self):
        self._cameras = OrderedDict()
        for config in self._config.cameras:
            config = MountedCameraConfig(**config)
            if config.name in self._cameras:
                raise KeyError("Non-unique camera name: {}".format(config.name))
            cam = create_mounted_camera(config, self._robot, self._scene)
            self._cameras[config.name] = cam

    def _initialize_sensors(self):
        self._sensors = OrderedDict()
        for config in self._config.sensors:
            config = MountedActiveLightSensorConfig(**config)
            if config.name in self._sensors:
                raise KeyError(
                    "Non-unique active light sensor name: {}".format(config.name)
                )
            sensor = create_mounted_sensor(config, self._robot, self._scene)
            self._sensors[config.name] = sensor

    def _initialize_controllers(self):
        self._combined_controllers = OrderedDict()
        for control_mode, controller_configs in self._config.controllers.items():
            self._combined_controllers[control_mode] = CombinedController(
                controller_configs, self._robot, self._control_freq
            )
        self._control_mode = self._config.default_control_mode
        self._combined_controllers[self._control_mode].set_joint_drive_property()

    def sample_ee_coords(self, num_sample=10) -> np.ndarray:
        """Uniformly sample points on the two finger meshes. Used for dense reward computation
        return: ee_coords (2, num_sample, 3)"""
        finger_points = (
            np.arange(num_sample) / (num_sample - 1) - 0.5
        ) * self.finger_size[1]
        finger_points = np.stack(
            [np.zeros(num_sample), finger_points, np.zeros(num_sample)], axis=1
        )  # (num_sample, 3)

        finger1_points = transform_points(
            self.finger1_link.get_pose().to_transformation_matrix(), finger_points
        )
        finger2_points = transform_points(
            self.finger2_link.get_pose().to_transformation_matrix(), finger_points
        )

        ee_coords = np.stack((finger1_points, finger2_points))

        return ee_coords

    def get_images(
        self,
        rgb=True,
        depth=False,
        visual_seg=False,
        actor_seg=False,
        irsensor_mode=False,
    ) -> Dict[str, Dict[str, np.ndarray]]:
        assert self._cameras
        all_images = OrderedDict()
        # self._scene.update_render()
        if not irsensor_mode:
            for cam_name, cam in self._cameras.items():
                cam.take_picture()
                cam_images = get_camera_images(
                    cam,
                    rgb=rgb,
                    depth=depth,
                    visual_seg=visual_seg,
                    actor_seg=actor_seg,
                )
                cam_images["camera_intrinsic"] = cam.get_intrinsic_matrix()
                cam_extrinsic_world_frame = cam.get_extrinsic_matrix()
                robot_base_frame = (
                    self._robot.get_root_pose().to_transformation_matrix()
                )  # robot base -> world
                cam_images["camera_extrinsic_base_frame"] = (
                    cam_extrinsic_world_frame @ robot_base_frame
                )  # robot base -> camera
                all_images[cam_name] = cam_images
        else:
            for sensor_name, sensor in self._sensors.items():
                sensor_dict = sensor.get_image_dict()
                # sensor_seg_dict = get_camera_images(
                #     sensor._cam_rgb,
                #     rgb=False,
                #     depth=False,
                #     visual_seg=visual_seg,
                #     actor_seg=actor_seg,
                # )
                # sensor_dict.update(sensor_seg_dict)
                sensor_dict["camera_intrinsic"] = sensor._cam_rgb.get_intrinsic_matrix()
                cam_extrinsic_world_frame = sensor._cam_rgb.get_extrinsic_matrix()
                robot_base_frame = (
                    self._robot.get_root_pose().to_transformation_matrix()
                )  # robot base -> world
                sensor_dict["camera_extrinsic_base_frame"] = (
                    cam_extrinsic_world_frame @ robot_base_frame
                )  # robot base -> camera
                all_images[sensor_name] = sensor_dict

        return all_images

    def get_proprioception(self):
        state_dict = OrderedDict()
        qpos = self._robot.get_qpos()
        qvel = self._robot.get_qvel()

        state_dict["qpos"] = qpos
        state_dict["qvel"] = qvel
        return state_dict

    @classmethod
    def from_config_file(cls, config_path: str, scene: sapien.Scene, control_freq: int):
        with open(config_path, "r") as f:
            config_dict = yaml.safe_load(f)
        config = AgentConfig(**config_dict)
        return cls(config, scene, control_freq)

    @property
    def control_mode(self) -> str:
        return self._control_mode

    def set_control_mode(self, control_mode: str):
        self._control_mode = control_mode

    @property
    def action_space(self):
        return spaces.Dict(
            {
                mode: spaces.Box(
                    controller.action_range[:, 0], controller.action_range[:, 1]
                )
                for mode, controller in self._combined_controllers.items()
            }
        )

    @property
    def action_range(self) -> spaces.Box:
        return spaces.Box(
            self._combined_controllers[self._control_mode].action_range[:, 0],
            self._combined_controllers[self._control_mode].action_range[:, 1],
        )
