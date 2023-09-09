import copy
from collections import OrderedDict, deque
from typing import Dict, List, Tuple

import attr
import numpy as np
import sapien.core as sapien
import yaml
from arti_mani import ASSET_DIR, DESCRIPTION_DIR
from arti_mani.agents.camera import get_camera_images, get_camera_pcd
from arti_mani.agents.controllers.combined_controller import CombinedController
from arti_mani.utils.cv_utils import depth2pts_np
from arti_mani.utils.sapien_utils import get_entity_by_name, make_actor_visible
from gym import spaces
from sapien.sensor import StereoDepthSensor, StereoDepthSensorConfig


@attr.s(auto_attribs=True, kw_only=True)
class MountedCameraConfig:
    name: str
    # extrinsic parameters
    mount: str  # name of link to mount
    mount_p: List[float]  # position relative to link
    mount_q: List[float]  # quaternion relative to link
    hide_mount_link: bool  # whether to hide the mount link
    # intrinsic parameters
    width: int
    height: int
    near: float
    far: float
    fx: float
    fy: float
    cx: float
    cy: float
    skew: float


@attr.s(auto_attribs=True, kw_only=True)
class MountedActiveLightSensorConfig:
    name: str
    # extrinsic parameters
    mount: str  # name of link to mount
    # intrinsic parameters
    rgb_resolution: Tuple[int, int]
    ir_resolution: Tuple[int, int]
    rgb_intrinsic: List[float]
    ir_intrinsic: List[float]
    trans_pose_l: List[float]
    trans_pose_r: List[float]
    light_pattern_path: str
    max_depth: float
    min_depth: float
    max_disp: int
    ir_ambient_strength: float
    ir_light_dim_factor: float
    ir_light_fov: float
    ir_intensity: float


def create_mounted_camera(
    config: MountedCameraConfig, robot: sapien.Articulation, scene: sapien.Scene
) -> sapien.CameraEntity:
    camera_mount = get_entity_by_name(robot.get_links(), config.mount)
    mount_pose = sapien.Pose(config.mount_p, config.mount_q)
    camera = scene.add_mounted_camera(
        config.name,
        camera_mount,
        mount_pose,
        width=config.width,
        height=config.height,
        fovy=0,  # focal will be set later.
        near=config.near,
        far=config.far,
    )
    camera.set_perspective_parameters(
        near=config.near,
        far=config.far,
        fx=config.fx,
        fy=config.fy,
        cx=config.cx,
        cy=config.cy,
        skew=config.skew,
    )
    if config.hide_mount_link:
        make_actor_visible(camera_mount, False)
    return camera


def create_mounted_sensor(
    config: MountedActiveLightSensorConfig,
    robot: sapien.Articulation,
    scene: sapien.Scene,
):
    sensor_mount = get_entity_by_name(robot.get_links(), config.mount)
    rgb_intrinsic = np.array(config.rgb_intrinsic).astype(np.float)
    ir_intrinsic = np.array(config.ir_intrinsic).astype(np.float)
    trans_pose_l = sapien.Pose(config.trans_pose_l[:3], config.trans_pose_l[3:])
    trans_pose_r = sapien.Pose(config.trans_pose_r[:3], config.trans_pose_r[3:])

    sensor_config = StereoDepthSensorConfig()
    sensor_config.rgb_resolution = config.rgb_resolution
    sensor_config.rgb_intrinsic = rgb_intrinsic
    sensor_config.ir_resolution = config.ir_resolution
    sensor_config.ir_intrinsic = ir_intrinsic
    sensor_config.trans_pose_l = trans_pose_l
    sensor_config.trans_pose_r = trans_pose_r
    sensor_config.light_pattern = str(ASSET_DIR / config.light_pattern_path)
    sensor_config.min_depth = config.min_depth
    sensor_config.max_depth = config.max_depth
    sensor_config.max_disp = config.max_disp
    sensor = StereoDepthSensor(config.name, scene, sensor_config, mount=sensor_mount)
    return sensor


@attr.s(auto_attribs=True, kw_only=True)
class AgentConfig:
    agent_class: str
    name: str
    urdf_file: str
    fix_root_link: bool = True
    urdf_config: Dict = attr.ib(factory=dict)

    default_init_qpos: List[float]
    default_control_mode: str

    controllers: Dict[str, List[Dict]]
    cameras: List[Dict] = attr.ib(factory=list)
    sensors: List[Dict] = attr.ib(factory=list)

    torque_freq: int


def parse_urdf_config(config_dict: Dict, scene: sapien.Scene) -> Dict:
    """Parse config from dict for SAPIEN URDF loader.

    Args:
        config_dict (Dict): dict containing link physical properties.
        scene (sapien.Scene): simualtion scene

    Returns:
        Dict: urdf config passed to `sapien.URDFLoader.load`.
    """
    config_dict = copy.deepcopy(config_dict)
    urdf_config = {}

    phy_mtls = {}
    for k, v in config_dict.get("materials", {}).items():
        phy_mtls[k] = scene.create_physical_material(**v)

    link_configs = config_dict.get("links", {})
    for link_name, link_config in link_configs.items():
        # replace with actual material
        link_config["material"] = phy_mtls[link_config["material"]]

    urdf_config["link"] = link_configs
    return urdf_config


class BaseAgent:
    """Base class for agents.

    Agent is an interface of the robot (sapien.Articulation).

    Args:
        config (AgentConfig):  agent configuration.
        scene (sapien.Scene): simulation scene instance.
        control_freq (int): control frequency (Hz).

    """

    _config: AgentConfig
    _scene: sapien.Scene
    _robot: sapien.Articulation
    _combined_controllers: Dict[str, CombinedController]
    _control_mode: str
    _cameras: Dict[str, sapien.CameraEntity]
    _sensors: Dict[str, StereoDepthSensor]
    _q_vel_ex: np.ndarray  # cached q velocity of previous step for external torque
    _sim_time_step: float

    def __init__(self, config: AgentConfig, scene: sapien.Scene, control_freq: int):
        self._config = copy.deepcopy(config)

        self._scene = scene
        self._sim_time_step = scene.timestep
        self._control_freq = control_freq
        assert control_freq % config.torque_freq == 0
        torque_queue_len = control_freq // config.torque_freq
        self._torque_queue = deque(maxlen=torque_queue_len)

        self._initialize_robot()
        self._initialize_controllers()
        self._initialize_cameras()
        self._initialize_sensors()

    def _initialize_robot(self):
        loader = self._scene.create_urdf_loader()
        loader.fix_root_link = self._config.fix_root_link

        urdf_file = DESCRIPTION_DIR / self._config.urdf_file
        urdf_config = parse_urdf_config(self._config.urdf_config, self._scene)
        self._robot = loader.load(str(urdf_file), urdf_config)
        self._robot.set_name(self._config.name)

    def _initialize_controllers(self):
        self._combined_controllers = OrderedDict()
        for control_mode, controller_configs in self._config.controllers.items():
            self._combined_controllers[control_mode] = CombinedController(
                controller_configs, self._robot, self._control_freq
            )
        self._control_mode = self._config.default_control_mode
        self._combined_controllers[self._control_mode].set_joint_drive_property()

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

    @classmethod
    def from_config_file(cls, config_path: str, scene: sapien.Scene, control_freq: int):
        with open(config_path, "r") as f:
            config_dict = yaml.safe_load(f)
        config = AgentConfig(**config_dict)
        return cls(config, scene, control_freq)

    @property
    def supported_control_modes(self) -> Tuple[str]:
        return tuple(self._combined_controllers.keys())

    @property
    def control_mode(self) -> str:
        return self._control_mode

    def set_control_mode(self, control_mode: str):
        assert (
            control_mode in self.supported_control_modes
        ), f"Unknown control mode: [{control_mode}]. All supported control modes: [{self.supported_control_modes}]"
        self._control_mode = control_mode
        self._combined_controllers[control_mode].set_joint_drive_property()
        self._combined_controllers[control_mode].reset()

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

    def set_action(self, action: np.ndarray):
        self._combined_controllers[self._control_mode].set_action(action)

    def simulation_step(self):
        self._combined_controllers[self._control_mode].simulation_step()
        self._q_vel_ex = self._robot.get_qvel()

    def reset(self, init_qpos=None):
        if init_qpos is None:
            init_qpos = self._config.default_init_qpos
        self._robot.set_qpos(init_qpos)
        self._robot.set_qvel(np.zeros(self._robot.dof))
        self._robot.set_qacc(np.zeros(self._robot.dof))
        self._robot.set_qf(np.zeros(self._robot.dof))
        self._combined_controllers[self._control_mode].reset()
        self._q_vel_ex = self._robot.get_qvel()
        self._torque_queue.clear()

    def get_proprioception(self) -> Dict:
        raise NotImplementedError

    def get_state(self) -> Dict:
        """Get current state for MPC, including robot state and controller state"""
        state = OrderedDict()

        # robot state
        root_link = self._robot.get_links()[0]
        state["robot_root_pose"] = root_link.get_pose()
        state["robot_root_vel"] = root_link.get_velocity()
        state["robot_root_qvel"] = root_link.get_angular_velocity()
        state["robot_qpos"] = self._robot.get_qpos()
        state["robot_qvel"] = self._robot.get_qvel()
        state["robot_qacc"] = self._robot.get_qacc()
        return state

    def set_state(self, state: Dict):
        # robot state
        self._robot.set_root_pose(state["robot_root_pose"])
        self._robot.set_root_velocity(state["robot_root_vel"])
        self._robot.set_root_angular_velocity(state["robot_root_qvel"])
        self._robot.set_qpos(state["robot_qpos"])
        self._robot.set_qvel(state["robot_qvel"])
        self._robot.set_qacc(state["robot_qacc"])

    def get_fused_pointcloud(
        self, rgb=True, visual_seg=False, actor_seg=False
    ) -> Dict[str, np.ndarray]:
        """get pointcloud from each camera, transform them to the *world* frame, and fuse together"""
        assert self._cameras
        pcds = []
        # self._scene.update_render()
        for cam in self._cameras.values():
            cam.take_picture()
            pcd = get_camera_pcd(cam, rgb, visual_seg, actor_seg)  # dict
            # Model matrix is the transformation from OpenGL camera space to SAPIEN world space
            # camera.get_model_matrix() must be called after scene.update_render()!
            T = cam.get_model_matrix()
            R = T[:3, :3]
            t = T[:3, 3]
            pcd["xyz"] = pcd["xyz"] @ R.transpose() + t
            pcds.append(pcd)
        fused_pcd = {}
        for key in pcds[0].keys():
            fused_pcd[key] = np.concatenate([pcd[key] for pcd in pcds], axis=0)
        return fused_pcd

    def get_sensor_fused_point_cloud(self, rgb=True, visual_seg=True, actor_seg=True):
        assert self._sensors
        pcds = []
        for sensor_name, sensor in self._sensors.items():
            sensor_dict = sensor.get_image_dict(rgb, visual_seg, actor_seg)
            camera_intrinsic = sensor._rgb_intrinsic
            cam_extrinsic_world_frame = sensor._cam_rgb.get_extrinsic_matrix()
            pcd = {}
            xyz = depth2pts_np(
                sensor_dict["stereo_depth"], camera_intrinsic, cam_extrinsic_world_frame
            )
            num_pts = xyz.shape[0]
            mask = sensor_dict["stereo_depth"].reshape(-1) > 0
            xyz = xyz[mask]
            pcd["xyz"] = xyz

            for k in ("rgb", "visual_seg", "actor_seg"):
                if k in sensor_dict:
                    pcd[k] = sensor_dict[k].reshape(num_pts, -1)[mask]
            pcds.append(pcd)

        fused_pcd = {}
        for k in pcds[0].keys():
            fused_pcd[k] = np.concatenate([pcd[k] for pcd in pcds], axis=0)

        return fused_pcd

    def get_images(
        self, rgb=True, depth=False, visual_seg=False, actor_seg=False
    ) -> Dict[str, Dict[str, np.ndarray]]:
        assert self._cameras
        all_images = OrderedDict()
        # self._scene.update_render()
        for cam_name, cam in self._cameras.items():
            cam.take_picture()
            cam_images = get_camera_images(
                cam, rgb=rgb, depth=depth, visual_seg=visual_seg, actor_seg=actor_seg
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

        if depth:
            for sensor_name, sensor in self._sensors.items():
                sensor_dict = OrderedDict()
                sensor.take_picture()
                sensor.compute_depth()
                sensor_dict["rgb"] = sensor.get_rgb()
                sensor_dict["depth"] = sensor.get_depth()[..., None]
                # sensor_dict = sensor.get_image_dict()
                # sensor_seg_dict = get_camera_images(
                #     sensor._cam_rgb,
                #     rgb=False,
                #     depth=False,
                #     visual_seg=visual_seg,
                #     actor_seg=actor_seg,
                # )
                # sensor_dict.update(sensor_seg_dict)
                # sensor_dict["camera_intrinsic"] = sensor._cam_rgb.get_intrinsic_matrix()
                # cam_extrinsic_world_frame = sensor._cam_rgb.get_extrinsic_matrix()
                # robot_base_frame = (
                #     self._robot.get_root_pose().to_transformation_matrix()
                # )  # robot base -> world
                # sensor_dict["camera_extrinsic_base_frame"] = (
                #     cam_extrinsic_world_frame @ robot_base_frame
                # )  # robot base -> camera
                all_images[sensor_name] = sensor_dict

        return all_images

    def get_camera_poses(self) -> Dict[str, np.ndarray]:
        assert self._cameras
        pose_list = OrderedDict()
        for cam_name, cam in self._cameras.items():
            pose_list[cam_name] = cam.get_pose().to_transformation_matrix()
        return pose_list

    def update_generalized_external_forces(self):
        """Note: only called in step_action()"""
        q_pos = self._robot.get_qpos()
        q_vel = self._robot.get_qvel()
        actual_acc = (q_vel - self._q_vel_ex) / self._sim_time_step
        actual_torques = self._robot.compute_inverse_dynamics(actual_acc)
        command_torques = []
        for j_idx, j in enumerate(self._robot.get_active_joints()):
            command_torque = j.stiffness * (
                j.get_drive_target() - q_pos[j_idx]
            ) + j.damping * (j.get_drive_velocity_target() - q_vel[j_idx])
            command_torques.append(command_torque)

        command_torques = np.array(command_torques)
        F = (
            actual_torques
            - command_torques
            - self._robot.get_qf()
            + self._robot.compute_passive_force()
        )
        self._torque_queue.append(F)

    def get_generalized_external_forces(self):
        if self._torque_queue:
            return np.mean(np.stack(self._torque_queue), axis=0)
        else:
            return np.zeros_like(self._robot.get_qf())
