from collections import OrderedDict
from typing import List

import numpy as np
import open3d as o3d
import sapien.core as sapien
import torch
import torch.nn.functional as F
import trimesh
import yaml
from arti_mani import ASSET_DIR
from arti_mani.agents.camera import get_camera_rgb, get_camera_seg, get_texture
from arti_mani.algorithms.rl_iam.Sampling import (
    FPSSampler,
    RandomSampler,
    ScoreSampler,
    UniDownsampler,
)
from arti_mani.algorithms.visual_net.Networks.Custom_Unet import (
    CustomUnet,
    CustomUnetNew,
    SplitUnet,
)
from arti_mani.algorithms.visual_net.Networks.keypoint_detection import (
    IntegralHumanPoseModel,
    soft_argmax,
)
from arti_mani.envs.fixed_robotenv import FixedXmate3RobotiqSensorLowResEnv
from arti_mani.utils.articulation_utils import ignore_collision
from arti_mani.utils.common import (
    convert_np_bool_to_float,
    flatten_state_dict,
    np_random,
    random_choice,
    register_gym_env,
)
from arti_mani.utils.contrib import (
    angle_distance_ms,
    apply_pose_to_points,
    norm,
    normalize_and_clip_in_interval,
    o3d_to_trimesh,
    trimesh_to_o3d,
    uv2xyz,
)
from arti_mani.utils.geometry import sample_grasp_multipoints_ee, transform_points
from arti_mani.utils.o3d_utils import (
    check_coplanar,
    get_visual_body_meshes,
    merge_mesh,
    merge_meshes,
    np2mesh,
    pcd_uni_down_sample_with_crop,
)
from arti_mani.utils.sapien_utils import (
    get_entity_by_name,
    hex2rgba,
    set_articulation_render_material,
)
from arti_mani.utils.trimesh_utils import get_actor_mesh
from sapien.core import Pose
from transforms3d.euler import euler2quat

CMAPS = [
    (1, 0, 0),  # "red"
    (0, 0, 1),  # "blue"
    (1, 1, 0),  # "yellow"
    (0, 1, 0),  # "green"
    (0.627, 0.125, 0.941),  # "purple"
    (0.753, 0.753, 0.753),  # "grey"
]


@register_gym_env("ArtiMani-v0", max_episode_steps=100)
class ArtiMani(FixedXmate3RobotiqSensorLowResEnv):
    SUPPORTED_OBS_MODES = (
        "state",
        "state_dict",
        "state_egorgbd",
        "state_ego_segpoints_gt",
        "state_egosegpoints",
        "state_egostereo_rgbd",
        "state_egostereo_segpoints",
        "state_egostereo_segpoints_gt",
        "state_egostereo_keypoints",
        "state_egostereo_keypoints_gt",
        "state_egostereo_dseg",
        "state_egostereo_dseg_gt",
        "state_egostereo_dexpoints",
    )
    SUPPORTED_REWARD_MODES = ("dense", "sparse")
    SUPPORTED_CONTROL_MODES = (
        "pd_joint_delta_pos",
        "pd_ee_delta_pose",
        "pd_ee_delta_pos",
    )

    def __init__(
        self,
        articulation_ids: List[int] = (),
        segmodel_path: str = None,
        sample_mode: str = None,
        frame_num: int = 1,
        sample_num: int = 30,
        other_handle_visible=False,
        num_classes: int = 6,
        obs_mode=None,
        reward_mode=None,
        sim_freq=500,
        control_freq=20,
        control_mode=None,
        add_eepadpts=False,
        device: str = "cuda:0",
    ):
        self._arti_info = OrderedDict()
        with open(
            ASSET_DIR / f"partnet_mobility_configs/fixed_artis_new.yml", "r"
        ) as f:
            self._arti_info = yaml.safe_load(f)
        # self._arti_fail_rate = dict.fromkeys(self._arti_info.keys(), 1)
        self._arti_mode = None
        self._articulation_info = None
        if isinstance(articulation_ids, int):
            articulation_ids = [articulation_ids]

        assert len(articulation_ids) > 0

        self._arti_fail_total = OrderedDict()
        for arti_id in articulation_ids:
            self._arti_fail_total[arti_id] = [0, 0]

        self.articulation_ids = articulation_ids
        self.articulation_id = None
        self.articulation_scale = None
        self.other_handle_visible = other_handle_visible
        self.num_classes = num_classes
        self.device = device
        self.sample_mode = sample_mode
        self.sample_num = sample_num
        if sample_mode == "full_downsample":
            self.sampler = UniDownsampler(sample_num=sample_num)
        elif sample_mode == "score_sample":
            self.sampler = ScoreSampler(sample_num=sample_num)
        elif sample_mode == "fps_sample":
            self.sampler = FPSSampler(sample_num=sample_num)
        elif sample_mode == "random_sample":
            self.sampler = RandomSampler(sample_num=sample_num)
        elif sample_mode == None:
            self.sampler = None
        else:
            raise NotImplementedError(sample_mode)

        if obs_mode in [
            "state_egosegpoints",
            "state_egostereo_segpoints",
            "state_egostereo_keypoints",
        ]:
            self._load_segmodel(segmodel_path, obs_mode)
        self.out_shape = (64, 40, 64)
        self.kpts_max = np.array([255, 143, 1.0])
        self.kpts_min = np.array([0, 0, 0.18])
        self.cam_para = OrderedDict()
        self.add_eepadpts = add_eepadpts

        super().__init__(
            obs_mode, reward_mode, control_mode, sim_freq, control_freq, device
        )

    def _load_segmodel(self, segmodel_path, obs_mode):
        if obs_mode == "state_egostereo_keypoints":
            model_path = f"{segmodel_path}/best.pth"
            self.kpt_model = IntegralHumanPoseModel(
                num_keypoints=3, num_deconv_layers=3, depth_dim=64, has_dropout=False
            )
            self.kpt_model.load_state_dict(torch.load(model_path))
            self.kpt_model.to(torch.device(self.device))
            self.kpt_model.eval()
        else:
            model_path = f"{segmodel_path}/best.pth"
            config_path = f"{segmodel_path}/config.yaml"
            with open(config_path, "r") as f:
                cfg = yaml.safe_load(f)
            assert self.num_classes == cfg["num_classes"]
            smp_cfg = cfg["smp_config"]
            if smp_cfg["mode"] == "RGBD":
                in_ch = 4
            elif smp_cfg["mode"] == "RGB":
                in_ch = 3
            elif smp_cfg["mode"] == "D":
                in_ch = 1
            else:
                raise NotImplementedError
            if "has_dropout" in smp_cfg.keys():
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
            elif smp_cfg["encoder"] == "splitnet":
                self.segmodel = SplitUnet(
                    dropout_p=smp_cfg["dropout_p"],
                    encoder_name=smp_cfg["encoder"],
                    encoder_depth=smp_cfg["encoder_depth"],
                    decoder_channels=smp_cfg["decoder_channels"],
                    encoder_weights=smp_cfg["encoder_weights"],
                    in_channels=in_ch,
                    classes=cfg["num_classes"],
                )
            else:
                self.segmodel = CustomUnetNew(
                    dropout_p=smp_cfg["dropout_p"],
                    encoder_name=smp_cfg["encoder"],
                    encoder_depth=smp_cfg["encoder_depth"],
                    decoder_channels=smp_cfg["decoder_channels"],
                    encoder_weights=smp_cfg["encoder_weights"],
                    in_channels=in_ch,
                    classes=cfg["num_classes"],
                )
            self.segmodel.load_state_dict(torch.load(model_path))
            self.segmodel.to(torch.device(self.device))
            self.segmodel.eval()

    def _setup_camera(self):
        if self._arti_mode in ["cabinet_door", "cabinet_drawer"]:
            self.render_camera = self._scene.add_camera(
                "topview", 512, 512, 1, 0.01, 10
            )
            self.render_camera.set_local_pose(
                Pose([-0.1, 0.0, 1.5], euler2quat(0, 1.57, 0))
            )
        elif self._arti_mode in ["faucet", "laptop", "kitchen_pot"]:
            self.render_camera = self._scene.add_camera(
                "front_view", 512, 512, 1, 0.01, 10
            )
            self.render_camera.set_local_pose(
                Pose([1, 0, 1.2], euler2quat(0, 0.6, 3.14))
            )
        else:
            raise NotImplementedError

    def _setup_viewer(self):
        super()._setup_viewer()
        if self._arti_mode in ["cabinet_door", "cabinet_drawer"]:
            ### top view
            self._viewer.set_camera_xyz(0.0, 0.0, 1.2)
            self._viewer.set_camera_rpy(0, -3.14, 0)
        elif self._arti_mode in ["faucet", "laptop", "kitchen_pot"]:
            ### front view
            self._viewer.set_camera_xyz(1.0, 0.0, 1.2)
            self._viewer.set_camera_rpy(0, -0.5, 3.14)
            ### top view
            # self._viewer.set_camera_xyz(0.0, 0.0, 1.2)
            # self._viewer.set_camera_rpy(0, -3.14, 0)
        else:
            raise NotImplementedError

    def reset(
        self,
        seed=None,
        reconfigure=False,
        articulation_id=None,
        articulation_scale=None,
    ):
        self.set_episode_rng(seed)
        _reconfigure = self._set_model(articulation_id, articulation_scale)
        reconfigure = _reconfigure or reconfigure
        ret = super().reset(seed=self._episode_seed, reconfigure=reconfigure)
        return ret

    def _set_model(self, articulation_id, articulation_scale):
        """Set the model id and scale. If not provided, choose one randomly."""
        reconfigure = False

        # Model ID
        if articulation_id is None:
            articulation_id = self.articulation_ids[
                self._episode_rng.choice(len(self.articulation_ids), p=None)
            ]
        if articulation_id != self.articulation_id:
            reconfigure = True
        self.articulation_id = articulation_id
        for mode in self._arti_info.keys():
            if self.articulation_id in self._arti_info[mode].keys():
                self._articulation_info = self._arti_info[mode]
                self._arti_mode = mode
                break
        assert self._articulation_info is not None
        self._articulation_config = self._articulation_info[self.articulation_id]

        if self._arti_mode in ["faucet", "laptop", "kitchen_pot"]:
            # Scale
            if articulation_scale is None:
                articulation_scale = self._articulation_config.get("scale")
            if articulation_scale is None:
                bbox_max, bbox_min = (
                    self._articulation_config["bbox_max"],
                    self._articulation_config["bbox_min"],
                )
                bbox_size = np.float32(bbox_max) - np.float32(bbox_min)
                articulation_scale = 0.3 / max(bbox_size)  # hardcode
            if articulation_scale != self.articulation_scale:
                reconfigure = True
            self.articulation_scale = articulation_scale

            if "offset" in self._articulation_config:
                self.articulation_offset = np.float32(
                    self._articulation_config["offset"]
                )
            else:
                bbox_min = self._articulation_config["bbox_min"]
                self.articulation_offset = -np.float32(bbox_min) * articulation_scale
            # Add a small clearance
            self.articulation_offset[2] += 0.01

        return reconfigure

    def _initialize_articulations(self):
        if self._arti_mode in ["cabinet_door", "cabinet_drawer"]:
            posxy_rotz_min = self._articulation_config["posxy_rotz_min"]
            posxy_rotz_max = self._articulation_config["posxy_rotz_max"]
            posxy_rotz = self._episode_rng.uniform(posxy_rotz_min, posxy_rotz_max)
            posz = (
                -self._articulation_config["scale"]
                * self._articulation_config["bbox_min"][2]
            )
            arti_root_pose = Pose(
                [posxy_rotz[0], posxy_rotz[1], posz],
                [np.sqrt(1 - posxy_rotz[2] ** 2), 0, 0, posxy_rotz[2]],
            )
            if self._articulation_config["flip"]:
                arti_root_pose = arti_root_pose * Pose(q=np.array([0, 1, 0, 0]))
            self._articulation.set_root_pose(arti_root_pose)

            # [[lmin, lmax]] = self._target_joint.get_limits()
            lmin, lmax = self._articulation_config["target_joint_range"]
            init_open_extent = self._episode_rng.uniform(
                0, self._articulation_config["init_open_extent_range"]
            )
            qpos = np.zeros(self._articulation.dof)
            for i in range(self._articulation.dof):
                qpos[i] = self._articulation.get_active_joints()[i].get_limits()[0][0]
            qpos[self._target_joint_idx] = lmin + (lmax - lmin) * init_open_extent
            self._articulation.set_qpos(qpos)

            self.target_qpos = (
                lmin + (lmax - lmin) * self._articulation_config["open_extent"]
            )
            self._get_handle_info_in_target_link()
        elif self._arti_mode == "faucet":
            p = np.array([0.1, 0, 0])
            p[:2] += self._episode_rng.uniform(-0.05, 0.05, [2])
            p[2] = self.articulation_offset[2]
            ori = self._episode_rng.uniform(0, np.pi / 3)
            q = euler2quat(0, 0, ori)
            self._articulation.set_pose(Pose(p, q))
            self._initialize_task()
        elif self._arti_mode == "laptop":
            p = np.array([0.15, 0, 0])
            p[:2] += self._episode_rng.uniform(-0.05, 0.05, [2])
            p[2] = self.articulation_offset[2]
            ori = self._episode_rng.uniform(-np.pi / 6, np.pi / 6)
            q = euler2quat(0, 0, ori)
            self._articulation.set_pose(Pose(p, q))

            # initialize task
            qmin, qmax, q_range = self.qmin_max_range
            self.init_angle = self._episode_rng.uniform(
                qmin + q_range / 6, qmin + q_range / 3
            )
            self.target_angle = qmin + q_range * 0.9
            # initialize qpos
            qpos = self._articulation_init_qpos.copy()
            qpos[self.target_joint_idx] = self.init_angle
            self._articulation.set_qpos(qpos)
            self.last_angle_diff = self.target_angle - self.current_angle
        elif self._arti_mode == "kitchen_pot":
            p = np.array([0.025, 0.05, 0.0])
            p[:2] += self._episode_rng.uniform(0.0, 0.05, [2])
            p[2] = self.articulation_offset[2]
            ori = np.pi / 2 + self._episode_rng.uniform(-np.pi / 12, np.pi / 12)
            q = euler2quat(0, 0, ori)
            self._articulation.set_pose(Pose(p, q))

            # initialize task
            qmin, qmax, q_range = self.qmin_max_range
            # self.init_angle = self._episode_rng.uniform(qmin + q_range / 6, qmin + q_range / 3)
            self.init_angle = 0
            self.target_angle = qmin + q_range * 0.9
            # initialize qpos
            qpos = self._articulation_init_qpos.copy()
            qpos[self.target_joint_idx] = self.init_angle
            self._articulation.set_qpos(qpos)
            self.last_angle_diff = self.target_angle - self.current_angle
        else:
            raise NotImplementedError

    def _initialize_task(self):
        self._set_target_link()
        self._set_init_and_target_angle()

        qpos = self._articulation_init_qpos.copy()
        qpos[self.target_joint_idx] = self.init_angle
        self._articulation.set_qpos(qpos)

        self.last_angle_diff = self.target_angle - self.current_angle

    def _set_target_link(self):
        n_switch_links = len(self.switch_link_names)
        idx = random_choice(np.arange(n_switch_links), self._episode_rng)

        self.target_link_name = self.switch_link_names[idx]
        self._target_link: sapien.Link = self.switch_links[idx]
        self.target_joint: sapien.Joint = self.switch_joints[idx]
        self.target_joint_idx = self._articulation.get_active_joints().index(
            self.target_joint
        )

        # x-axis is the revolute joint direction
        assert self.target_joint.type == "revolute", self.target_joint.type
        joint_pose = self.target_joint.get_global_pose().to_transformation_matrix()
        self.target_joint_axis = joint_pose[:3, 0]

        self.target_link_mesh = self.switch_links_mesh[idx]
        with np_random(self._episode_seed):
            self.target_pcd = self.target_link_mesh.sample(1000)
        # trimesh.PointCloud(self.target_pcd).show()

        # get points center
        target_link_frame_pos = self._target_link.get_pose().p
        # get points far, near from joint axis
        pointing_dir = (
            transform_points(
                self.target_link.pose.to_transformation_matrix(), self.target_pcd
            )
            - target_link_frame_pos
        )  # N X 3
        distance = np.sqrt(
            np.linalg.norm(pointing_dir, axis=1) ** 2
            - np.dot(pointing_dir, self.target_joint_axis) ** 2
        )
        ind_sort = np.argsort(distance)
        points_num = int(0.5 * self.target_pcd.shape[0])
        tip_index = ind_sort[-points_num:]
        self.tip_points = self.target_pcd[tip_index]

        # get tip, med, bottom keypoints
        if self.articulation_id in [5024, 5034]:
            norm_sw_pcd = apply_pose_to_points(
                self.target_pcd, self._target_joint.get_pose_in_parent().inv()
            )
            x_max, x_min = norm_sw_pcd[:, 0].max(), norm_sw_pcd[:, 0].min()
            center_pts, bottom_pts = (
                norm_sw_pcd[norm_sw_pcd[:, 0] - x_max > -0.01],
                norm_sw_pcd[norm_sw_pcd[:, 0] - x_min < 0.01],
            )
            center = np.mean(center_pts, axis=0)
            center[0] = x_max
            bottom = np.mean(bottom_pts, axis=0)
            bottom[0] = x_min
            tip = norm_sw_pcd[norm_sw_pcd[:, 1].argmin()]
            self._handle_keypoints = apply_pose_to_points(
                np.stack([tip, center, bottom]), self._target_joint.get_pose_in_parent()
            )
        else:
            xyzmax_val, xyzmin_val = self.target_pcd.max(0), self.target_pcd.min(0)
            xyzmax_id, xyzmin_id = np.argmax(self.target_pcd, 0), np.argmin(
                self.target_pcd, 0
            )
            bbox_size = xyzmax_val - xyzmin_val
            long_side = np.argmax(bbox_size)
            tip = self.target_pcd[xyzmax_id[long_side]]

            joint_pts_ids = np.where(
                np.linalg.norm(self.target_pcd[:, [0, 2]], axis=1) < 0.02
            )  # dist to joint axis < 0.01
            joint_pts = self.target_pcd[joint_pts_ids][:, 1]
            joint_pts_max = joint_pts.max()
            center = np.array([0, joint_pts_max, 0])
            bottom = np.array([0, self.target_pcd[:, 1].min(), 0])
            self._handle_keypoints = np.stack([tip, center, bottom])

        cmass_pose = self.target_link.pose * self.target_link.cmass_local_pose
        self.target_link_pos = cmass_pose.p

    def _set_init_and_target_angle(self):
        qmin, qmax = self.target_joint.get_limits()[0]
        qmin = 0 if np.isinf(qmin) else qmin
        qmax = np.pi / 2 if np.isinf(qmax) else qmax
        q_range = qmax - qmin
        self.init_angle = self._episode_rng.uniform(
            qmin + q_range / 6, qmin + q_range / 3
        )
        self.target_angle = qmin + q_range * 0.9

        # The angle to go
        self.target_angle_diff = self.target_angle - self.init_angle

    def _initialize_agent(self):
        if self._arti_mode in ["cabinet_door", "cabinet_drawer"]:
            ### close to handle 1, half open gripper
            qpos = np.array([1.4, -1.053, -2.394, 1.662, 1.217, 1.05, -0.8, 0.04, 0.04])
        elif self._arti_mode in ["faucet", "laptop"]:
            qpos = np.array([-0.5, -0.143, 0, np.pi / 3, 0, 1.57, 1.57, 0.068, 0.068])
            # qpos = np.array([-0.488665, 0.340129, -1.14341, 1.18789, 0.346452, 1.73497, 0.7321, 0.068, 0.068])
        elif self._arti_mode == "kitchen_pot":
            qpos = np.array([-0.5, -0.143, 0, np.pi / 3, 0, 1.57, 1.57, 0.04, 0.04])
        else:
            raise NotImplementedError
        qpos[:-2] += self._episode_rng.normal(0, 0.02, len(qpos) - 2)
        # qpos = np.array([0., 0., 0., 0., 0., 0., 0., 0.04, 0.04])
        self._agent.reset(qpos)
        self._agent._robot.set_pose(Pose([-0.6, 0.4, 0]))
        ### get agent camera extrinsic, intrinsic para
        self.update_render()
        for cam_name, cam in self.agent._cameras.items():
            self.cam_para[cam_name + "_intrinsic"] = cam.get_intrinsic_matrix()
            cam_extrinsic_world_frame = cam.get_model_matrix()  # cam -> world
            if cam_name == "base_camera":
                base_frame = (
                    self.agent._robot.get_root_pose().inv().to_transformation_matrix()
                )  # world -> robot_base
            elif cam_name == "hand_camera":
                base_frame = (
                    self.agent.grasp_site.get_pose().inv().to_transformation_matrix()
                )  # world -> ee
            self.cam_para[cam_name + "_extrinsic_base_frame"] = (
                base_frame @ cam_extrinsic_world_frame
            )  # cam -> robot_base/ee

    def _load_articulations(self):
        if self._arti_mode in ["cabinet_door", "cabinet_drawer"]:
            self._articulation = self._load_cabinet()
        elif self._arti_mode == "faucet":
            self._articulation = self._load_faucet()
            # Cache qpos to restore
            self._articulation_init_qpos = self._articulation.get_qpos()
        elif self._arti_mode == "laptop":
            self._articulation = self._load_laptop()
            # Cache qpos to restore
            self._articulation_init_qpos = self._articulation.get_qpos()
        elif self._arti_mode == "kitchen_pot":
            self._articulation = self._load_kitchenpot()
            # Cache qpos to restore
            self._articulation_init_qpos = self._articulation.get_qpos()
        else:
            raise NotImplementedError

        # set physical properties for all the joints
        self._joint_stiffness_range = (0.0, 0.0)
        if self._arti_mode in ["cabinet_door", "laptop"]:
            self._joint_friction_range = (0.8, 1.0)
            self._joint_damping_range = (20.0, 30.0)
        elif self._arti_mode in ["cabinet_drawer", "kitchen_pot"]:
            self._joint_friction_range = (0.2, 0.8)
            self._joint_damping_range = (70.0, 80.0)
        elif self._arti_mode == "faucet":
            self._joint_friction_range = (0.2, 0.8)
            self._joint_damping_range = (4.0, 5.0)  # faucet
        else:
            raise NotImplementedError

        joint_friction = self._episode_rng.uniform(
            self._joint_friction_range[0], self._joint_friction_range[1]
        )
        joint_stiffness = self._episode_rng.uniform(
            self._joint_stiffness_range[0], self._joint_stiffness_range[1]
        )
        joint_damping = self._episode_rng.uniform(
            self._joint_damping_range[0], self._joint_damping_range[1]
        )
        for joint in self._articulation.get_active_joints():
            joint.set_friction(joint_friction)
            joint.set_drive_property(joint_stiffness, joint_damping)
        ignore_collision(self._articulation)

        self._set_switch_links()

    def _load_cabinet(self):
        loader = self._scene.create_urdf_loader()
        loader.load_multiple_collisions_from_file = self._articulation_config[
            "multiple_collisions"
        ]
        loader.scale = self._articulation_config["scale"]
        loader.fix_root_link = True

        articulation_path = (
            ASSET_DIR
            / f"partnet_mobility_dataset/{self._arti_mode}/{self.articulation_id}"
        )
        urdf_path = articulation_path / "mobility_cvx.urdf"
        assert (
            urdf_path.exists()
        ), f"{urdf_path} is not found. Please download Partnet-Mobility (ManiSkill2022) first."
        config = {
            "material": self._physical_materials["default"],
            # "density": 8e3
        }
        articulation = loader.load(str(urdf_path), config=config)
        articulation.set_name("cabinet")
        return articulation

    def _load_faucet(self):
        loader = self._scene.create_urdf_loader()
        loader.scale = self.articulation_scale
        loader.fix_root_link = True

        model_dir = (
            ASSET_DIR
            / f"partnet_mobility_dataset/{self._arti_mode}/{self.articulation_id}"
        )
        urdf_path = model_dir / "mobility_cvx.urdf"
        loader.load_multiple_collisions_from_file = True
        assert (
            urdf_path.exists()
        ), f"{urdf_path} is not found. Please download Partnet-Mobility (ManiSkill2022) first."

        density = self._articulation_config.get("density", 8e3)
        articulation = loader.load(str(urdf_path), config={"density": density})
        articulation.set_name("faucet")

        cam = self._scene.add_camera("dummy_camera", 1, 1, 1, 0.01, 10)
        self._scene.remove_camera(cam)
        set_articulation_render_material(
            articulation, color=hex2rgba("#AAAAAA"), metallic=1, roughness=0.4
        )

        return articulation

    def _load_laptop(self):
        loader = self._scene.create_urdf_loader()
        loader.scale = self.articulation_scale
        loader.fix_root_link = True

        model_dir = (
            ASSET_DIR
            / f"partnet_mobility_dataset/{self._arti_mode}/{self.articulation_id}"
        )
        urdf_path = model_dir / "mobility.urdf"
        loader.load_multiple_collisions_from_file = True
        assert (
            urdf_path.exists()
        ), f"{urdf_path} is not found. Please download Partnet-Mobility Dataset first."

        density = self._articulation_config.get("density", 8e3)
        articulation = loader.load(str(urdf_path), config={"density": density})
        articulation.set_name("laptop_kitchenpot")

        cam = self._scene.add_camera("dummy_camera", 1, 1, 1, 0.01, 10)
        self._scene.remove_camera(cam)
        set_articulation_render_material(
            articulation, color=hex2rgba("#AAAAAA"), metallic=1, roughness=0.4
        )

        return articulation

    def _load_kitchenpot(self):
        loader = self._scene.create_urdf_loader()
        loader.scale = self.articulation_scale
        loader.fix_root_link = True

        model_dir = (
            ASSET_DIR
            / f"partnet_mobility_dataset/{self._arti_mode}/{self.articulation_id}"
        )
        urdf_path = model_dir / "mobility.urdf"
        loader.load_multiple_collisions_from_file = True
        assert (
            urdf_path.exists()
        ), f"{urdf_path} is not found. Please download Partnet-Mobility Dataset first."

        density = self._articulation_config.get("density", 1e3)
        articulation = loader.load(str(urdf_path), config={"density": density})
        articulation.set_name("laptop_kitchenpot")

        cam = self._scene.add_camera("dummy_camera", 1, 1, 1, 0.01, 10)
        self._scene.remove_camera(cam)
        set_articulation_render_material(
            articulation, color=hex2rgba("#AAAAAA"), metallic=1, roughness=0.4
        )

        return articulation

    def _set_switch_links(self):
        if self._arti_mode in ["cabinet_door", "cabinet_drawer"]:
            all_links = self._articulation.get_links()
            all_active_joints = self._articulation.get_active_joints()

            # set target link & joint, handle & door of target_link
            self._target_joint_idx = self._articulation_config["target_joint_idx"]
            self._target_joint = all_active_joints.pop(self._target_joint_idx)
            self._target_link = self._target_joint.get_child_link()
            all_links.remove(self._target_link)

            handle_visid = []
            door_visid = []
            for visual_body in self._target_link.get_visual_bodies():
                if "handle" in visual_body.get_name():
                    handle_visid.append(visual_body.get_visual_id())
                else:
                    door_visid.append(visual_body.get_visual_id())
            self.handle_visid = handle_visid
            self.door_visid = door_visid

            other_handle_visid = []
            other_door_visid = []
            for ajoint in all_active_joints:
                door_link = ajoint.get_child_link()
                all_links.remove(door_link)
                for visual_body in door_link.get_visual_bodies():
                    if "handle" in visual_body.get_name():
                        other_handle_visid.append(visual_body.get_visual_id())
                        visual_body.set_visibility(
                            1.0 if self.other_handle_visible else 0.0
                        )
                    else:
                        other_door_visid.append(visual_body.get_visual_id())
            self.other_handle_visid = other_handle_visid
            self.other_door_visid = other_door_visid

            cabinet_visid = []
            for link in all_links:
                for visual_body in link.get_visual_bodies():
                    cabinet_visid.append(visual_body.get_visual_id())
            self.cabinet_visid = cabinet_visid
        elif self._arti_mode == "faucet":
            all_links = self._articulation.get_links()
            all_joints = self._articulation.get_joints()

            switch_link_names = []
            fix_link_visid = []
            # self.fix_links = []
            # self.fix_links_mesh = []
            for semantic in self._articulation_config["semantics"]:
                if semantic[2] == "switch":
                    switch_link_names.append(semantic[0])
                else:
                    fix_link = get_entity_by_name(all_links, semantic[0])
                    # self.fix_links.append(fix_link)
                    # link_mesh = get_actor_mesh(fix_link, False)
                    # self.fix_links_mesh.append(link_mesh)
                    for visual_body in fix_link.get_visual_bodies():
                        fix_link_visid.append(visual_body.get_visual_id())
            self.fix_link_visid = fix_link_visid

            if len(switch_link_names) == 0:
                raise RuntimeError(self.articulation_id)
            self.switch_link_names = switch_link_names

            self.switch_links = []
            self.switch_links_visid = []
            self.switch_links_mesh = []
            self.switch_joints = []
            for name in self.switch_link_names:
                link = get_entity_by_name(all_links, name)
                self.switch_links.append(link)
                for visual_body in link.get_visual_bodies():
                    self.switch_links_visid.append(visual_body.get_visual_id())

                # cache mesh
                link_mesh = get_actor_mesh(link, False)
                self.switch_links_mesh.append(link_mesh)

                # hardcode
                joint = all_joints[link.get_index()]
                # joint.set_friction(0.1)
                # joint.set_drive_property(0.0, 2.0)
                self.switch_joints.append(joint)
        elif self._arti_mode == "laptop":
            all_links = self._articulation.get_links()
            all_joints = self._articulation.get_joints()
            switch_link_names = []
            fix_link_visid = []
            for semantic in self._articulation_config["semantics"]:
                if semantic[2] == "switch":
                    switch_link_names.append(semantic[0])
                else:
                    fix_link = get_entity_by_name(all_links, semantic[0])
                    for visual_body in fix_link.get_visual_bodies():
                        fix_link_visid.append(visual_body.get_visual_id())
            self.fix_link_visid = fix_link_visid
            self.fix_link = fix_link
            if len(switch_link_names) == 0:
                raise RuntimeError(self.articulation_id)
            self.switch_link_names = switch_link_names
            self.switch_links = []
            self.switch_links_visid = []
            switch_links_mesh = []
            switch_joints = []
            for name in self.switch_link_names:
                link = get_entity_by_name(all_links, name)
                self.switch_links.append(link)
                for visual_body in link.get_visual_bodies():
                    self.switch_links_visid.append(visual_body.get_visual_id())
                # cache mesh
                link_mesh = get_actor_mesh(link, False)
                switch_links_mesh.append(link_mesh)
                # hardcode
                joint = all_joints[link.get_index()]

                switch_joints.append(joint)

            n_switch_links = len(self.switch_link_names)
            idx = random_choice(np.arange(n_switch_links), self._episode_rng)
            self.target_link_name = self.switch_link_names[idx]
            self._target_link: sapien.Link = self.switch_links[idx]
            self.target_link_mesh: trimesh.Trimesh = switch_links_mesh[idx]
            self.target_joint: sapien.Joint = switch_joints[idx]
            self.target_joint_idx = self._articulation.get_active_joints().index(
                self.target_joint
            )

            joint_pose = self.target_joint.get_global_pose().to_transformation_matrix()
            self.target_joint_axis = joint_pose[:3, 0]
            with np_random(self._episode_seed):
                self.target_pcd = self.target_link_mesh.sample(1000)

            distance = self.target_pcd[:, 1]
            ind_sort = np.argsort(distance)
            points_num = int(0.2 * self.target_pcd.shape[0])
            tip_index = ind_sort[-points_num:]
            self.tip_points = self.target_pcd[tip_index]

            qmin, qmax = self.target_joint.get_limits()[0]
            qmin = 0 if np.isinf(qmin) else qmin
            qmax = np.pi / 2 if np.isinf(qmax) else qmax
            q_range = qmax - qmin
            self.qmin_max_range = [qmin, qmax, q_range]
        elif self._arti_mode == "kitchen_pot":
            all_links = self._articulation.get_links()
            all_joints = self._articulation.get_joints()
            switch_link_names = []
            fix_link_visid = []
            for semantic in self._articulation_config["semantics"]:
                if semantic[2] == "switch":
                    switch_link_names.append(semantic[0])
                else:
                    fix_link = get_entity_by_name(all_links, semantic[0])
                    for visual_body in fix_link.get_visual_bodies():
                        fix_link_visid.append(visual_body.get_visual_id())
            self.fix_link_visid = fix_link_visid
            self.fix_link = fix_link
            if len(switch_link_names) == 0:
                raise RuntimeError(self.articulation_id)
            self.switch_link_names = switch_link_names

            # set target link, joint
            n_switch_links = len(self.switch_link_names)
            idx = random_choice(np.arange(n_switch_links), self._episode_rng)
            self.target_link_name = self.switch_link_names[idx]
            self._target_link: sapien.Link = get_entity_by_name(
                all_links, self.target_link_name
            )
            self.target_joint: sapien.Joint = all_joints[self._target_link.get_index()]
            self.target_joint_idx = self._articulation.get_active_joints().index(
                self.target_joint
            )

            self.lid_visid, lid_meshes = [], []
            self.handle_visid, handle_meshes = [], []

            for visual_body in self._target_link.get_visual_bodies():
                vis_id = visual_body.get_visual_id()
                if "handle" in visual_body.get_name():
                    self.handle_visid.append(vis_id)
                    # cache mesh
                    handle_meshes.extend(get_visual_body_meshes(visual_body))
                else:
                    self.lid_visid.append(vis_id)
                    # cache mesh
                    lid_meshes.extend(get_visual_body_meshes(visual_body))
            self.handle_mesh = merge_meshes(handle_meshes)
            self.lid_mesh = merge_meshes(lid_meshes)

            ### get handle grasp
            handle_convex_mesh = trimesh.convex.convex_hull(self.handle_mesh)
            handle_pcd = handle_convex_mesh.sample(200)
            self.handle_pcd = handle_pcd
            handle_pcd_world = apply_pose_to_points(
                handle_pcd, self._target_link.get_pose()
            )
            bbox_size = (handle_pcd_world.max(0) - handle_pcd_world.min(0)) / 2
            center = (handle_pcd_world.max(0) + handle_pcd_world.min(0)) / 2 + np.array(
                [0, 0.04, 0]
            )
            forward = np.array([0, -1, 0])
            if bbox_size[0] > bbox_size[2]:
                flat = np.array([0, 0, 1])
            else:
                flat = np.array([-1, 0, 0])
            grasp_pose = (
                self.agent.build_grasp_pose(forward, flat, center),
                self.agent.build_grasp_pose(forward, -flat, center),
            )
            self.handle_grasp = grasp_pose

            qmin, qmax = self.target_joint.get_limits()[0]
            qmin = 0 if np.isinf(qmin) else qmin
            qmax = 0.2 if np.isinf(qmax) else qmax
            q_range = qmax - qmin
            self.qmin_max_range = [qmin, qmax, q_range]
        else:
            raise NotImplementedError

    def _get_handle_info_in_target_link(self):
        """
        build a mesh and a point cloud of handle in target link, compute grasp poses of handle
        """
        handle_meshes = []
        others_meshes = []
        for visual_body in self._target_link.get_visual_bodies():
            for render_shape in visual_body.get_render_shapes():
                vertices = apply_pose_to_points(
                    render_shape.mesh.vertices * visual_body.scale,
                    visual_body.local_pose,
                )
                shape_mesh = np2mesh(vertices, render_shape.mesh.indices.reshape(-1, 3))
                if "handle" in visual_body.get_name():
                    handle_meshes.append(shape_mesh)
                else:
                    others_meshes.append(shape_mesh)
        handle_mesh = merge_mesh(handle_meshes)

        ### get handle grasp
        handle_convex_mesh = trimesh.convex.convex_hull(o3d_to_trimesh(handle_mesh))
        handle_pcd = handle_convex_mesh.sample(200)
        handle_pcd_world = apply_pose_to_points(
            handle_pcd, self._target_link.get_pose()
        )
        bbox_size = (handle_pcd_world.max(0) - handle_pcd_world.min(0)) / 2
        center = (handle_pcd_world.max(0) + handle_pcd_world.min(0)) / 2
        R = self._target_link.get_pose().to_transformation_matrix()[:3, :3]
        # choose the axis closest to X
        idx = np.argmax(np.abs(R[0]))
        forward = R[:3, idx]
        if forward[0] < 0:
            forward *= -1
        if bbox_size[1] > bbox_size[2]:
            flat = np.array([0, 0, 1])
        else:
            flat = np.cross(forward, np.array([0, 0, 1]))
        grasp_pose = (
            self._target_link.get_pose().inv()
            * self._agent.build_grasp_pose(forward, flat, center),
            self._target_link.get_pose().inv()
            * self._agent.build_grasp_pose(forward, -flat, center),
        )

        # trans_handle_mesh = handle_mesh.transform(self._target_link.get_pose().to_transformation_matrix())
        self.target_link_mesh = o3d_to_trimesh(handle_mesh)
        self.target_pcd = self.target_link_mesh.sample(300)  ## target_link frame

        # get tip, med, bottom keypoints
        xyzmax_val, xyzmin_val = self.target_pcd.max(0), self.target_pcd.min(0)
        xyzmax_id, xyzmin_id = np.argmax(self.target_pcd, 0), np.argmin(
            self.target_pcd, 0
        )
        bbox_size = xyzmax_val - xyzmin_val
        long_side = np.argmax(bbox_size)
        tip, bottom = (
            self.target_pcd[xyzmax_id[long_side]],
            self.target_pcd[xyzmin_id[long_side]],
        )
        center = self.target_pcd.mean(0)
        if self.articulation_id in [0, 1]:
            center[1] = xyzmax_val[1]
        else:
            center[2] = xyzmax_val[2]
        self._handle_keypoints = np.stack([tip, center, bottom])

        self._handle_info = OrderedDict()
        self._handle_info["mesh"] = trimesh_to_o3d(handle_convex_mesh)
        self._handle_info["pcd"] = handle_pcd
        self._handle_info["grasp"] = grasp_pose
        self._handle_info["center"] = handle_pcd.mean(0)

    def _get_keypart_vid(self, part_name):
        """
        get the visual id from each link in (cabinet, door, robot, gripper)
        """
        assert part_name in ["cabinet", "robot", "gripper"]

        vis_ids = []
        for link in self._part_info[part_name]["links"]:
            for visual_body in link.get_visual_bodies():
                vis_ids.append(visual_body.get_visual_id())
        self._part_info[part_name]["vis_id"] = vis_ids
        if part_name == "cabinet":
            hanele_ids = []
            others_ids = []
            for visual_body in self._target_link.get_visual_bodies():
                if "handle" in visual_body.get_name():
                    hanele_ids.append(visual_body.get_visual_id())
                else:
                    others_ids.append(visual_body.get_visual_id())
            self._part_info["handle"]["vis_id"] = hanele_ids
            self._part_info["door"]["vis_id"] = others_ids

    def _get_keypart_info(self, part_name="cabinet"):
        """
        get the visual id, sampled pcd from each link in (cabinet, door, robot, gripper)
        """
        assert part_name in ["cabinet", "robot", "gripper"]

        idx = 0
        for link in self._part_info[part_name]["links"]:
            meshes = []
            for visual_body in link.get_visual_bodies():
                if visual_body.type == "mesh":  ### for gripper finger pad
                    scale = visual_body.scale
                elif visual_body.type == "box":
                    scale = visual_body.half_lengths
                elif visual_body.type == "sphere":  ### from maniskill actor_to_o3d
                    scale = visual_body.radius
                else:
                    raise TypeError
                for render_shape in visual_body.get_render_shapes():
                    if render_shape.mesh.indices.reshape(-1, 3).shape[
                        0
                    ] < 4 or check_coplanar(render_shape.mesh.vertices * scale):
                        continue
                    vertices = apply_pose_to_points(
                        render_shape.mesh.vertices * scale, visual_body.local_pose
                    )
                    meshes.append(
                        np2mesh(vertices, render_shape.mesh.indices.reshape(-1, 3))
                    )
            if len(meshes) > 0:
                link_mesh = merge_mesh(meshes)  # link frame
                num_faces = 1024
                if np.asarray(link_mesh.triangles).shape[0] > num_faces:
                    link_mesh = link_mesh.simplify_vertex_clustering(
                        voxel_size=0.02,
                        contraction=o3d.geometry.SimplificationContraction.Average,
                    )
                link_trimesh = o3d_to_trimesh(link_mesh)
                link_pcd_gt = link_trimesh.sample(
                    self._part_info[part_name]["pts_num"][idx]
                )  ## link frame
                self._part_info[part_name]["pcd"][link.get_name()] = link_pcd_gt
            idx += 1

    def _get_obs_state_dict(self) -> OrderedDict:
        state_dict = OrderedDict()
        ## propriception
        arm_qpos = self.agent._robot.get_qpos()
        state_dict["qpos"] = arm_qpos.astype(np.float32)
        ee_pos_robot = (
            self._agent._robot.get_root_pose().inv() * self._agent.grasp_site.get_pose()
        ).p
        state_dict["ee_pos_base"] = ee_pos_robot.astype(np.float32)

        return state_dict

    def get_handcam_seg(self):
        self.update_render()
        ego_cam = self._agent._cameras["hand_camera"]
        ego_cam.take_picture()
        visual_id_seg = get_camera_seg(ego_cam)[..., 0]
        cam_shape = visual_id_seg.shape
        cam_seg = (
            np.ones(cam_shape, dtype=np.uint8) * 5
        )  # gripper, handle, door, cabient, ..., ..., other
        if self._arti_mode in ["cabinet_door", "cabinet_drawer"]:
            seg_visids = [
                self.handle_visid,
                self.door_visid + self.other_door_visid,
                self.cabinet_visid,
            ]
            seg_id = [0, 1, 2]
            for id, seg_visid in enumerate(seg_visids):
                mask = np.zeros(cam_shape, dtype=np.bool)
                for visual_id in seg_visid:
                    mask = mask | (visual_id_seg == visual_id)
                cam_seg[mask] = seg_id[id]
        elif self._arti_mode == "faucet":
            seg_visids = [self.switch_links_visid, self.fix_link_visid]
            seg_id = [3, 4]
            for id, seg_visid in enumerate(seg_visids):
                mask = np.zeros(cam_shape, dtype=np.bool)
                for visual_id in seg_visid:
                    mask = mask | (visual_id_seg == visual_id)
                cam_seg[mask] = seg_id[id]
        else:
            raise NotImplementedError
        return cam_seg

    def get_handcam_rgbdseg(self):
        rgbdseg = OrderedDict()
        ### get visual info
        self.update_render()
        ego_cam = self._agent._cameras["hand_camera"]
        ego_cam.take_picture()
        ## process rgb
        cam_rgb = get_camera_rgb(ego_cam)  # (H, W, 3)
        cam_H, cam_W = cam_rgb.shape[:2]
        rgbdseg["rgb"] = cam_rgb.transpose((2, 0, 1)).astype(np.uint8)  # (3, H, W)
        ## process depth & point clouds
        position = get_texture(ego_cam, "Position")
        cam_depth = -position[..., [2]]  # [H, W, 1]
        rgbdseg["depth"] = (cam_depth.transpose((2, 0, 1))).astype(
            np.float32
        )  # (1, H, W)

        visual_id_seg = get_camera_seg(ego_cam)[..., 0]
        cam_seg = (
            np.ones((cam_H, cam_W), dtype=np.bool) * 5
        )  # gripper, handle, door, cabient, ..., ..., other
        if self._arti_mode in ["cabinet_door", "cabinet_drawer"]:
            seg_visids = [
                self.handle_visid,
                self.door_visid + self.other_door_visid,
                self.cabinet_visid,
            ]
            seg_id = [0, 1, 2]
            for id, seg_visid in enumerate(seg_visids):
                mask = np.zeros((cam_H, cam_W), dtype=np.bool)
                for visual_id in seg_visid:
                    mask = mask | (visual_id_seg == visual_id)
                cam_seg[mask] = seg_id[id]
            rgbdseg["seg"] = convert_np_bool_to_float(cam_seg).astype(
                np.uint8
            )  # (H, W)
        elif self._arti_mode == "faucet":
            seg_visids = [self.switch_links_visid, self.fix_link_visid]
            seg_id = [3, 4]
            for id, seg_visid in enumerate(seg_visids):
                mask = np.zeros((cam_H, cam_W), dtype=np.bool)
                for visual_id in seg_visid:
                    mask = mask | (visual_id_seg == visual_id)
                cam_seg[mask] = seg_id[id]
            rgbdseg["seg"] = convert_np_bool_to_float(cam_seg).astype(
                np.uint8
            )  # (H, W)
        else:
            raise NotImplementedError
        return rgbdseg

    def get_handsensor_rgb(self):
        self.update_render()
        hand_sensor = self.agent._sensors["hand"]
        sensor_rgb = hand_sensor._cam_rgb
        hand_sensor.take_picture()
        rgb = (hand_sensor.get_rgb().transpose(2, 0, 1) * 255).astype(
            np.uint8
        )  # (3, H, W)
        return rgb

    def get_handsensor_rgbdseg(self):
        state_egopts = OrderedDict()
        self.update_render()
        hand_sensor = self.agent._sensors["hand"]
        sensor_rgb = hand_sensor._cam_rgb
        hand_sensor.take_picture()
        hand_sensor.compute_depth()

        state_egopts["rgb"] = (hand_sensor.get_rgb().transpose(2, 0, 1) * 255).astype(
            np.uint8
        )  # (3, H, W)
        state_egopts["depth"] = hand_sensor.get_depth().astype(np.float32)

        # process seg groundtruth
        visual_id_seg = get_camera_seg(sensor_rgb)[..., 0]  # [H, W]
        cam_H, cam_W = visual_id_seg.shape
        # gripper, handle, door, cabient, switch_link, fix_link, other
        cam_seg = np.ones((cam_H, cam_W), dtype=np.bool) * 5
        if self._arti_mode in ["cabinet_door", "cabinet_drawer"]:
            seg_visids = [
                self.handle_visid,
                self.door_visid + self.other_door_visid,
                self.cabinet_visid,
            ]
            seg_id = [0, 1, 2]
        elif self._arti_mode == "faucet":
            seg_visids = [self.switch_links_visid, self.fix_link_visid]
            seg_id = [3, 4]
        elif self._arti_mode == "laptop":
            seg_visids = [self.switch_links_visid, self.fix_link_visid]
            seg_id = [5, 6]
        elif self._arti_mode == "kitchen_pot":
            seg_visids = [self.handle_visid, self.lid_visid, self.fix_link_visid]
            seg_id = [7, 8, 9]
        else:
            raise NotImplementedError
        for id, seg_visid in enumerate(seg_visids):
            mask = np.zeros((cam_H, cam_W), dtype=np.bool)
            for visual_id in seg_visid:
                mask = mask | (visual_id_seg == visual_id)
            cam_seg[mask] = seg_id[id]
        state_egopts["seg"] = convert_np_bool_to_float(cam_seg).astype(
            np.uint8
        )  # (H, W)
        return state_egopts

    def get_handsensor_pc(self):
        hand_sensor = self.agent._sensors["hand"]
        sensor_rgb = hand_sensor._cam_rgb
        hand_sensor.take_picture()
        hand_sensor.compute_depth()
        # process point cloud
        cam_xyz = hand_sensor.get_pointcloud(False)  # [H*W, 3]
        ## cam to world
        T_cam2world = sensor_rgb.get_model_matrix()
        world_xyz = transform_points(
            T_cam2world, cam_xyz * np.array([1, -1, -1])
        )  # [H*W, 3]
        return world_xyz

    def get_egostereo_segpts(self):
        # get state
        state_egopts = self._get_obs_state_dict()

        ## ee points in ee frame
        if self.add_eepadpts:
            ee_cords = self._agent.sample_ee_coords(num_sample=20).reshape(
                -1, 3
            )  # [40, 3], world frame
            eepad_pts = apply_pose_to_points(
                ee_cords, self.agent.grasp_site.get_pose().inv()
            )  # [40, 3], ee frame
            eepad_pts_t = eepad_pts
            eepat_pts_label = np.zeros(
                (eepad_pts.shape[0], self.num_classes + 1)
            )  # (40, C+1)
            eepat_pts_label[:, -1] = 1.0
            eepad_ptsC = np.concatenate(
                (eepad_pts_t, eepat_pts_label), axis=1
            )  # (40, 3+C+1)

        # get visual info
        self.update_render()
        hand_sensor = self.agent._sensors["hand"]
        sensor_rgb = hand_sensor._cam_rgb
        hand_sensor.take_picture()
        hand_sensor.compute_depth()

        # process seg groundtruth
        visual_id_seg = get_camera_seg(sensor_rgb)[..., 0]  # [H, W]
        cam_H, cam_W = visual_id_seg.shape
        cam_seg = (
            np.ones((cam_H, cam_W), dtype=np.bool) * 5
        )  # gripper, handle, door, cabient, switch_link, fix_link, other
        if self._arti_mode in ["cabinet_door", "cabinet_drawer"]:
            seg_visids = [
                self.handle_visid,
                self.door_visid + self.other_door_visid,
                self.cabinet_visid,
            ]
            seg_id = [0, 1, 2]
        elif self._arti_mode == "faucet":
            seg_visids = [self.switch_links_visid, self.fix_link_visid]
            seg_id = [3, 4]
        else:
            raise NotImplementedError

        for id, seg_visid in enumerate(seg_visids):
            mask = np.zeros((cam_H, cam_W), dtype=np.bool)
            for visual_id in seg_visid:
                mask = mask | (visual_id_seg == visual_id)
            cam_seg[mask] = seg_id[id]
        if self.add_eepadpts:
            state_egopts["seg"] = convert_np_bool_to_float(cam_seg).astype(
                np.float32
            )  # (H, W)
        else:
            state_egopts["seg"] = convert_np_bool_to_float(cam_seg).astype(
                np.uint8
            )  # (H, W)

        # process seg model prediction
        cam_rgb_tensor = (
            torch.utils.dlpack.from_dlpack(hand_sensor.get_rgba_dl_tensor())
            .clone()[None, ..., :3]
            .clip(0, 1)
            .permute((0, 3, 1, 2))
        )  # [1,3,H,W]
        cam_depth_tensor = torch.utils.dlpack.from_dlpack(
            hand_sensor.get_depth_dl_tensor()
        ).clone()[
            None, None
        ]  # [1,1,H,W]

        # predict seg & sample K points
        seg_map = self.segmodel.predict(
            torch.cat([cam_rgb_tensor, cam_depth_tensor], dim=1)
        )[
            0
        ]  # (C, H, W)
        # seg_map_noise = (
        #     seg_map + torch.randn(seg_map.shape, device=seg_map.device) * 0.1
        # )
        seg_mc_label = torch.argmax(seg_map, dim=0).reshape(-1)  # (H*W)
        if self.add_eepadpts:
            seg_onehot_label = (
                F.one_hot(seg_mc_label, num_classes=self.num_classes + 1)
                .cpu()
                .detach()
                .numpy()
            )  # (H*W, C+1)
        else:
            seg_onehot_label = (
                F.one_hot(seg_mc_label, num_classes=self.num_classes)
                .cpu()
                .detach()
                .numpy()
            )  # (H*W, C)
        # if self._arti_mode == "cabinet_drawer":
        #     seg_onehot_label *= np.array([-1, -1, -1, 1, 1, 1])

        # process point cloud
        cam_xyz = hand_sensor.get_pointcloud(False)  # [H*W, 3]
        ## cam to world
        T_cam2world = sensor_rgb.get_model_matrix()
        world_xyz = transform_points(
            T_cam2world, cam_xyz * np.array([1, -1, -1])
        )  # [H*W, 3]
        T_world2ee = (
            self.agent.grasp_site.get_pose().inv().to_transformation_matrix()
        )  ## world => ee
        if self.sample_mode == "full_downsample":
            # ## crop the pts & uniform downsample to num pts
            sample_indices = self.sampler.sampling(world_xyz, self._episode_rng)
            ds_world_xyz = world_xyz[sample_indices]
            ds_onehot_label = seg_onehot_label[sample_indices]
            ## world to ee
            ds_ee_xyz = transform_points(T_world2ee, ds_world_xyz)
            # add noise on downsampled ee_xyz
            # with np_random(self._episode_seed):
            #     ds_ee_xyz = ds_ee_xyz + np.random.normal(loc=0, scale=0.01, size=ds_ee_xyz.shape)
            segptsC = np.concatenate((ds_ee_xyz, ds_onehot_label), axis=1)  # (SN, 3+C)
            if self.add_eepadpts:
                state_egopts["segsampled_ptsC"] = np.concatenate(
                    (segptsC, eepad_ptsC), axis=0
                ).astype(
                    np.float32
                )  # (SN+40, 3+C+1)
            else:
                state_egopts["segsampled_ptsC"] = segptsC.astype(
                    np.float32
                )  # (SN, 3+C)
        else:
            # pts_index = self.sampler.sampling(world_xyz, seg_map, self._episode_rng)
            if self.sample_mode == "score_sample":
                pts_index = self.sampler.sampling(seg_map)
            elif self.sample_mode == "random_sample":
                seg_mc_map = seg_mc_label.cpu().numpy()
                pts_index = self.sampler.sampling(
                    seg_mc_map, self.num_classes, self._episode_rng
                )
            elif self.sample_mode == "fps_sample":
                seg_mc_map = seg_mc_label.cpu().numpy()
                pts_index = self.sampler.sampling(
                    world_xyz, self.num_classes, seg_mc_map, self._episode_rng
                )
            else:
                raise NotImplementedError
            world_segsampled_xyz = world_xyz[pts_index]  # (C*sample_num, 3)
            sampled_onehot_label = seg_onehot_label[pts_index]  # (C*sample_num, C)
            ee_segsampled_xyz = transform_points(
                T_world2ee, world_segsampled_xyz
            )  # (C*sample_num, 3)
            # add noise on downsampled ee_xyz
            # with np_random(self._episode_seed):
            #     ee_segsampled_xyz = ee_segsampled_xyz + np.random.normal(loc=0, scale=0.01, size=ee_segsampled_xyz.shape)
            segptsC = np.concatenate(
                (ee_segsampled_xyz, sampled_onehot_label), axis=1
            )  # (C*sample_num, 3+C)
            if self.add_eepadpts:
                state_egopts["segsampled_ptsC"] = np.concatenate(
                    (segptsC, eepad_ptsC), axis=0
                ).astype(
                    np.float32
                )  # (C*sample_num+40, 3+C+1)
            else:
                state_egopts["segsampled_ptsC"] = segptsC.astype(
                    np.float32
                )  # (C*sample_num, 3+C)
        return state_egopts

    def get_egostereo_segpts_gt(self):
        # get state
        state_egopts = self._get_obs_state_dict()

        ## ee points in ee frame
        # ee_cords = self._agent.sample_ee_coords(num_sample=20).reshape(
        #     -1, 3
        # )  # [40, 3], world frame
        # eepad_pts = apply_pose_to_points(
        #     ee_cords, self.agent.grasp_site.get_pose().inv()
        # )  # [40, 3], ee frame
        # eepad_pts_t = eepad_pts
        # eepat_pts_label = np.zeros((eepad_pts.shape[0], self.num_classes + 1))  # (40, C+1)
        # eepat_pts_label[:, -1] = 1.0
        # eepad_ptsC = np.concatenate((eepad_pts_t, eepat_pts_label), axis=1)  # (40, 3+C+1)

        # get visual info
        self.update_render()
        hand_sensor = self.agent._sensors["hand"]
        sensor_rgb = hand_sensor._cam_rgb
        hand_sensor.take_picture(infrared_only=False)
        hand_sensor.compute_depth()

        # get set gt
        visual_id_seg = get_camera_seg(sensor_rgb)[..., 0]  # [H, W]
        cam_H, cam_W = visual_id_seg.shape
        cam_seg = (
            np.ones((cam_H, cam_W), dtype=np.bool) * 5
        )  # gripper, handle, door, cabient, switch_link, fix_link, other
        if self._arti_mode in ["cabinet_door", "cabinet_drawer"]:
            seg_visids = [
                self.handle_visid,
                self.door_visid + self.other_door_visid,
                self.cabinet_visid,
            ]
            seg_id = [0, 1, 2]
        elif self._arti_mode == "faucet":
            seg_visids = [self.switch_links_visid, self.fix_link_visid]
            seg_id = [3, 4]
        elif self._arti_mode == "laptop":
            seg_visids = [self.switch_links_visid, self.fix_link_visid]
            seg_id = [5, 6]
        elif self._arti_mode == "kitchen_pot":
            seg_visids = [self.handle_visid, self.lid_visid, self.fix_link_visid]
            seg_id = [7, 8, 9]
        else:
            raise NotImplementedError
        for id, seg_visid in enumerate(seg_visids):
            mask = np.zeros((cam_H, cam_W), dtype=np.bool)
            for visual_id in seg_visid:
                mask = mask | (visual_id_seg == visual_id)
            cam_seg[mask] = seg_id[id]
        state_egopts["seg"] = convert_np_bool_to_float(cam_seg).astype(
            np.uint8
        )  # (H, W)

        # get C * sample_num indices from cam_seg
        seg_label = np.eye(self.num_classes)
        seg_onehot_label = seg_label[cam_seg.reshape(-1)].astype(np.float_)  # (H*W, C)

        indices = []
        for ind in range(seg_onehot_label.shape[1]):
            one_indices = np.nonzero(seg_onehot_label[:, ind])[0]  # S
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
                cur_indices = np.random.choice(one_indices, self.sample_num)
            indices.append(cur_indices)
        pts_index = np.concatenate(indices)  # (C*sample_num)

        # get sampled point cloud (ee frame)
        cam_xyz = hand_sensor.get_pointcloud(with_rgb=False)
        cam_xyz = cam_xyz * np.array([1, -1, -1])
        ds_cam_xyz = cam_xyz[pts_index]  # (C*sample_num, 3)
        T_cam2ee = self.cam_para["hand_camera_extrinsic_base_frame"]
        ds_ee_xyz = transform_points(T_cam2ee, ds_cam_xyz)  # (C*sample_num, 3)
        # get sampled one-hot label
        sampled_onehot_label = seg_onehot_label[pts_index]  # (C*sample_num, C)
        segptsC = np.concatenate(
            (ds_ee_xyz, sampled_onehot_label), axis=1
        )  # (C*sample_num, 3+C)
        state_egopts["segsampled_ptsC"] = segptsC.astype(
            np.float_
        )  # (C*sample_num+40, 3+C)
        # state_egopts["segsampled_ptsC"] = np.concatenate((segptsC, eepad_ptsC), axis=0).astype(
        #     np.float_)  # (C*sample_num+40, 3+C+1)
        return state_egopts

    def get_egostereo_rgbd_seg_kpts(self):
        # get state
        state_egopts = self._get_obs_state_dict()

        # get visual info
        self.update_render()
        hand_sensor = self.agent._sensors["hand"]
        sensor_rgb = hand_sensor._cam_rgb
        hand_sensor.take_picture()
        hand_sensor.compute_depth()
        cam_intrin = self.cam_para["hand_camera_intrinsic"]
        T_cam2ee = self.cam_para["hand_camera_extrinsic_base_frame"]

        # process seg groundtruth
        visual_id_seg = get_camera_seg(sensor_rgb)[..., 0]  # [H, W]
        cam_H, cam_W = visual_id_seg.shape
        cam_seg = (
            np.ones((cam_H, cam_W), dtype=np.bool) * 5
        )  # gripper, handle, door, cabient, switch_link, fix_link, other
        if self._arti_mode in ["cabinet_door", "cabinet_drawer"]:
            seg_visids = [
                self.handle_visid,
                self.door_visid + self.other_door_visid,
                self.cabinet_visid,
            ]
            seg_id = [0, 1, 2]
        elif self._arti_mode == "faucet":
            seg_visids = [self.switch_links_visid, self.fix_link_visid]
            seg_id = [3, 4]
        else:
            raise NotImplementedError

        for id, seg_visid in enumerate(seg_visids):
            mask = np.zeros((cam_H, cam_W), dtype=np.bool)
            for visual_id in seg_visid:
                mask = mask | (visual_id_seg == visual_id)
            cam_seg[mask] = seg_id[id]
        state_egopts["seg"] = convert_np_bool_to_float(cam_seg).astype(
            np.uint8
        )  # (H, W)

        # process seg model prediction
        cam_rgb_tensor = (
            torch.utils.dlpack.from_dlpack(hand_sensor.get_rgba_dl_tensor())
            .clone()[None, ..., :3]
            .clip(0, 1)
            .permute((0, 3, 1, 2))
        )  # [1,3,H,W]
        cam_depth_tensor = torch.utils.dlpack.from_dlpack(
            hand_sensor.get_depth_dl_tensor()
        ).clone()[
            None, None
        ]  # [1,1,H,W]

        # predict seg & sample K points
        heatmap_out = self.kpt_model.forward(
            torch.cat([cam_rgb_tensor, cam_depth_tensor], dim=1)
        )  # (1, J*D', H', W')
        uvz_pred_norm = (
            soft_argmax(heatmap_out, 3, self.out_shape)[0].detach().cpu().numpy()
        )  # (J, 3)
        uvz_pred = uvz_pred_norm * (self.kpts_max - self.kpts_min) + self.kpts_min
        # state_egopts["uvz_pred"] = uvz_pred.astype(np.float32)  # (J, 3)
        xyz_cam_pred = uv2xyz(uvz_pred, cam_intrin)
        xyz_ee_pred = transform_points(T_cam2ee, xyz_cam_pred)
        state_egopts["xyz_ee_pred"] = xyz_ee_pred.astype(np.float32)  # [J, 3]
        return state_egopts

    def get_egostereo_rgbd_seg_kpts_gt(self):
        # get state
        state_egopts = self._get_obs_state_dict()

        # get visual info
        self.update_render()
        hand_sensor = self.agent._sensors["hand"]
        sensor_rgb = hand_sensor._cam_rgb
        hand_sensor.take_picture()

        # process seg groundtruth
        visual_id_seg = get_camera_seg(sensor_rgb)[..., 0]  # [H, W]
        cam_H, cam_W = visual_id_seg.shape
        cam_seg = (
            np.ones((cam_H, cam_W), dtype=np.bool) * 5
        )  # gripper, handle, door, cabient, switch_link, fix_link, other
        if self._arti_mode in ["cabinet_door", "cabinet_drawer"]:
            seg_visids = [
                self.handle_visid,
                self.door_visid + self.other_door_visid,
                self.cabinet_visid,
            ]
            seg_id = [0, 1, 2]
        elif self._arti_mode == "faucet":
            seg_visids = [self.switch_links_visid, self.fix_link_visid]
            seg_id = [3, 4]
        else:
            raise NotImplementedError

        for id, seg_visid in enumerate(seg_visids):
            mask = np.zeros((cam_H, cam_W), dtype=np.bool)
            for visual_id in seg_visid:
                mask = mask | (visual_id_seg == visual_id)
            cam_seg[mask] = seg_id[id]
        state_egopts["seg"] = convert_np_bool_to_float(cam_seg).astype(
            np.uint8
        )  # (H, W)

        kpts_world = apply_pose_to_points(
            self._handle_keypoints, self._target_link.get_pose()
        )  ## world frame
        kpts_ee = apply_pose_to_points(
            kpts_world, self.agent.grasp_site.get_pose().inv()
        )  # ee frame
        kpts = kpts_ee + np.random.normal(loc=0, scale=0.015, size=kpts_ee.shape)
        state_egopts["kpts"] = kpts.astype(np.float32)  # (J, 3)
        return state_egopts

    def get_egostereo_rgbdseg(self):
        # get state
        state_egorgbdseg = self._get_obs_state_dict()

        self.update_render()
        hand_sensor = self.agent._sensors["hand"]
        sensor_rgb = hand_sensor._cam_rgb
        hand_sensor.take_picture()
        hand_sensor.compute_depth()

        state_egorgbdseg["rgb"] = (
            hand_sensor.get_rgb().transpose(2, 0, 1) * 255
        ).astype(
            np.uint8
        )  # [3, H, W]
        state_egorgbdseg["depth"] = hand_sensor.get_depth()[None].astype(
            np.float32
        )  # [1, H, W]

        # process seg groundtruth
        visual_id_seg = get_camera_seg(sensor_rgb)[..., 0]  # [H, W]
        cam_H, cam_W = visual_id_seg.shape
        # gripper, handle, door, cabient, switch_link, fix_link, other
        cam_seg = np.ones((cam_H, cam_W), dtype=np.bool) * 5
        if self._arti_mode in ["cabinet_door", "cabinet_drawer"]:
            seg_visids = [
                self.handle_visid,
                self.door_visid + self.other_door_visid,
                self.cabinet_visid,
            ]
            seg_id = [0, 1, 2]
        elif self._arti_mode == "faucet":
            seg_visids = [self.switch_links_visid, self.fix_link_visid]
            seg_id = [3, 4]
        else:
            raise NotImplementedError
        for id, seg_visid in enumerate(seg_visids):
            mask = np.zeros((cam_H, cam_W), dtype=np.bool)
            for visual_id in seg_visid:
                mask = mask | (visual_id_seg == visual_id)
            cam_seg[mask] = seg_id[id]
        state_egorgbdseg["seg"] = convert_np_bool_to_float(cam_seg).astype(
            np.uint8
        )  # (H, W)
        return state_egorgbdseg

    def get_egostereo_rgbdseg_gt(self):
        # get state
        state_egorgbdseg = self._get_obs_state_dict()

        self.update_render()
        hand_sensor = self.agent._sensors["hand"]
        sensor_rgb = hand_sensor._cam_rgb
        hand_sensor.take_picture()
        hand_sensor.compute_depth()

        # state_egorgbdseg["rgb"] = (hand_sensor.get_rgb().transpose(2, 0, 1) * 255).astype(np.uint8)  # [3, H, W]
        state_egorgbdseg["depth"] = hand_sensor.get_depth()[None].astype(
            np.float32
        )  # [1, H, W]

        # process seg groundtruth
        visual_id_seg = get_camera_seg(sensor_rgb)[..., 0]  # [H, W]
        cam_H, cam_W = visual_id_seg.shape
        # gripper, handle, door, cabient, switch_link, fix_link, other
        cam_seg = np.ones((cam_H, cam_W), dtype=np.bool) * 5
        if self._arti_mode in ["cabinet_door", "cabinet_drawer"]:
            seg_visids = [self.handle_visid, self.door_visid, self.cabinet_visid]
            seg_id = [0, 1, 2]
        elif self._arti_mode == "faucet":
            seg_visids = [self.switch_links_visid, self.fix_link_visid]
            seg_id = [3, 4]
        else:
            raise NotImplementedError
        for id, seg_visid in enumerate(seg_visids):
            mask = np.zeros((cam_H, cam_W), dtype=np.bool)
            for visual_id in seg_visid:
                mask = mask | (visual_id_seg == visual_id)
            cam_seg[mask] = seg_id[id]
        state_egorgbdseg["seg"] = convert_np_bool_to_float(cam_seg).astype(
            np.uint8
        )  # (H, W)
        return state_egorgbdseg

    def get_egostereo_state_rgbdseg(self):
        # get state
        state_egorgbdseg = self._get_obs_state_dict()

        self.update_render()
        hand_sensor = self.agent._sensors["hand"]
        sensor_rgb = hand_sensor._cam_rgb
        hand_sensor.take_picture()
        hand_sensor.compute_depth()

        state_egorgbdseg["rgb"] = (
            hand_sensor.get_rgb().transpose(2, 0, 1) * 255
        ).astype(
            np.uint8
        )  # [3, H, W]
        state_egorgbdseg["depth"] = hand_sensor.get_depth()[None]  # [1, H, W]

        if self.sample_mode in ["full_downsample", "score_sample", "random_sample"]:
            # process point cloud
            cam_xyz = hand_sensor.get_pointcloud(False)  # [H*W, 3]
            if self.sample_mode in ["score_sample", "random_sample"]:
                # cam to ee frame directly
                T_cam2ee = self.cam_para["hand_camera_extrinsic_base_frame"]
                ee_xyz = (
                    transform_points(T_cam2ee, cam_xyz)
                    .reshape(-1, 3)
                    .astype(np.float32)
                )  # [H*W, 3]
                state_egorgbdseg["ee_xyz"] = ee_xyz
            else:
                ## cam to world
                T_cam2world = sensor_rgb.get_model_matrix()
                world_xyz = transform_points(
                    T_cam2world, cam_xyz * np.array([1, -1, -1])
                )  # [H*W, 3]
                T_world2ee = (
                    self.agent.grasp_site.get_pose().inv().to_transformation_matrix()
                )  ## world => ee
                state_egorgbdseg["world_xyz"] = world_xyz.astype(np.float32)  # [H*W, 3]
                state_egorgbdseg["world_ee"] = T_world2ee

        # process seg groundtruth
        visual_id_seg = get_camera_seg(sensor_rgb)[..., 0]  # [H, W]
        cam_H, cam_W = visual_id_seg.shape
        # gripper, handle, door, cabient, switch_link, fix_link, other
        cam_seg = np.ones((cam_H, cam_W), dtype=np.bool) * 5
        if self._arti_mode in ["cabinet_door", "cabinet_drawer"]:
            seg_visids = [
                self.handle_visid,
                self.door_visid + self.other_door_visid,
                self.cabinet_visid,
            ]
            seg_id = [0, 1, 2]
        elif self._arti_mode == "faucet":
            seg_visids = [self.switch_links_visid, self.fix_link_visid]
            seg_id = [3, 4]
        else:
            raise NotImplementedError
        for id, seg_visid in enumerate(seg_visids):
            mask = np.zeros((cam_H, cam_W), dtype=np.bool)
            for visual_id in seg_visid:
                mask = mask | (visual_id_seg == visual_id)
            cam_seg[mask] = seg_id[id]
        state_egorgbdseg["seg"] = convert_np_bool_to_float(cam_seg).astype(
            np.uint8
        )  # (H, W)
        return state_egorgbdseg

    def get_egostereo_dexpoints(self):
        # get state
        state_egodexpts = self._get_obs_state_dict()

        # get ee pad sampled points
        eepad_pts_world = self._agent.sample_ee_coords(num_sample=20).reshape(
            -1, 3
        )  # [40, 3], world frame
        eepad_pts_robot = apply_pose_to_points(
            eepad_pts_world, self._agent._robot.get_root_pose().inv()
        )
        # eepad_pts = apply_pose_to_points(
        #     eepad_pts_world, self.agent.grasp_site.get_pose().inv()
        # )  # [40, 3], ee frame
        eepat_pts_label = np.zeros((eepad_pts_robot.shape[0], 2))  # (40, 2)
        eepat_pts_label[:, -1] = 1.0
        eepad_ptsC_robot = np.concatenate(
            (eepad_pts_robot, eepat_pts_label), axis=1
        )  # (40, 3+2)

        # get visual info
        self.update_render()
        hand_sensor = self.agent._sensors["hand"]
        sensor_rgb = hand_sensor._cam_rgb
        hand_sensor.take_picture()
        hand_sensor.compute_depth()

        # process seg groundtruth
        visual_id_seg = get_camera_seg(sensor_rgb)[..., 0]  # [H, W]
        cam_H, cam_W = visual_id_seg.shape
        cam_seg = (
            np.ones((cam_H, cam_W), dtype=np.bool) * 5
        )  # gripper, handle, door, cabient, switch_link, fix_link, other
        if self._arti_mode in ["cabinet_door", "cabinet_drawer"]:
            seg_visids = [
                self.handle_visid,
                self.door_visid + self.other_door_visid,
                self.cabinet_visid,
            ]
            seg_id = [0, 1, 2]
        elif self._arti_mode in ["faucet", "laptop", "kitchen_pot"]:
            seg_visids = [self.switch_links_visid, self.fix_link_visid]
            seg_id = [3, 4]
        else:
            raise NotImplementedError

        for id, seg_visid in enumerate(seg_visids):
            mask = np.zeros((cam_H, cam_W), dtype=np.bool)
            for visual_id in seg_visid:
                mask = mask | (visual_id_seg == visual_id)
            cam_seg[mask] = seg_id[id]
            state_egodexpts["seg"] = convert_np_bool_to_float(cam_seg).astype(
                np.uint8
            )  # (H, W)

        # process point cloud
        cam_xyz = hand_sensor.get_pointcloud(False)  # [H*W, 3]
        ## cam to world
        T_cam2world = sensor_rgb.get_model_matrix()
        world_xyz = transform_points(
            T_cam2world, cam_xyz * np.array([1, -1, -1])
        )  # [H*W, 3]
        robot_xyz = apply_pose_to_points(
            world_xyz, self._agent._robot.get_root_pose().inv()
        )
        sample_indices = pcd_uni_down_sample_with_crop(
            world_xyz,
            1000,
            min_bound=np.array([-0.5, -1, 1e-3]),
            max_bound=np.array([1, 1, 2]),
        )
        indices_len = len(sample_indices)
        if indices_len < 1000:
            sample_indices.extend(
                self._episode_rng.choice(world_xyz.shape[0], 1000 - indices_len)
            )
        ds_robot_xyz = robot_xyz[sample_indices]
        ds_robot_label = np.zeros((ds_robot_xyz.shape[0], 2))  # (1000, 2)
        ds_robot_label[:, 0] = 1.0
        ds_ptsC_robot = np.concatenate(
            (ds_robot_xyz, ds_robot_label), axis=1
        )  # (1000, 3+2)

        state_egodexpts["segsampled_ptsC"] = np.concatenate(
            (ds_ptsC_robot, eepad_ptsC_robot), axis=0
        ).astype(
            np.float32
        )  # (1000+40, 3+2)
        return state_egodexpts

    def get_obs(self):
        if self._obs_mode == "state":
            state_dict = self._get_obs_state_dict()
            return flatten_state_dict(state_dict)
        elif self._obs_mode == "state_dict":
            state_dict = self._get_obs_state_dict()
            return state_dict
        elif self._obs_mode == "state_egorgbd":
            ### get state
            state_rgbd = self._get_obs_state_dict()

            ### get visual info
            self.update_render()
            ego_cam = self._agent._cameras["hand_camera"]
            ego_cam.take_picture()
            ## process rgb
            cam_rgb = get_camera_rgb(ego_cam)  # (H, W, 3)
            cam_H, cam_W = cam_rgb.shape[:2]
            state_rgbd["rgb"] = (cam_rgb.transpose((2, 0, 1))).astype(
                np.uint8
            )  # (3, H, W)
            ## process depth & point clouds
            position = get_texture(ego_cam, "Position")
            cam_depth = -position[..., [2]]  # [H, W, 1]
            state_rgbd["depth"] = (cam_depth.transpose((2, 0, 1))).astype(
                np.float32
            )  # (1, H, W)

            visual_id_seg = get_camera_seg(ego_cam)[..., 0]
            cam_seg = (
                np.ones((cam_H, cam_W), dtype=np.bool) * 5
            )  # gripper, handle, door, cabient, ..., ..., other
            if self._arti_mode in ["cabinet_door", "cabinet_drawer"]:
                seg_visids = [self.handle_visid, self.door_visid, self.cabinet_visid]
                seg_id = [0, 1, 2]
                for id, seg_visid in enumerate(seg_visids):
                    mask = np.zeros((cam_H, cam_W), dtype=np.bool)
                    for visual_id in seg_visid:
                        mask = mask | (visual_id_seg == visual_id)
                    cam_seg[mask] = seg_id[id]
                state_rgbd["seg"] = convert_np_bool_to_float(cam_seg).astype(
                    np.float32
                )  # (H, W)
            elif self._arti_mode in ["faucet", "laptop", "kitchen_pot"]:
                seg_visids = [self.switch_links_visid, self.fix_link_visid]
                seg_id = [3, 4]
                for id, seg_visid in enumerate(seg_visids):
                    mask = np.zeros((cam_H, cam_W), dtype=np.bool)
                    for visual_id in seg_visid:
                        mask = mask | (visual_id_seg == visual_id)
                    cam_seg[mask] = seg_id[id]
                state_rgbd["seg"] = convert_np_bool_to_float(cam_seg).astype(
                    np.float32
                )  # (H, W)
            else:
                raise NotImplementedError

            return state_rgbd
        elif self._obs_mode == "state_egosegpoints":
            # get state
            state_egopts = self._get_obs_state_dict()

            # get visual info
            self.update_render()
            ego_cam = self._agent._cameras["hand_camera"]
            ego_cam.take_picture()
            # process rgb
            cam_rgb = get_camera_rgb(ego_cam)  # (H, W, 3)
            cam_H, cam_W = cam_rgb.shape[:2]
            rgb_proc = torch.from_numpy(
                (cam_rgb.transpose((2, 0, 1)) / 255.0)[None].astype(np.float32)
            ).to(
                torch.device(self.device)
            )  # (1, 3, H, W)

            # process depth
            position = get_texture(ego_cam, "Position")  # [H, W, 4]
            cam_depth = -position[..., [2]]  # [H, W, 1]
            depth_proc = torch.from_numpy(
                (cam_depth.transpose((2, 0, 1)))[None].astype(np.float32)
            ).to(
                torch.device(self.device)
            )  # (1, 1, H, W)
            # process points
            cam_xyz = position[..., :3]
            with np_random(self._episode_seed):
                cam_xyz_noise = cam_xyz + np.random.normal(
                    loc=0, scale=0.01, size=cam_xyz.shape
                )
            # invalid_mask = position[..., -1] < 1  # Remove invalid points
            # filter_cam_xyz = cam_xyz[invalid_mask]  # [H, W, 3]
            # cam to ee frame directly
            T_cam2ee = self.cam_para["hand_camera_extrinsic_base_frame"]
            ee_xyz = (
                (cam_xyz_noise @ T_cam2ee[:3, :3].transpose() + T_cam2ee[:3, 3])
                .reshape(-1, 3)
                .astype(np.float32)
            )  # [H*W, 3]

            # predict seg & sample K points
            seg_map = self.segmodel.predict(torch.cat([rgb_proc, depth_proc], dim=1))[
                0
            ]  # (C, H, W)
            seg_map_noise = (
                seg_map + torch.randn(seg_map.shape, device=seg_map.device) * 0.1
            )
            _, indices = torch.topk(
                seg_map_noise.view(self.num_classes, -1), self.sample_num, dim=1
            )  # (C, sample_num)
            pts_index = indices.detach().cpu().numpy().reshape(-1)  # (C*sample_num)
            segsampled_pts = ee_xyz[pts_index]  # (C*sample_num, 3)
            state_egopts["segsampled_pts"] = segsampled_pts

            visual_id_seg = get_camera_seg(ego_cam)[..., 0]
            cam_seg = (
                np.ones((cam_H, cam_W), dtype=np.bool) * 5
            )  # gripper, handle, door, cabient, switch_link, fix_link, other
            if self._arti_mode in ["cabinet_door", "cabinet_drawer"]:
                seg_visids = [self.handle_visid, self.door_visid, self.cabinet_visid]
                seg_id = [0, 1, 2]
            elif self._arti_mode == "faucet":
                seg_visids = [self.switch_links_visid, self.fix_link_visid]
                seg_id = [3, 4]
            else:
                raise NotImplementedError

            for id, seg_visid in enumerate(seg_visids):
                mask = np.zeros((cam_H, cam_W), dtype=np.bool)
                for visual_id in seg_visid:
                    mask = mask | (visual_id_seg == visual_id)
                cam_seg[mask] = seg_id[id]
            state_egopts["seg"] = convert_np_bool_to_float(cam_seg).astype(
                np.float32
            )  # (H, W)

            return state_egopts
        elif self._obs_mode == "state_ego_segpoints_gt":
            ### get state
            state_ego_segpoints_gt = self._get_obs_state_dict()

            ### get visual info
            self.update_render()
            ego_cam = self._agent._cameras["hand_camera"]
            ego_cam.take_picture()
            ## process rgb
            # cam_rgb = get_camera_rgb(ego_cam)  # (H, W, 3)
            # cam_H, cam_W = cam_rgb.shape[:2]
            # state_ego_segpoints_gt["rgb"] = (cam_rgb.transpose((2, 0, 1))).astype(np.uint8)  # (3, H, W)
            ## process depth & point clouds
            position = get_texture(ego_cam, "Position")
            # cam_depth = -position[..., [2]]  # [H, W, 1]
            # state_rgbd["depth"] = (cam_depth.transpose((2, 0, 1))).astype(np.float32)  # (1, H, W)
            cam_xyz = position[..., :3].reshape(-1, 3)  # (H*W, 3)

            # get set gt
            visual_id_seg = get_camera_seg(ego_cam)[..., 0]  # [H, W]
            cam_H, cam_W = visual_id_seg.shape
            cam_seg = np.ones((cam_H, cam_W), dtype=np.bool) * (
                self.num_classes - 1
            )  # gripper, handle, door, cabient, switch_link, fix_link, other
            if self._arti_mode == "cabinet_door":
                seg_visids = [
                    self.handle_visid,
                    self.door_visid + self.other_door_visid,
                    self.cabinet_visid,
                ]
                seg_id = [0, 2, 3]
            elif self._arti_mode == "cabinet_drawer":
                seg_visids = [
                    self.handle_visid,
                    self.door_visid + self.other_door_visid,
                    self.cabinet_visid,
                ]
                seg_id = [0, 2, 3]
            elif self._arti_mode == "faucet":
                seg_visids = [self.switch_links_visid, self.fix_link_visid]
                seg_id = [1, 3]
            elif self._arti_mode == "laptop":
                seg_visids = [self.switch_links_visid, self.fix_link_visid]
                seg_id = [1, 3]
            elif self._arti_mode == "kitchen_pot":
                seg_visids = [self.handle_visid, self.lid_visid, self.fix_link_visid]
                seg_id = [0, 2, 3]
            else:
                raise NotImplementedError
            for id, seg_visid in enumerate(seg_visids):
                mask = np.zeros((cam_H, cam_W), dtype=np.bool)
                for visual_id in seg_visid:
                    mask = mask | (visual_id_seg == visual_id)
                cam_seg[mask] = seg_id[id]
            state_ego_segpoints_gt["seg"] = convert_np_bool_to_float(cam_seg).astype(
                np.uint8
            )  # (H, W)
            # get C * sample_num indices from cam_seg
            seg_label = np.eye(self.num_classes)
            seg_onehot_label = seg_label[cam_seg.reshape(-1)].astype(
                np.float_
            )  # (H*W, C)
            indices = []
            for ind in range(seg_onehot_label.shape[1]):
                one_indices = np.nonzero(seg_onehot_label[:, ind])[0]  # S
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
                    cur_indices = np.random.choice(one_indices, self.sample_num)
                indices.append(cur_indices)
            pts_index = np.concatenate(indices)  # (C*sample_num)

            # get sampled point cloud (ee frame)
            # cam_xyz = cam_xyz * np.array([1, -1, -1])
            ds_cam_xyz = cam_xyz[pts_index]  # (C*sample_num, 3)
            T_cam2ee = self.cam_para["hand_camera_extrinsic_base_frame"]
            ds_ee_xyz = transform_points(T_cam2ee, ds_cam_xyz)  # (C*sample_num, 3)
            # get sampled one-hot label
            sampled_onehot_label = seg_onehot_label[pts_index]  # (C*sample_num, C)
            segptsC = np.concatenate(
                (ds_ee_xyz, sampled_onehot_label), axis=1
            )  # (C*sample_num, 3+C)
            state_ego_segpoints_gt["segsampled_ptsC"] = segptsC.astype(
                np.float_
            )  # (C*sample_num+40, 3+C)

            return state_ego_segpoints_gt
        elif self._obs_mode == "state_egostereo_rgbd":
            state_egorgbdseg = self.get_egostereo_state_rgbdseg()
            return state_egorgbdseg
        elif self._obs_mode == "state_egostereo_segpoints":
            state_egosegpts = self.get_egostereo_segpts()
            return state_egosegpts
        elif self._obs_mode == "state_egostereo_segpoints_gt":
            state_egosegptsgt = self.get_egostereo_segpts_gt()
            return state_egosegptsgt
        elif self._obs_mode == "state_egostereo_keypoints":
            state_egokpts = self.get_egostereo_rgbd_seg_kpts()
            return state_egokpts
        elif self._obs_mode == "state_egostereo_keypoints_gt":
            state_egokptsgt = self.get_egostereo_rgbd_seg_kpts_gt()
            return state_egokptsgt
        elif self._obs_mode == "state_egostereo_dseg":
            state_egodseg = self.get_egostereo_rgbdseg()
            return state_egodseg
        elif self._obs_mode == "state_egostereo_dseg_gt":
            state_egodseggt = self.get_egostereo_rgbdseg_gt()
            return state_egodseggt
        elif self._obs_mode == "state_egostereo_dexpoints":
            state_egodexpoints = self.get_egostereo_dexpoints()
            return state_egodexpoints
        else:
            raise NotImplementedError(self._obs_mode)

    @property
    def current_angle(self):
        return self._articulation.get_qpos()[self.target_joint_idx]

    def _compute_distance(self):
        ## transfer to target frame
        pcdee_wf = self._agent.sample_ee_coords(num_sample=20).reshape(-1, 3)  # (40, 3)
        T = self.target_link.get_pose().inv().to_transformation_matrix()
        pcdee_tf = transform_points(T, pcdee_wf)  # (40, 3), target frame

        pcdee_center_tf = pcdee_tf.mean(0)
        # pcdtip_center_tf = self.tip_points.mean(0)
        # l1_distance = pcdee_center_tf - pcdtip_center_tf
        ee_tip_dist = pcdee_center_tf - self.tip_points
        l1x_distance = np.max(ee_tip_dist[:, 0])
        l2_distance = np.min(np.linalg.norm(ee_tip_dist[:, 1:], axis=1))
        return l1x_distance, l2_distance

    def _get_visual_state(self) -> OrderedDict:  # from ManiSkill1
        joint_pos = (
            self._articulation.get_qpos()[self._target_joint_idx] / self.target_qpos
        )
        current_handle = apply_pose_to_points(
            self._handle_info["pcd"], self.target_link.get_pose()
        ).mean(0)

        visual_state = OrderedDict()
        visual_state["joint_pos"] = np.array([joint_pos])  # 1
        visual_state["open_enough"] = convert_np_bool_to_float(joint_pos > 1.0)  # 1
        visual_state["current_handle"] = current_handle  # 3

        return visual_state

    def check_handle_exist(self, obs):
        assert "seg" in obs.keys()
        if self._arti_mode in ["cabinet_door", "cabinet_drawer"]:
            partseg_id = 0
        elif self._arti_mode == "faucet":
            partseg_id = 3
        elif self._arti_mode == "laptop":
            partseg_id = 5
        elif self._arti_mode == "kitchen_pot":
            partseg_id = 7
        else:
            raise NotImplementedError
        handle_exist_rew = 1 if np.any(obs["seg"] == partseg_id) else 0
        return handle_exist_rew

    def compute_other_flag_dict(self):
        ee_cords = self._agent.sample_ee_coords()  # [2, 10, 3]
        current_handle = apply_pose_to_points(
            self._handle_info["pcd"], self._target_link.get_pose()
        )  # [200, 3]
        ee_to_handle = (
            ee_cords[..., None, :] - current_handle
        )  # [2, 10, 200, 3] = [2, 10, 1, 3] - [200, 3]
        dist_ee_to_handle = (
            np.linalg.norm(ee_to_handle, axis=-1).min(-1).min(-1)
        )  # [2, 10, 200, 3] -> [2, 10, 200](dist) -> [2, 10] -> [2] min_dist between (left_finger, right_finger) <-> handle

        handle_center_world = apply_pose_to_points(
            self._handle_info["center"][None], self._target_link.get_pose()
        )[0]

        handle_mesh = trimesh.Trimesh(
            vertices=apply_pose_to_points(
                np.asarray(self._handle_info["mesh"].vertices),
                self._target_link.get_pose(),
            ),
            faces=np.asarray(np.asarray(self._handle_info["mesh"].triangles)),
        )
        handle_obj = trimesh.proximity.ProximityQuery(handle_mesh)
        # sd between ee_left, ee_right <-> handle, >0 means inside the mesh; <0 means outside the mesh
        sd_ee_to_handle = (
            handle_obj.signed_distance(ee_cords.reshape(-1, 3)).reshape(2, -1).max(1)
        )  # [2,10,3]->[20,3]->[20]->[2,10]->2
        # sd between mid_ee <-> handle, >0 means inside the mesh; <0 means outside the mesh
        sd_ee_mid_to_handle = handle_obj.signed_distance(
            ee_cords.mean(0)
        ).max()  # [2,10,3]->[10,3]->[10]->1

        ee_close_to_handle = convert_np_bool_to_float(
            dist_ee_to_handle.max() <= 0.025
            and sd_ee_mid_to_handle > -0.01  # 0 or -0.01
        )

        # Grasp = mid close almost in cvx and both sides has similar sign distance.
        close_to_grasp = (
            sd_ee_to_handle.min() > -1e-2
        )  ## grasp happen: one side close to handle
        ee_in_grasp_pose = (
            sd_ee_mid_to_handle > 0
        )  ## one mean gripper point in handle mesh, 0/-1e-2
        grasp_happen = ee_in_grasp_pose and close_to_grasp

        other_info = {
            "dist_ee_to_handle": np.array(dist_ee_to_handle, dtype=np.float_),  ## 2
            "handle_center_world": np.array(handle_center_world, dtype=np.float_),  ## 1
            "ee_close_to_handle": convert_np_bool_to_float(ee_close_to_handle),  ## 1
            "sd_ee_to_handle": np.array(sd_ee_to_handle, dtype=np.float_),  ## 2
            "sd_ee_mid_to_handle": np.array(sd_ee_mid_to_handle, dtype=np.float_),  ## 1
            "grasp_happen": convert_np_bool_to_float(grasp_happen),  ## 1
        }
        return other_info

    def compute_eval_flag_dict(self):
        flag_dict = OrderedDict()
        if self._arti_mode in ["cabinet_door", "cabinet_drawer"]:
            flag_dict["cabinet_static"] = convert_np_bool_to_float(
                self.check_actor_static(self._target_link, max_v=0.1, max_ang_v=1.0)
            )
            flag_dict["open_enough"] = convert_np_bool_to_float(
                self._articulation.get_qpos()[
                    self._articulation_config["target_joint_idx"]
                ]
                >= 0.9 * self.target_qpos
            )
        elif self._arti_mode in ["faucet", "laptop", "kitchen_pot"]:
            cur_qpos = self._articulation.get_qpos()[self.target_joint_idx]
            flag_dict["target_achieved"] = (
                (cur_qpos - self.init_angle) / (self.target_angle - self.init_angle)
            ) > 0.5
        else:
            raise NotImplementedError

        flag_dict["is_success"] = convert_np_bool_to_float(all(flag_dict.values()))
        return flag_dict

    def check_success(self):
        flag_dict = self.compute_eval_flag_dict()
        return flag_dict["is_success"]

    def compute_dense_reward_cabinet(self, **kwargs):
        np.set_printoptions(suppress=True)
        flag_dict = self.compute_eval_flag_dict()
        other_info = self.compute_other_flag_dict()
        dist_ee_to_handle = other_info["dist_ee_to_handle"]
        sd_ee_mid_to_handle = other_info["sd_ee_mid_to_handle"]  # > 0 means good
        sd_ee_to_handle = other_info[
            "sd_ee_to_handle"
        ]  # [2,], min > -1e-2 means grasp happen: one side close to handle

        agent_proprio = self.agent.get_proprioception()
        gripper_qpos = agent_proprio["qpos"][-2:]

        grasp_site_pose = self._agent.grasp_site.get_pose()
        target_pose_2 = self._target_link.get_pose() * self._handle_info["grasp"][1]
        gripper_angle_err = (
            np.abs(angle_distance_ms(grasp_site_pose, target_pose_2)) / np.pi
        )

        handle_center_world = other_info["handle_center_world"]
        ## handle center xyz in EE frame, abs: 0 ~ (0.4, 0.1, 0.1)
        handle_center_ee = apply_pose_to_points(
            handle_center_world[None], grasp_site_pose.inv()
        )[0]

        cabinet_qpos = self._articulation.get_qpos()[self._target_joint_idx]
        cabinet_qvel = self._articulation.get_qvel()[self._target_joint_idx]

        door_vel_norm = min(norm(self._target_link.get_velocity()), 1)  ## [0, 1]
        door_ang_vel_norm = min(
            norm(self._target_link.get_angular_velocity()), 1
        )  ## [0, 1]

        scale = 0.5
        stage_index = 0

        handle_exist_rew = self.check_handle_exist(kwargs["obs"])

        # [-1.5, 0]: (0.01, 0.5) -> [-0.5, 0] + (-0.35, 0.01) => [0, 1]
        close_to_handle_rew = (
            -1
            - dist_ee_to_handle.mean()
            + normalize_and_clip_in_interval(sd_ee_mid_to_handle, -0.35, 0.01)
        )
        dir_rew = (
            -np.clip(gripper_angle_err, 0, 0.5)
            * 2
            # - np.clip(np.sum(np.abs(handle_center_ee[:2])), 0, 0.1) * 5
        )

        cabinet_qvel_rew = 0
        cabinet_qpos_rew = 0
        cabinet_static_rew = 0
        ## open half, [-1, 0]
        gripper_rew = -np.abs(gripper_qpos.mean() - 0.034) / 0.034

        if dist_ee_to_handle.max() < 0.03 and sd_ee_mid_to_handle > -0.01:
            stage_index = 1
            ## close, [-2, 2]
            gripper_rew = (gripper_qpos.mean() - 0.034) / 0.034 * 2
            cabinet_qvel_rew = (
                np.clip(cabinet_qvel, -0.5, 0.5) * 2
            )  ## Push vel to positive, [-1, 1], init 0
            cabinet_qpos_rew = (
                normalize_and_clip_in_interval(cabinet_qpos, 0, self.target_qpos * 1.1)
                * 2
            )  ## [0, 2]
            if flag_dict["open_enough"]:
                ## cabinet_qpos >= self.target_qpos
                stage_index = 2
                gripper_rew = 2
                cabinet_qvel_rew = 1
                cabinet_qvel_rew = 2
                cabinet_static_rew = -(door_vel_norm + door_ang_vel_norm)  ## [-2, 0]

        reward = (
            handle_exist_rew
            + close_to_handle_rew
            + dir_rew
            + gripper_rew
            + cabinet_qvel_rew
            + cabinet_qpos_rew
            + cabinet_static_rew
        ) * scale

        action = kwargs["action"]
        if isinstance(action, dict):
            cur_action = action["action"]
        elif isinstance(action, np.ndarray):
            cur_action = action
        else:
            raise NotImplementedError
        info_dict = {
            "stage_index": np.array(stage_index),
            ### stage 0
            "dist_ee_to_handle": np.array(dist_ee_to_handle, dtype=np.float_),
            "sd_ee_to_handle": np.array(sd_ee_to_handle, dtype=np.float_),
            "sd_ee_mid_to_handle": np.array(sd_ee_mid_to_handle, dtype=np.float_),
            "gripper_angle_err": np.array(gripper_angle_err * 180, dtype=np.float_),
            "handle_center_ee": np.array(handle_center_ee, dtype=np.float_),
            "close_to_handle_rew": np.array(close_to_handle_rew, dtype=np.float_),
            "dir_rew": np.array(dir_rew, dtype=np.float_),
            "gripper_rew": np.array(gripper_rew, dtype=np.float_),
            ### stage 1
            "cabinet_qpos": np.array(cabinet_qpos, dtype=np.float_),
            "cabinet_qvel": np.array(cabinet_qvel, dtype=np.float_),
            ##  np.clip(cabinet_qvel, -0.5, 0.5) * 2
            "cabinet_qvel_rew": np.array(cabinet_qvel_rew, dtype=np.float_),
            ## normalize_and_clip_in_interval(cabinet_qpos, 0, self.target_qpos * 1.1) * 2
            "cabinet_qpos_rew": np.array(cabinet_qpos_rew, dtype=np.float_),
            ### stage 2
            "door_vel_norm": np.array(door_vel_norm, dtype=np.float_),
            "door_ang_vel_norm": np.array(door_ang_vel_norm, dtype=np.float_),
            ## -(door_vel_norm + door_ang_vel_norm * 0.5)
            "cabinet_static_rew": np.array(cabinet_static_rew, dtype=np.float_),
            "handle_exist_rew": np.array(handle_exist_rew, dtype=np.float_),
            "reward_raw": np.array(reward, dtype=np.float_),
            "check_grasp": np.array(agent_proprio["gripper_grasp"], dtype=np.float_),
            "action": np.array(cur_action, dtype=np.float_),
        }
        self._cache_info = info_dict
        return reward

    def compute_dense_reward_door(self, **kwargs):
        np.set_printoptions(suppress=True)
        flag_dict = self.compute_eval_flag_dict()
        other_info = self.compute_other_flag_dict()
        dist_ee_to_handle = other_info["dist_ee_to_handle"]
        sd_ee_mid_to_handle = other_info["sd_ee_mid_to_handle"]  # > 0 means good
        sd_ee_to_handle = other_info[
            "sd_ee_to_handle"
        ]  # [2,], min > -1e-2 means grasp happen: one side close to handle

        agent_proprio = self.agent.get_proprioception()
        gripper_qpos = agent_proprio["qpos"][-2:]

        grasp_site_pose = self._agent.grasp_site.get_pose()
        target_pose = self._target_link.get_pose() * self._handle_info["grasp"][0]
        target_pose_2 = self._target_link.get_pose() * self._handle_info["grasp"][1]
        angle1 = np.abs(angle_distance_ms(grasp_site_pose, target_pose))
        angle2 = np.abs(angle_distance_ms(grasp_site_pose, target_pose_2))
        gripper_angle_err = min(angle1, angle2) / np.pi

        handle_center_world = other_info["handle_center_world"]
        ## handle center xyz in EE frame, abs: 0 ~ (0.4, 0.1, 0.1)
        handle_center_ee = apply_pose_to_points(
            handle_center_world[None], grasp_site_pose.inv()
        )[0]

        cabinet_qpos = self._articulation.get_qpos()[self._target_joint_idx]
        cabinet_qvel = self._articulation.get_qvel()[self._target_joint_idx]

        door_vel_norm = min(norm(self._target_link.get_velocity()), 1)  ## [0, 1]
        door_ang_vel_norm = min(
            norm(self._target_link.get_angular_velocity()), 1
        )  ## [0, 1]

        scale = 0.5
        stage_index = 0

        handle_exist_rew = self.check_handle_exist(kwargs["obs"])

        # (0.07, 0.35) -> [-oo, 0], (-0.35, 0.01) => [-0.7, 0]
        close_to_handle_rew = (
            -dist_ee_to_handle.mean()
            + normalize_and_clip_in_interval(sd_ee_mid_to_handle, -0.5, 0.01)
        )  # direction reward, [0, 0.25] -> [-1, 0]
        dir_rew = (
            -np.clip(gripper_angle_err, 0, 0.5)
            - np.clip(np.sum(np.abs(handle_center_ee[:2])), 0, 0.1) * 10
        )

        ## open half, [-0.5, 0.5]
        open_gripper_rew = 0.5 - np.abs(gripper_qpos.mean() - 0.034) / 0.034
        ## close, [-1, 1]
        close_gripper_rew = (gripper_qpos.mean() - 0.034) / 0.034

        cabinet_qvel_rew = 0
        cabinet_qpos_rew = 0
        cabinet_static_rew = 0
        gripper_rew = open_gripper_rew

        if dist_ee_to_handle.max() < 0.03 and sd_ee_mid_to_handle > 0:
            ## dist_ee_to_handle.max() <= 0.025 and sd_ee_mid_to_handle > -0.01 & sd_ee_mid_to_handle > 0
            stage_index = 1
            dir_rew = 0.5
            # close_to_handle_rew = 0.5
            gripper_rew = close_gripper_rew
            cabinet_qvel_rew = (
                np.clip(cabinet_qvel, -0.5, 0.5) * 2
            )  ## Push vel to positive, [-1, 1], init 0
            cabinet_qpos_rew = (
                normalize_and_clip_in_interval(cabinet_qpos, 0, self.target_qpos * 1.1)
                * 2
            )  ## [0, 2]
            if flag_dict["open_enough"]:
                ## cabinet_qpos >= self.target_qpos
                stage_index = 2
                gripper_rew = 1
                cabinet_qvel_rew = 1
                cabinet_qvel_rew = 2
                cabinet_static_rew = -(door_vel_norm + door_ang_vel_norm)  ## [-2, 0]

        reward = (
            +handle_exist_rew
            + close_to_handle_rew
            + dir_rew
            + gripper_rew
            + cabinet_qvel_rew
            + cabinet_qpos_rew
            + cabinet_static_rew
        ) * scale

        action = kwargs["action"]
        if isinstance(action, dict):
            cur_action = action["action"]
        elif isinstance(action, np.ndarray):
            cur_action = action
        else:
            raise NotImplementedError
        info_dict = {
            "stage_index": np.array(stage_index),
            ### stage 0
            "dist_ee_to_handle": np.array(dist_ee_to_handle, dtype=np.float_),
            "sd_ee_to_handle": np.array(sd_ee_to_handle, dtype=np.float_),
            "sd_ee_mid_to_handle": np.array(sd_ee_mid_to_handle, dtype=np.float_),
            "gripper_angle_err": np.array(gripper_angle_err * 180, dtype=np.float_),
            "close_to_handle_rew": np.array(close_to_handle_rew, dtype=np.float_),
            "dir_rew": np.array(dir_rew, dtype=np.float_),
            "gripper_rew": np.array(gripper_rew, dtype=np.float_),
            ### stage 1
            "cabinet_qpos": np.array(cabinet_qpos, dtype=np.float_),
            "cabinet_qvel": np.array(cabinet_qvel, dtype=np.float_),
            ##  np.clip(cabinet_qvel, -0.5, 0.5) * 2
            "cabinet_qvel_rew": np.array(cabinet_qvel_rew, dtype=np.float_),
            ## normalize_and_clip_in_interval(cabinet_qpos, 0, self.target_qpos * 1.1) * 2
            "cabinet_qpos_rew": np.array(cabinet_qpos_rew, dtype=np.float_),
            ### stage 2
            "door_vel_norm": np.array(door_vel_norm, dtype=np.float_),
            "door_ang_vel_norm": np.array(door_ang_vel_norm, dtype=np.float_),
            ## -(door_vel_norm + door_ang_vel_norm * 0.5)
            "cabinet_static_rew": np.array(cabinet_static_rew, dtype=np.float_),
            "handle_exist_rew": np.array(handle_exist_rew, dtype=np.float_),
            "reward_raw": np.array(reward, dtype=np.float_),
            "check_grasp": np.array(agent_proprio["gripper_grasp"], dtype=np.float_),
            "action": np.array(cur_action, dtype=np.float_),
        }
        self._cache_info = info_dict
        return reward

    def compute_dense_reward_drawer(self, **kwargs):
        np.set_printoptions(suppress=True)
        flag_dict = self.compute_eval_flag_dict()
        other_info = self.compute_other_flag_dict()
        dist_ee_to_handle = other_info["dist_ee_to_handle"]
        sd_ee_mid_to_handle = other_info["sd_ee_mid_to_handle"]  # > 0 means good
        sd_ee_to_handle = other_info[
            "sd_ee_to_handle"
        ]  # [2,], min > -1e-2 means grasp happen: one side close to handle

        agent_proprio = self.agent.get_proprioception()
        gripper_qpos = agent_proprio["qpos"][-2:]

        grasp_site_pose = self._agent.grasp_site.get_pose()
        target_pose_2 = self._target_link.get_pose() * self._handle_info["grasp"][1]
        gripper_angle_err = (
            np.abs(angle_distance_ms(grasp_site_pose, target_pose_2)) / np.pi
        )

        handle_center_world = other_info["handle_center_world"]
        ## handle center xyz in EE frame, abs: 0 ~ (0.4, 0.1, 0.1)
        handle_center_ee = apply_pose_to_points(
            handle_center_world[None], grasp_site_pose.inv()
        )[0]

        cabinet_qpos = self._articulation.get_qpos()[self._target_joint_idx]
        cabinet_qvel = self._articulation.get_qvel()[self._target_joint_idx]

        door_vel_norm = min(norm(self._target_link.get_velocity()), 1)  ## [0, 1]
        door_ang_vel_norm = min(
            norm(self._target_link.get_angular_velocity()), 1
        )  ## [0, 1]

        scale = 0.5
        stage_index = 0

        handle_exist_rew = self.check_handle_exist(kwargs["obs"])

        # [-1.5, 0]: (0.01, 0.5) -> [-0.5, 0] + (-0.35, 0.01) => [0, 1]
        close_to_handle_rew = (
            -1
            - dist_ee_to_handle.mean()
            + normalize_and_clip_in_interval(sd_ee_mid_to_handle, -0.35, 0.01)
        )
        dir_rew = -np.clip(gripper_angle_err, 0, 0.5) * 2

        cabinet_qvel_rew = 0
        cabinet_qpos_rew = 0
        cabinet_static_rew = 0
        ## open half, [-1, 0]
        gripper_rew = -np.abs(gripper_qpos.mean() - 0.034) / 0.034

        if gripper_angle_err < 0.15:  # 27degree
            dir_rew = 0.5
            if dist_ee_to_handle.max() < 0.03 and sd_ee_mid_to_handle > -0.01:
                stage_index = 1
                # dir_rew = 0.5
                close_to_handle_rew = 0.5
                ## close, [-2, 2]
                gripper_rew = (gripper_qpos.mean() - 0.034) / 0.034 * 2
                cabinet_qvel_rew = (
                    np.clip(cabinet_qvel, -0.5, 0.5) * 2
                )  ## Push vel to positive, [-1, 1], init 0
                cabinet_qpos_rew = (
                    normalize_and_clip_in_interval(
                        cabinet_qpos, 0, self.target_qpos * 1.1
                    )
                    * 2
                )  ## [0, 2]
                if flag_dict["open_enough"]:
                    ## cabinet_qpos >= self.target_qpos
                    stage_index = 2
                    gripper_rew = 2
                    cabinet_qvel_rew = 1
                    cabinet_qvel_rew = 2
                    cabinet_static_rew = -(
                        door_vel_norm + door_ang_vel_norm
                    )  ## [-2, 0]

        reward = (
            handle_exist_rew
            + close_to_handle_rew
            + dir_rew
            + gripper_rew
            + cabinet_qvel_rew
            + cabinet_qpos_rew
            + cabinet_static_rew
        ) * scale

        action = kwargs["action"]
        if isinstance(action, dict):
            cur_action = action["action"]
        elif isinstance(action, np.ndarray):
            cur_action = action
        else:
            raise NotImplementedError
        info_dict = {
            "stage_index": np.array(stage_index),
            ### stage 0
            "dist_ee_to_handle": np.array(dist_ee_to_handle, dtype=np.float_),
            "sd_ee_to_handle": np.array(sd_ee_to_handle, dtype=np.float_),
            "sd_ee_mid_to_handle": np.array(sd_ee_mid_to_handle, dtype=np.float_),
            "gripper_angle_err": np.array(gripper_angle_err * 180, dtype=np.float_),
            "handle_center_ee": np.array(handle_center_ee, dtype=np.float_),
            "close_to_handle_rew": np.array(close_to_handle_rew, dtype=np.float_),
            "dir_rew": np.array(dir_rew, dtype=np.float_),
            "gripper_rew": np.array(gripper_rew, dtype=np.float_),
            ### stage 1
            "cabinet_qpos": np.array(cabinet_qpos, dtype=np.float_),
            "cabinet_qvel": np.array(cabinet_qvel, dtype=np.float_),
            ##  np.clip(cabinet_qvel, -0.5, 0.5) * 2
            "cabinet_qvel_rew": np.array(cabinet_qvel_rew, dtype=np.float_),
            ## normalize_and_clip_in_interval(cabinet_qpos, 0, self.target_qpos * 1.1) * 2
            "cabinet_qpos_rew": np.array(cabinet_qpos_rew, dtype=np.float_),
            ### stage 2
            "door_vel_norm": np.array(door_vel_norm, dtype=np.float_),
            "door_ang_vel_norm": np.array(door_ang_vel_norm, dtype=np.float_),
            ## -(door_vel_norm + door_ang_vel_norm * 0.5)
            "cabinet_static_rew": np.array(cabinet_static_rew, dtype=np.float_),
            "handle_exist_rew": np.array(handle_exist_rew, dtype=np.float_),
            "reward_raw": np.array(reward, dtype=np.float_),
            "check_grasp": np.array(agent_proprio["gripper_grasp"], dtype=np.float_),
            "action": np.array(cur_action, dtype=np.float_),
        }
        self._cache_info = info_dict
        return reward

    def compute_dense_reward_faucet(self, **kwargs):
        reward = 0.0

        flag_dict = self.compute_eval_flag_dict()
        if flag_dict["is_success"]:
            return 4.0

        handle_exist_rew = self.check_handle_exist(kwargs["obs"])
        reward += handle_exist_rew
        l1_distance, l2_distance = self._compute_distance()
        close_reward = 1 - 10 * np.clip(l2_distance, 0, 0.2)  # [-1, 1]
        reward += close_reward
        left_reward = -20.0 * np.clip(l1_distance, 0, 0.1)  # [-2, 0]
        reward += left_reward

        angle_diff = self.target_angle - self.current_angle
        turn_reward_1 = 5 * (1 - np.tanh(max(angle_diff, 0) * 2.0))  # [0, 1] => [0, 5]
        reward += turn_reward_1

        delta_angle = angle_diff - self.last_angle_diff  # <=> last_angle - cur_angle
        if angle_diff > 0:
            turn_reward_2 = -np.tanh(delta_angle * 5)  # [0, 1]
        else:
            turn_reward_2 = np.tanh(delta_angle * 5)
        turn_reward_2 *= 5
        reward += turn_reward_2
        self.last_angle_diff = angle_diff

        # scale
        reward = reward * 0.5

        action = kwargs["action"]
        if isinstance(action, dict):
            cur_action = action["action"]
        elif isinstance(action, np.ndarray):
            cur_action = action
        else:
            raise NotImplementedError
        info_dict = {
            "handle_exist_rew": np.array(handle_exist_rew, dtype=np.float_),
            ### stage 0
            "l1_distance": np.array(l1_distance, dtype=np.float_),
            "left_reward": np.array(left_reward, dtype=np.float_),
            "distance": np.array(l2_distance, dtype=np.float_),
            "close_reward": np.array(close_reward, dtype=np.float_),
            ### stage 1
            "angle_diff": np.array(angle_diff, dtype=np.float_),
            "turn_reward_1": np.array(turn_reward_1, dtype=np.float_),
            ### stage 2
            "delta_angle": np.array(delta_angle, dtype=np.float_),
            "turn_reward_2": np.array(turn_reward_2, dtype=np.float_),
            "reward": np.array(reward, dtype=np.float_),
            # other info
            "cur_angle": np.array(self.current_angle, dtype=np.float_),
            "init_angle": np.array(self.init_angle, dtype=np.float_),
            "target_angle": np.array(self.target_angle, dtype=np.float_),
            "action": np.array(cur_action, dtype=np.float_),
        }
        self._cache_info = info_dict

        return reward

    def compute_dense_reward_laptop(self, **kwargs):
        reward = 0.0
        flag_dict = self.compute_eval_flag_dict()
        if flag_dict["is_success"]:
            return 4.0

        handle_exist_rew = self.check_handle_exist(kwargs["obs"])
        reward += handle_exist_rew

        ## transfer to target frame
        pcdee_wf = self._agent.sample_ee_coords(num_sample=20).reshape(-1, 3)  # (40, 3)
        T = self.target_link.get_pose().inv().to_transformation_matrix()
        pcdee_tf = transform_points(T, pcdee_wf)  # (40, 3), target frame
        pcdee_center_tf = pcdee_tf.mean(0)
        ee_tip_dist = self.tip_points - pcdee_center_tf
        l1z_distance = np.max(ee_tip_dist[:, 2])
        l2_distance = np.min(np.linalg.norm(ee_tip_dist[:, :2], axis=1))

        close_reward = 1 - 10 * np.clip(l2_distance, 0, 0.2)  # [-1, 1]
        down_reward = -20.0 * np.clip(l1z_distance, 0, 0.1)  # [-2, 0]
        reward += close_reward + down_reward

        angle_diff = self.target_angle - self.current_angle
        turn_reward_1 = 5 * (1 - np.tanh(max(angle_diff, 0) * 2.0))  # [0, 1] => [0, 5]
        reward += turn_reward_1

        delta_angle = angle_diff - self.last_angle_diff  # <=> last_angle - cur_angle
        if angle_diff > 0:
            turn_reward_2 = -np.tanh(delta_angle * 5)  # [0, 1]
        else:
            turn_reward_2 = np.tanh(delta_angle * 5)
        turn_reward_2 *= 5
        reward += turn_reward_2
        self.last_angle_diff = angle_diff

        # scale
        reward = reward * 0.5

        action = kwargs["action"]
        if isinstance(action, dict):
            cur_action = action["action"]
        elif isinstance(action, np.ndarray):
            cur_action = action
        else:
            raise NotImplementedError
        info_dict = {
            "handle_exist_rew": np.array(handle_exist_rew, dtype=np.float_),
            ### stage 0
            "l1z_distance": np.array(l1z_distance, dtype=np.float_),
            "down_reward": np.array(down_reward, dtype=np.float_),
            "l2_distance": np.array(l2_distance, dtype=np.float_),
            "close_reward": np.array(close_reward, dtype=np.float_),
            ### stage 1
            "angle_diff": np.array(angle_diff, dtype=np.float_),
            "turn_reward_1": np.array(turn_reward_1, dtype=np.float_),
            ### stage 2
            "delta_angle": np.array(delta_angle, dtype=np.float_),
            "turn_reward_2": np.array(turn_reward_2, dtype=np.float_),
            "reward": np.array(reward, dtype=np.float_),
            # other info
            "cur_angle": np.array(self.current_angle, dtype=np.float_),
            "init_angle": np.array(self.init_angle, dtype=np.float_),
            "target_angle": np.array(self.target_angle, dtype=np.float_),
            "action": np.array(cur_action, dtype=np.float_),
        }
        self._cache_info = info_dict

        return reward

    def compute_dense_reward_kitchenpot(self, **kwargs):
        flag_dict = self.compute_eval_flag_dict()
        if flag_dict["is_success"]:
            return 4.0

        handle_exist_rew = self.check_handle_exist(kwargs["obs"])

        ## transfer to target frame
        gripper_pts_ee = sample_grasp_multipoints_ee(
            0.03, num_points_perlink=10, x_offset=0.02
        )
        trans_tarlink2ee = (
            self._agent.grasp_site.get_pose().inv().to_transformation_matrix()
            @ self.target_link.get_pose().to_transformation_matrix()
        )
        grasp_mat_ee = (
            trans_tarlink2ee @ self.handle_grasp[0].to_transformation_matrix()
        )
        gripper_pts_targetpose_ee = transform_points(grasp_mat_ee, gripper_pts_ee)
        gripper_pts_dist = np.linalg.norm(
            gripper_pts_targetpose_ee - gripper_pts_ee, axis=-1
        )  # (k, )
        grasp_dist = np.mean(gripper_pts_dist, axis=-1)
        approach_reward = 1 - 3.0 * np.tanh(10.0 * grasp_dist)  # [-2, 1]

        grasp_reward, turn_reward_1, turn_reward_2 = 0, 0, 0

        angle_diff = self.target_angle - self.current_angle
        delta_angle = angle_diff - self.last_angle_diff  # <=> last_angle - cur_angle
        if grasp_dist < 0.02:
            # grasp reward
            is_grasped = self.agent.check_grasp(self.target_link, max_angle=30)
            grasp_reward = 1.0 if is_grasped else 0.0

            # reach-goal reward
            if is_grasped:
                ## close, [-2, 2]
                turn_reward_1 = 1 * (1 - np.tanh(max(angle_diff, 0) * 2.0))  # [0, 1]

                if angle_diff > 0:
                    turn_reward_2 = -np.tanh(delta_angle * 5)  # [0, 1]
                else:
                    turn_reward_2 = np.tanh(delta_angle * 5)
                turn_reward_2 *= 1  # [0, 1]

        self.last_angle_diff = angle_diff

        # scale
        reward = 0.5 * (
            approach_reward
            + grasp_reward
            + turn_reward_1
            + turn_reward_2
            + handle_exist_rew
        )

        action = kwargs["action"]
        if isinstance(action, dict):
            cur_action = action["action"]
        elif isinstance(action, np.ndarray):
            cur_action = action
        else:
            raise NotImplementedError
        info_dict = {
            "handle_exist_rew": np.array(handle_exist_rew, dtype=np.float_),
            ### stage 0
            "grasp_dist": np.array(grasp_dist, dtype=np.float_),
            "approach_reward": np.array(approach_reward, dtype=np.float_),
            "grasp_reward": np.array(grasp_reward, dtype=np.float_),
            ### stage 2
            "angle_diff": np.array(angle_diff, dtype=np.float_),
            "turn_reward_1": np.array(turn_reward_1, dtype=np.float_),
            "delta_angle": np.array(delta_angle, dtype=np.float_),
            "turn_reward_2": np.array(turn_reward_2, dtype=np.float_),
            "reward": np.array(reward, dtype=np.float_),
            # other info
            "cur_angle": np.array(self.current_angle, dtype=np.float_),
            "init_angle": np.array(self.init_angle, dtype=np.float_),
            "target_angle": np.array(self.target_angle, dtype=np.float_),
            "action": np.array(cur_action, dtype=np.float_),
        }
        self._cache_info = info_dict

        return reward

    def compute_dense_reward_kitchenpot_aligndrawer(self, **kwargs):
        np.set_printoptions(suppress=True)

        ee_cords = self._agent.sample_ee_coords()  # [2, 10, 3]
        current_handle = apply_pose_to_points(
            self.handle_pcd, self._target_link.get_pose()
        )  # [200, 3]
        ee_to_handle = (
            ee_cords[..., None, :] - current_handle
        )  # [2, 10, 200, 3] = [2, 10, 1, 3] - [200, 3]
        # [2, 10, 200, 3] -> [2, 10, 200](dist) -> [2, 10] -> [2] min_dist between (left_finger, right_finger) <-> handle
        dist_ee_to_handle = np.linalg.norm(ee_to_handle, axis=-1).min(-1).min(-1)
        dist_ee_mid_to_handle = -np.linalg.norm(ee_to_handle.mean(0), axis=-1).min()
        # [-1.5, 0]: (0.01, 0.5) -> [-0.5, 0] + (-0.35, 0.01) => [0, 1]
        close_to_handle_rew = (
            -1
            - dist_ee_to_handle.mean()
            + normalize_and_clip_in_interval(dist_ee_mid_to_handle, -0.35, 0.01)
        )

        grasp_site_pose = self._agent.grasp_site.get_pose()
        target_pose_2 = self._target_link.get_pose() * self.handle_grasp[1]
        grasp_site_pose = self._agent.grasp_site.get_pose()
        gripper_angle_err = (
            np.abs(angle_distance_ms(grasp_site_pose, target_pose_2)) / np.pi
        )
        # direction reward, [-1.5, 0]
        dir_rew = -np.clip(gripper_angle_err, 0, 0.5) * 2

        cabinet_qpos = self._articulation.get_qpos()[self.target_joint_idx]
        cabinet_qvel = self._articulation.get_qvel()[self.target_joint_idx]
        door_vel_norm = min(norm(self.target_link.get_velocity()), 1)  ## [0, 1]
        door_ang_vel_norm = min(
            norm(self.target_link.get_angular_velocity()), 1
        )  ## [0, 1]

        scale = 0.5
        stage_index = 0
        cabinet_qvel_rew = 0
        cabinet_qpos_rew = 0
        cabinet_static_rew = 0
        handle_exist_rew = self.check_handle_exist(kwargs["obs"])
        agent_proprio = self.agent.get_proprioception()
        gripper_qpos = agent_proprio["qpos"][-2:]
        ## open half, [-1, 0]
        gripper_rew = -np.abs(gripper_qpos.mean() - 0.034) / 0.034

        if gripper_angle_err < 0.15:  # 27degree
            dir_rew = 0.5
            if dist_ee_to_handle.max() < 0.03 and dist_ee_mid_to_handle > -0.01:
                stage_index = 1
                # dir_rew = 0.5
                close_to_handle_rew = 0.5
                ## close, [-2, 2]
                gripper_rew = (gripper_qpos.mean() - 0.034) / 0.034 * 2
                cabinet_qvel_rew = (
                    np.clip(cabinet_qvel, -0.5, 0.5) * 2
                )  ## Push vel to positive, [-1, 1], init 0
                cabinet_qpos_rew = (
                    normalize_and_clip_in_interval(
                        cabinet_qpos, 0, self.target_qpos * 1.1
                    )
                    * 2
                )  ## [0, 2]

                cur_qpos = self._articulation.get_qpos()[self.target_joint_idx]
                target_achieved = (
                    (cur_qpos - self.init_angle) / (self.target_angle - self.init_angle)
                ) > 0.5
                if target_achieved:
                    stage_index = 2
                    gripper_rew = 2
                    cabinet_qvel_rew = 2
                    cabinet_static_rew = -(
                        door_vel_norm + door_ang_vel_norm
                    )  ## [-2, 0]

        reward = (
            handle_exist_rew
            + close_to_handle_rew
            + dir_rew
            + gripper_rew
            + cabinet_qvel_rew
            + cabinet_qpos_rew
            + cabinet_static_rew
        ) * scale

        action = kwargs["action"]
        if isinstance(action, dict):
            cur_action = action["action"]
        elif isinstance(action, np.ndarray):
            cur_action = action
        else:
            raise NotImplementedError
        info_dict = {
            "stage_index": np.array(stage_index),
            ### stage 0
            "dist_ee_to_handle": np.array(dist_ee_to_handle, dtype=np.float_),
            "dist_ee_mid_to_handle": np.array(dist_ee_mid_to_handle, dtype=np.float_),
            "gripper_angle_err": np.array(gripper_angle_err * 180, dtype=np.float_),
            "close_to_handle_rew": np.array(close_to_handle_rew, dtype=np.float_),
            "dir_rew": np.array(dir_rew, dtype=np.float_),
            "gripper_rew": np.array(gripper_rew, dtype=np.float_),
            ### stage 1
            "cabinet_qpos": np.array(cabinet_qpos, dtype=np.float_),
            "cabinet_qvel": np.array(cabinet_qvel, dtype=np.float_),
            ##  np.clip(cabinet_qvel, -0.5, 0.5) * 2
            "cabinet_qvel_rew": np.array(cabinet_qvel_rew, dtype=np.float_),
            ## normalize_and_clip_in_interval(cabinet_qpos, 0, self.target_qpos * 1.1) * 2
            "cabinet_qpos_rew": np.array(cabinet_qpos_rew, dtype=np.float_),
            ### stage 2
            "door_vel_norm": np.array(door_vel_norm, dtype=np.float_),
            "door_ang_vel_norm": np.array(door_ang_vel_norm, dtype=np.float_),
            ## -(door_vel_norm + door_ang_vel_norm * 0.5)
            "cabinet_static_rew": np.array(cabinet_static_rew, dtype=np.float_),
            "handle_exist_rew": np.array(handle_exist_rew, dtype=np.float_),
            "reward_raw": np.array(reward, dtype=np.float_),
            "action": np.array(cur_action, dtype=np.float_),
        }
        self._cache_info = info_dict
        return reward

    def step_action(self, action):
        if action is None:  # simulation without action
            pass
        elif isinstance(action, np.ndarray):
            if self._arti_mode in ["faucet", "laptop"]:
                action[-1] = 1  # set gripper close forcibly
            self._agent.set_action(action)
        elif isinstance(action, dict):
            if action["control_mode"] != self._agent.control_mode:
                self._agent.set_control_mode(action["control_mode"])
            if self._arti_mode in ["faucet", "laptop"]:
                action["action"][-1] = 1  # set gripper close forcibly
            self._agent.set_action(action["action"])
        else:
            raise TypeError(type(action))

        for _ in range(self._sim_step_per_control):
            self._agent.simulation_step()
            self._scene.step()

        self._agent.update_generalized_external_forces()

    def get_reward(self, **kwargs):
        if self._reward_mode == "sparse":
            return float(self.check_success())
        elif self._reward_mode == "dense":
            if self._arti_mode == "cabinet_door":
                return self.compute_dense_reward_door(**kwargs)
                # return self.compute_dense_reward_cabinet(**kwargs)
            elif self._arti_mode == "cabinet_drawer":
                return self.compute_dense_reward_drawer(**kwargs)
                # return self.compute_dense_reward_cabinet(**kwargs)
            elif self._arti_mode == "faucet":
                return self.compute_dense_reward_faucet(**kwargs)
            elif self._arti_mode == "laptop":
                return self.compute_dense_reward_laptop(**kwargs)
            elif self._arti_mode == "kitchen_pot":
                return self.compute_dense_reward_kitchenpot(**kwargs)
            else:
                raise NotImplementedError
        else:
            raise NotImplementedError(self._reward_mode)

    def get_info(self, **kwargs):
        info = super().get_info()
        info.update(self._cache_info)
        return info

    def get_state(self) -> np.ndarray:
        state = super().get_state()
        if self._arti_mode in ["cabinet_door", "cabinet_drawer"]:
            return state
        elif self._arti_mode == "faucet":
            return np.hstack([state, self.target_angle])
        else:
            raise NotImplementedError

    def set_state(self, state):
        if self._arti_mode in ["cabinet_door", "cabinet_drawer"]:
            super().set_state(state)
        elif self._arti_mode == "faucet":
            self.target_angle = state[-1]
            super().set_state(state[:-1])
            self.last_angle_diff = self.target_angle - self.current_angle
        else:
            raise NotImplementedError

    @property
    def articulation(self):
        return self._articulation

    @property
    def target_link(self):
        return self._target_link

    @property
    def handle_info(self):
        return self._handle_info

    @property
    def table(self):
        return self._table
