import os
from collections import OrderedDict
from typing import List, Tuple

import numpy as np
import sapien.core as sapien
import yaml
from arti_mani import AGENT_CONFIG_DIR, ASSET_DIR
from arti_mani.agents.camera import get_camera_rgb, get_camera_seg, get_texture
from arti_mani.agents.floating_robotiq import FloatingRobotiq
from arti_mani.envs.base import BaseEnv
from arti_mani.utils.articulation_utils import (
    get_actor_state,
    get_articulation_state,
    get_entity_by_name,
    ignore_collision,
    set_actor_state,
    set_articulation_state,
)
from arti_mani.utils.common import (
    convert_np_bool_to_float,
    np_random,
    random_choice,
    register_gym_env,
)
from arti_mani.utils.contrib import apply_pose_to_points, o3d_to_trimesh, xyz2uv
from arti_mani.utils.geometry import angle_distance, transform_points
from arti_mani.utils.o3d_utils import merge_mesh, np2mesh
from arti_mani.utils.sapien_utils import (
    get_entity_by_name,
    hex2rgba,
    look_at,
    set_actor_render_material,
    set_articulation_render_material,
    set_cabinet_render_material,
)
from arti_mani.utils.trimesh_utils import get_actor_mesh
from transforms3d.euler import euler2quat


class FloatingRobotiqEnv(BaseEnv):
    SUPPORTED_OBS_MODES: Tuple[str] = ()
    SUPPORTED_REWARD_MODES: Tuple[str] = ()
    SUPPORTED_CONTROL_MODES: Tuple[str] = ()
    _agent: FloatingRobotiq

    def __init__(
        self,
        obs_mode=None,
        reward_mode=None,
        control_mode=None,
        irsensor_mode=False,
        sim_freq=500,
        control_freq=20,
    ):
        self._articulation = None
        self.step_in_ep = 0
        self._prev_actor_pose = sapien.Pose()
        self._cache_obs_state_dict: OrderedDict = (
            OrderedDict()
        )  # save obs state dict to save reward computation time
        self._cache_info = {}
        self.irsensor_mode = irsensor_mode
        self.height, self.width = 144, 256
        self.texture_filelist = os.listdir(ASSET_DIR / "textures/random_textures")
        self.colortexture_filelist = os.listdir(ASSET_DIR / "textures/color_textures")
        super().__init__(obs_mode, reward_mode, control_mode, sim_freq, control_freq)

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

    def _random_lighting(self):
        for light in self._scene.get_all_lights():
            if type(light) in [sapien.PointLightEntity, sapien.DirectionalLightEntity]:
                self._scene.remove_light(light)
        pointlight_num = self._episode_rng.choice(np.arange(1, 4))
        direclight_num = self._episode_rng.choice(np.arange(1, 4))
        for num in range(pointlight_num):
            self._scene.add_point_light(
                [
                    self._episode_rng.uniform(-1, 1),
                    self._episode_rng.uniform(-0.5, 0.5),
                    self._episode_rng.uniform(1, 3),
                ],
                [self._episode_rng.uniform(0.5, 1)] * 3,
                shadow=True,
            )
        for num in range(direclight_num):
            self._scene.add_directional_light(
                [
                    self._episode_rng.uniform(0.1, 1),
                    self._episode_rng.uniform(-1, 1),
                    self._episode_rng.uniform(-1, -0.25),
                ],
                [self._episode_rng.uniform(0.5, 1)] * 3,
                shadow=True,
            )

    def _domain_random(self):
        # self._random_lighting()
        if self._episode_rng.rand() > 0.2:
            rand_img_file = self._episode_rng.choice(self.texture_filelist, 1)[0]
            set_actor_render_material(
                self._table,
                metallic=self._episode_rng.uniform(0.4, 0.9),  # 1, 0
                roughness=self._episode_rng.uniform(0.4, 0.9),  # 0.4, 0.9
                specular=self._episode_rng.uniform(0.01, 0.8),  # /, 0.0
                diffuse_texture_filename=str(
                    ASSET_DIR / f"textures/random_textures/{rand_img_file}"
                ),
            )

    def _load_articulations(self):
        pass

    def _load_agent(self):
        self._agent = FloatingRobotiq.from_config_file(
            AGENT_CONFIG_DIR / "floating_robotiq_low_res.yml",
            self._scene,
            self._control_freq,
        )

    def _initialize_articulations(self):
        pass

    def _initialize_agent(self):
        qpos = self._episode_rng.uniform(0, 0.068, [2])
        self._agent._robot.set_qpos(qpos)
        self._agent._robot.set_pose(sapien.Pose([-0.6, 0.4, 0], [1, 0, 0, 0]))

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
        self._load_articulations()
        self._load_agent()
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

    # ---------------------------------------------------------------------------- #
    # Visualization
    # ---------------------------------------------------------------------------- #
    def _setup_lighting(self):
        self._scene.set_ambient_light([0.3, 0.3, 0.3])
        self._scene.add_point_light([2, 2, 2], [1, 1, 1])
        self._scene.add_point_light([2, -2, 2], [1, 1, 1])
        self._scene.add_point_light([-2, 0, 2], [1, 1, 1])
        self._scene.add_point_light([1, -1, 1], [1, 1, 1])
        self._scene.add_directional_light([1, -1, -1], [0.3, 0.3, 0.3])
        self._scene.add_directional_light([0, 0, -1], [1, 1, 1])

    def _setup_viewer(self):
        super()._setup_viewer()
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


@register_gym_env("CabinetDoor-v0", max_episode_steps=200)
class CabinetDoor(FloatingRobotiqEnv):
    SUPPORTED_OBS_MODES = ("state", "state_dict", "state_egorgbd", "egorgbd_pc")
    SUPPORTED_REWARD_MODES = ("dense", "sparse")
    SUPPORTED_CONTROL_MODES = ("pd_joint_delta_pos", "pd_ee_delta_pose")
    DEFAULT_MODEL_PATH = (
        ASSET_DIR / "partnet_mobility_configs/fixed_cabinet_doors_new.yml"
    )

    def __init__(
        self,
        articulation_ids: List[int] = (),
        articulation_config_path="",
        obs_mode=None,
        reward_mode=None,
        irsensor_mode=False,
        other_handle_visible=False,
        domain_random=False,
        sim_freq=500,
        control_freq=20,
        control_mode=None,
    ):
        if articulation_config_path is None:
            articulation_config_path = self.DEFAULT_MODEL_PATH
        with open(articulation_config_path, "r") as f:
            self._articulation_info = yaml.safe_load(f)
        if isinstance(articulation_ids, int):
            articulation_ids = [articulation_ids]
        if len(articulation_ids) == 0:
            articulation_ids = sorted(self._articulation_info.keys())
        self.articulation_ids = articulation_ids
        self.articulation_id = None
        self.other_handle_visible = other_handle_visible
        self.domain_random = domain_random

        super().__init__(
            obs_mode,
            reward_mode,
            control_mode,
            irsensor_mode,
            sim_freq,
            control_freq,
        )

    def _domain_random(self):
        self._scene.remove_articulation(self._articulation)
        self._load_articulations()
        self._initialize_articulations()
        if self._episode_rng.rand() > 0.2:
            color_rand_val = self._episode_rng.uniform(0, 1)
            a_rand_val = self._episode_rng.uniform(0.9, 1)
            rand_img_file = self._episode_rng.choice(self.texture_filelist, 1)[0]
            rand_color_file = self._episode_rng.choice(self.colortexture_filelist, 1)[0]
            set_cabinet_render_material(
                self._articulation,
                # color=[color_rand_val, color_rand_val, color_rand_val, a_rand_val],
                color=[
                    self._episode_rng.uniform(0, 1),
                    self._episode_rng.uniform(0, 1),
                    self._episode_rng.uniform(0, 1),
                    1,
                ],  # hex2rgba("#AAAAAA")<=>[0.40982574, 0.40982574, 0.40982574, 1.], [0.8, 0.8, 0.8, 1]
                color_texture_filename=str(
                    ASSET_DIR / f"textures/color_textures/{rand_color_file}"
                ),
                metallic=self._episode_rng.uniform(0, 0.5),  # 1, 0
                roughness=self._episode_rng.uniform(0, 1),  # 0.4, 0.9
                specular=self._episode_rng.uniform(0, 0.8),  # /, 0.0
                diffuse_texture_filename=str(
                    ASSET_DIR / f"textures/random_textures/{rand_img_file}"
                ),
            )
        super()._domain_random()

    def _initialize_agent(self):
        qpos = self._episode_rng.uniform(0, 0.068, [2])
        self._agent._robot.set_qpos(qpos)
        self._agent._robot.set_pose(
            look_at(
                eye=[-0.3, 0, 0.5],
                target=self._articulation.get_root_pose().p,
                up=[0, 0, 1],
            )
        )

    def _load_cabinet(self):
        # with open(self._articulation_config_path.joinpath(self.articulation_id+".yml"), "r") as f:
        #     config_dict = yaml.safe_load(f)
        #     self._articulation_config = ArticulationConfig(**config_dict)
        loader = self._scene.create_urdf_loader()
        loader.load_multiple_collisions_from_file = self._articulation_config[
            "multiple_collisions"
        ]
        loader.scale = self._articulation_config["scale"]
        loader.fix_root_link = True

        articulation_path = (
            ASSET_DIR / f"partnet_mobility_dataset/{self.articulation_id}"
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

    def _load_articulations(self):
        self._articulation = self._load_cabinet()

        # set physical properties for all the joints
        joint_friction = self._episode_rng.uniform(
            self._articulation_config["joint_friction_range"][0],
            self._articulation_config["joint_friction_range"][1],
        )
        joint_stiffness = self._episode_rng.uniform(
            self._articulation_config["joint_stiffness_range"][0],
            self._articulation_config["joint_stiffness_range"][1],
        )
        joint_damping = self._episode_rng.uniform(
            self._articulation_config["joint_damping_range"][0],
            self._articulation_config["joint_damping_range"][1],
        )
        for joint in self._articulation.get_active_joints():
            joint.set_friction(joint_friction)
            joint.set_drive_property(joint_stiffness, joint_damping)
        ignore_collision(self._articulation)

        self._set_semantic_visid()

    def _set_semantic_visid(self):
        all_links = self._articulation.get_links()
        all_active_joints = self._articulation.get_active_joints()

        # set target link & joint, handle & door of target_link
        self._target_joint_idx = self._articulation_config["target_joint_idx"]
        self._target_joint = all_active_joints[self._target_joint_idx]
        self._target_link = self._target_joint.get_child_link()
        # all_links.remove(self._target_link)

        door_links = []
        for joint in all_active_joints:
            link = joint.get_child_link()
            door_links.append(link)

        self.handles_visid = []
        self.doors_visid = []
        self.cabinet_visid = []
        for link in all_links:
            if link in door_links:
                for visual_body in link.get_visual_bodies():
                    if "handle" in visual_body.get_name():
                        self.handles_visid.append(visual_body.get_visual_id())
                        if link != self._target_link:
                            visual_body.set_visibility(
                                1.0 if self.other_handle_visible else 0.0
                            )
                    else:
                        self.doors_visid.append(visual_body.get_visual_id())
            else:
                for visual_body in link.get_visual_bodies():
                    self.cabinet_visid.append(visual_body.get_visual_id())

    def _initialize_articulations(self):
        # sim init
        posz = (
            -self._articulation_config["scale"]
            * self._articulation_config["bbox_min"][2]
        )
        if self.articulation_id == 0:
            posx, posy, rotz = 0.152, -0.188, 0.7
        elif self.articulation_id == 1:
            posx, posy, rotz = 0.237, -0.14, 0
        else:
            posx, posy, rotz = 0, 0, 0
        arti_root_pose = sapien.Pose(
            [posx + 0.6, posy, posz], [np.sqrt(1 - rotz**2), 0, 0, rotz]
        )
        if self._articulation_config["flip"]:
            arti_root_pose = arti_root_pose * sapien.Pose(q=np.array([0, 1, 0, 0]))
        self._articulation.set_root_pose(arti_root_pose)

        # init_open_extent = self._episode_rng.uniform(0, self._articulation_config["init_open_extent_range"])
        init_open_extent = self._episode_rng.uniform(0, 0.1, self._articulation.dof)
        qpos = np.zeros(self._articulation.dof)
        for i in range(self._articulation.dof):
            [[lmin, lmax]] = self._articulation.get_active_joints()[i].get_limits()
            qpos[i] = lmin + (lmax - lmin) * init_open_extent[i]
            # print(f"++++ {self.articulation_id}, activejoint_{i}", lmin, lmax)
        ## sim init
        self._articulation.set_qpos(qpos)

        # [[lmin, lmax]] = self._target_joint.get_limits()
        lmin, lmax = self._articulation_config["target_joint_range"]
        # print("++++", lmin, lmax)
        self.target_qpos = (
            lmin + (lmax - lmin) * self._articulation_config["open_extent"]
        )

        self._get_handle_info_in_target_link()

    def _get_handle_info_in_target_link(self):
        handle_meshes = []
        for visual_body in self._target_link.get_visual_bodies():
            for render_shape in visual_body.get_render_shapes():
                vertices = apply_pose_to_points(
                    render_shape.mesh.vertices * visual_body.scale,
                    visual_body.local_pose,
                )
                shape_mesh = np2mesh(vertices, render_shape.mesh.indices.reshape(-1, 3))
                if "handle" in visual_body.get_name():
                    handle_meshes.append(shape_mesh)
        handle_mesh = merge_mesh(handle_meshes)
        handle_trimesh = o3d_to_trimesh(handle_mesh)
        handle_pcd = handle_trimesh.sample(300)  ## target_link frame, (300, 3)
        # get tip, med, bottom keypoints
        xyzmax_val, xyzmin_val = handle_pcd.max(0), handle_pcd.min(0)
        xyzmax_id, xyzmin_id = np.argmax(handle_pcd, 0), np.argmin(handle_pcd, 0)
        bbox_size = xyzmax_val - xyzmin_val
        long_side = np.argmax(bbox_size)
        tip, bottom = handle_pcd[xyzmax_id[long_side]], handle_pcd[xyzmin_id[long_side]]
        center = handle_pcd.mean(0)
        if self.articulation_id in [0, 1]:
            center[1] = xyzmax_val[1]
        else:
            center[2] = xyzmax_val[2]
        self._handle_keypoints = np.stack([tip, center, bottom])
        self._handle_pcd = handle_pcd

    def reset(self, seed=None, reconfigure=False, articulation_id=None):
        self.set_episode_rng(seed)
        _reconfigure = self._set_model(articulation_id)
        reconfigure = _reconfigure or reconfigure
        ret = super().reset(seed=self._episode_seed, reconfigure=reconfigure)
        return ret

    def _set_model(self, articulation_id):
        """Set the model id and scale. If not provided, choose one randomly."""
        reconfigure = False

        # Model ID
        if articulation_id is None:
            articulation_id = random_choice(self.articulation_ids, self._episode_rng)
        if articulation_id != self.articulation_id:
            reconfigure = True
        self.articulation_id = articulation_id
        self._articulation_config = self._articulation_info[self.articulation_id]

        return reconfigure

    def get_obs_state_egorgbd(self):
        ### get state
        state_rgbd = self._get_obs_state_dict()

        ### get visual info
        self.update_render()
        hand_sensor = self.agent._sensors["hand"]
        sensor_rgb = hand_sensor._cam_rgb
        hand_sensor.take_picture()
        hand_sensor.compute_depth()

        rgb = (hand_sensor.get_rgb() * 255).astype(np.uint8)  # [H, W, 3]
        ir_l, ir_r = hand_sensor.get_ir()  # [H, W]
        depth = hand_sensor.get_depth()  # [H, W]
        clean_depth = -get_texture(sensor_rgb, "Position")[..., 2]  # [H, W]
        visual_id_seg = get_camera_seg(sensor_rgb)[..., 0]  # [H, W]
        imh, imw = visual_id_seg.shape

        state_rgbd["rgb"] = rgb.transpose((2, 0, 1))
        if self.irsensor_mode:
            state_rgbd["ir_l"] = ir_l.astype(np.float32)
            state_rgbd["ir_r"] = ir_r.astype(np.float32)
            state_rgbd["depth"] = depth.astype(np.float32)
            state_rgbd["clean_depth"] = clean_depth.astype(np.float32)
        else:
            state_rgbd["clean_depth"] = depth.astype(np.float32)

        seg_visids = [self.handles_visid, self.doors_visid, self.cabinet_visid]
        cam_seg = np.ones(visual_id_seg.shape, dtype=np.float32) * 5
        seg_id = [0, 1, 2]
        for id, seg_visid in enumerate(seg_visids):
            mask = np.zeros(visual_id_seg.shape, dtype=np.bool)
            for visual_id in seg_visid:
                mask = mask | (visual_id_seg == visual_id)
            cam_seg[mask] = seg_id[id]
        state_rgbd["seg"] = convert_np_bool_to_float(cam_seg).astype(np.uint8)  # (H, W)
        background_seg = np.zeros(visual_id_seg.shape, dtype=np.bool)
        bg_visids = [
            0,
            self._bg_front.get_id(),
            self._bg_left.get_id(),
            self._bg_right.get_id(),
        ]
        for bg_visid in bg_visids:
            background_seg[visual_id_seg == bg_visid] = True
        state_rgbd["bg_seg"] = convert_np_bool_to_float(background_seg).astype(
            np.uint8
        )  # (H, W)
        # got pcd target_link => world => ee
        # handle_pcd_world = apply_pose_to_points(self._handle_pcd, self._target_link.get_pose())  ## world frame
        kpts_world = apply_pose_to_points(
            self._handle_keypoints, self._target_link.get_pose()
        )  ## world frame
        # handle_pcd_ee = apply_pose_to_points(handle_pcd_world, self.agent.grasp_site.get_pose().inv())  ## ee frame
        # kpts_ee = apply_pose_to_points(handle_keypoints_world, self.agent.grasp_site.get_pose().inv())  ## ee frame
        # kpts_cam = apply_pose_to_points(kpts_ee, sensor_rgb.get_pose().inv())  ## cam frame
        kpts_cam = transform_points(sensor_rgb.get_extrinsic_matrix(), kpts_world)
        uvz = xyz2uv(kpts_cam, sensor_rgb.get_camera_matrix())
        u_visable = (uvz[:, 0] >= 0) & (uvz[:, 0] <= imw - 1)
        v_visable = (uvz[:, 1] >= 0) & (uvz[:, 1] <= imh - 1)
        uvz_visable = np.stack(
            [u_visable, v_visable, u_visable & v_visable]
        ).T  # (3, 3)
        for ind in range(uvz_visable.shape[0]):
            if uvz_visable[ind, 2]:  # not out of img range
                u, v = round(uvz[ind, 0]), round(uvz[ind, 1])
                if cam_seg[v, u] not in seg_id:  # handle keypoint is occluded
                    uvz_visable[ind, 2] = False
        state_rgbd["kpts"] = kpts_cam.astype(np.float32)
        state_rgbd["uvz"] = uvz.astype(np.float32)
        state_rgbd["uvz_visable"] = uvz_visable
        return state_rgbd

    def get_obs_egorgbd_pc(self):
        ### get state
        rgbd_pc = OrderedDict()

        ### get visual info
        self.update_render()
        hand_sensor = self.agent._sensors["hand"]
        sensor_rgb = hand_sensor._cam_rgb
        hand_sensor.take_picture()
        hand_sensor.compute_depth()

        rgb = (hand_sensor.get_rgb() * 255).astype(np.uint8)  # [H, W, 3]
        ir_l, ir_r = hand_sensor.get_ir()  # [H, W]
        depth = hand_sensor.get_depth()  # [H, W]
        clean_depth = -get_texture(sensor_rgb, "Position")[..., 2]  # [H, W]
        visual_id_seg = get_camera_seg(sensor_rgb)[..., 0]  # [H, W]
        imh, imw = visual_id_seg.shape

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
        rgbd_pc["world_xyz"] = world_xyz

        rgbd_pc["rgb"] = rgb.transpose((2, 0, 1))
        if self.irsensor_mode:
            rgbd_pc["ir_l"] = ir_l.astype(np.float32)
            rgbd_pc["ir_r"] = ir_r.astype(np.float32)
            rgbd_pc["depth"] = depth.astype(np.float32)
            rgbd_pc["clean_depth"] = clean_depth.astype(np.float32)
        else:
            rgbd_pc["clean_depth"] = depth.astype(np.float32)

        seg_visids = [self.handles_visid, self.doors_visid, self.cabinet_visid]
        cam_seg = np.ones(visual_id_seg.shape, dtype=np.float32) * 5
        seg_id = [0, 1, 2]
        for id, seg_visid in enumerate(seg_visids):
            mask = np.zeros(visual_id_seg.shape, dtype=np.bool)
            for visual_id in seg_visid:
                mask = mask | (visual_id_seg == visual_id)
            cam_seg[mask] = seg_id[id]
        rgbd_pc["seg"] = convert_np_bool_to_float(cam_seg).astype(np.uint8)  # (H, W)
        background_seg = np.zeros(visual_id_seg.shape, dtype=np.bool)
        bg_visids = [
            0,
            self._bg_front.get_id(),
            self._bg_left.get_id(),
            self._bg_right.get_id(),
        ]
        for bg_visid in bg_visids:
            background_seg[visual_id_seg == bg_visid] = True
        rgbd_pc["bg_seg"] = convert_np_bool_to_float(background_seg).astype(
            np.uint8
        )  # (H, W)
        return rgbd_pc

    def get_obs(self):
        if self.domain_random:
            self._domain_random()
        if self._obs_mode == "state_dict":
            state_dict = self._get_obs_state_dict()
            return state_dict
        elif self._obs_mode == "state_egorgbd":
            state_rgbd = self.get_obs_state_egorgbd()
            return state_rgbd
        elif self._obs_mode == "egorgbd_pc":
            rgbd_pc = self.get_obs_egorgbd_pc()
            return rgbd_pc
        else:
            raise NotImplementedError(self._obs_mode)


@register_gym_env("CabinetDrawer-v0", max_episode_steps=200)
class CabinetDrawer(CabinetDoor):
    DEFAULT_MODEL_PATH = (
        ASSET_DIR / "partnet_mobility_configs/fixed_cabinet_drawers_new.yml"
    )

    def __init__(self, *args, **kwargs):
        super(CabinetDrawer, self).__init__(*args, **kwargs)


@register_gym_env("Faucet-v0", max_episode_steps=200)
class Faucet(FloatingRobotiqEnv):
    SUPPORTED_OBS_MODES = ("state", "state_dict", "state_egorgbd", "egorgbd_pc")
    SUPPORTED_REWARD_MODES = ("dense", "sparse")
    SUPPORTED_CONTROL_MODES = ("pd_joint_delta_pos", "pd_ee_delta_pose")
    DEFAULT_MODEL_PATH = ASSET_DIR / "partnet_mobility_configs/fixed_faucets_new.yml"

    def __init__(
        self,
        articulation_ids: List[int] = (),
        articulation_config_path="",
        obs_mode=None,
        reward_mode=None,
        irsensor_mode=False,
        other_handle_visible=False,
        domain_random=False,
        sim_freq=500,
        control_freq=20,
        control_mode=None,
    ):
        if articulation_config_path is None:
            articulation_config_path = self.DEFAULT_MODEL_PATH
        with open(articulation_config_path, "r") as f:
            self._articulation_info = yaml.safe_load(f)
        if isinstance(articulation_ids, int):
            articulation_ids = [articulation_ids]
        if len(articulation_ids) == 0:
            articulation_ids = sorted(self._articulation_info.keys())
        self.articulation_ids = articulation_ids
        self.articulation_id = None
        self.articulation_scale = None
        self.domain_random = domain_random

        super().__init__(
            obs_mode,
            reward_mode,
            control_mode,
            irsensor_mode,
            sim_freq,
            control_freq,
        )

    def _domain_random(self):
        self._scene.remove_articulation(self._articulation)
        self._load_articulations()
        self._initialize_articulations()
        if self._episode_rng.rand() > 0.2:
            rand_img_file = self._episode_rng.choice(self.texture_filelist, 1)[0]
            set_articulation_render_material(
                self._articulation,
                metallic=self._episode_rng.uniform(0.4, 1),  # 1, 0
                roughness=self._episode_rng.uniform(0.4, 0.9),  # 0.4, 0.9
                specular=self._episode_rng.uniform(0.0, 0.8),  # /, 0.0
                diffuse_texture_filename=str(
                    ASSET_DIR / f"textures/random_textures/{rand_img_file}"
                ),
            )
        super()._domain_random()

    def _initialize_agent(self):
        qpos = self._episode_rng.uniform(0, 0.068, [2])
        self._agent._robot.set_qpos(qpos)
        self._agent._robot.set_root_pose(
            look_at(eye=[-0.2, 0, 0.8], target=[-0.1, 0, 0], up=[0, 0, 1])
        )

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
        """Set the articulation id and scale. If not provided, choose one randomly."""
        reconfigure = False

        # Model ID
        if articulation_id is None:
            articulation_id = random_choice(self.articulation_ids, self._episode_rng)
        if articulation_id != self.articulation_id:
            reconfigure = True
        self.articulation_id = articulation_id
        self._articulation_config = self._articulation_info[self.articulation_id]

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
            self.articulation_offset = np.float32(self._articulation_config["offset"])
        else:
            bbox_min = self._articulation_config["bbox_min"]
            self.articulation_offset = -np.float32(bbox_min) * articulation_scale
        # Add a small clearance
        self.articulation_offset[2] += 0.01

        return reconfigure

    def _load_articulations(self):
        self._articulation = self._load_faucet()
        # Cache qpos to restore
        self._articulation_init_qpos = self._articulation.get_qpos()

        # Set friction and damping for all joints
        for joint in self._articulation.get_active_joints():
            joint.set_friction(1.0)
            joint.set_drive_property(0.0, 10.0)

        self._set_switch_links()

    def _load_faucet(self):
        loader = self._scene.create_urdf_loader()
        loader.scale = self.articulation_scale
        loader.fix_root_link = True

        model_dir = ASSET_DIR / f"partnet_mobility_dataset/{self.articulation_id}"
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

    def _set_switch_links(self):
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
            joint.set_friction(0.1)
            joint.set_drive_property(0.0, 2.0)
            self.switch_joints.append(joint)

    def _initialize_articulations(self):
        p = np.zeros(3)
        p[2] = self.articulation_offset[2]
        ori = self._episode_rng.uniform(-np.pi / 12, np.pi / 12)
        q = euler2quat(0, 0, ori)
        self._articulation.set_pose(sapien.Pose(p, q))

        # init_open_extent = 1
        init_open_extent = self._episode_rng.uniform(0, 1)
        # lmin, lmax = 0, 3.14
        qpos = np.zeros(self._articulation.dof)
        for i, joint in enumerate(self.switch_joints):
            [[lmin, lmax]] = joint.get_limits()
            lmin = 0 if np.isinf(lmin) else lmin
            lmax = 3.14 if np.isinf(lmax) else lmax
            # print(f"{self.articulation_id} joint limits: {lmin, lmax}")
            qpos[i] = lmin + (lmax - lmin) * init_open_extent
        self._articulation.set_qpos(qpos)
        self._set_target_link()

    def _set_target_link(self):
        n_switch_links = len(self.switch_link_names)
        idx = random_choice(np.arange(n_switch_links), self._episode_rng)

        self._target_link_name = self.switch_link_names[idx]
        self._target_link: sapien.Link = self.switch_links[idx]
        self._target_joint: sapien.Joint = self.switch_joints[idx]
        self._target_joint_idx = self._articulation.get_active_joints().index(
            self._target_joint
        )

        # x-axis is the revolute joint direction
        assert self._target_joint.type == "revolute", self._target_joint.type
        joint_pose = self._target_joint.get_global_pose().to_transformation_matrix()
        self.target_joint_axis = joint_pose[:3, 0]

        self.target_link_mesh = self.switch_links_mesh[idx]
        with np_random(self._episode_seed):
            self._sw_pcd = self.target_link_mesh.sample(1000)

        # get tip, med, bottom keypoints
        if self.articulation_id in [5024, 5034]:
            norm_sw_pcd = apply_pose_to_points(
                self._sw_pcd, self._target_joint.get_pose_in_parent().inv()
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
            self._sw_keypoints = apply_pose_to_points(
                np.stack([tip, center, bottom]), self._target_joint.get_pose_in_parent()
            )
        else:
            xyzmax_val, xyzmin_val = self._sw_pcd.max(0), self._sw_pcd.min(0)
            xyzmax_id, xyzmin_id = np.argmax(self._sw_pcd, 0), np.argmin(
                self._sw_pcd, 0
            )
            bbox_size = xyzmax_val - xyzmin_val
            long_side = np.argmax(bbox_size)
            tip = self._sw_pcd[xyzmax_id[long_side]]
            # bottom = self._sw_pcd[xyzmin_id[long_side]]
            # center = self._sw_pcd.mean(0)
            # center[1] = xyzmax_val[1]
            joint_pts_ids = np.where(
                np.linalg.norm(self._sw_pcd[:, [0, 2]], axis=1) < 0.02
            )  # dist to joint axis < 0.01
            joint_pts = self._sw_pcd[joint_pts_ids][:, 1]
            joint_pts_max = joint_pts.max()
            center = np.array([0, joint_pts_max, 0])
            bottom = np.array([0, self._sw_pcd[:, 1].min(), 0])
            self._sw_keypoints = np.stack([tip, center, bottom])

    def get_obs_state_egorgbd(self):
        ### get state
        state_rgbd = self._get_obs_state_dict()

        ### get visual info
        self.update_render()
        hand_sensor = self.agent._sensors["hand"]
        sensor_rgb = hand_sensor._cam_rgb
        hand_sensor.take_picture()
        hand_sensor.compute_depth()

        rgb = (hand_sensor.get_rgb() * 255).astype(np.uint8)  # [H, W, 3]
        ir_l, ir_r = hand_sensor.get_ir()  # [H, W]
        depth = hand_sensor.get_depth()  # [H, W]
        clean_depth = -get_texture(sensor_rgb, "Position")[..., 2]  # [H, W]
        visual_id_seg = get_camera_seg(sensor_rgb)[..., 0]  # [H, W]
        imh, imw = visual_id_seg.shape

        state_rgbd["rgb"] = rgb.transpose((2, 0, 1))
        if self.irsensor_mode:
            state_rgbd["ir_l"] = ir_l.astype(np.float32)
            state_rgbd["ir_r"] = ir_r.astype(np.float32)
            state_rgbd["depth"] = depth.astype(np.float32)
            state_rgbd["clean_depth"] = clean_depth.astype(np.float32)
        else:
            state_rgbd["clean_depth"] = depth.astype(np.float32)

        cam_seg = np.ones(visual_id_seg.shape, dtype=np.float32) * 5
        seg_visids = [self.switch_links_visid, self.fix_link_visid]
        seg_id = [3, 4]
        for id, seg_visid in enumerate(seg_visids):
            mask = np.zeros(visual_id_seg.shape, dtype=np.bool)
            for visual_id in seg_visid:
                mask = mask | (visual_id_seg == visual_id)
            cam_seg[mask] = seg_id[id]
        state_rgbd["seg"] = convert_np_bool_to_float(cam_seg).astype(np.uint8)  # (H, W)
        background_seg = np.zeros(visual_id_seg.shape, dtype=np.bool)
        bg_visids = [
            0,
            self._bg_front.get_id(),
            self._bg_left.get_id(),
            self._bg_right.get_id(),
        ]
        for bg_visid in bg_visids:
            background_seg[visual_id_seg == bg_visid] = True
        state_rgbd["bg_seg"] = convert_np_bool_to_float(background_seg).astype(
            np.uint8
        )  # (H, W)
        # got pcd target_link => world => ee
        # sw_pcd_world = apply_pose_to_points(self._sw_pcd, self._target_link.get_pose())  ## world frame
        kpts_world = apply_pose_to_points(
            self._sw_keypoints, self._target_link.get_pose()
        )  ## world frame
        # sw_pcd_ee = apply_pose_to_points(sw_pcd_world, self.agent.grasp_site.get_pose().inv())  ## world frame
        # sw_kpts_ee = apply_pose_to_points(sw_kpts_world, self.agent.grasp_site.get_pose().inv())  ## ee frame
        kpts_cam = transform_points(sensor_rgb.get_extrinsic_matrix(), kpts_world)
        uvz = xyz2uv(kpts_cam, sensor_rgb.get_camera_matrix())
        u_visable = (uvz[:, 0] >= 0) & (uvz[:, 0] <= imw - 1)
        v_visable = (uvz[:, 1] >= 0) & (uvz[:, 1] <= imh - 1)
        uvz_visable = np.stack(
            [u_visable, v_visable, u_visable & v_visable]
        ).T  # (3, 3)
        for ind in range(uvz_visable.shape[0]):
            if uvz_visable[ind, 2]:  # not out of img range
                u, v = round(uvz[ind, 0]), round(uvz[ind, 1])
                if cam_seg[v, u] not in seg_id:  # handle keypoint is occluded
                    uvz_visable[ind, 2] = False
        state_rgbd["kpts"] = kpts_cam.astype(np.float32)
        state_rgbd["uvz"] = uvz.astype(np.float32)
        state_rgbd["uvz_visable"] = uvz_visable
        return state_rgbd

    def get_obs_egorgbd_pc(self):
        rgbd_pc = OrderedDict()

        ### get visual info
        self.update_render()
        hand_sensor = self.agent._sensors["hand"]
        sensor_rgb = hand_sensor._cam_rgb
        hand_sensor.take_picture()
        hand_sensor.compute_depth()

        rgb = (hand_sensor.get_rgb() * 255).astype(np.uint8)  # [H, W, 3]
        ir_l, ir_r = hand_sensor.get_ir()  # [H, W]
        depth = hand_sensor.get_depth()  # [H, W]
        clean_depth = -get_texture(sensor_rgb, "Position")[..., 2]  # [H, W]
        visual_id_seg = get_camera_seg(sensor_rgb)[..., 0]  # [H, W]
        imh, imw = visual_id_seg.shape

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
        rgbd_pc["world_xyz"] = world_xyz

        rgbd_pc["rgb"] = rgb.transpose((2, 0, 1))
        if self.irsensor_mode:
            rgbd_pc["ir_l"] = ir_l.astype(np.float32)
            rgbd_pc["ir_r"] = ir_r.astype(np.float32)
            rgbd_pc["depth"] = depth.astype(np.float32)
            rgbd_pc["clean_depth"] = clean_depth.astype(np.float32)
        else:
            rgbd_pc["clean_depth"] = depth.astype(np.float32)

        background_seg = np.zeros(visual_id_seg.shape, dtype=np.bool)
        bg_visids = [
            0,
            self._bg_front.get_id(),
            self._bg_left.get_id(),
            self._bg_right.get_id(),
        ]
        for bg_visid in bg_visids:
            background_seg[visual_id_seg == bg_visid] = True
        rgbd_pc["bg_seg"] = convert_np_bool_to_float(background_seg).astype(
            np.uint8
        )  # (H, W)

        cam_seg = np.ones(visual_id_seg.shape, dtype=np.float32) * 5
        seg_visids = [self.switch_links_visid, self.fix_link_visid]
        seg_id = [3, 4]
        for id, seg_visid in enumerate(seg_visids):
            mask = np.zeros(visual_id_seg.shape, dtype=np.bool)
            for visual_id in seg_visid:
                mask = mask | (visual_id_seg == visual_id)
            cam_seg[mask] = seg_id[id]
        rgbd_pc["seg"] = convert_np_bool_to_float(cam_seg).astype(np.uint8)  # (H, W)
        return rgbd_pc

    def get_obs(self):
        if self.domain_random:
            self._domain_random()
        if self._obs_mode == "state_dict":
            return self._get_obs_state_dict()
        elif self._obs_mode == "state_egorgbd":
            return self.get_obs_state_egorgbd()
        elif self._obs_mode == "egorgbd_pc":
            return self.get_obs_egorgbd_pc()
        else:
            raise NotImplementedError(self._obs_mode)
