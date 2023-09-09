import numpy as np
import sapien.core as sapien
from arti_mani.agents.controllers.base_controller import BaseController
from transforms3d.quaternions import axangle2quat, qmult


class FloatGripperPoseController(BaseController):
    def __init__(
        self, controller_config: dict, robot: sapien.Articulation, control_freq: int
    ):
        super().__init__(controller_config, robot, control_freq)
        self.action_dimension = 6
        self.control_type = "pos"
        assert not self.interpolate, "Do NOT support interpolation"

        self.ee_pos_min = self.nums2array(controller_config["ee_pos_min"], 3)
        self.ee_pos_max = self.nums2array(controller_config["ee_pos_max"], 3)
        self.ee_rot_min = self.nums2array(controller_config["ee_rot_min"], 3)
        self.ee_rot_max = self.nums2array(controller_config["ee_rot_max"], 3)
        self.joint_stiffness = self.nums2array(
            controller_config["joint_stiffness"], self.num_control_joints
        )
        self.joint_damping = self.nums2array(
            controller_config["joint_damping"], self.num_control_joints
        )
        self.joint_friction = self.nums2array(
            controller_config["joint_friction"], self.num_control_joints
        )

        # self.pmodel = self.robot.create_pinocchio_model()
        self.qmask = np.zeros(self.robot.dof)
        self.qmask[self.control_joint_index] = 1
        self.target_joint_pose = self._get_curr_joint_pos()

    def set_joint_drive_property(self):
        for j_idx, j in enumerate(self.control_joints):
            j.set_drive_property(self.joint_stiffness[j_idx], self.joint_damping[j_idx])
            j.set_friction(self.joint_friction[j_idx])

    def reset(self):
        self.target_joint_pose = self._get_curr_joint_pos()

    @property
    def action_range(self) -> np.ndarray:
        lower_bound = np.hstack([self.ee_pos_min, self.ee_rot_min])
        upper_bound = np.hstack([self.ee_pos_max, self.ee_rot_max])
        return np.stack([lower_bound, upper_bound], axis=1)

    def set_action(self, action: np.ndarray):
        assert action.shape[0] == self.action_dimension
        target_ee_pos = self.end_link.pose.p + action[0:3]

        angle = np.linalg.norm(action[3:6])
        if angle < 1e-6:
            axis = (0, 0, 1)
            angle = 0
        else:
            axis = action[3:6] / angle
        delta_quat = axangle2quat(axis, angle)
        target_ee_quat = qmult(self.end_link.pose.q, delta_quat)
        target_ee_pose = sapien.Pose(target_ee_pos, target_ee_quat)
        # print(self.target_ee_pos, self.end_link.pose.p)
        if target_ee_pose is not None:
            self.target_joint_pos = target_ee_pose

    def simulation_step(self):
        for j_idx, j in enumerate(self.control_joints):
            j.set_drive_target(self.target_joint_pos[j_idx])
