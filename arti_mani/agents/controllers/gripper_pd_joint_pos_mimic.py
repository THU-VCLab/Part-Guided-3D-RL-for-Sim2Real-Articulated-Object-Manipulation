import numpy as np
import sapien.core as sapien
from arti_mani.agents.controllers.base_controller import BaseController


class GripperPDJointPosMimicController(BaseController):
    def __init__(
        self, controller_config: dict, robot: sapien.Articulation, control_freq: int
    ):
        super(GripperPDJointPosMimicController, self).__init__(
            controller_config, robot, control_freq
        )
        self.action_dimension = 1
        self.control_type = "pos"
        self.joint_stiffness = controller_config["joint_stiffness"]
        self.joint_damping = controller_config["joint_damping"]
        self.joint_friction = controller_config["joint_friction"]
        self.joint_force_limit = controller_config["joint_force_limit"]

        self.joint_pos_min = controller_config["joint_pos_min"]
        self.joint_pos_max = controller_config["joint_pos_max"]

        self.curr_joint_pos: np.ndarray = self._get_curr_joint_pos()
        self.start_joint_pos: np.ndarray = self.curr_joint_pos.copy()
        self.final_target_joint_pos: np.ndarray = self.curr_joint_pos.copy()

    @property
    def action_range(self) -> np.ndarray:
        return np.array([self.joint_pos_min, self.joint_pos_max]).reshape(1, 2)

    def set_action(self, action: np.ndarray):
        assert action.shape[0] == self.action_dimension
        self.curr_step = 0
        self.start_joint_pos = self._get_curr_joint_pos()
        self.final_target_joint_pos = np.ones(self.num_control_joints) * action

    def simulation_step(self):
        self.curr_step += 1
        if self.interpolate:
            curr_target_joint_pos = (
                self.final_target_joint_pos - self.start_joint_pos
            ) / self.control_step * self.curr_step + self.start_joint_pos
        else:
            curr_target_joint_pos = self.final_target_joint_pos

        for j_idx, j in enumerate(self.control_joints):
            j.set_drive_target(curr_target_joint_pos[j_idx])

    def set_joint_drive_property(self):
        for j_idx, j in enumerate(self.control_joints):
            j.set_drive_property(
                self.joint_stiffness, self.joint_damping, self.joint_force_limit
            )
            j.set_friction(self.joint_friction)

    def reset(self):
        self.curr_joint_pos: np.ndarray = self._get_curr_joint_pos()
        self.start_joint_pos: np.ndarray = self.curr_joint_pos.copy()
        self.final_target_joint_pos: np.ndarray = self.curr_joint_pos.copy()
