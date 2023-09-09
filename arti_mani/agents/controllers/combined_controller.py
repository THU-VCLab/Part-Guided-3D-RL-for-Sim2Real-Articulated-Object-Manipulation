from copy import deepcopy
from typing import Dict, List, Tuple

import arti_mani.agents.controllers as controller_zoo
import numpy as np
import sapien.core as sapien
from arti_mani.agents.controllers.base_controller import BaseController


class CombinedController:
    """A class to combine a list of controllers"""

    _controllers: List[BaseController]
    _controller_action_idx: List[Tuple[int]]

    def __init__(
        self,
        controller_configs: List[Dict],
        robot: sapien.Articulation,
        control_freq: int,
    ):
        self._robot = robot
        self._control_freq = control_freq
        self._controllers = []
        self._controller_action_idx = []
        self.total_torque = None
        self.total_torque_imp = None

        idx_before = 0
        controlled_joint_idx = []
        combined_action_range = []

        for controller_config in controller_configs:
            ControllerClass = getattr(
                controller_zoo, controller_config["controller_type"]
            )
            controller: BaseController = ControllerClass(
                controller_config, robot, control_freq
            )
            self._controllers.append(controller)
            controlled_joint_idx.extend(list(controller.control_joint_index))
            combined_action_range.append(controller.action_range)
            self._controller_action_idx.append(
                (idx_before, idx_before + controller.action_dimension)
            )
            idx_before += controller.action_dimension

        controlled_joint_idx = np.array(controlled_joint_idx)
        assert np.allclose(
            controlled_joint_idx, np.arange(len(self._robot.get_active_joints()))
        ), "All active joints should be controlled by a controller"

        self._action_range = np.concatenate(combined_action_range)

    @property
    def action_range(self) -> np.ndarray:
        return self._action_range

    def simulation_step(self):
        qf_total = self._robot.compute_passive_force(external=False)
        self.total_torque_imp = deepcopy(qf_total)
        for i, controller in enumerate(self._controllers):
            if controller.control_type == "torque":
                qf_total += controller.simulation_step()
            else:
                controller.simulation_step()
        self.total_torque = deepcopy(qf_total)
        # torque_max = np.max(np.abs(qf_total))
        # torque_min = np.min(np.abs(qf_total))
        # print("qf_total max, min:", torque_max, torque_min)
        # self._robot.torque_max_min[0] = torque_max if self.torque_max_min[0] < torque_max else self.torque_max_min[0]
        # self._robot.torque_max_min[1] = torque_min if self.torque_max_min[1] > torque_min else self.torque_max_min[1]
        # print("qf_total: ", qf_total - self._robot.compute_passive_force(external=False))
        qf_total_addnoise = qf_total.copy()
        mean, sigma = 0, 0
        noise = np.clip(
            np.random.normal(mean, sigma, qf_total[:-2].size), -2 * sigma, 2 * sigma
        )
        qf_total_addnoise[:-2] += noise
        # add (m=0, s=100, 10, 1, 0.1, 0.01) norm noise
        self._robot.set_qf(qf_total_addnoise)

    def set_action(self, action: np.ndarray):
        assert action.shape == self._action_range.shape[:1]
        for i, (idx_start, idx_end) in enumerate(self._controller_action_idx):
            self._controllers[i].set_action(action[idx_start:idx_end])

    def set_joint_drive_property(self):
        """set the joint drive property of all controllers"""
        for controller in self._controllers:
            controller.set_joint_drive_property()

    def reset(self):
        for controller in self._controllers:
            controller.reset()
