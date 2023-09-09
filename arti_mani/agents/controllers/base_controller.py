# Controller takes in action and transforms it to sapien joint control
from collections.abc import Iterable
from typing import List

import numpy as np
import sapien.core as sapien


class BaseController:
    """Base controller class"""

    action_dimension: int
    control_type: str

    def __init__(
        self, controller_config: dict, robot: sapien.Articulation, control_freq: int
    ):
        self.robot = robot
        self.control_freq = control_freq
        self.sim_timestep = self.robot.get_builder().get_scene().get_timestep()
        self.control_step = (
            round(1 / self.sim_timestep) // control_freq
        )  # used for action interpolation
        self.curr_step = 0
        self.interpolate = controller_config[
            "interpolate"
        ]  # whether to linear interpolate between simulation steps

        # list of strings
        self.control_joint_names = controller_config["control_joints"]
        self.control_joints: List[sapien.Joint] = []
        self.control_joint_index = []  # the index of controlled joint in `robot`
        for joint_name in self.control_joint_names:
            found = False
            for j_idx, j in enumerate(self.robot.get_active_joints()):
                if j.get_name() == joint_name:
                    self.control_joints.append(j)
                    self.control_joint_index.append(j_idx)
                    found = True
                    break
            if not found:
                raise AssertionError(
                    f"Can NOT find joint [{joint_name}] in robot."
                    f" All joint names: {[j.get_name() for j in self.robot.get_active_joints()]}"
                )
        self.control_joint_index = np.array(self.control_joint_index)
        # check the control joint index is continuous
        assert (
            len(self.control_joint_index)
            == self.control_joint_index.max() - self.control_joint_index.min() + 1
        )
        self.num_control_joints = len(self.control_joints)
        self.start_link = self.control_joints[0].get_parent_link()
        self.end_link = self.control_joints[-1].get_child_link()
        self.start_link_idx = 0
        self.end_link_idx = 0
        for link_idx, link in enumerate(robot.get_links()):
            if link == self.start_link:
                self.start_link_idx = link_idx
            if link == self.end_link:
                self.end_link_idx = link_idx
        self.start_joint_idx = 0
        self.end_joint_idx = 0

        for joint_idx, j in enumerate(
            robot.get_joints()
        ):  # find the joint idx in all the robot joints (including fix)
            if joint_idx == 0:  # first joint is the joint from None to base link
                continue
            if j.get_name() == self.control_joint_names[0]:
                self.start_joint_idx = joint_idx
                continue
            if j.get_name() == self.control_joint_names[-1]:
                self.end_joint_idx = joint_idx
                continue

        assert self.end_joint_idx > self.start_joint_idx > 0
        # create a new articulation to compute mass matrix and Jacobian
        builder: sapien.ArticulationBuilder = (
            self.robot.get_builder().get_scene().create_articulation_builder()
        )
        root: sapien.LinkBuilder = builder.create_link_builder()
        root.set_mass_and_inertia(
            self.start_link.get_mass(),
            self.start_link.cmass_local_pose,
            self.start_link.get_inertia(),
        )
        links = [root]
        all_joints = robot.get_joints()[self.start_joint_idx : self.end_joint_idx + 1]
        for j_idx, j in enumerate(all_joints):
            link = builder.create_link_builder(links[-1])
            link.set_mass_and_inertia(
                j.get_child_link().get_mass(),
                j.get_child_link().cmass_local_pose,
                j.get_child_link().get_inertia(),
            )
            link.set_joint_properties(
                j.type, j.get_limits(), j.get_pose_in_parent(), j.get_pose_in_child()
            )
            links.append(link)

        self.controller_articulation = builder.build(
            fix_root_link=True
        )  # fix=False will make inverse dynamic wrong
        self.controller_articulation.set_name(controller_config["controller_name"])

    @staticmethod
    def nums2array(nums, dim):
        """
        Convert input @nums into numpy array of length @dim. If @nums is a single number, broadcasts it to the
        corresponding dimension size @dim before converting into a numpy array

        Args:
            nums (numeric or Iterable): Either single value or array of numbers
            dim (int): Size of array to broadcast input to env.sim.data.actuator_force

        Returns:
            np.array: Array filled with values specified in @nums
        """
        # First run sanity check to make sure no strings are being inputted
        if isinstance(nums, str):
            raise TypeError(
                "Error: Only numeric inputs are supported for this function, nums2array!"
            )
        if isinstance(nums, Iterable):
            if len(nums) != dim:
                raise TypeError(
                    f"Error: the dimension of nums [{len(nums)}]is not consistent with dim[{dim}]."
                )
        # Check if input is an Iterable, if so, we simply convert the input to np.array and return
        # Else, input is a single value, so we map to a numpy array of correct size and return
        return np.array(nums) if isinstance(nums, Iterable) else np.ones(dim) * nums

    def set_action(self, action: np.ndarray):
        pass

    def simulation_step(self):
        pass

    @property
    def action_range(self) -> np.ndarray:
        return np.arange([])

    def _get_curr_joint_pos(self):
        return self.robot.get_qpos()[self.control_joint_index].copy()

    def _get_curr_joint_vel(self):
        return self.robot.get_qvel()[self.control_joint_index].copy()

    def _sync_articulation(self):
        # sync the robot qpos to controller articulation to compute mass matrix and Jacobian
        self.controller_articulation.set_qpos(self._get_curr_joint_pos())
        self.controller_articulation.set_qvel(self._get_curr_joint_vel())
        self.controller_articulation.set_root_pose(self.start_link.get_pose())
        self.controller_articulation.set_root_velocity(self.start_link.get_velocity())
        self.controller_articulation.set_root_angular_velocity(
            self.start_link.get_angular_velocity()
        )

    def set_joint_drive_property(self):
        """Set the joint drive propertry according to the controller type"""
        raise NotImplementedError()

    def reset(self):
        raise NotImplementedError()
