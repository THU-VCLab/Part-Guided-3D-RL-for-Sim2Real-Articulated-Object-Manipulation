from collections import OrderedDict

import numpy as np
import sapien.core as sapien
from arti_mani.agents.base_agent import BaseAgent
from arti_mani.utils.common import compute_angle_between
from arti_mani.utils.geometry import transform_points
from arti_mani.utils.sapien_utils import (
    check_joint_stuck,
    get_actor_by_name,
    get_entity_by_name,
    get_pairwise_contact_impulse,
)
from arti_mani.utils.trimesh_utils import get_actor_mesh
from sapien.core import Pose


class FixedXmate3Robotiq(BaseAgent):
    def __init__(self, *args, **kwargs):
        super(FixedXmate3Robotiq, self).__init__(*args, **kwargs)
        self.finger1_link: sapien.LinkBase = get_actor_by_name(
            self._robot.get_links(), "left_inner_finger_pad"
        )
        self.finger2_link: sapien.LinkBase = get_actor_by_name(
            self._robot.get_links(), "right_inner_finger_pad"
        )
        self.finger1_mesh = get_actor_mesh(self.finger1_link, False)
        self.finger2_mesh = get_actor_mesh(self.finger2_link, False)
        self.fingers_visid = []
        for visual_body in self.finger1_link.get_visual_bodies():
            self.fingers_visid.append(visual_body.get_visual_id())
        for visual_body in self.finger2_link.get_visual_bodies():
            self.fingers_visid.append(visual_body.get_visual_id())
        self.finger_size = (0.03, 0.07, 0.0075)  # values from URDF
        self.grasp_site: sapien.Link = get_entity_by_name(
            self._robot.get_links(), "grasp_convenient_link"
        )

    def get_proprioception(self):
        state_dict = OrderedDict()
        qpos = self._robot.get_qpos()
        qvel = self._robot.get_qvel()

        state_dict["qpos"] = qpos
        state_dict["qvel"] = qvel
        state_dict["tcp_wrench"] = self.get_tcp_wrench()
        state_dict["joint_external_torque"] = self.get_generalized_external_forces()
        state_dict["gripper_grasp"] = np.array(
            [self.check_gripper_grasp_real()]
        ).astype(np.float)

        return state_dict

    def check_grasp(self, actor: sapien.ActorBase, min_impulse=1e-6, max_angle=85):
        assert isinstance(actor, sapien.ActorBase), type(actor)
        contacts = self._scene.get_contacts()

        limpulse = get_pairwise_contact_impulse(contacts, self.finger1_link, actor)
        rimpulse = get_pairwise_contact_impulse(contacts, self.finger2_link, actor)

        # direction to open the gripper
        ldirection = self.finger1_link.pose.to_transformation_matrix()[:3, 2]
        rdirection = self.finger2_link.pose.to_transformation_matrix()[:3, 2]

        # angle between impulse and open direction
        langle = compute_angle_between(ldirection, limpulse)
        rangle = compute_angle_between(rdirection, rimpulse)

        lflag = (
            np.linalg.norm(limpulse) >= min_impulse and np.rad2deg(langle) <= max_angle
        )
        rflag = (
            np.linalg.norm(rimpulse) >= min_impulse and np.rad2deg(rangle) <= max_angle
        )

        return all([lflag, rflag])

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

    def get_tcp_wrench(self):
        joint_tau = self.get_generalized_external_forces()[:7]
        controller = self._combined_controllers[self._control_mode]._controllers[0]
        assert controller.control_joint_names == [
            "joint1",
            "joint2",
            "joint3",
            "joint4",
            "joint5",
            "joint6",
            "joint7",
        ]
        controller._sync_articulation()
        J_full = controller.controller_articulation.compute_world_cartesian_jacobian()[
            -6:
        ]
        J = np.linalg.pinv(J_full.T)

        TCP_wrench = J @ joint_tau
        return TCP_wrench

    @staticmethod
    def build_grasp_pose(forward, flat, center):
        extra = np.cross(flat, forward)
        ans = np.eye(4)
        ans[:3, :3] = np.array([forward, flat, -extra]).T
        ans[:3, 3] = center
        return Pose.from_transformation_matrix(ans)

    def check_gripper_grasp_real(self) -> bool:
        """check whether the gripper is grasping something by checking the joint position and velocity"""
        from arti_mani.agents.controllers import GripperPDJointPosMimicController

        assert isinstance(
            self._combined_controllers[self._control_mode]._controllers[1],
            GripperPDJointPosMimicController,
        )
        for joint_idx, joint in enumerate(self._robot.get_active_joints()):
            if joint.name == "robotiq_2f_140_left_driver_joint":
                active_joint1_idx = joint_idx
            if joint.name == "robotiq_2f_140_right_driver_joint":
                active_joint2_idx = joint_idx

        joint1_stuck = check_joint_stuck(
            self._robot, active_joint1_idx, pos_diff_threshold=1e-3, vel_threshold=1
        )
        joint2_stuck = check_joint_stuck(
            self._robot, active_joint2_idx, pos_diff_threshold=1e-3, vel_threshold=1
        )

        return joint1_stuck and joint2_stuck
