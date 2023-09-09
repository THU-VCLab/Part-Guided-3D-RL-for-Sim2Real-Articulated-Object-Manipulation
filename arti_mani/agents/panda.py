from collections import OrderedDict

import numpy as np
import sapien.core as sapien
from arti_mani.agents.base_agent import BaseAgent
from arti_mani.utils.common import compute_angle_between
from arti_mani.utils.sapien_utils import (
    check_joint_stuck,
    get_actor_by_name,
    get_entity_by_name,
    get_pairwise_contact_impulse,
)
from sapien.core import Pose


class Panda(BaseAgent):
    def __init__(self, *args, **kwargs):
        super(Panda, self).__init__(*args, **kwargs)
        self.finger1_link, self.finger2_link = get_actor_by_name(
            self._robot.get_links(), ["panda_leftfinger", "panda_rightfinger"]
        )
        self.finger1_joint, self.finger2_joint = get_actor_by_name(
            self._robot.get_joints(), ["panda_finger_joint1", "panda_finger_joint2"]
        )
        self.grasp_site: sapien.Link = get_entity_by_name(
            self._robot.get_links(), "grasp_site"
        )

    def get_proprioception(self):
        state_dict = OrderedDict()
        qpos = self._robot.get_qpos()
        qvel = self._robot.get_qvel()

        state_dict["qpos"] = qpos
        state_dict["qvel"] = qvel
        state_dict["tcp_wrench"] = self.get_tcp_wrench().astype(np.float32)
        state_dict["joint_external_torque"] = self.get_generalized_external_forces()
        state_dict["gripper_grasp"] = np.array(
            [self.check_gripper_grasp_real()], dtype=np.float32
        )

        return state_dict

    def check_grasp(self, actor: sapien.ActorBase, min_impulse=1e-6, max_angle=85):
        assert isinstance(actor, sapien.ActorBase), type(actor)
        contacts = self._scene.get_contacts()

        limpulse = get_pairwise_contact_impulse(contacts, self.finger1_link, actor)
        rimpulse = get_pairwise_contact_impulse(contacts, self.finger2_link, actor)

        # direction to open the gripper
        ldirection = self.finger1_link.pose.to_transformation_matrix()[:3, 1]
        rdirection = -self.finger2_link.pose.to_transformation_matrix()[:3, 1]

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

    def check_contact_fingers(self, actor: sapien.ActorBase, min_impulse=1e-6):
        assert isinstance(actor, sapien.ActorBase), type(actor)
        contacts = self._scene.get_contacts()

        limpulse = get_pairwise_contact_impulse(contacts, self.finger1_link, actor)
        rimpulse = get_pairwise_contact_impulse(contacts, self.finger2_link, actor)

        return (
            np.linalg.norm(limpulse) >= min_impulse,
            np.linalg.norm(rimpulse) >= min_impulse,
        )

    def sample_ee_coords(self, num_sample=10) -> np.ndarray:
        """Uniformly sample points on the two finger meshes. Used for dense reward computation
        return: ee_coords (2, num_sample, 3)"""
        l = 0.0355
        r = 0.052
        ret = []
        for i in range(num_sample):
            x = (l * i + (4 - i) * r) / 4
            finger_tips = [
                self.finger2_joint.get_global_pose().transform(Pose([0, x, 0])).p,
                self.finger1_joint.get_global_pose().transform(Pose([0, -x, 0])).p,
            ]
            ret.append(finger_tips)
        return np.array(ret).transpose((1, 0, 2))

    def get_tcp_wrench(self):
        joint_tau = self.get_generalized_external_forces()[:7]
        controller = self._combined_controllers[self._control_mode]._controllers[0]
        assert controller.control_joint_names == [
            "panda_joint1",
            "panda_joint2",
            "panda_joint3",
            "panda_joint4",
            "panda_joint5",
            "panda_joint6",
            "panda_joint7",
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
        ans[:3, :3] = np.array([extra, flat, forward]).T
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
            if joint.name == "panda_finger_joint1":
                active_joint1_idx = joint_idx
            if joint.name == "panda_finger_joint2":
                active_joint2_idx = joint_idx

        joint1_stuck = check_joint_stuck(self._robot, active_joint1_idx)
        joint2_stuck = check_joint_stuck(self._robot, active_joint2_idx)

        return joint1_stuck or joint2_stuck
