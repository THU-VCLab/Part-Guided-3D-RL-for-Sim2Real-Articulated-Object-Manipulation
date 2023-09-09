import numpy as np
import rospy
import sapien.core as sapien
import transforms3d
from impedance_control.msg import (
    CartesianControlCommand,
    ImpedanceRobotState,
    JointControlCommand,
)
from robotiq_2f_gripper_control.msg import (
    Robotiq2FGripper_robot_input,
    Robotiq2FGripper_robot_output,
)


def mat2pose(mat: np.ndarray):
    quat = transforms3d.quaternions.mat2quat(mat[:3, :3])
    pos = mat[:3, 3]
    # pose = np.concatenate((pos, quat))
    pose = sapien.Pose(pos, quat)
    return pose


def trans_axangle2pose(trans_axangle):
    RT = np.eye(4)
    RT[:3, 3] = trans_axangle[:3]
    angle = np.linalg.norm(trans_axangle[3:6])
    if angle < 1e-6:
        axis = (0, 0, 1)
    else:
        axis = trans_axangle[3:6] / angle
    RT[:3, :3] = transforms3d.axangles.axangle2mat(axis, angle)
    return mat2pose(RT)


class Robotiq2FGripper_Listener:
    def __init__(self) -> None:
        self.gripper_is_ready = True
        self.gPO = 0

    def listen(self, input):
        # print("input.gACT: ", input.gACT)
        # print("input.gOBJ: ", input.gOBJ)
        if input.gACT == 1 and input.gOBJ > 0:
            self.gripper_is_ready = True
        else:
            self.gripper_is_ready = False
        self.gPO = input.gPO


class ROS_ImpOpendoor:
    def __init__(self, gripper_listener, control_mode="pd_joint_delta_pos") -> None:
        rospy.init_node("open_cabinet_node")
        self.gripper_listener = gripper_listener
        self.rate = rospy.Rate(10)  # 10hz
        self.control_mode = control_mode
        self._initialize_robotcontrol()
        self._initialize_gripper()

    def _initialize_robotcontrol(self):
        if self.control_mode == "pd_joint_delta_pos":
            self.command_pub = rospy.Publisher(
                "joint_control_command", JointControlCommand, queue_size=1
            )
        elif self.control_mode == "pd_ee_pose":
            self.command_pub = rospy.Publisher(
                "cartesian_control_command", CartesianControlCommand, queue_size=1
            )
        else:
            raise NotImplementedError

    def _initialize_gripper(self):
        rospy.Subscriber(
            "Robotiq2FGripperRobotInput",
            Robotiq2FGripper_robot_input,
            self.gripper_listener.listen,
        )

        self.gripper_input = Robotiq2FGripper_robot_input()
        self.gripper_output = Robotiq2FGripper_robot_output()
        self.gripper_pub = rospy.Publisher(
            "Robotiq2FGripperRobotOutput", Robotiq2FGripper_robot_output
        )
        self.gripper_output.rACT = 1  # 1: Active, 0: Not active
        self.gripper_output.rGTO = 1
        self.gripper_output.rATR = 0
        self.gripper_output.rPR = 150  # 0~255: Placement
        self.gripper_output.rFR = 50  # 0~255: Force
        self.gripper_output.rSP = 150  # 0~255: Speed

        while self.gripper_listener.gripper_is_ready:
            self.gripper_pub.publish(self.gripper_output)
        self.gripper_output.rPR = 0  # Open Gripper
        while not self.gripper_listener.gripper_is_ready:
            pass
        while self.gripper_listener.gripper_is_ready:
            self.gripper_pub.publish(self.gripper_output)

    def get_realstate(self):
        self.real_state: ImpedanceRobotState = rospy.wait_for_message(
            "impedance_robot_state", ImpedanceRobotState, timeout=10000
        )
        if self.real_state == None:
            print("no real robot state received in 10000s")
            return

    def exec_gripper(self, gripper_norm_rPR):
        self.gripper_output.rPR = gripper_norm_rPR
        while not self.gripper_listener.gripper_is_ready:
            pass
        # while self.gripper_listener.gripper_is_ready: self.gripper_pub.publish(self.gripper_output)
        self.gripper_pub.publish(self.gripper_output)
        while not self.gripper_listener.gripper_is_ready:
            pass

    def exec_trajs(self, target, stiffness=None, damping=None):
        """
        target: (joint_pos): qpos[7], (ee_pose): mat(16)
        stiffness: (joint_): 7, (ee_): 6
        damping: (joint_): 7, (ee_): 6
        """
        self.get_realstate()
        assert (
            stiffness is not None and damping is not None
        ), "joint/ee stiffness and damping must be set!"
        if self.control_mode == "pd_joint_delta_pos":
            msg = JointControlCommand(target, stiffness, damping, False)
            # print("++++ get robot arm control msg ++++")
        elif self.control_mode == "pd_ee_pose":
            msg = CartesianControlCommand(target, stiffness, damping, False)
        else:
            raise NotImplementedError
        self.command_pub.publish(msg)
        # print("++++ robot arm executation finished ++++")
        self.rate.sleep()


if __name__ == "__main__":
    np.set_printoptions(suppress=True, precision=4)
    control_mode = "pd_ee_pose"  # "pd_joint_delta_pos",

    if control_mode == "pd_joint_delta_pos":
        kp = [200.0, 200.0, 200.0, 200.0, 50.0, 50.0, 50.0]
        kd = [20.0, 20.0, 20.0, 20.0, 10.0, 10.0, 10.0]
    elif control_mode == "pd_ee_pose":
        kp = [300.0, 300.0, 300.0, 50.0, 50.0, 50.0]
        kd = [20.0, 20.0, 20.0, 10.0, 10.0, 10.0]
    else:
        raise NotImplementedError
    print("current kp: ", kp)
    print("current kd: ", kd)

    print("+++++ Setup robotiq controller +++++++")
    gripper_listener = Robotiq2FGripper_Listener()
    print("+++++ Setup robot arm controller +++++++")
    imp_opendoor = ROS_ImpOpendoor(
        gripper_listener=gripper_listener, control_mode=control_mode
    )

    imp_opendoor.get_realstate()
    ## robot qpos
    robot_arm_state = imp_opendoor.real_state.q_m
    gripper_state = np.repeat(gripper_listener.gPO / 255 * 0.068, 2)  # 0-255 => 0-0.068
    real_robot_qpos = np.concatenate((robot_arm_state, gripper_state))
    print("=== real robot current qpos: ", real_robot_qpos)
    ## robot ee pose (robot frame)
    real_ee_pose_mat = imp_opendoor.real_state.toolTobase_pos_m
    real_ee_pose_base = mat2pose(np.reshape(real_ee_pose_mat, (4, 4)))
    print("=== real robot ee pose: ", real_ee_pose_base)
    ## target ee pose
    target_link7_pose = sapien.Pose(
        np.array([0.6, 0, 0.4]), np.array([0, 0.707, 0.707, 0])
    )
    ### link7 pose: Pose([0, 0, 0.431991], [0, -0.707107, 0.707107, 0])
    ### handcolorframe pose: Pose([-0.0773266, -0.0402823, 0.419322], [-0.00315852, 0.704288, 0.00583035, -0.709884])
    target_ee_mat = np.reshape(target_link7_pose.to_transformation_matrix(), 16)
    print("=== target robot ee pose: ", target_link7_pose)
    print("=== target robot ee mat: ", target_ee_mat)
    for k in range(10):
        imp_opendoor.exec_trajs(target_ee_mat, stiffness=kp, damping=kd)
    print("Done!")

    ## camera pos (robot frame): (0.685, 0.025, 0.393~0.4 - 0.03 = 0.363~0.37)
    ## xmate7_link pos (robot frame): (0.6, 0, 0.4)
    ## diff: (0.085, 0.025, -0.037~-0.03)
    # <joint name="realsense_hand_joint" type="fixed">
    #     <origin rpy="2.12574149 -1.55582192  2.58286879" xyz="0.04028231  0.07736792  0.01266915"/>
    #     <!--  <origin rpy="3.9290746 ,   -1.6206431 , -124.82640542" xyz="-0.0188 0.0531 -0.0298"/>  11 samples from d435 -->
    #     <parent link="xmate3_link7"/>
    #     <child link="camera_hand_color_frame"/>
    # </joint>
