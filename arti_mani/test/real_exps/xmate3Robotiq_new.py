import threading
from copy import deepcopy

import rospy
from impedance_control.msg import JointControlCommand, CartesianControlCommand, ImpedanceRobotState
from robotiq_2f_gripper_control.msg import Robotiq2FGripper_robot_output, Robotiq2FGripper_robot_input


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
        rospy.init_node('open_cabinet_node')
        self.gripper_listener = gripper_listener
        self.rate = rospy.Rate(10)  # 10hz
        self.control_mode = control_mode
        self._initialize_robotcontrol()
        self._initialize_gripper()

        self.sub = None
        self._robot_state = rospy.wait_for_message("impedance_robot_state", ImpedanceRobotState, timeout=10000)
        self._lock = threading.Lock()
        self._thread = threading.Thread(target=self.init_robot_state_subscriber)
        self._thread.start()

    def init_robot_state_subscriber(self):
        self.sub = rospy.Subscriber("impedance_robot_state", ImpedanceRobotState, self.update_robot_state)
        rospy.spin()

    def update_robot_state(self, state: ImpedanceRobotState):
        with self._lock:
            self._robot_state = state

    def get_realstate(self):
        with self._lock:
            return deepcopy(self._robot_state)

    def _initialize_robotcontrol(self):
        if self.control_mode == "pd_joint_delta_pos":
            self.command_pub = rospy.Publisher('joint_control_command', JointControlCommand, queue_size=1)
        elif self.control_mode == "pd_ee_pose":
            self.command_pub = rospy.Publisher('cartesian_control_command', CartesianControlCommand, queue_size=1)
        else:
            raise NotImplementedError

    def _initialize_gripper(self):
        rospy.Subscriber("Robotiq2FGripperRobotInput", Robotiq2FGripper_robot_input, self.gripper_listener.listen)

        self.gripper_input = Robotiq2FGripper_robot_input()
        self.gripper_output = Robotiq2FGripper_robot_output()
        self.gripper_pub = rospy.Publisher('Robotiq2FGripperRobotOutput', Robotiq2FGripper_robot_output)
        self.gripper_output.rACT = 1  # 1: Active, 0: Not active
        self.gripper_output.rGTO = 1
        self.gripper_output.rATR = 0
        self.gripper_output.rPR = 150  # 0~255: Placement
        self.gripper_output.rFR = 50  # 0~255: Force
        self.gripper_output.rSP = 150  # 0~255: Speed

        while self.gripper_listener.gripper_is_ready: self.gripper_pub.publish(self.gripper_output)
        self.gripper_output.rPR = 0  # Open Gripper
        while not self.gripper_listener.gripper_is_ready: pass
        while self.gripper_listener.gripper_is_ready: self.gripper_pub.publish(self.gripper_output)

    def exec_gripper(self, gripper_norm_rPR):
        self.gripper_output.rPR = gripper_norm_rPR
        # while not self.gripper_listener.gripper_is_ready: pass
        # while self.gripper_listener.gripper_is_ready: self.gripper_pub.publish(self.gripper_output)
        self.gripper_pub.publish(self.gripper_output)
        # while not self.gripper_listener.gripper_is_ready: pass

    def exec_trajs(self, target, stiffness=None, damping=None):
        '''
        target: (joint_pos): qpos[7], (ee_pose): mat(16)
        stiffness: (joint_): 7, (ee_): 6
        damping: (joint_): 7, (ee_): 6
        '''
        assert (stiffness is not None and damping is not None), "joint/ee stiffness and damping must be set!"
        if self.control_mode == "pd_joint_delta_pos":
            msg = JointControlCommand(target, stiffness, damping, False)
            # print("++++ get robot arm control msg ++++")
        elif self.control_mode == "pd_ee_pose":
            msg = CartesianControlCommand(target, stiffness, damping, False)
        else:
            raise NotImplementedError
        self.command_pub.publish(msg)
        # print("++++ robot arm executation finished ++++")
        # self.rate.sleep()

    def close(self):
        rospy.signal_shutdown("Terminated by user.")
        self._thread.join()
