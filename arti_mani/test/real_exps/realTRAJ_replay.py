import os
import time
import warnings
from collections import OrderedDict
from copy import deepcopy

import cv2
import gym
import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d
import sapien.core as sapien
import segmentation_models_pytorch as smp
import torch
import torch.nn.functional as F
import yaml
from arti_mani import ASSET_DIR, RLMODEL_DIR, VISUALMODEL_DIR
from arti_mani.algorithms.rl_iam.Sampling import (
    FCUWSampler,
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
from arti_mani.envs.arti_mani import ArtiMani
from arti_mani.test.real_exps.utils import mat2pose
from arti_mani.utils.contrib import apply_pose_to_points
from arti_mani.utils.cv_utils import visualize_depth, visualize_seg
from arti_mani.utils.geometry import transform_points
from arti_mani.utils.o3d_utils import pcd_uni_down_sample_with_crop
from arti_mani.utils.sapien_utils import get_entity_by_name
from arti_mani.utils.visualization import images_to_video
from arti_mani.utils.wrappers import NormalizeActionWrapper
from observer import Observer
from stable_baselines3 import SAC
from tqdm import tqdm
from xmate3Robotiq import Robotiq2FGripper_Listener, ROS_ImpOpendoor


def interp_steps(traj_qpos_init, traj_qpos_end, step_num):
    delta_qpos = traj_qpos_end - traj_qpos_init
    interp_qpos = delta_qpos / step_num
    traj_qpos_seq = np.tile(traj_qpos_init, (step_num, 1))
    for step in range(step_num):
        traj_qpos_seq[step] += interp_qpos * (step + 1)
    return traj_qpos_seq


if __name__ == "__main__":
    np.set_printoptions(suppress=True, precision=4)

    test_name = "drawer0_demo2_frameweight_sample_3_32_smp0327"
    rl_exp_name = "sac2_tr4eval1drawer_state_egostereo_segpoints_random_samplef1s32_100steps_padfriction1_p3000d100_tr16up16_noeepadpts_newextric_smp0219_adjustreward_2close_initrot"
    real_results_path = RLMODEL_DIR / f"{rl_exp_name}/real_result/{test_name}/"
    real_qpos = []
    for ind in range(43):
        real_qpos_ind = np.load(real_results_path / f"{ind:02}_qpos_eepos_action.npy")
        real_qpos.append(real_qpos_ind[:9])

    control_mode = "pd_joint_delta_pos"

    # logger.info(f"Reward mode {env.reward_mode}")
    print("+++++ Setup robotiq controller +++++++")
    gripper_listener = Robotiq2FGripper_Listener()
    print("+++++ Setup robot arm controller +++++++")
    imp_opendoor = ROS_ImpOpendoor(
        gripper_listener=gripper_listener, control_mode=control_mode
    )
    if control_mode == "pd_joint_delta_pos":
        # joint_kp = [200.0, 200.0, 200.0, 200.0, 50.0, 50.0, 50.0]
        # joint_kd = [20.0, 20.0, 20.0, 20.0, 10.0, 10.0, 10.0]
        joint_kp = [300, 300, 300, 300, 100, 100, 100]
        joint_kd = [20, 20, 20, 20, 5, 5, 5]
    else:
        raise NotImplementedError
    print("current joint kp: ", joint_kp)
    print("current joint kd: ", joint_kd)
    qpos_interp = interp_steps(real_qpos[16], real_qpos[29], 5)
    real_interp_qpos = np.concatenate(
        (real_qpos[:17], qpos_interp, real_qpos[29:]), axis=0
    )

    step = 0
    for real_robot_qpos in real_interp_qpos:
        # if step > 15 and step < 29:
        #     step += 1
        #     continue
        print(f"+++++++++ step {step} +++++++++++")
        print("=== real robot current qpos: ", real_robot_qpos)
        target_qpos = real_robot_qpos[:-2]

        print("target robot qpos: ", np.asarray(target_qpos))
        for k in range(5):
            imp_opendoor.exec_trajs(target_qpos, stiffness=joint_kp, damping=joint_kd)
        target_gripper_pos = int(real_robot_qpos[-1] / 0.068 * 255)
        print("target gripper qpos: ", target_gripper_pos)
        imp_opendoor.exec_gripper(target_gripper_pos)
        time.sleep(0.4)
        real_gripper_pos = gripper_listener.gPO
        print("real gripper qpos: ", real_gripper_pos)

        step += 1
