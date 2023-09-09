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


def setup_seed(seed=1029):
    os.environ["PYTHONHASHSEED"] = str(seed)  # 为了禁止hash随机化，使得实验可复现
    np.random.seed(seed)
    rng = np.random.RandomState(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    return rng


def scale_para(para, output_low, output_high):
    """
    para: norm to [-1, 1]
    output_low, output_high: reproject to (output_low, output_high)
    """
    bias = 0.5 * (output_high + output_low)
    weight = 0.5 * (output_high - output_low)
    output = weight * para + bias

    return output


def load_vismodel(vismodel_path, load_device):
    model_path = VISUALMODEL_DIR / f"smp_model/{vismodel_path}/best.pth"
    config_path = VISUALMODEL_DIR / f"smp_model/{vismodel_path}/config.yaml"
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)
    smp_cfg = cfg["smp_config"]
    if smp_cfg["mode"] == "RGBD":
        in_ch = 4
    elif smp_cfg["mode"] == "RGB":
        in_ch = 3
    elif smp_cfg["mode"] == "D":
        in_ch = 1
    else:
        raise NotImplementedError
    segmodel = CustomUnet(
        has_dropout=False,
        encoder_name=smp_cfg["encoder"],
        encoder_depth=smp_cfg["encoder_depth"],
        decoder_channels=smp_cfg["decoder_channels"],
        encoder_weights=smp_cfg["encoder_weights"],
        in_channels=in_ch,
        classes=cfg["num_classes"],
        activation=smp_cfg["activation"],
    )
    segmodel.load_state_dict(torch.load(model_path))
    segmodel.to(torch.device(load_device))
    segmodel.eval()
    return segmodel


if __name__ == "__main__":
    np.set_printoptions(suppress=True, precision=4)
    eval_seed = np.random.RandomState().randint(2**32)
    rng = setup_seed(eval_seed)
    print("experiment eval random seed: ", eval_seed)
    device = "cuda:0"

    record_id = 3
    sim2real_step_num = 5
    gripper_wait = 0.5
    name_suffix = (
        f"_repeatstep{sim2real_step_num}_gripperwait{gripper_wait}_realreset_kp500kd45"
    )
    mode = "door"  # door, drawer, faucet
    save_mode = True
    render_mode = False
    MAX_STEPS = 5

    smp_exp_name = "20230219_000940_train52-val18_384_noDR_randombg_aug_stereo_bs16_focalloss_0.5step50lr0.001_RGBDunet-163264128_mobilenet_v2"
    vis_model = load_vismodel(smp_exp_name, device)
    segmodel_path = VISUALMODEL_DIR / f"smp_model/{smp_exp_name}"

    if mode == "door":
        # rl_exp_name = "sac4_tr4eval1door_state_egostereo_segpoints_random_samplef1s32_100steps_padfriction1_p3000d100_tr16up16_noeepadpts_newextric_ptsaddnoise0.01_smp0219"
        rl_exp_name = "sac2_tr12eval3arti_state_egostereo_segpoints_random_samplef1s32_100steps_smp0219_largerange_cabinetrew"
    elif mode == "drawer":
        rl_exp_name = "sac2_tr4eval1drawer_state_egostereo_segpoints_random_samplef1s32_100steps_padfriction1_p3000d100_tr16up16_noeepadpts_newextric_smp0219_adjustreward_2close_initrot"
    elif mode == "faucet":
        rl_exp_name = "sac4_tr4eval1faucet_state_egostereo_segpoints_random_samplef1s32_100steps_padfriction1_p3000d100_tr16up16_noeepadpts_newextric_ptsaddnoise0.01_smp0219"
    else:
        raise NotImplementedError
    model_name = "best_model"  # "rl_model_1000000_steps"
    model_path = RLMODEL_DIR / f"{rl_exp_name}/{model_name}"
    eval_videos_path = RLMODEL_DIR / f"{rl_exp_name}/sim2real_videos/{model_name}/"
    if not os.path.exists(eval_videos_path):
        os.makedirs(eval_videos_path)

    print("+++++ Build Real2Sim Env +++++++")
    if mode == "door":
        arti_ids = [0]  # 0
    elif mode == "drawer":
        arti_ids = [1]  # 1
    elif mode == "faucet":
        arti_ids = [5052]
    elif mode == "arti":
        arti_ids = [0, 1, 5052]
    else:
        raise NotImplementedError
    obs_mode = "state_egostereo_segpoints"
    control_mode = "pd_joint_delta_pos"
    sample_mode = "random_sample"  # full_downsample, score_sample, random_sample, frameweight_sample, fps_sample
    frame_num = 1
    pts_sample_num = 32
    env: ArtiMani = gym.make(
        "ArtiMani-v0",
        articulation_ids=arti_ids,
        segmodel_path=segmodel_path,
        sample_mode=sample_mode,
        frame_num=frame_num,
        sample_num=pts_sample_num,
        other_handle_visible=False,
        obs_mode=obs_mode,
        control_mode=control_mode,
        reward_mode="dense",  # "sparse", "dense"
        device=device,
    )
    env = NormalizeActionWrapper(env)
    env.seed(eval_seed)

    obs = env.reset()
    robot_ee: sapien.Link = get_entity_by_name(
        env.agent._robot.get_links(), "xmate3_link7"
    )
    ### Set Cabinet Pose
    # env.unwrapped._articulation.set_root_pose(
    #     sapien.Pose([2, 0, 0], [1, 0, 0, 0])
    # )

    ### load RL model
    RL_model = SAC.load(
        model_path,
        env=env,
        print_system_info=True,
    )

    ### Set Robot Pose
    ## real robot init
    if mode == "door" or mode == "drawer":
        qpos = np.array([1.4, -1.053, -2.394, 1.662, 1.217, 1.05, -0.8, 0.0, 0.0])
    elif mode == "faucet":
        qpos = np.array([-0.5, -0.143, 0, np.pi / 3, 0, 1.57, 1.57, 0.068, 0.068])
    else:
        raise NotImplementedError(mode)
    env.unwrapped._agent._robot.set_qpos(qpos)

    if render_mode:
        viewer = env.unwrapped.render()
        print("Press [b] to start")
        while True:
            if viewer.window.key_down("b"):
                break
            env.render()

    # logger.info(f"Observation space {env.observation_space}")
    print(f"Action space {env.action_space}")
    print(f"Control mode {env.control_mode}")
    action_range_low = env.unwrapped.action_space.low
    action_range_high = env.unwrapped.action_space.high
    print("Input Action space range (low):", action_range_low)
    print("Input Action space range (high):", action_range_high)

    # logger.info(f"Reward mode {env.reward_mode}")
    print("+++++ Setup robotiq controller +++++++")
    gripper_listener = Robotiq2FGripper_Listener()
    print("+++++ Setup robot arm controller +++++++")
    imp_opendoor = ROS_ImpOpendoor(
        gripper_listener=gripper_listener, control_mode=control_mode
    )
    if control_mode == "pd_joint_delta_pos":
        # joint_kp = [100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0]
        # joint_kd = [5.5, 5.5, 5.5, 5.5, 5.5, 5.5, 5.5]
        joint_kp = [200.0, 200.0, 200.0, 200.0, 50.0, 50.0, 50.0]
        joint_kd = [20.0, 20.0, 20.0, 20.0, 10.0, 10.0, 10.0]
        # joint_kp = [300.0, 300.0, 300.0, 300.0, 100.0, 100.0, 100.0]
        # joint_kd = [20.0, 20.0, 20.0, 20.0, 10.0, 10.0, 10.0]
        # joint_kp = [1000, 1000, 1000, 1000, 100, 100, 100]
        # joint_kd = [60, 60, 60, 60, 20, 20, 20]
        # joint_kp = [500, 500, 500, 500, 50, 50, 50]
        # joint_kd = [45, 45, 45, 45, 14, 14, 14]
        # joint_kp = [200, 200, 200, 200, 20, 20, 20]
        # joint_kd = [20, 20, 20, 20, 5, 5, 5]
    else:
        raise NotImplementedError
    print("current joint kp: ", joint_kp)
    print("current joint kd: ", joint_kd)

    ### load RL model
    RL_model = SAC.load(model_path, env=env, print_system_info=True)

    obs = env.reset()
    success_flag = False
    if save_mode:
        frames = []
        base_rgbd = env.render("cameras")
        frames.append(base_rgbd)
        sim_qpos_traj = []
        real_qpos_traj = []
    for step in tqdm(range(MAX_STEPS), colour="green", leave=False):
        ## get current real state
        imp_opendoor.get_realstate()
        robot_arm_state = imp_opendoor.real_state.q_m
        gripper_state = np.repeat(
            gripper_listener.gPO / 255 * 0.068, 2
        )  # 0-255 => 0-0.068
        real_robot_qpos = np.concatenate((robot_arm_state, gripper_state))
        ### reset agent qpos and get grasp_site pos in sim
        # env.agent._robot.set_qpos(real_robot_qpos)

        action, _states = RL_model.predict(obs, deterministic=True)
        ## sim step
        obs, rewards, dones, info = env.step(action)

        if render_mode:
            viewer = env.unwrapped.render()
            print("Press [c] to start")
            while True:
                if viewer.window.key_down("c"):
                    break
                env.render()
        if save_mode:
            base_rgbd = env.render("cameras")
            frames.append(base_rgbd)
            sim_qpos_traj.append(env.unwrapped._agent._robot.get_qpos())

        ## get current real state
        imp_opendoor.get_realstate()
        robot_arm_state = imp_opendoor.real_state.q_m
        gripper_state = np.repeat(
            gripper_listener.gPO / 255 * 0.068, 2
        )  # 0-255 => 0-0.068
        real_robot_qpos = np.concatenate((robot_arm_state, gripper_state))
        print("=== real robot current qpos: ", real_robot_qpos)

        ## real step
        real_delta_qpos = scale_para(
            action[:7], action_range_low[:7], action_range_high[:7]
        )
        target_qpos = real_robot_qpos[:-2] + real_delta_qpos
        print("target robot qpos: ", np.asarray(target_qpos))
        for k in range(sim2real_step_num):
            imp_opendoor.exec_trajs(target_qpos, stiffness=joint_kp, damping=joint_kd)
        if mode == "door" or mode == "drawer":
            target_gripper_pos = int(
                (action[-1] + 1) / 2 * 255
            )  # Gripper Open->Close: -1->1 => 0->255
        elif mode == "faucet":
            target_gripper_pos = 255
        else:
            raise NotImplementedError(mode)
        print("target gripper qpos: ", target_gripper_pos)
        imp_opendoor.exec_gripper(target_gripper_pos)
        if mode != "faucet":
            time.sleep(gripper_wait)
        real_gripper_pos = gripper_listener.gPO
        print("real gripper qpos: ", real_gripper_pos)

        ## get current real state
        imp_opendoor.get_realstate()
        robot_arm_state = imp_opendoor.real_state.q_m
        gripper_state = np.repeat(
            gripper_listener.gPO / 255 * 0.068, 2
        )  # 0-255 => 0-0.068
        real_robot_qpos = np.concatenate((robot_arm_state, gripper_state))
        if save_mode:
            real_qpos_traj.append(real_robot_qpos)

        if not success_flag and info["is_success"] == 1.0:
            success_flag = True
            total_steps = step
            break

    if save_mode:
        sim_qpos_traj = np.array(sim_qpos_traj)
        real_qpos_traj = np.array(real_qpos_traj)
        qpos_err = real_qpos_traj - sim_qpos_traj
        labels = np.array([f"j{i}" for i in range(7)])
        plt.subplot(2, 1, 1)
        plt.plot(np.arange(sim_qpos_traj.shape[0]), qpos_err[:, :-2], label=labels)
        plt.title("qpos err (real -sim)")
        plt.legend()
        plt.subplot(2, 1, 2)
        plt.plot(np.arange(sim_qpos_traj.shape[0]), qpos_err[:, -2:])
        plt.title("gripper pos err (real -sim)")
        plt.savefig(
            eval_videos_path / f"{record_id}_{mode}_sim2real_err_{name_suffix}.png",
            dpi=300,
            pad_inches=0.2,
        )
        images_to_video(
            images=frames,
            output_dir=eval_videos_path,
            video_name=f"{record_id}_{mode}_sim2real_{name_suffix}",
            verbose=False,
        )
        np.save(
            eval_videos_path / f"{record_id}_{mode}_sim2real_qpos_{name_suffix}.npy",
            np.concatenate((sim_qpos_traj, real_qpos_traj), axis=0),
        )
