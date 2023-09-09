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
from arti_mani.utils.visualization import images_to_video
from arti_mani.utils.wrappers import NormalizeActionWrapper
from stable_baselines3 import SAC
from tqdm import tqdm


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

    record_id = 5
    mode = "door"  # door, drawer, faucet
    save_mode = True
    render_mode = True
    MAX_STEPS = 30

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
    eval_videos_path = RLMODEL_DIR / f"{rl_exp_name}/sim_trajs/{model_name}/"
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

    success_flag = False
    if save_mode:
        frames = []
        base_rgbd = env.render("cameras")
        frames.append(base_rgbd)
        sim_qpos_action = []
    for step in tqdm(range(MAX_STEPS), colour="green", leave=False):
        action, _states = RL_model.predict(obs, deterministic=True)
        # action[-1] = 1
        if step % 2 == 0:
            action[-1] = -1
        else:
            action[-1] = 1
        print("+++", action)
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
            sim_qpos_action.append(
                np.concatenate((env.unwrapped._agent._robot.get_qpos(), action), axis=0)
            )

        if not success_flag and info["is_success"] == 1.0:
            success_flag = True
            total_steps = step
            break

    if save_mode:
        sim_qpos_traj = np.array(sim_qpos_action)
        images_to_video(
            images=frames,
            output_dir=eval_videos_path,
            video_name=f"{record_id}_{mode}_sim2real",
            verbose=False,
        )
        np.save(
            eval_videos_path / f"{record_id}_{mode}_sim2real_qpos_action.npy",
            sim_qpos_traj,
        )
