import os
import time
import warnings
from collections import OrderedDict, defaultdict
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
from arti_mani.algorithms.visual_net.Networks.keypoint_detection import (
    IntegralHumanPoseModel,
    soft_argmax,
)
from arti_mani.envs.arti_mani import ArtiMani
from arti_mani.test.real_exps.utils import mat2pose
from arti_mani.utils.contrib import uv2xyz
from arti_mani.utils.cv_utils import visualize_depth, visualize_seg
from arti_mani.utils.geometry import transform_points
from arti_mani.utils.o3d_utils import pcd_uni_down_sample_with_crop
from arti_mani.utils.sapien_utils import get_entity_by_name
from arti_mani.utils.wrappers import NormalizeActionWrapper
from observer import Observer
from stable_baselines3 import SAC
from xmate3Robotiq import Robotiq2FGripper_Listener, ROS_ImpOpendoor

# from xmate3Robotiq_new import Robotiq2FGripper_Listener, ROS_ImpOpendoor

warnings.filterwarnings("ignore")


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


def load_kpt_model(path):
    model_path = VISUALMODEL_DIR / f"kpt_model/{path}/best.pth"
    kpt_model = IntegralHumanPoseModel(
        num_keypoints=3, num_deconv_layers=3, depth_dim=64, has_dropout=False
    )
    kpt_model.load_state_dict(torch.load(model_path))
    kpt_model.to(torch.device("cuda:0"))
    kpt_model.eval()
    return kpt_model


def save_subplots(real_results_path, step, subplot_num, rgb, depth, rgbplot, mode=None):
    plt.subplot(subplot_num[0], subplot_num[1], 1)
    plt.imshow(rgb)
    plt.title("rgb")
    plt.axis("off")
    plt.subplot(subplot_num[0], subplot_num[1], 2)
    plt.imshow(visualize_depth(depth)[..., ::-1])
    plt.title("depth")
    plt.axis("off")
    plt.subplot(subplot_num[0], subplot_num[1], 3)
    plt.imshow(rgbplot)
    plt.title("keypoints")
    plt.axis("off")
    if mode == "plot":
        plt.show()
    elif mode == "save":
        plt.savefig(
            real_results_path / f"test_{step:02}.png",
            bbox_inches="tight",
            pad_inches=0.1,
        )
    else:
        raise NotImplementedError


if __name__ == "__main__":
    np.set_printoptions(suppress=True, precision=4)
    device = "cuda:0"
    mode = "drawer"  # door, drawer, faucet
    rs_mode = "rgbd"  # "rgbd", "depth", "rgb"

    eval_seed = np.random.RandomState().randint(2**32)
    rng = setup_seed(eval_seed)
    print("experiment eval random seed: ", eval_seed)

    save_plot_mode = "plot"  # None, "plot", "save"
    record_id = 0
    gripper_wait = 0.3
    MAX_STEPS = 50
    subplot_num = [1, 3]
    out_shape = (64, 40, 64)
    kpts_max = np.array([255, 143, 1.0])
    kpts_min = np.array([0, 0, 0.18])
    test_name = f"{mode}0_demo{record_id}_3dkpt"
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
    obs_mode = "state_egostereo_keypoints"
    control_mode = "pd_joint_delta_pos"

    if mode == "door":
        rl_exp_name = "baselines/sac4_tr4eval1door_state_egostereo_keypoints3_bs256_tr16up16_0"  # 0, 1, 2
    elif mode == "drawer":
        rl_exp_name = "baselines/sac4_tr4eval1drawer_state_egostereo_keypoints3_bs256_tr16up16_0"  # 0, 1, 2
    elif mode == "faucet":
        rl_exp_name = "baselines/sac4_tr4eval1faucet_state_egostereo_keypoints3_bs256_tr16up16_0"  # 0, 1, 2
    else:
        raise NotImplementedError
    model_path = (
        RLMODEL_DIR / f"{rl_exp_name}/best_model"
    )  # best_model, rl_model_1750000_steps, 2000000, 2250000, 2500000

    real_results_path = RLMODEL_DIR / f"{rl_exp_name}/real_result/{test_name}/"
    if save_plot_mode == "save":
        if not os.path.exists(real_results_path):
            os.makedirs(real_results_path)

    ### load kpt model
    kpt_exp_name = "20230307_182821_D64H40W64_deconv3_kpts3norm01addvis_uvz_lr1e-3_mobilenetv2_dropout0.2_newdatafilter2drawer"
    kpt_model = load_kpt_model(kpt_exp_name)
    kpt_model.eval()

    print("log name: ", rl_exp_name)

    print("+++++ Build Real2Sim Env +++++++")
    env: ArtiMani = gym.make(
        "ArtiMani-v0",
        articulation_ids=arti_ids,
        segmodel_path=VISUALMODEL_DIR / f"kpt_model/{kpt_exp_name}",
        sample_mode=None,
        frame_num=0,
        sample_num=0,
        other_handle_visible=False,
        obs_mode=obs_mode,
        control_mode=control_mode,
        reward_mode="dense",  # "sparse", "dense"
        add_eepadpts=False,
        device=device,
    )
    env = NormalizeActionWrapper(env)
    env.seed(eval_seed)

    obs = env.reset()
    robot_ee: sapien.Link = get_entity_by_name(
        env.agent._robot.get_links(), "xmate3_link7"
    )
    ### Set Cabinet Pose
    env.unwrapped._articulation.set_root_pose(sapien.Pose([2, 0, 0], [1, 0, 0, 0]))

    ### Set Robot Pose
    ## real robot init
    if mode == "door" or mode == "drawer":
        qpos = np.array([1.4, -1.053, -2.394, 1.662, 1.217, 1.05, -0.8, 0.0, 0.0])
    elif mode == "faucet":
        qpos = np.array([-0.5, -0.143, 0, np.pi / 3, 0, 1.57, 1.57, 0.068, 0.068])
    else:
        raise NotImplementedError(mode)
    env.unwrapped._agent._robot.set_qpos(qpos)

    print(f"Action space {env.action_space}")
    print(f"Control mode {env.control_mode}")
    action_range_low = env.unwrapped.action_space.low
    action_range_high = env.unwrapped.action_space.high
    print("Input Action space range (low):", action_range_low)
    print("Input Action space range (high):", action_range_high)

    print("+++++ Setup robotiq controller +++++++")
    gripper_listener = Robotiq2FGripper_Listener()
    print("+++++ Setup robot arm controller +++++++")
    imp_opendoor = ROS_ImpOpendoor(
        gripper_listener=gripper_listener, control_mode=control_mode
    )
    if control_mode == "pd_joint_delta_pos":
        joint_kp = [200.0, 200.0, 200.0, 200.0, 50.0, 50.0, 50.0]
        joint_kd = [20.0, 20.0, 20.0, 20.0, 10.0, 10.0, 10.0]
        # joint_kp = [300.0, 300.0, 300.0, 300.0, 100.0, 100.0, 100.0]
        # joint_kd = [10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0]
    else:
        raise NotImplementedError
    print("current joint kp: ", joint_kp)
    print("current joint kd: ", joint_kd)

    ### load seg model
    kpt_exp_name = "20230307_182821_D64H40W64_deconv3_kpts3norm01addvis_uvz_lr1e-3_mobilenetv2_dropout0.2_newdatafilter2drawer"
    kpt_model = load_kpt_model(kpt_exp_name)
    kpt_model.eval()

    ### load RL model
    print("RL model path: ", model_path)
    RL_model = SAC.load(model_path, env=env, print_system_info=True)

    step = 0
    sim2real_step_num = 5
    prev_target_qpos = None
    prev_target_gripper_pos = None

    figs_dict = defaultdict(list)

    with torch.no_grad():
        rs_observer = Observer(
            base_camera_SN="918512071887",
            mode=rs_mode,
        )  # d435

        while step < MAX_STEPS:
            ### get agent qpos in real
            imp_opendoor.get_realstate()
            robot_arm_state = imp_opendoor.real_state.q_m
            gripper_state = np.repeat(
                gripper_listener.gPO / 255 * 0.068, 2
            )  # 0-255 => 0-0.068
            real_robot_qpos = np.concatenate((robot_arm_state, gripper_state))
            print("=== real robot current qpos: ", real_robot_qpos)
            ### reset agent qpos and get grasp_site pos in sim
            env.agent._robot.set_qpos(real_robot_qpos)
            sim_ee_pose_base = (
                env.agent._robot.get_root_pose().inv() * env.agent.grasp_site.get_pose()
            )  # in robot root coord
            sim_robotee_pose_base = (
                env.agent._robot.get_root_pose().inv() * robot_ee.get_pose()
            )  # in robot root coord
            real_ee_pose_mat = imp_opendoor.real_state.toolTobase_pos_m
            real_ee_pose_base = mat2pose(np.reshape(real_ee_pose_mat, (4, 4)))
            ee_pos_base = sim_ee_pose_base.p
            print("sim robot current qpos: ", env.agent._robot.get_qpos())
            print("sim ee pose (robot root frame): ", sim_ee_pose_base)
            print("sim robot link7 pose (robot root frame): ", sim_robotee_pose_base)
            print("real ee pose (robot root frame): ", real_ee_pose_base)

            ### concate to total states
            frame, _, _, cam_xyz, rs_depth_intrinsics = rs_observer.get_observation()
            cur_rgb, cur_depth = frame[:, :, :3] / 255.0, frame[:, :, 3]
            camH, camW = cur_depth.shape

            state_egokpts = OrderedDict()
            state_egokpts["qpos"] = torch.from_numpy(
                real_robot_qpos
            ).float()  # .to(device)  # (9)
            state_egokpts["ee_pos_base"] = torch.from_numpy(
                ee_pos_base
            ).float()  # .to(device)  # (3)

            rgb_tensor = (
                torch.from_numpy(cur_rgb[None].transpose(0, 3, 1, 2)).float().to(device)
            )  # 1,3,H,W
            depth_tensor = (
                torch.from_numpy(cur_depth[None, None]).float().to(device)
            )  # 1,1,H,W
            heatmap_out = kpt_model.forward(
                torch.cat([rgb_tensor, depth_tensor], dim=1)
            )  # (1, 3*D', H', W')
            uvz_pred_norm = (
                soft_argmax(heatmap_out, 3, out_shape)[0].detach().cpu().numpy()
            )  # (3, 3)
            uvz_pred = uvz_pred_norm * (kpts_max - kpts_min) + kpts_min
            state_egokpts["uvz_pred"] = uvz_pred.astype(np.float32)  # (J, 3)
            # xyz_cam_pred = uv2xyz(uvz_pred, rs_depth_intrinsics)
            # T_cam2ee = env.cam_para["hand_camera_extrinsic_base_frame"]
            # print("=== cam to ee: ", T_cam2ee)
            # xyz_ee_pred = transform_points(T_cam2ee, xyz_cam_pred)
            # state_egokpts["xyz_ee_pred"] = xyz_ee_pred.astype(np.float32)  # [J, 3]

            print(f"+++++++++ step {step} +++++++++++")
            if save_plot_mode:
                rgb_plot = cur_rgb.copy()
                for kpt_ind in range(uvz_pred.shape[0]):
                    cv2.circle(
                        rgb_plot,
                        (int(uvz_pred[kpt_ind, 1]), int(uvz_pred[kpt_ind, 0])),
                        radius=2,
                        color=(1, 0, 0),
                        thickness=-1,
                    )
                save_subplots(
                    real_results_path,
                    step,
                    subplot_num,
                    cur_rgb,
                    cur_depth,
                    rgb_plot,
                    save_plot_mode,
                )

            action, _states = RL_model.predict(state_egokpts, deterministic=True)
            if save_plot_mode == "save":
                real_robot_qpos_eepos_action = np.concatenate(
                    (real_robot_qpos, ee_pos_base, action), axis=0
                )  # 9 + 3 + 8
                np.save(
                    real_results_path / f"{step:02}_qpos_eepos_action.npy",
                    real_robot_qpos_eepos_action,
                )

            real_delta_qpos = scale_para(
                action[:7], action_range_low[:7], action_range_high[:7]
            )
            target_qpos = real_robot_qpos[:-2] + real_delta_qpos
            print("real qpos, action: ", real_robot_qpos[:-2], action)
            print("prev target robot qpos: ", prev_target_qpos)
            print("target robot qpos: ", np.asarray(target_qpos))
            for k in range(sim2real_step_num):
                imp_opendoor.exec_trajs(
                    target_qpos, stiffness=joint_kp, damping=joint_kd
                )

            ## control gripper
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

            step += 1
    print("Total steps: ", step)
