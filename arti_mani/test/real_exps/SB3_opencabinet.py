import os
import time
import warnings
from collections import OrderedDict, defaultdict

import cv2
import gym
import numpy as np
import open3d as o3d
import sapien.core as sapien
import torch
import torch.nn.functional as F
import yaml
from arti_mani import RLMODEL_DIR, VISUALMODEL_DIR
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
from arti_mani.utils.geometry import transform_points
from arti_mani.utils.sapien_utils import get_entity_by_name
from arti_mani.utils.wrappers import NormalizeActionWrapper
from observer import Observer
from stable_baselines3 import SAC
from xmate3Robotiq import Robotiq2FGripper_Listener, ROS_ImpOpendoor

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


def load_segmodel(path):
    ### load visual model
    smp_model_path = VISUALMODEL_DIR / f"smp_model/{path}/best.pth"
    config_path = VISUALMODEL_DIR / f"smp_model/{path}/config.yaml"
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)
    smp_cfg = cfg["smp_config"]
    data_mode = smp_cfg["mode"]
    if data_mode == "RGBD":
        in_ch = 4
    elif data_mode == "RGB":
        in_ch = 3
    elif data_mode == "D":
        in_ch = 1
    else:
        raise NotImplementedError

    if "has_dropout" in smp_cfg.keys():
        unet_model = CustomUnet(
            has_dropout=True,
            encoder_name=smp_cfg["encoder"],
            encoder_depth=smp_cfg["encoder_depth"],
            decoder_channels=smp_cfg["decoder_channels"],
            encoder_weights=smp_cfg["encoder_weights"],
            in_channels=in_ch,
            classes=cfg["num_classes"],
            activation=None,
            # activation=smp_cfg["activation"],
        )
    elif smp_cfg["encoder"] == "splitnet":
        unet_model = SplitUnet(
            dropout_p=smp_cfg["dropout_p"],
            encoder_name=smp_cfg["encoder"],
            encoder_depth=smp_cfg["encoder_depth"],
            decoder_channels=smp_cfg["decoder_channels"],
            encoder_weights=smp_cfg["encoder_weights"],
            in_channels=in_ch,
            classes=cfg["num_classes"],
        )
    else:
        unet_model = CustomUnetNew(
            dropout_p=smp_cfg["dropout_p"],
            encoder_name=smp_cfg["encoder"],
            encoder_depth=smp_cfg["encoder_depth"],
            decoder_channels=smp_cfg["decoder_channels"],
            encoder_weights=smp_cfg["encoder_weights"],
            in_channels=in_ch,
            classes=cfg["num_classes"],
        )
    unet_model.load_state_dict(torch.load(smp_model_path))
    unet_model.to(device)
    return unet_model, data_mode


def get_uncertainty(seg_model, rgbd_tensor):
    uncertain_pred = []
    for ind in range(4):
        pred_seg_mtcsamp = seg_model(rgbd_tensor, True)  # (N, 6, H, W)
        uncertain_pred.append(F.softmax(pred_seg_mtcsamp))
    uncertain_mean = torch.mean(torch.stack(uncertain_pred), dim=0)  # (N, 6, H, W)
    print("mean uncertainty ==", (uncertain_pred[0] - uncertain_mean).mean())
    uncertain_map = -1.0 * torch.sum(
        uncertain_mean * torch.log(uncertain_mean + 1e-6), dim=1
    )  # (N, H, W)
    uncertain_map = uncertain_map.cpu().numpy().squeeze()  # (N, H, W)
    norm_uncertain_map = (uncertain_map - uncertain_map.min()) / (
        uncertain_map.max() - uncertain_map.min()
    )
    norm_uncertain_map -= norm_uncertain_map[norm_uncertain_map > 1e-3].mean()
    norm_uncertain_map = np.clip(norm_uncertain_map, 0, 1)
    return norm_uncertain_map


if __name__ == "__main__":
    np.set_printoptions(suppress=True, precision=4)
    device = "cuda:0"
    mode = "door"  # door, drawer, faucet
    rs_mode = "rgbd"  # "rgbd", "depth", "rgb"

    eval_seed = np.random.RandomState().randint(2**32)
    rng = setup_seed(eval_seed)
    print("experiment eval random seed: ", eval_seed)

    save_mode = False
    record_id = 0
    plot_mode = False
    points_vis_mode = False
    add_eepadpts = False
    gripper_wait = 0.3
    MAX_STEPS = 50
    subplot_num = [2, 3]
    camH, camW = 144, 256
    sample_mode = "frameweight_sample"  # full_downsample, score_sample, fps_sample, random_sample, frameweight_sample
    frame_num = 3
    sample_num = 32
    num_classes = 6
    test_name = (
        f"{mode}0_demo{record_id}_{sample_mode}_{frame_num}_{sample_num}_smp0327"
    )
    fig_modes = [
        "rgb",
        "depth",
        "segpred",
        "samplepts",
        "uncertainty_norm",
        "weight_norm",
    ]
    cmaps = [
        (1, 0, 0),  # "red"
        (0, 0, 1),  # "blue"
        (1, 1, 0),  # "yellow"
        (0, 1, 0),  # "green"
        (0.627, 0.125, 0.941),  # "purple"
        (0.753, 0.753, 0.753),  # "grey"
    ]
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
    smp_exps = ["share/20230327_231728_full_splitnet_new_data_aug_ce0.0_dropout0.2"]
    smp_exp_name = smp_exps[0]
    if mode == "door":
        rl_exp_name = "sac4_tr4eval1door_state_egostereo_segpoints_random_samplef1s32_100steps_padfriction1_p3000d100_tr16up16_noeepadpts_newextric_ptsaddnoise0.01_smp0219"
    elif mode == "drawer":
        rl_exp_name = "sac2_tr4eval1drawer_state_egostereo_segpoints_random_samplef1s32_100steps_padfriction1_p3000d100_tr16up16_noeepadpts_newextric_smp0219_adjustreward_2close_initrot"
    elif mode == "faucet":
        rl_exp_name = "sac4_tr12eval3arti_state_egostereo_segpoints_random_samplef1s32_100steps_smp0219_largerange"
    else:
        raise NotImplementedError
    model_path = RLMODEL_DIR / f"{rl_exp_name}/best_model"

    real_results_path = RLMODEL_DIR / f"{rl_exp_name}/real_result/{test_name}/"
    if save_mode:
        if not os.path.exists(real_results_path):
            os.makedirs(real_results_path)

    if sample_mode == "full_downsample":
        sampler = UniDownsampler(sample_num=sample_num)
    elif sample_mode == "score_sample":
        sampler = ScoreSampler(sample_num=sample_num)
    elif sample_mode == "fps_sample":
        sampler = FPSSampler(sample_num=sample_num)
    elif sample_mode == "random_sample":
        sampler = RandomSampler(sample_num=sample_num)
    elif sample_mode == "frameweight_sample":
        sampler = FCUWSampler(frame_num=frame_num, sample_num=sample_num)
    else:
        raise NotImplementedError(sample_mode)

    print("log name: ", rl_exp_name)

    print("+++++ Build Real2Sim Env +++++++")
    env: ArtiMani = gym.make(
        "ArtiMani-v0",
        articulation_ids=arti_ids,
        segmodel_path=VISUALMODEL_DIR / f"smp_model/{smp_exp_name}",
        sample_mode="random_sample",
        frame_num=frame_num,
        sample_num=sample_num,
        other_handle_visible=False,
        obs_mode=obs_mode,
        control_mode=control_mode,
        reward_mode="dense",  # "sparse", "dense"
        add_eepadpts=add_eepadpts,
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

    # viewer = env.unwrapped.render()
    # print("Press [b] to start")
    # while True:
    #     if viewer.window.key_down("b"):
    #         break
    #     env.render()

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

    ### load RL model
    RL_model = SAC.load(model_path, env=env, print_system_info=True)

    unet_model, data_mode = load_segmodel(smp_exp_name)
    unet_model.eval()

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

            # # get ee pad pts
            ee_cords = env.agent.sample_ee_coords(num_sample=20).reshape(
                -1, 3
            )  # [40, 3]
            T_world2ee = (
                env.agent.grasp_site.get_pose().inv().to_transformation_matrix()
            )
            eepad_pts = transform_points(T_world2ee, ee_cords)  # [40, 3]

            eepat_pts_label = np.zeros(
                (eepad_pts.shape[0], num_classes + 1)
            )  # (40, C+1)
            eepat_pts_label[:, -1] = 1.0
            eepad_ptsC = np.concatenate(
                (eepad_pts, eepat_pts_label), axis=1
            )  # (40, 10)

            ### concate to total states
            frame, _, _, cam_xyz, rs_depth_intrinsics = rs_observer.get_observation()
            cur_rgb, cur_depth = frame[:, :, :3] / 255.0, frame[:, :, 3]
            H, W = cur_depth.shape

            # get pred masks
            rgb_tensor = (
                torch.from_numpy(cur_rgb[None].transpose(0, 3, 1, 2)).float().to(device)
            )  # 1,3,H,W
            depth_tensor = (
                torch.from_numpy(cur_depth[None, None]).float().to(device)
            )  # 1,1,H,W
            if data_mode == "RGBD":
                img = torch.cat([rgb_tensor, depth_tensor], dim=1)
            elif data_mode == "RGB":
                img = rgb_tensor
            elif data_mode == "D":
                img = depth_tensor
            else:
                raise NotImplementedError
            seg_map = unet_model.predict(img, T=8, dropout=True)[0]  # (C, H, W)
            seg_mc_label = torch.argmax(seg_map, dim=0).reshape(-1)  # (H*W)
            if add_eepadpts:
                seg_onehot_label = (
                    F.one_hot(seg_mc_label, num_classes=num_classes + 1)
                    .cpu()
                    .detach()
                    .numpy()
                )  # (H*W, C+1)
            else:
                seg_onehot_label = (
                    F.one_hot(seg_mc_label, num_classes=num_classes)
                    .cpu()
                    .detach()
                    .numpy()
                )  # (H*W, C)
            seg_mc_map = seg_mc_label.cpu().detach().numpy()  # (H*W)

            # uncertainty map
            uncertainty_norm = get_uncertainty(unet_model, img)

            ## cam to world
            hand_sensor = env.agent._sensors["hand"]
            sensor_rgb = hand_sensor._cam_rgb
            T_cam2world_sim = sensor_rgb.get_model_matrix()
            print("=== cam to world (wrong): ", T_cam2world_sim)
            T_cam2ee = env.cam_para["hand_camera_extrinsic_base_frame"]
            print("=== cam to ee: ", T_cam2ee)
            T_ee2world = (
                env.agent.grasp_site.get_pose().to_transformation_matrix()
            )  ## world => ee
            print("=== ee to world: ", T_ee2world)
            T_cam2world = T_ee2world @ T_cam2ee
            print("=== cam to world (correct): ", T_cam2world)
            world_xyz = transform_points(
                T_cam2world, cam_xyz * np.array([1, -1, -1])
            )  # [H*W, 3]

            weight_map = np.zeros((1, camH, camW))
            if sample_mode == "full_downsample":
                pts_index = sampler.sampling(world_xyz, rng)
                ds_world_xyz = world_xyz[pts_index]
                ds_onehot_label = seg_onehot_label[pts_index]
                # ## world to ee
                T_world2ee = (
                    env.unwrapped.agent.grasp_site.get_pose()
                    .inv()
                    .to_transformation_matrix()
                )  ## world => ee
                ds_ee_xyz = transform_points(T_world2ee, ds_world_xyz)
                segptsC = np.concatenate(
                    (ds_ee_xyz, ds_onehot_label), axis=1
                )  # (SN, 3+C+1)
                if points_vis_mode:
                    # pcd0 = o3d.geometry.PointCloud()
                    # pcd0.points = o3d.utility.Vector3dVector(transform_points(T_world2ee, world_xyz))
                    # pcd0.paint_uniform_color([0,0,0])
                    pcd1 = o3d.geometry.PointCloud()
                    pcd1.points = o3d.utility.Vector3dVector(eepad_pts)
                    pcd1.paint_uniform_color([0, 1, 0])
                    ds_mc_label = seg_mc_map[pts_index]  # (1000, )
                    pcd_parts = [
                        o3d.geometry.PointCloud(),
                        o3d.geometry.PointCloud(),
                        o3d.geometry.PointCloud(),
                        o3d.geometry.PointCloud(),
                        o3d.geometry.PointCloud(),
                        o3d.geometry.PointCloud(),
                    ]
                    for ind in range(6):
                        pcd_parts[ind].points = o3d.utility.Vector3dVector(
                            ds_ee_xyz[ds_mc_label == ind]
                        )
                        pcd_parts[ind].paint_uniform_color(cmaps[ind])
                    XYZ = o3d.geometry.TriangleMesh.create_coordinate_frame(
                        size=0.1, origin=[0.0, 0.0, 0.0]
                    )
                    objs = [
                        pcd1,
                        XYZ,
                        pcd_parts[0],
                        pcd_parts[1],
                        pcd_parts[2],
                        pcd_parts[3],
                        pcd_parts[4],
                        pcd_parts[5],
                    ]
                    o3d.visualization.draw_geometries(objs)
            else:
                # get seg points
                if sample_mode == "score_sample":
                    pts_index = sampler.sampling(seg_map)
                elif sample_mode == "random_sample":
                    pts_index = sampler.sampling(seg_mc_map, num_classes, rng)
                elif sample_mode == "fps_sample":
                    pts_index = sampler.sampling(
                        world_xyz, num_classes, seg_mc_map, rng
                    )
                elif sample_mode == "frameweight_sample":
                    pts_index, weight_map = sampler.sampling(
                        world_xyz, num_classes, seg_mc_map, uncertainty_norm, rng
                    )
                world_segsampled_xyz = world_xyz[pts_index]  # (C*sample_num, 3)
                ee_segsampled_xyz = transform_points(
                    T_world2ee, world_segsampled_xyz
                )  # (C*sample_num, 3)
                sampled_onehot_label = seg_onehot_label[
                    pts_index
                ]  # (C*sample_num, C+1)
                segptsC = np.concatenate(
                    (ee_segsampled_xyz, sampled_onehot_label), axis=1
                )  # (C*sample_num, 3+C+1)
                # debug point clouds
                if points_vis_mode:
                    ee_xyz = transform_points(T_world2ee, world_xyz)
                    # pcd0 = o3d.geometry.PointCloud()
                    # pcd0.points = o3d.utility.Vector3dVector(ee_xyz)
                    # pcd0.paint_uniform_color([0, 0, 0])
                    pcd1 = o3d.geometry.PointCloud()
                    pcd1.points = o3d.utility.Vector3dVector(eepad_pts)
                    pcd1.paint_uniform_color([0, 1, 0])
                    XYZ = o3d.geometry.TriangleMesh.create_coordinate_frame(
                        size=0.1, origin=[0.0, 0.0, 0.0]
                    )
                    # objs = [pcd0, pcd1, XYZ]
                    pcd_parts = [
                        o3d.geometry.PointCloud(),
                        o3d.geometry.PointCloud(),
                        o3d.geometry.PointCloud(),
                        o3d.geometry.PointCloud(),
                        o3d.geometry.PointCloud(),
                        o3d.geometry.PointCloud(),
                    ]
                    for ind in range(6):
                        pcd_parts[ind].points = o3d.utility.Vector3dVector(
                            segptsC[ind * 30 : (ind + 1) * 30, :3]
                        )
                        pcd_parts[ind].paint_uniform_color(cmaps[ind])
                    objs = [
                        pcd1,
                        XYZ,
                        pcd_parts[0],
                        pcd_parts[1],
                        pcd_parts[2],
                        pcd_parts[3],
                        pcd_parts[4],
                        pcd_parts[5],
                    ]
                    o3d.visualization.draw_geometries(objs)

            state_egopts = OrderedDict()
            state_egopts["qpos"] = torch.from_numpy(
                real_robot_qpos
            ).float()  # .to(device)  # (9)
            state_egopts["ee_pos_base"] = torch.from_numpy(
                ee_pos_base
            ).float()  # .to(device)  # (3)
            if add_eepadpts:
                state_egopts["segsampled_ptsC"] = np.concatenate(
                    (segptsC, eepad_ptsC), axis=0
                ).astype(
                    np.float32
                )  # (SN+40, 3+C+1)
            else:
                state_egopts["segsampled_ptsC"] = segptsC.astype(
                    np.float32
                )  # (C*sample_num+40, 3+C)

            rgb_plot = cur_rgb.copy()
            if sample_mode == "full_downsample":
                pts_inds = np.array(pts_index)
                u = pts_inds // camW
                v = pts_inds - u * camW
                for idn in range(u.shape[0]):
                    cls = np.argmax(seg_onehot_label, axis=1)[pts_inds[idn]]
                    cv2.circle(
                        rgb_plot,
                        (int(v[idn]), int(u[idn])),
                        radius=2,
                        color=cmaps[cls],
                        thickness=-1,
                    )
            else:
                for cls in range(num_classes):
                    cls_inds = np.array(
                        pts_index[cls * sample_num : (cls + 1) * sample_num]
                    )
                    u = cls_inds // camW
                    v = cls_inds - u * camW
                    for idn in range(u.shape[0]):
                        cv2.circle(
                            rgb_plot,
                            (int(v[idn]), int(u[idn])),
                            radius=2,
                            color=cmaps[cls],
                            thickness=-1,
                        )
            print(f"+++++++++ step {step} +++++++++++")

            action, _states = RL_model.predict(state_egopts, deterministic=True)
            if save_mode:
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
