import os
import random

import cv2
import gym
import matplotlib.pyplot as plt
import numpy as np
from arti_mani import ASSET_DIR
from arti_mani.envs.floating_robotiqenv import (
    CabinetDoor,
    CabinetDrawer,
    Faucet,
)  # register gym envs
from arti_mani.utils.cv_utils import visualize_depth, visualize_seg
from arti_mani.utils.sapien_utils import look_at
from PIL import Image


def gen_grid_campos(dist_xy_xz_num, length, width, height, mode):
    dist_limit = np.sqrt(width**2 + length**2) / 2
    dist_num, rotxy_num, rotxz_num = dist_xy_xz_num

    if mode == "faucet":
        # get more data for faucet
        dist_range = np.linspace(
            dist_limit + 0.125, dist_limit + 0.6, int(1.5 * dist_num)
        )  # 0.475
        rot_xy_range = np.linspace(
            45 / 180 * np.pi, 135 / 180 * np.pi, int(1.5 * rotxy_num)
        )  # 90
        rot_xz_range = np.linspace(
            -60 / 180 * np.pi, 60 / 180 * np.pi, int(1.5 * rotxz_num)
        )  # 120
    else:
        dist_range = np.linspace(dist_limit + 0.2, dist_limit + 0.6, dist_num)  # 0.4
        rot_xy_range = np.linspace(-10 / 180 * np.pi, 30 / 180 * np.pi, rotxy_num)  # 40
        rot_xz_range = np.linspace(-30 / 180 * np.pi, 30 / 180 * np.pi, rotxz_num)  # 60

    grid_sample = np.array(np.meshgrid(dist_range, rot_xy_range, rot_xz_range)).reshape(
        3, -1
    )
    ori_sample = grid_sample.copy()

    # add some noise to lookat angle
    grid_sample[-2:] += (
        np.random.normal(0, 2, size=grid_sample[-2:].shape) / 180 * np.pi
    )
    print(np.abs(grid_sample[-2:] - ori_sample[-2:]).mean() * 180 / np.pi)

    sample_pos = np.stack(
        (
            -grid_sample[0] * np.cos(grid_sample[1]) * np.cos(grid_sample[2]),
            grid_sample[0] * np.cos(grid_sample[1]) * np.sin(grid_sample[2]),
            grid_sample[0] * np.sin(grid_sample[1]) + height / 2,
        ),
        axis=1,
    )
    sample_campos = sample_pos[sample_pos[:, 2] > 0]
    return sample_campos


def random_crop(image, crop_shape):
    nh = random.randint(0, image.shape[0] - crop_shape[0])
    nw = random.randint(0, image.shape[1] - crop_shape[1])
    image_crop = image[nh : nh + crop_shape[0], nw : nw + crop_shape[1]]
    return image_crop


if __name__ == "__main__":
    np.set_printoptions(suppress=True, precision=4)
    data_mode = "seg"  # "seg" "keypoints"
    assert data_mode in ["seg", "keypoints"]
    save_mode = True
    ir_mode = True
    domain_random = True  # random textures & materials on artis
    random_bg_mode = False  # random background RGBD images
    render_mode = False  # render each RGB-D-seg images
    H, W = 144, 256
    dist_xy_xz_num = (8, 8, 12)
    num_classes = 6
    data_dir_name = f"artidata_rebalance_{dist_xy_xz_num}_{data_mode}"
    other_handle_visible = True if data_mode == "seg" else False
    print("data_dir_name:", data_dir_name)

    if domain_random:
        data_dir_name = data_dir_name + "_DR"
    else:
        data_dir_name = data_dir_name + "_noDR"

    if random_bg_mode:
        data_dir_name = data_dir_name + "_randombg"
        random_bg_num = 180
        bg_rgbs = []
        bg_ds = []
        for ind in range(random_bg_num):
            rgb_data = np.asarray(Image.open(ASSET_DIR / f"random_bg/{ind:03}.jpg"))
            d_data = np.asarray(Image.open(ASSET_DIR / f"random_bg/{ind:03}.png"))
            # rgb_data = rgb_data / 255.0
            d_data = (d_data.astype(np.float32) / 1000.0).clip(0.2, 2)
            bg_rgbs.append(rgb_data)
            bg_ds.append(d_data)
    else:
        data_dir_name = data_dir_name + "_norandombg"

    mode_list = ["door", "drawer", "faucet"]
    arti_items = {
        "door": [
            0,
            1000,
            1001,
            1002,
            1006,
            1007,
            1014,
            1017,
            1018,
            1025,
            1026,
            1027,
            1030,
            1031,
            1034,
            1036,
            1038,
            1039,
            1041,
            1042,
            1044,
            1045,
            1047,
            1049,
            1051,
            1052,
            1054,
            1057,
            1060,
            1061,
            1062,
            1063,
            1064,
            1065,
            1067,
            1073,
            1075,
            1077,
            1078,
            1081,
            # (1028, 1046, 1068)
        ],
        # 40 + 3
        "drawer": [
            1,
            1004,
            1005,
            1013,
            1016,
            1021,
            1024,
            1032,
            1033,
            1035,
            1040,
            1056,
            1066,
            1076,
            1079,
            1082,
        ],
        # 16,
        "faucet": [
            5002,
            5004,
            5005,
            5007,
            5018,
            5023,
            5024,
            5028,
            5034,
            5049,
            5052,
            5053,
            5063,
            5069,
        ],
        # 14 + 1 (60)
    }

    train_list = [
        0,
        1,
        1000,
        1001,
        1004,
        1006,
        1013,
        1014,
        1016,
        1018,
        1024,
        1025,
        1026,
        1031,
        1032,
        1034,
        1035,
        1038,
        1039,
        1040,
        1041,
        1042,
        1044,
        1047,
        1049,
        1051,
        1052,
        1057,
        1060,
        1061,
        1062,
        1064,
        1065,
        1066,
        1073,
        1076,
        1077,
        1078,
        1082,
        5002,
        5004,
        5005,
        5018,
        5023,
        5024,
        5034,
        5049,
        5052,
        5053,
        5063,
    ]  # 39+11

    val_list = [
        1002,
        1005,
        1007,
        1017,
        1021,
        1027,
        1030,
        1033,
        1036,
        1045,
        1054,
        1056,
        1063,
        1067,
        1075,
        1079,
        1081,
        5007,
        5028,
        5069,
    ]  # 17+3

    if data_mode == "keypoints":
        train_list.remove(1035)
        val_list.remove(1056)
        arti_items["drawer"].remove(1035)
        arti_items["drawer"].remove(1056)
        plt_x, plt_y = 2, 2
    else:
        plt_x, plt_y = 1, 3

    obs_mode = "state_egorgbd"
    control_mode = "pd_joint_delta_pos"
    for mode in mode_list:
        if mode == "door":
            env_id = "CabinetDoor-v0"
        elif mode == "drawer":
            env_id = "CabinetDrawer-v0"
        elif mode == "faucet":
            env_id = "Faucet-v0"
        else:
            raise NotImplementedError(mode)
        env = gym.make(
            env_id,
            articulation_ids=[arti_items[mode][0]],
            articulation_config_path=None,
            obs_mode=obs_mode,
            control_mode=control_mode,
            reward_mode="dense",
            irsensor_mode=ir_mode,
            other_handle_visible=other_handle_visible,
            domain_random=domain_random,
        )
        # random.shuffle(arti_items[mode])
        for arti_ind in arti_items[mode]:
            render_rgbd_path = f"./{data_dir_name}/{mode}/{arti_ind}/"
            print(render_rgbd_path)
            if not os.path.exists(render_rgbd_path):
                os.makedirs(render_rgbd_path)

            obs_global = env.reset(articulation_id=arti_ind)
            if arti_ind == 0:
                width, length, height = 0.304, 0.751, 0.54
                obj_center_pos = np.array([0, 0, 0.27])
            elif arti_ind == 1:
                width, length, height = 0.474, 0.28, 0.453
                obj_center_pos = np.array([0, 0, 0.23])
            else:
                obj_center_pos = env.unwrapped._articulation.get_root_pose().p
                bbox_max, bbox_min = (
                    np.array(env.unwrapped._articulation_config["bbox_max"]),
                    np.array(env.unwrapped._articulation_config["bbox_min"]),
                )
                scale = env.unwrapped._articulation_config["scale"]
                width, length, height = (bbox_max - bbox_min) * scale

            # sample camera position
            sample_campos = gen_grid_campos(dist_xy_xz_num, length, width, height, mode)

            rgbd_seg_files = {"rgb": [], "depth": [], "seg": []}
            part_keypoints = []
            for ind, pos_it in enumerate(sample_campos):
                gripper_qpos = np.random.uniform(0, 0.068)
                env.unwrapped._agent._robot.set_qpos([gripper_qpos, gripper_qpos])
                env.unwrapped._initialize_articulations()

                # door and draw have a 0.6 offset on x-axis
                if mode != "faucet":
                    # random lookat center
                    sample_poscenter = obj_center_pos + np.hstack(
                        (
                            np.random.normal(loc=0, scale=width / 12, size=1),
                            np.random.normal(loc=0, scale=length / 10, size=1),
                            np.random.normal(loc=0, scale=height / 10, size=1),
                        )
                    )
                    pos_it[0] += 0.6
                    sample_poscenter[0] += 0.6
                else:
                    # random lookat center
                    sample_poscenter = obj_center_pos + np.hstack(
                        (
                            np.random.normal(loc=0, scale=0.05, size=1),
                            np.random.normal(loc=0, scale=0.05, size=1),
                            np.random.normal(loc=0, scale=0.01, size=1),
                        )
                    )

                # random sample camera in-plane rotation
                sample_gripper_roty = (np.random.rand() - 0.5) * 2  # (-1, 1)
                sample_gripper_rotz = np.sqrt(
                    1 - sample_gripper_roty**2
                ) * np.random.choice([-1, 1])
                env.unwrapped._agent._robot.set_root_pose(
                    look_at(
                        eye=pos_it,
                        target=sample_poscenter,
                        up=[0, sample_gripper_roty, sample_gripper_rotz],
                    )
                )
                obs_global = env.get_obs()
                rgb = obs_global["rgb"]
                clean_depth = obs_global["clean_depth"]
                depth = obs_global["depth"]
                seg = obs_global["seg"]
                bg_seg = obs_global["bg_seg"]
                if random_bg_mode:
                    random_id = np.random.choice(range(random_bg_num))
                    bg_rgb_sample, bg_d_sample = bg_rgbs[random_id], bg_ds[random_id]
                    bg_rgb_sample_crop = random_crop(bg_rgb_sample, (H, W))
                    bg_d_sample_crop = random_crop(bg_d_sample, (H, W))
                    bg_seg_mask = bg_seg.astype(np.bool_)
                    rgb = (
                        rgb * ~bg_seg_mask
                        + bg_rgb_sample_crop.transpose(2, 0, 1) * bg_seg_mask
                    )
                    clean_depth = (
                        clean_depth * ~bg_seg_mask + bg_d_sample_crop * bg_seg_mask
                    )
                    depth = depth * ~bg_seg_mask + bg_d_sample_crop * bg_seg_mask

                if data_mode == "seg":
                    rgbd_seg_files["rgb"].append(rgb)
                    rgbd_seg_files["depth"].append(depth)
                    rgbd_seg_files["seg"].append(seg)
                elif data_mode == "keypoints":
                    kpts = obs_global["kpts"]
                    uvz = obs_global["uvz"]
                    uvz_visable = obs_global["uvz_visable"]  # (3, 3)
                    print(
                        "+++ tip camxyz, uvz, visable: ",
                        kpts[0],
                        uvz[0],
                        uvz_visable[0],
                    )
                    print(
                        "+++ center camxyz, uvz, visable: ",
                        kpts[1],
                        uvz[1],
                        uvz_visable[1],
                    )
                    print(
                        "+++ bottom camxyz, uvz, visable: ",
                        kpts[2],
                        uvz[2],
                        uvz_visable[2],
                    )
                    if not np.any(uvz_visable[:, 2]):
                        continue
                    part_keypoints.append(
                        np.stack((kpts, uvz, uvz_visable))
                    )  # (3, 3, 3)

                    rgbd_seg_files["rgb"].append(rgb)
                    rgbd_seg_files["depth"].append(depth)

                    plot_rgb = rgb.transpose((1, 2, 0)).copy()
                    for idn in range(uvz.shape[0]):
                        if uvz_visable[idn, 0] and uvz_visable[idn, 1]:
                            if uvz_visable[idn, 2]:
                                cv2.circle(
                                    plot_rgb,
                                    (int(uvz[idn, 0]), int(uvz[idn, 1])),
                                    radius=2,
                                    color=(255, 0, 0),
                                    thickness=-1,
                                )
                            else:
                                cv2.circle(
                                    plot_rgb,
                                    (int(uvz[idn, 0]), int(uvz[idn, 1])),
                                    radius=2,
                                    color=(0, 255, 0),
                                    thickness=-1,
                                )
                else:
                    raise NotImplementedError(data_mode)

                plt.figure()
                plt.subplot(plt_x, plt_y, 1)
                plt.imshow(rgb.transpose((1, 2, 0)) / 255.0)
                plt.title("rgb")
                plt.axis("off")
                plt.subplot(plt_x, plt_y, 2)
                plt.imshow(visualize_depth(depth.squeeze())[..., ::-1])
                plt.title("depth")
                plt.axis("off")
                plt.subplot(plt_x, plt_y, 3)
                plt.imshow(visualize_seg(seg, num_classes))
                plt.title("seg")
                plt.axis("off")
                if data_mode == "keypoints":
                    plt.subplot(plt_x, plt_y, 4)
                    plt.imshow(plot_rgb)
                    plt.title("keypoints")
                    plt.axis("off")
                # plt.tight_layout()
                if save_mode:
                    plt.savefig(
                        f"./{render_rgbd_path}/{arti_ind}_{ind}.jpg",
                        bbox_inches="tight",
                        pad_inches=0.0,
                    )
                else:
                    plt.show()
                plt.close()

                if render_mode:
                    viewer = env.render()
                    print("Press [e] to start")
                    while True:
                        if viewer.window.key_down("e"):
                            break
                        env.render()
                print(mode, arti_ind, ind)
            if save_mode:
                if arti_ind in train_list:
                    data_rgbd_path = f"./{data_dir_name}/train/"
                    data_kpts_path = f"./{data_dir_name}/train_kpts/"
                elif arti_ind in val_list:
                    data_rgbd_path = f"./{data_dir_name}/val/"
                    data_kpts_path = f"./{data_dir_name}/val_kpts/"
                else:
                    raise NotImplementedError
                if not os.path.exists(data_rgbd_path):
                    os.makedirs(data_rgbd_path)
                if not os.path.exists(data_kpts_path):
                    os.makedirs(data_kpts_path)

                np.savez(data_rgbd_path + f"{arti_ind}.npz", **rgbd_seg_files)
                if data_mode == "keypoints":
                    kpts_data = np.array(part_keypoints)
                    np.save(data_kpts_path + f"{arti_ind}_keypoints.npy", kpts_data)
