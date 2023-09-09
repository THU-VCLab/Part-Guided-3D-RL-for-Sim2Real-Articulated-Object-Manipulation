import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from arti_mani import KPTDATA_DIR, VISUALMODEL_DIR
from arti_mani.algorithms.visual_net.Networks.keypoint_detection import (
    IntegralHumanPoseModel,
    soft_argmax,
)
from torch.nn import functional as F


def load_kptmodel(log_name, device="cuda:0"):
    vis_result_path = (
        VISUALMODEL_DIR / f"kpt_model/{log_name}/visual_results/test_kpts_val/"
    )
    if not os.path.exists(vis_result_path):
        os.makedirs(vis_result_path)
    # load model
    model_path = VISUALMODEL_DIR / f"kpt_model/{log_name}/best.pth"
    kpt_model = IntegralHumanPoseModel(
        num_keypoints=3, num_deconv_layers=3, depth_dim=64, has_dropout=False
    )
    kpt_model.load_state_dict(torch.load(model_path))
    kpt_model.to(torch.device(device))
    kpt_model.eval()
    model_params = sum(p.numel() for p in kpt_model.parameters())
    print(f"model params: {model_params / 1e6:.2f} M")
    return kpt_model


def load_data(arti_ids):
    rgbd_data = []
    kpts_data = []
    train_ids = [int(name[:-14]) for name in os.listdir(KPTDATA_DIR / "train_kpts")]
    val_ids = [int(name[:-14]) for name in os.listdir(KPTDATA_DIR / "val_kpts")]
    for val_id in arti_ids:
        if val_id in val_ids:
            rgbd = np.load(KPTDATA_DIR / f"val/{val_id}.npz")
            kpts = np.load(KPTDATA_DIR / f"val_kpts/{val_id}_keypoints.npy")
        elif val_id in train_ids:
            rgbd = np.load(KPTDATA_DIR / f"train/{val_id}.npz")
            kpts = np.load(KPTDATA_DIR / f"train_kpts/{val_id}_keypoints.npy")
        rgbd_data.extend(
            np.concatenate((rgbd["rgb"] / 255.0, rgbd["depth"][:, None]), axis=1)
        )  # SN, 4, H, W
        kpts_data.extend(kpts)  # SN, 3, 3, 3
    rgbd_data = np.array(rgbd_data)  # SN, 4, H, W
    kpts_data = np.array(kpts_data)  # SN, 3, 3, 3
    return rgbd_data, kpts_data


def vis_kpt(rgb, depth, ind, save_dir):
    # save qualified results
    rgb_kptgt = rgb.copy()
    for idn in range(joint_num):
        if uvz_visable[idn, 0] and uvz_visable[idn, 1]:
            if uvz_visable[idn, 2]:
                cv2.circle(
                    rgb_kptgt,
                    (round(uvz[idn, 0]), round(uvz[idn, 1])),
                    radius=2,
                    color=(1, 0, 0),
                    thickness=-1,
                )
            else:
                cv2.circle(
                    rgb_kptgt,
                    (round(uvz[idn, 0]), round(uvz[idn, 1])),
                    radius=2,
                    color=(0, 1, 0),
                    thickness=-1,
                )

    rgb_kptpred = rgb.copy()
    uvz_pred_norm = uvz_pred_norm_tensor.detach().cpu().numpy()
    uvz_pred = uvz_pred_norm * (kpts_max - kpts_min) + kpts_min
    for idn in range(joint_num):
        if uvz_visable[idn, 0] and uvz_visable[idn, 1]:
            if uvz_visable[idn, 2]:
                cv2.circle(
                    rgb_kptpred,
                    (round(uvz_pred[idn, 0]), round(uvz_pred[idn, 1])),
                    radius=2,
                    color=(1, 0, 0),
                    thickness=-1,
                )
            else:
                cv2.circle(
                    rgb_kptpred,
                    (round(uvz_pred[idn, 0]), round(uvz_pred[idn, 1])),
                    radius=2,
                    color=(0, 1, 0),
                    thickness=-1,
                )
    plt.subplot(plt_x, plt_y, 1)
    plt.imshow(rgb)
    plt.subplot(plt_x, plt_y, 2)
    plt.imshow(depth)
    plt.subplot(plt_x, plt_y, 3)
    plt.imshow(rgb_kptgt)
    plt.subplot(plt_x, plt_y, 4)
    plt.imshow(rgb_kptpred)
    plt.tight_layout()
    plt.savefig(save_dir / f"val_{ind}.jpg")
    plt.close()


def compute_err(uvz_err, xyz, fx, fy):
    uvz_mean_err = uvz_err.mean(0)
    print("mean uvz error: ", uvz_mean_err)
    xy_mean_err = uvz_mean_err[:2] * xyz[:, :, 2].mean() / np.array([fx, fy])
    xyz_mean_err = np.append(xy_mean_err, uvz_mean_err[2])
    print("mean xyz error: ", xyz_mean_err)
    print("mean kpt error: ", xyz_mean_err.mean())


if __name__ == "__main__":
    np.set_printoptions(suppress=True, precision=4)
    device = torch.device("cuda:0")
    joint_num = 3
    batch_size = 256
    IM_H, IM_W = 144, 256
    out_shape = (64, 40, 64)
    kpts_min = np.array([0, 0, 0.18])
    kpts_max = np.array([255, 143, 1.0])
    fx, fy = 183.406, 183.491
    cx, cy = 126.608, 73.768
    plt_x, plt_y = 2, 2

    # load model
    log_name = "20230307_182821_D64H40W64_deconv3_kpts3norm01addvis_uvz_lr1e-3_mobilenetv2_dropout0.2_newdatafilter2drawer"
    kpt_model = load_kptmodel(log_name)

    # load sim data and preprocess
    val_list = [
        0,
        1006,
        1030,
        1047,
        1081,
        1,
        1005,
        1016,
        1024,
        1076,
        5004,
        5007,
        5023,
        5069,
        5052,
    ]
    # rgbd: (N, 4, H, W), kpts: (N, 3, 3, 3)
    rgbd_data, kpts_data = load_data(val_list)
    data_num = kpts_data.shape[0]
    xyz = kpts_data[:, 0]  # N, 3, 3
    uvz = kpts_data[:, 1]  # N, 3, 3
    uvz_norm = (uvz - kpts_min) / (kpts_max - kpts_min)  # N, 3, 3
    uvz_visable = kpts_data[:, 2]  # N, 3, 3

    ids_slice = np.append(np.arange(data_num)[::batch_size], data_num)
    loss, uvz_error_all, uvz_error_occlu, uvz_error_noocclu = [], [], [], []
    with torch.no_grad():
        for id_start, id_end in zip(ids_slice[:-1], ids_slice[1:]):
            uvz_gt, xyz_gt = uvz[id_start:id_end], xyz[id_start:id_end]
            uvz_visable_gt = uvz_visable[id_start:id_end]  # (bs, 3, 3)
            sim_rgbd = (
                torch.from_numpy(rgbd_data[id_start:id_end]).float().to(device)
            )  # (bs, ch, H, W)
            uvz_norm_tensor = torch.from_numpy(uvz_norm[id_start:id_end]).to(device)
            # pred
            pred_heatmap = kpt_model.forward(sim_rgbd)  # (bs, 3*D, H', W')
            uvz_pred_norm_tensor = soft_argmax(
                pred_heatmap, joint_num, out_shape
            )  # (bs, 3, 3)
            loss.append(
                F.l1_loss(
                    uvz_pred_norm_tensor, uvz_norm_tensor, reduction="mean"
                ).item()
            )

            uvz_pred_norm = uvz_pred_norm_tensor.detach().cpu().numpy()  # (bs, 3, 3)
            uvz_pred = uvz_pred_norm * (kpts_max - kpts_min) + kpts_min  # (bs, 3, 3)
            uvz_error_all.append(np.abs(uvz_pred - uvz_gt))  # (bs, 3, 3) total
            uvz_error_noocclu.append(
                np.abs(uvz_pred - uvz_gt) * uvz_visable_gt
            )  # (bs, 3, 3) w/o occlusion
            uvz_error_occlu.append(
                np.abs(uvz_pred - uvz_gt) * (1 - uvz_visable_gt)
            )  # (bs, 3, 3) only occlusion

    print("loss: ", np.mean(loss))
    total_nums = uvz_visable.shape[0]
    visable_nums = uvz_visable.sum(0)  # (3,3)
    invisable_nums = total_nums - uvz_visable.sum(0)
    print(invisable_nums)
    mean_uvz_err = np.concatenate(np.array(uvz_error_all), axis=0).sum(0) / total_nums
    mean_uvz_noocclu_err = (
        np.concatenate(np.array(uvz_error_noocclu), axis=0).sum(0) / visable_nums
    )
    mean_uvz_occlu_err = (
        np.concatenate(np.array(uvz_error_occlu), axis=0).sum(0) / invisable_nums
    )
    compute_err(mean_uvz_err, xyz, fx, fy)
    compute_err(mean_uvz_noocclu_err, xyz, fx, fy)
    compute_err(mean_uvz_occlu_err, xyz, fx, fy)
