import argparse
import logging
import os
import time

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


def main(sample_num, logger, log_name, device):
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
    logger.info(f"model params: {model_params / 1e6:.2f} M")
    IM_H, IM_W = 144, 256
    out_shape = (64, 40, 64)
    kpts_min = np.array([0, 0, 0.18])
    kpts_max = np.array([255, 143, 1.0])
    plt_x, plt_y = 2, 2

    # load sim data and preprocess
    rgbd_data = []
    kpts_data = []
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
        1063,
        1067,
        1075,
        1079,
        1081,
        5007,
        5028,
        5069,
    ]  # 1056,
    num_perinst = sample_num // len(val_list)
    for val_id in val_list:
        rgbd = np.load(KPTDATA_DIR / f"val/{val_id}.npz")
        kpts = np.load(KPTDATA_DIR / f"val_kpts/{val_id}_keypoints.npy")
        indices = np.random.choice(kpts.shape[0], num_perinst)
        rgbd_data.extend(
            np.concatenate(
                (rgbd["rgb"][indices] / 255.0, rgbd["depth"][indices][:, None]), axis=1
            )
        )  # SN, 4, H, W
        kpts_data.extend(kpts[indices])  # SN, 3, 3, 3
    rgbd_data = np.array(rgbd_data)  # SN, 4, H, W
    kpts_data = np.array(kpts_data)  # SN, 3, 3, 3
    uvz = kpts_data[:, 1]  # SN, 3, 3
    uvz_norm = (uvz - kpts_min) / (kpts_max - kpts_min)
    uvz_visable = kpts_data[:, 2]  # SN, 3, 3

    # transfer to torch data
    sim_rgbd = torch.from_numpy(rgbd_data).float().to(device)  # (SN, ch, H, W)
    uvz_norm_tensor = torch.from_numpy(uvz_norm).to(device)

    pred_heatmap = kpt_model.forward(sim_rgbd)  # (SN, J*D, H', W')
    joint_num = uvz.shape[1]

    uvz_pred_norm_tensor = soft_argmax(pred_heatmap, joint_num, out_shape)  # (SN, J, 3)

    loss = F.l1_loss(uvz_pred_norm_tensor, uvz_norm_tensor, reduction="mean")
    print("loss: ", loss.item())

    uvz_error = []
    for ind in range(kpts_data.shape[0]):
        rgb = rgbd_data[ind, :3].transpose(1, 2, 0)
        depth = rgbd_data[ind, 3]
        # save qualified results
        rgb_kptgt = rgb.copy()
        for idn in range(joint_num):
            if uvz_visable[ind, idn, 0] and uvz_visable[ind, idn, 1]:
                if uvz_visable[ind, idn, 2]:
                    cv2.circle(
                        rgb_kptgt,
                        (round(uvz[ind, idn, 0]), round(uvz[ind, idn, 1])),
                        radius=2,
                        color=(1, 0, 0),
                        thickness=-1,
                    )
                else:
                    cv2.circle(
                        rgb_kptgt,
                        (round(uvz[ind, idn, 0]), round(uvz[ind, idn, 1])),
                        radius=2,
                        color=(0, 1, 0),
                        thickness=-1,
                    )

        rgb_kptpred = rgb.copy()
        uvz_pred_norm = uvz_pred_norm_tensor.detach().cpu().numpy()
        uvz_pred = uvz_pred_norm * (kpts_max - kpts_min) + kpts_min
        for idn in range(joint_num):
            if uvz_visable[ind, idn, 0] and uvz_visable[ind, idn, 1]:
                if uvz_visable[ind, idn, 2]:
                    cv2.circle(
                        rgb_kptpred,
                        (round(uvz_pred[ind, idn, 0]), round(uvz_pred[ind, idn, 1])),
                        radius=2,
                        color=(1, 0, 0),
                        thickness=-1,
                    )
                else:
                    cv2.circle(
                        rgb_kptpred,
                        (round(uvz_pred[ind, idn, 0]), round(uvz_pred[ind, idn, 1])),
                        radius=2,
                        color=(0, 1, 0),
                        thickness=-1,
                    )
        uvz_error.append(np.abs(uvz_pred[ind] - uvz[ind]))
        plt.subplot(plt_x, plt_y, 1)
        plt.imshow(rgb)
        plt.subplot(plt_x, plt_y, 2)
        plt.imshow(depth)
        plt.subplot(plt_x, plt_y, 3)
        plt.imshow(rgb_kptgt)
        plt.subplot(plt_x, plt_y, 4)
        plt.imshow(rgb_kptpred)
        plt.tight_layout()
        plt.savefig(vis_result_path / f"val_{ind}.jpg")
        plt.close()
    mean_kpt_err = np.array(uvz_error).mean(0)
    print("mean uvz error: ", mean_kpt_err)


if __name__ == "__main__":
    np.set_printoptions(suppress=True, precision=4)
    device = torch.device("cuda:0")
    test_num = 80
    parser = argparse.ArgumentParser(description=f"test for {test_num} sim images")
    # log_name = "20230305_173407_D64H40W64_deconv3_kpts3norm01addvis_uvz_lr1e-3"
    # log_name = "20230305_181802_D64H40W64_deconv3_kpts3norm01addvis_uvz_lr1e-3_resnet18"
    # log_name = "20230305_185213_D64H40W64_deconv3_kpts3norm01addvis_uvz_lr1e-3_mobilenetv2"
    # log_name = "20230305_193703_D64H40W64_deconv3_kpts3norm01addvis_uvz_lr1e-3_mobilenetv2_dropout0.2"
    # log_name = "20230305_201857_D64H40W64_deconv3_kpts3norm01addvis_uvz_lr1e-3_mobilenetv2_block8_dropout0.2"
    # log_name = "20230307_175848_D64H40W64_deconv3_kpts3norm01addvis_uvz_lr1e-3_mobilenetv2_block8_dropout0.2_newdatafilter2drawer"
    log_name = "20230307_182821_D64H40W64_deconv3_kpts3norm01addvis_uvz_lr1e-3_mobilenetv2_dropout0.2_newdatafilter2drawer"
    # log_name = "20230308_135416_kpts3_uvz_modeRGBDepoch60bs32lr0.001_mobilenetv2_dropout0.2_newdatafilter2drawer"
    # log_name = "20230308_145353_kpts3_uvz_modeRGBDepoch60bs32lr0.001_mobilenetv2_dropout0.2_newdatafilter2drawer_smoothl1"
    parser.add_argument("--log-name", "-ln", type=str, default=log_name)
    parser.add_argument("--sample-num", type=int, default=test_num)
    args = parser.parse_args()

    # generate log
    logfile_path = VISUALMODEL_DIR / f"kpt_model/{args.log_name}/simval_results/"
    if not os.path.exists(logfile_path):
        os.makedirs(logfile_path)
    logfile_name = f"{time.strftime('%Y%m%d_%H%M%S', time.localtime())}"
    logger = logging.getLogger(logfile_name)
    logger.setLevel(logging.DEBUG)
    log_file = "{}.log".format(logfile_name)
    fileHandler = logging.FileHandler(os.path.join(logfile_path, log_file), mode="w")
    fileHandler.setLevel(logging.DEBUG)
    formatter = logging.Formatter(
        "[%(asctime)s] {%(filename)s:%(lineno)d} %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    fileHandler.setFormatter(formatter)
    logger.addHandler(fileHandler)

    main(
        sample_num=args.sample_num,
        logger=logger,
        log_name=args.log_name,
        device=device,
    )
