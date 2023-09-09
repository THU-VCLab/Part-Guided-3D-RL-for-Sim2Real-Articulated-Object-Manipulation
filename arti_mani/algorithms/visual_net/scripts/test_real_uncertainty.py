import argparse
import glob
import json
import logging
import os
import time

import cv2
import matplotlib.pyplot as plt
import numpy as np
import segmentation_models_pytorch as smp
import torch
import yaml
from arti_mani import REAL_DIR, VISUALMODEL_DIR
from arti_mani.algorithms.visual_net.Networks.Custom_Unet import CustomUnet, SplitUnet
from arti_mani.utils.cv_utils import visualize_depth, visualize_seg
from path import Path
from PIL import Image
from torch.nn import functional as F
from tqdm import tqdm


def load_real_data_old(
    REAL_DIR="/ssd/pwral_real_seg_old",
    PAD_SIZE=(256, 144),
    IM_SIZE=(256, 144),
    in_ch=4,
    only_vis=False,
):
    # load real data and preprocess
    REAL_DIR = Path(REAL_DIR)
    leftrightpad = int((PAD_SIZE[0] - IM_SIZE[0]) / 2)
    topbottompad = int((PAD_SIZE[1] - IM_SIZE[1]) / 2)
    real_img = {"rgb": [], "depth": [], "masks": []}
    for mode in ["door", "drawer", "faucet"]:
        for im_mode in ["rgb", "depth"]:
            files = glob.glob(str(REAL_DIR / f"{mode}/*{im_mode}.npy"))
            files.sort()
            for file in files:
                data = np.load(file)
                data = cv2.resize(
                    src=data, dsize=IM_SIZE, interpolation=cv2.INTER_NEAREST
                )
                if leftrightpad > 0 or topbottompad > 0:
                    data = cv2.copyMakeBorder(
                        data,
                        topbottompad,
                        topbottompad,
                        leftrightpad,
                        leftrightpad,
                        cv2.BORDER_DEFAULT,
                    )
                if im_mode == "rgb":
                    data = data[..., ::-1] / 255.0
                real_img[im_mode].append(data)
    # load real masks
    if not only_vis:
        mask_files = []
        for mode in ["door", "drawer", "faucet"]:
            files = glob.glob(str(REAL_DIR / f"masks/*{mode}.npy"))
            files.sort()
            mask_files.extend(files)
        for mf in mask_files:
            data = np.load(mf)
            data = cv2.resize(src=data, dsize=IM_SIZE, interpolation=cv2.INTER_NEAREST)
            if leftrightpad > 0 or topbottompad > 0:
                data = cv2.copyMakeBorder(
                    data,
                    topbottompad,
                    topbottompad,
                    leftrightpad,
                    leftrightpad,
                    cv2.BORDER_DEFAULT,
                )
            real_img["masks"].append(data)

    for key in real_img.keys():
        real_img[key] = np.array(real_img[key])

    if in_ch == 1:
        real_data = np.expand_dims(real_img["depth"], 3)
    elif in_ch == 3:
        real_data = np.array(real_img["rgb"])
    elif in_ch == 4:
        real_data = np.concatenate(
            [real_img["rgb"], np.expand_dims(real_img["depth"], 3)], axis=3
        )
    else:
        raise NotImplementedError
    # transfer to torch data
    real_rgbd = torch.from_numpy(real_data).permute(0, 3, 1, 2)  # (N, ch, H, W)
    return real_rgbd, real_img


def load_real_data(REAL_DIR="/ssd/pwral_real", in_ch=4):
    # load real data and preprocess
    REAL_DIR = Path(REAL_DIR)
    real_img = {"rgbd": [], "mask": []}
    corr_map = {"handle": 0, "door": 1, "cabinet": 2, "switch_link": 4, "fix_link": 3}
    for mode in ["doors", "drawers", "faucets"]:
        cate_dir = str(REAL_DIR / f"{mode}_labels/")
        dirs = os.listdir(cate_dir)
        for dir in sorted(dirs):
            # load data
            data = np.load(cate_dir + f"{dir}/{dir}_rgbd.npy")
            label = np.array(Image.open(cate_dir + f"{dir}/label.png"))
            # load color config
            with open(str(REAL_DIR / f"{mode}/") + f"{dir}_rgb.json") as f:
                conf = json.load(f)
            label_names = []
            for shape in conf["shapes"]:
                if shape["label"] not in label_names:
                    label_names.append(shape["label"])

            real_img["rgbd"].append(data)
            # fix mask index
            fix_label = label.copy()
            fix_label[label == 0] = 5
            for i in range(1, len(label_names) + 1):
                fix_label[label == i] = corr_map[label_names[i - 1]]
            real_img["mask"].append(fix_label)

    for key in real_img.keys():
        real_img[key] = np.array(real_img[key])
    print(real_img["rgbd"].shape, real_img["mask"].shape)

    # transfer to torch data
    real_rgbd = torch.from_numpy(real_img["rgbd"][..., :in_ch]).permute(0, 3, 1, 2)
    return real_rgbd, real_img


def main(logger, log_name, logfile_path, device, classes, only_vis):
    # load model
    model_path = VISUALMODEL_DIR / f"{log_name}/best.pth"
    config_path = VISUALMODEL_DIR / f"{log_name}/config.yaml"
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
    seg_model = SplitUnet(
        dropout_p=smp_cfg["dropout_p"],
        encoder_name=smp_cfg["encoder"],
        encoder_depth=smp_cfg["encoder_depth"],
        decoder_channels=smp_cfg["decoder_channels"],
        encoder_weights=smp_cfg["encoder_weights"],
        in_channels=in_ch,
        classes=cfg["num_classes"],
    )
    seg_model.load_state_dict(torch.load(model_path))
    seg_model.to(device)
    seg_model.eval()
    model_params = sum(p.numel() for p in seg_model.parameters())
    logger.info(f"model params: {model_params / 1e6:.2f} M")
    num_classes = cfg["num_classes"]

    real_data, real_img = load_real_data(in_ch=in_ch)

    print("real data:", real_data.shape)

    pred_mc_img = []
    uncertain_map_img = []

    with torch.no_grad():
        for i in tqdm(range(len(real_data)), desc="Evaluating"):
            d = real_data[i : i + 1].to(device=device, dtype=torch.float32)
            # get uncertainty
            uncertain_pred = []
            for _ in range(4):
                pred_seg_mtcsamp = seg_model(d, activate_dropout=True)  # (1, 6, H, W)
                uncertain_pred.append(F.softmax(pred_seg_mtcsamp, 1))
            uncertain_mean = torch.mean(
                torch.stack(uncertain_pred), dim=0
            )  # (1, 6, H, W)
            uncertain_map = -1.0 * torch.sum(
                uncertain_mean * torch.log(uncertain_mean + 1e-6), dim=1
            )  # (1, H, W)

            # predict segmentation
            seg_model.eval()
            pred_seg = seg_model.predict(
                d, T=10, dropout=True
            )  # (1, 6, H, W) # test-time aug
            # pred_seg = torch.mean(torch.stack(uncertain_pred), dim=0) # test-time dropout
            pred_mc = torch.argmax(pred_seg, dim=1)  # (1, H, W)
            pred_mc_img.append(pred_mc.cpu().numpy())

            uncertain_map -= uncertain_map.mean()
            uncertain_map = torch.maximum(
                uncertain_map, torch.zeros(uncertain_map.shape, device="cuda")
            )
            uncertain_map_img.append(uncertain_map.cpu().numpy())

        # save qualified results
        subplot_num = [2, 3]
        # seg_pred_img = pred_seg.detach().cpu().numpy()
        pred_mc_img = np.concatenate(pred_mc_img, 0)  # (N, H, W)
        uncertain_map_img = np.concatenate(uncertain_map_img, 0)  # (N, H, W)
        uncertain_map_img = (uncertain_map_img - uncertain_map_img.min()) / (
            uncertain_map_img.max() - uncertain_map_img.min()
        )
        kernel1 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
        kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
        for ind in tqdm(range(real_data.shape[0]), desc="Saving"):
            # print(f'saving image {ind}')
            uncertain_fix = cv2.dilate(uncertain_map_img[ind], kernel, iterations=1)
            uncertain_fix = cv2.morphologyEx(
                uncertain_fix, cv2.MORPH_OPEN, kernel1, iterations=1
            )
            plt.subplot(subplot_num[0], subplot_num[1], 1)
            plt.imshow(real_img["rgbd"][ind][..., :3])
            plt.subplot(subplot_num[0], subplot_num[1], 2)
            plt.imshow(visualize_depth(real_img["rgbd"][ind][..., 3])[..., ::-1])
            plt.subplot(subplot_num[0], subplot_num[1], 3)
            plt.imshow(visualize_seg(real_img["mask"][ind], num_classes))
            plt.subplot(subplot_num[0], subplot_num[1], 4)
            plt.imshow(visualize_seg(pred_mc_img[ind], num_classes=num_classes))
            plt.subplot(subplot_num[0], subplot_num[1], 5)
            plt.imshow(uncertain_map_img[ind], cmap="jet")
            plt.subplot(subplot_num[0], subplot_num[1], 6)
            plt.imshow(uncertain_fix, cmap="jet")

            plt.savefig(f"{logfile_path}/images/{ind:03}.jpg", dpi=200)
            plt.close()

        # save quantified results
        real_seg = torch.from_numpy(real_img["mask"]).to(device)  # (N, H, W)
        tp, fp, fn, tn = smp.metrics.get_stats(
            torch.from_numpy(pred_mc_img).long().to(device),
            real_seg.long(),
            mode="multiclass",
            num_classes=num_classes,
        )
        iou_metric = smp.metrics.iou_score(tp, fp, fn, tn, reduction="none")  # (bs, C)
        # f1_metric = smp.metrics.f1_score(tp, fp, fn, tn, reduction='none')  # (bs, C)
        logger.info(
            f"\n miou: {smp.metrics.iou_score(tp, fp, fn, tn, reduction='micro').item()}"
        )
        for ind, key in enumerate(classes):
            logger.info(f"iou_{key}: {torch.mean(iou_metric, dim=0)[ind].item()}")


if __name__ == "__main__":
    device = torch.device("cuda:0")
    classes_name = ["handle", "door", "cabinet", "switchlink", "fixlink", "other"]

    parser = argparse.ArgumentParser(description="test for 90 real images")
    parser.add_argument(
        "--log-name",
        "-ln",
        type=str,
        default="smp_model/20230327_231728_full_splitnet_new_data_aug_ce0.0_dropout0.2",
    )
    parser.add_argument(
        "--only-vis", action="store_true", help="only visualize results or not"
    )
    args = parser.parse_args()

    # generate log
    logfile_path = VISUALMODEL_DIR / f"{args.log_name}/real_results"
    if not os.path.exists(logfile_path):
        os.makedirs(logfile_path / "images")
    logfile_name = f"{time.strftime('%Y%m%d_%H%M%S', time.localtime())}_47arti"
    logger = logging.getLogger(logfile_name)
    logger.setLevel(logging.DEBUG)
    log_file = "{}.log".format(logfile_name)
    fileHandler = logging.FileHandler(os.path.join(logfile_path, log_file), mode="w")
    fileHandler.setLevel(logging.DEBUG)
    consoleHandler = logging.StreamHandler()
    consoleHandler.setLevel(logging.DEBUG)
    formatter = logging.Formatter(
        "[%(asctime)s] {%(filename)s:%(lineno)d} %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    fileHandler.setFormatter(formatter)
    consoleHandler.setFormatter(formatter)
    logger.addHandler(fileHandler)
    logger.addHandler(consoleHandler)

    main(
        logger=logger,
        log_name=args.log_name,
        logfile_path=logfile_path,
        device=device,
        classes=classes_name,
        only_vis=args.only_vis,
    )
