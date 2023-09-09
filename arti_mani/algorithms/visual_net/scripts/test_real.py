import argparse
import glob
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
from arti_mani.utils.cv_utils import visualize_depth, visualize_seg


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
    seg_model = smp.Unet(
        encoder_name=smp_cfg["encoder"],
        encoder_depth=smp_cfg["encoder_depth"],
        decoder_channels=smp_cfg["decoder_channels"],
        encoder_weights=smp_cfg["encoder_weights"],
        in_channels=in_ch,
        classes=cfg["num_classes"],
        activation=smp_cfg["activation"],
    )
    seg_model.load_state_dict(torch.load(model_path))
    seg_model.to(device)
    seg_model.eval()
    model_params = sum(p.numel() for p in seg_model.parameters())
    logger.info(f"model params: {model_params / 1e6:.2f} M")
    num_classes = cfg["num_classes"]
    IM_SIZE = (256, 144)
    if smp_cfg["encoder_depth"] == 5:
        PAD_SIZE = (256, 160)
    else:
        PAD_SIZE = (256, 144)

    # load real data and preprocess
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
                    data /= 255.0
                    data = data[..., ::-1]
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
    real_rgbd = (
        torch.from_numpy(real_data).float().permute(0, 3, 1, 2).to(device)
    )  # (N, ch, H, W)
    pred_seg = seg_model.forward(real_rgbd)  # (N, 6, H, W)
    pred_mc = torch.argmax(pred_seg, dim=1).to(device)  # (N, H, W)

    # save qualified results
    subplot_num = [3, 3]
    seg_pred_img = pred_seg.detach().cpu().numpy()
    for ind in range(real_rgbd.shape[0]):
        plt.subplot(subplot_num[0], subplot_num[1], 1)
        plt.imshow(real_img["rgb"][ind])
        plt.subplot(subplot_num[0], subplot_num[1], 2)
        plt.imshow(visualize_depth(real_img["depth"][ind])[..., ::-1])
        plt.subplot(subplot_num[0], subplot_num[1], 3)
        plt.imshow(visualize_seg(real_img["masks"][ind], num_classes))
        for pid in range(num_classes):
            plt.subplot(subplot_num[0], subplot_num[1], pid + 4)
            plt.imshow(seg_pred_img[ind][pid])

        plt.savefig(f"{logfile_path}/{ind:03}.jpg", dpi=200)

    # save quantified results
    if not only_vis:
        real_seg = torch.from_numpy(real_img["masks"]).to(device)  # (N, H, W)
        tp, fp, fn, tn = smp.metrics.get_stats(
            pred_mc.long(), real_seg.long(), mode="multiclass", num_classes=num_classes
        )
        iou_metric = smp.metrics.iou_score(tp, fp, fn, tn, reduction="none")  # (bs, C)
        f1_metric = smp.metrics.f1_score(tp, fp, fn, tn, reduction="none")  # (bs, C)
        logger.info(
            f"\n miou: {smp.metrics.iou_score(tp, fp, fn, tn, reduction='micro').item()}"
        )
        for ind, key in enumerate(classes):
            logger.info(f"iou_{key}: {torch.mean(iou_metric, dim=0)[ind].item()}")
        logger.info(
            f"\n mf1: {smp.metrics.f1_score(tp, fp, fn, tn, reduction='micro').item()}"
        )
        for ind, key in enumerate(classes):
            logger.info(f"f1_{key}: {torch.mean(f1_metric, dim=0)[ind].item()}")


if __name__ == "__main__":
    device = torch.device("cuda:0")
    classes_name = ["handle", "door", "cabinet", "switchlink", "fixlink", "other"]

    parser = argparse.ArgumentParser(description="test for 90 real images")
    parser.add_argument(
        "--log-name",
        "-ln",
        type=str,
        default="smp_model/20230216_151449_train50-val20_noDR_norandombg_aug_stereo_bs64_cel_0.5step50lr0.001_RGBDunet4-163264128_mobilenet_v2",
    )
    parser.add_argument(
        "--only-vis", action="store_true", help="only visualize results or not"
    )
    args = parser.parse_args()

    # generate log
    logfile_path = VISUALMODEL_DIR / f"{args.log_name}/real_results/"
    if not os.path.exists(logfile_path):
        os.makedirs(logfile_path)
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
