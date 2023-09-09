import glob
import logging
import os
import time
from collections import defaultdict

import cv2
import gym
import matplotlib.pyplot as plt
import segmentation_models_pytorch as smp
from arti_mani import RLMODEL_DIR, VISUALMODEL_DIR
from arti_mani.envs import *
from arti_mani.utils.cv_utils import visualize_depth
from arti_mani.utils.wrappers import NormalizeActionWrapper
from PIL import Image
from stable_baselines3.sac import SAC


def make_gif(frame_path, result_prefix):
    file_list = sorted(
        glob.glob(f"{frame_path}/*.jpg"), key=lambda name: int(name[-7:-4])
    )
    frames = [Image.open(image) for image in file_list]
    frame_one = frames[0]
    frame_one.save(
        f"{frame_path}/{result_prefix}.gif",
        format="GIF",
        append_images=frames,
        save_all=True,
        duration=500,
        loop=0,
    )


if __name__ == "__main__":
    device = torch.device("cuda:0")
    # visual_models = ["51200_epo", "102400_epo", "716800_epo", "1024000_epo"]
    eval_steps = np.array([5000000])
    color_maps = [
        (0, 255, 0),
        (0, 0, 255),
        (255, 255, 0),
        (0, 255, 255),
        (255, 0, 0),
        (255, 255, 255),
    ]
    subplot_num = [3, 3]
    num_classes = 6
    obs_mode = "state_egostereo_rgbd"
    control_mode = "pd_joint_delta_pos"
    rl_exp_name = "sac4_4doors_rgbdunet-mobilev2_6segmap180_pn256ln_state12pad10"

    smp_exps = [
        "20230111_005201_cab39-17_faucet11-3_stereo_bs256_0.5step50lr0.001_RGBDunet3-163264_mobilenet_v2",
    ]

    logfile_path = VISUALMODEL_DIR / f"smp_model"
    logfile_name = f"{time.strftime('%Y%m%d_%H%M%S', time.localtime())}_eval_smp_result"
    logger = logging.getLogger(logfile_name)
    logger.setLevel(logging.DEBUG)
    # set two handlers
    log_file = "{}.log".format(logfile_name)
    # rm_file(log_file)
    fileHandler = logging.FileHandler(os.path.join(logfile_path, log_file), mode="w")
    fileHandler.setLevel(logging.DEBUG)
    # set formatter
    formatter = logging.Formatter(
        "[%(asctime)s] {%(filename)s:%(lineno)d} %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    fileHandler.setFormatter(formatter)
    # add
    logger.addHandler(fileHandler)

    for smp_exp_name in smp_exps:
        smp_model_path = VISUALMODEL_DIR / f"smp_model/{smp_exp_name}/best.pth"
        config_path = VISUALMODEL_DIR / f"smp_model/{smp_exp_name}/config.yaml"
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
        unet_model = smp.Unet(
            encoder_name=smp_cfg["encoder"],
            encoder_depth=smp_cfg["encoder_depth"],
            decoder_channels=smp_cfg["decoder_channels"],
            encoder_weights=smp_cfg["encoder_weights"],
            in_channels=in_ch,
            classes=cfg["num_classes"],
            activation=smp_cfg["activation"],
        )
        unet_model.load_state_dict(torch.load(smp_model_path))
        unet_model.to(device)
        unet_model.eval()
        model_params = sum(p.numel() for p in unet_model.parameters())

        # door
        env_id = "OpenCabinetDoor-v0"
        arti_ids = [0, 1006, 1030, 1047, 1081]
        arti_config_path = (
            ASSET_DIR / "partnet_mobility_configs/fixed_cabinet_doors_new.yml"
        )

        # create env
        env = gym.make(
            env_id,
            articulation_ids=arti_ids,
            articulation_config_path=arti_config_path,
            obs_mode=obs_mode,
            control_mode=control_mode,
            reward_mode="dense",
        )
        env = NormalizeActionWrapper(env)

        for eval_step in eval_steps:
            model_path = RLMODEL_DIR / f"{rl_exp_name}/rl_model_{eval_step}_steps.zip"

            # load models
            model = SAC.load(
                model_path,
                env=env,
                print_system_info=True,
            )
            sample_pts = model.actor.features_extractor.sample_num

            for arti_id in arti_ids:
                eval_result_path = (
                    VISUALMODEL_DIR
                    / f"smp_model/{smp_exp_name}/vis_results/{arti_id}_{eval_step / 1000000}M_new"
                )
                # print("visual results dir: ", eval_result_path)
                logger.info(
                    f"{smp_exp_name}: {arti_id}, {rl_exp_name}-{eval_step / 1000000}M"
                )
                if not os.path.exists(eval_result_path):
                    os.makedirs(eval_result_path)

                logger.info(f"model params: {model_params/1e6:.2f} M")

                eval_seed = np.random.RandomState().randint(2**32)
                # print("experiment eval random seed: ", eval_seed)
                logger.info(f"experiment eval random seed: {eval_seed}")
                env.seed(eval_seed)

                with torch.no_grad():
                    obs = env.reset(articulation_id=arti_id)
                    infer_times = []
                    eval_items = defaultdict(list)
                    for ind in range(30):
                        rgb = obs["rgb"]  # (3,72,128)
                        rgb_norm = torch.from_numpy((rgb / 255.0)[None]).to(
                            device
                        )  # Normalization
                        ch, h, w = rgb.shape

                        depth = obs["depth"]  # (1,72,128)
                        # depth_norm = torch.from_numpy((depth / np.max(depth))[None]).to(device=device)
                        depth_norm = torch.from_numpy(depth[None]).to(device=device)
                        seg = obs["seg"].astype(np.uint8)  # (72, 128)
                        seg_torch = torch.from_numpy(seg[None]).long().to(device)
                        seg_gt_onehot = F.one_hot(
                            seg_torch, num_classes=cfg["num_classes"]
                        ).permute(
                            0, 3, 1, 2
                        )  # (1,C,72,128)
                        # other_seg = obs["other_handleseg"]
                        # print(rgb_norm.shape, depth_norm.shape, seg.shape)
                        if smp_cfg["mode"] == "RGBD":
                            img = torch.cat([rgb_norm, depth_norm], dim=1)
                        elif smp_cfg["mode"] == "RGB":
                            img = rgb_norm
                        elif smp_cfg["mode"] == "D":
                            img = depth_norm
                        else:
                            raise NotImplementedError
                        time0 = time.time()
                        # seg_feat = unet_model.predict(img)  # (1, C, H, W)
                        # attnmap = F.softmax(seg_feat, dim=1)  # (1, C, H, W)
                        attnmap = unet_model.predict(img)  # (1, C, H, W)
                        time1 = time.time()
                        infer_times.append(time1 - time0)

                        tp, fp, fn, tn = smp.metrics.get_stats(
                            attnmap, seg_gt_onehot, mode="multilabel", threshold=0.5
                        )

                        iou_metric = smp.metrics.iou_score(
                            tp, fp, fn, tn, reduction="none"
                        )  # (bs, C)
                        f1_metric = smp.metrics.f1_score(
                            tp, fp, fn, tn, reduction="none"
                        )  # (bs, C)
                        for cls_ind, key in enumerate(smp_cfg["classes"]):
                            eval_items[f"iou_{key}"].append(
                                torch.mean(iou_metric, dim=0)[cls_ind].item()
                            )
                            eval_items[f"f1_{key}"].append(
                                torch.mean(f1_metric, dim=0)[cls_ind].item()
                            )
                        # then compute metrics with required reduction (see metric docs)
                        eval_items["iou"].append(
                            smp.metrics.iou_score(
                                tp, fp, fn, tn, reduction="micro"
                            ).item()
                        )
                        eval_items["f1"].append(
                            smp.metrics.f1_score(
                                tp, fp, fn, tn, reduction="micro"
                            ).item()
                        )
                        eval_items["acc"].append(
                            smp.metrics.accuracy(
                                tp, fp, fn, tn, reduction="macro"
                            ).item()
                        )
                        eval_items["prec"].append(
                            smp.metrics.precision(
                                tp, fp, fn, tn, reduction="micro-imagewise"
                            ).item()
                        )
                        eval_items["recall"].append(
                            smp.metrics.recall(
                                tp, fp, fn, tn, reduction="micro-imagewise"
                            ).item()
                        )

                        plot_rgb = (rgb.transpose((1, 2, 0))).astype(np.uint8)
                        attnmap = attnmap.squeeze().cpu().numpy()

                        plt.subplot(subplot_num[0], subplot_num[1], 1)
                        plt.imshow(plot_rgb / 255.0)
                        plt.subplot(subplot_num[0], subplot_num[1], 2)
                        plt.imshow(depth.squeeze())
                        plt.subplot(subplot_num[0], subplot_num[1], 3)
                        plt.imshow(seg)
                        for pid in range(num_classes):
                            plt.subplot(subplot_num[0], subplot_num[1], pid + 4)
                            plt.imshow(attnmap[pid])

                        plt.savefig(f"{eval_result_path}/{ind:03}.jpg", dpi=200)
                        plt.close()

                        depth_colormap = visualize_depth(depth.squeeze())
                        cv2.imwrite(
                            f"{eval_result_path}/{ind:03}_depth_colormap.png",
                            depth_colormap,
                        )

                        action, _states = model.predict(obs, deterministic=True)
                        obs, rewards, dones, info = env.step(action)
                        # print(f"{ind} step: {info['is_success']}, {info['check_grasp']}")
                        if info["is_success"] == 1.0:
                            break
                # print(f"inference time: {np.mean(infer_times)}")
                logger.info(f"inference time: {np.mean(infer_times)*1000:.2f} ms")
                for key in eval_items.keys():
                    logger.info(f"metrics {key}: {np.mean(eval_items[key]):.4f}")
                make_gif(eval_result_path, f"{eval_step / 1000000}M")
