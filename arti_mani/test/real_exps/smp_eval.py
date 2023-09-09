import glob
import logging
import os
import time

import cv2
import gym
import matplotlib.pyplot as plt
import segmentation_models_pytorch as smp
from arti_mani import ROOT_DIR, VISUALMODEL_DIR
from arti_mani.envs import *
from arti_mani.utils.cv_utils import visualize_depth
from arti_mani.utils.wrappers import NormalizeActionWrapper
from PIL import Image


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
    inds = [92]  # list(range(0, 16))
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

    smp_exps = [
        # '20230106_222106_cab41-15_faucet15-3_newaug_bs256_0.8step20lr0.001_unet3-mobilenet_v2_good',
        # '20230108_150002_cab41-15_faucet15-3_newaug_bs256_0.5step50lr0.001_unet3_3264128-mobilenet_v2_good',
        # '20230108_161133_cab41-15_faucet15-3_newaug_bs256_0.5step50lr0.001_unet3_163264-mobilenet_v2_good',
        # '20230109_133653_cab41-15_faucet11-3_newaug_bs256_0.5step50lr0.001_unet3-163264_mobilenet_v2_good',
        #
        # '20230110_021737_cab39-17_faucet11-3_newaug_bs256_0.5step50lr0.001_RGBDunet3-163264_mobilenet_v2',
        # '20230110_103746_cab39-17_faucet11-3_newaug_bs256_0.5step50lr0.001_RGBDunet3-3264128_mobilenet_v2',
        # '20230110_030002_cab39-17_faucet11-3_newaug_bs256_0.5step50lr0.001_RGBunet3-163264_mobilenet_v2',
        # '20230110_110619_cab39-17_faucet11-3_newaug_bs256_0.5step50lr0.001_RGBunet3-3264128_mobilenet_v2',
        # '20230110_021934_cab39-17_faucet11-3_newaug_bs256_0.5step50lr0.001_Dunet3-163264_mobilenet_v2',
        # '20230110_133947_cab39-17_faucet11-3_newaug_bs256_0.5step50lr0.001_Dunet3-3264128_mobilenet_v2',
        "20230111_005201_cab39-17_faucet11-3_stereo_bs256_0.5step50lr0.001_RGBDunet3-163264_mobilenet_v2",
        # '20230111_014201_cab39-17_faucet11-3_stereo_bs256_0.5step50lr0.001_RGBDunet3-3264128_mobilenet_v2',
        # '20230111_130024_cab39-17_faucet11-3_stereo_bs256_0.5step50lr0.001_RGBunet3-163264_mobilenet_v2',
        # '20230111_100917_cab39-17_faucet11-3_stereo_bs256_0.5step50lr0.001_RGBunet3-3264128_mobilenet_v2',
        # '20230111_005327_cab39-17_faucet11-3_stereo_bs256_0.5step50lr0.001_Dunet3-163264_mobilenet_v2',
        # '20230111_012400_cab39-17_faucet11-3_stereo_bs256_0.5step50lr0.001_Dunet3-3264128_mobilenet_v2',
    ]

    logfile_path = VISUALMODEL_DIR / f"smp_model/"
    logfile_name = (
        f"{time.strftime('%Y%m%d_%H%M%S', time.localtime())}_real_smp_evalresults"
    )
    logger = logging.getLogger(logfile_name)
    logger.setLevel(logging.DEBUG)
    # set two handlers
    log_file = "{}.log".format(logfile_name)
    # rm_file(log_file)
    fileHandler = logging.FileHandler(os.path.join(logfile_path, log_file), mode="w")
    fileHandler.setLevel(logging.DEBUG)
    # consoleHandler = logging.StreamHandler()
    # consoleHandler.setLevel(logging.DEBUG)
    # set formatter
    formatter = logging.Formatter(
        "[%(asctime)s] {%(filename)s:%(lineno)d} %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    # consoleHandler.setFormatter(formatter)
    fileHandler.setFormatter(formatter)
    # add
    logger.addHandler(fileHandler)
    # logger.addHandler(consoleHandler)

    # "state", "state_dict", "rgbd", "pointcloud", "state_depth", "state_egorgbd", "state_egopcrgb", "state_crossview"
    obs_mode = "state_egorgbd"
    # "pd_joint_delta_pos", "pd_ee_delta_pose", "pd_ee_delta_pos"
    control_mode = "pd_joint_delta_pos"
    env_id = "OpenCabinetDoor-v0"
    arti_ids = [0]
    arti_config_path = (
        ASSET_DIR / "partnet_mobility_configs/fixed_cabinet_doors_new.yml"
    )

    env = gym.make(
        env_id,
        articulation_ids=arti_ids,
        articulation_config_path=arti_config_path,
        obs_mode=obs_mode,
        control_mode=control_mode,
        reward_mode="dense",
    )
    env = NormalizeActionWrapper(env)
    eval_seed = np.random.RandomState().randint(2**32)
    logger.info(f"eval env random seed: {eval_seed}")
    env.seed(eval_seed)
    imgs = env.unwrapped._agent.get_images()
    hand_camera_extrinsic_base_frame = np.asarray(
        imgs["hand_camera"]["camera_extrinsic_base_frame"]
    )
    logger.info(f"hand_camera_extrinsic_base_frame: {hand_camera_extrinsic_base_frame}")

    for smp_exp_name in smp_exps:
        logger.info(f"{smp_exp_name} real visual results.")
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

        eval_result_path = (
            VISUALMODEL_DIR / f"smp_model/{smp_exp_name}/real_results_final/"
        )
        if not os.path.exists(eval_result_path):
            os.makedirs(eval_result_path)

        logger.info(f"model params: {model_params/1e6:.2f} M")

        for ind in inds:
            real_data_path = ROOT_DIR / f"test/real_exps/real_images/"
            # cur_rgb = np.array(Image.open(real_data_path / f"{ind:02}_rgb.png"))
            # cur_depth = np.array(Image.open(real_data_path / f"{ind:02}_depth.png"))
            cur_rgb = np.load(real_data_path / f"{ind:02}_rgb.npy")[..., ::-1]
            cur_depth = np.load(real_data_path / f"{ind:02}_depth.npy")
            logger.info(f"real rgb, depth shape: {cur_rgb.shape}, {cur_depth.shape}")
            logger.info(f"real rgb min, max: {np.min(cur_rgb)}, {np.max(cur_rgb)}")
            logger.info(
                f"real depth min, max: {np.min(cur_depth)}, {np.max(cur_depth)}"
            )
            with torch.no_grad():
                # proc_rgb = cv2.resize(cur_rgb, (128, 72)) / 255.0
                # proc_depth = cv2.resize(cur_depth, (128, 72))
                proc_rgb = cur_rgb / 255.0
                proc_depth = cur_depth

                real_rgb = (
                    torch.from_numpy(proc_rgb).float().to(device).permute((2, 0, 1))
                )  # (3, H, W)
                real_depth = (
                    torch.from_numpy(proc_depth[None]).float().to(device)
                )  # (1, H, W)

                if smp_cfg["mode"] == "RGBD":
                    img_input = torch.cat([real_rgb, real_depth], dim=0)[None]
                elif smp_cfg["mode"] == "RGB":
                    img_input = real_rgb[None]
                elif smp_cfg["mode"] == "D":
                    img_input = real_depth[None]
                else:
                    raise NotImplementedError

                time0 = time.time()
                attnmap = unet_model.predict(img_input)  # (1, 6, H, W)
                time1 = time.time()
                infer_times = (time1 - time0) * 1000

                plot_rgb = real_rgb.detach().cpu().numpy().transpose((1, 2, 0))
                plot_depth = real_depth.detach().squeeze().cpu().numpy()
                attnmap = attnmap.squeeze().cpu().numpy()

                plt.subplot(subplot_num[0], subplot_num[1], 1)
                plt.imshow(plot_rgb)
                plt.subplot(subplot_num[0], subplot_num[1], 2)
                plt.imshow(plot_depth)
                for pid in range(num_classes):
                    plt.subplot(subplot_num[0], subplot_num[1], pid + 3)
                    plt.imshow(attnmap[pid])

                plt.savefig(f"{eval_result_path}/test_{ind:02}.jpg", dpi=200)
                plt.close()

                depth_colormap = visualize_depth(cur_depth)
                cv2.imwrite(
                    f"{eval_result_path}/{ind:03}_depth_colormap.png", depth_colormap
                )

                logger.info(f"inference time: {infer_times:.2f} ms")
