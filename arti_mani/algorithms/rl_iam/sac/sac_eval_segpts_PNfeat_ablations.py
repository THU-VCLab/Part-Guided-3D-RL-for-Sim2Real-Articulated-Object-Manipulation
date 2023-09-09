import os
from collections import defaultdict

import cv2
import gym
import matplotlib.pyplot as plt
from arti_mani import RLMODEL_DIR, VISUALMODEL_DIR
from arti_mani.algorithms.visual_net.Networks.Custom_Unet import CustomUnet
from arti_mani.envs import *
from arti_mani.utils.common import convert_np_bool_to_float
from arti_mani.utils.cv_utils import visualize_depth, visualize_seg
from arti_mani.utils.visualization import images_to_video
from arti_mani.utils.wrappers import NormalizeActionWrapper, RenderInfoWrapper
from stable_baselines3.sac import SAC
from tqdm import tqdm

CMAPS = [
    (1, 0, 0),  # "red"
    (0, 0, 1),  # "blue"
    (1, 1, 0),  # "yellow"
    (0, 1, 0),  # "green"
    (0.627, 0.125, 0.941),  # "purple"
    (0.753, 0.753, 0.753),  # "grey"
]


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


def vis_sample_pts(
    rgbdseg, indices, num_classes, pts_sample_num, vis_model, device, save_filename
):
    assert indices.shape[0] == num_classes * pts_sample_num
    rgb, depth, seg = (
        rgbdseg["rgb"].transpose((1, 2, 0)) / 255.0,
        rgbdseg["depth"],
        rgbdseg["seg"],
    )
    h, w = 144, 256
    v = indices // w  # (C * sample_num)
    u = indices - v * w  # (C * sample_num)
    rgb_plt = rgb.copy()
    for idn in range(num_classes):
        for pt_num in range(pts_sample_num):
            cv2.circle(
                rgb_plt,
                (
                    round(u[idn * pts_sample_num + pt_num]),
                    round(v[idn * pts_sample_num + pt_num]),
                ),
                radius=2,
                color=CMAPS[idn],
                thickness=-1,
            )

    with torch.no_grad():
        rgb_tensor = torch.from_numpy(rgb.transpose(2, 0, 1)[None]).to(
            device, dtype=torch.float32
        )
        depth_tensor = torch.from_numpy(depth[None, None]).to(
            device, dtype=torch.float32
        )
        seg_pred = vis_model.predict(torch.cat((rgb_tensor, depth_tensor), dim=1))[0]
        seg_mc_label = torch.argmax(seg_pred, dim=0).cpu().numpy()

    plt.subplot(231)
    plt.imshow(rgb)
    plt.title("rgb")
    plt.axis("off")
    plt.subplot(232)
    plt.imshow(visualize_depth(depth))
    plt.title("depth")
    plt.axis("off")
    plt.subplot(233)
    plt.imshow(visualize_seg(seg))
    plt.title("seg")
    plt.axis("off")
    plt.subplot(234)
    plt.imshow(visualize_seg(seg_mc_label, 6))
    plt.title("segpred")
    plt.axis("off")
    plt.subplot(235)
    plt.imshow(rgb_plt)
    plt.title("sample_pts")
    plt.axis("off")
    plt.savefig(save_filename, bbox_inches="tight", pad_inches=0.0)


if __name__ == "__main__":
    device = "cuda:0"
    mode = "door"  # "door", "drawer", "faucet", "arti"
    test_trials = 3
    test_num = 50
    save_num = 0
    MAX_STEPS = 100
    arti_id = 0

    smp_exp_name = "20230219_000940_train52-val18_384_noDR_randombg_aug_stereo_bs16_focalloss_0.5step50lr0.001_RGBDunet-163264128_mobilenet_v2"
    vis_model = load_vismodel(smp_exp_name, device)
    segmodel_path = VISUALMODEL_DIR / f"smp_model/{smp_exp_name}"
    other_handle_visible = False
    obs_mode = "state_egostereo_segpoints"
    control_mode = "pd_joint_delta_pos"
    sample_mode = "random_sample"  # full_downsample, score_sample, random_sample, frameweight_sample, fps_sample
    frame_num = 1
    num_classes = 6
    state_repeat_times = 10
    pts_sample_num = 8

    rl_exp_name = "ablations/sac4_tr4eval1door_state_egostereo_segpoints_random_samplef1s8_smp0219_2"
    model_path = (
        RLMODEL_DIR / f"{rl_exp_name}/best_model"
    )  # rl_model_2000000_steps, best_model

    env = gym.make(
        "ArtiMani-v0",
        articulation_ids=[arti_id],
        segmodel_path=segmodel_path,
        sample_mode=sample_mode,
        frame_num=frame_num,
        sample_num=pts_sample_num,
        other_handle_visible=other_handle_visible,
        obs_mode=obs_mode,
        control_mode=control_mode,
        device=device,
        reward_mode="dense",  # "sparse", "dense"
    )
    env = NormalizeActionWrapper(env)
    env = RenderInfoWrapper(env)
    RL_model = SAC.load(
        model_path,
        env=env,
        print_system_info=True,
    )

    success_rates = defaultdict(list)
    success_steps = defaultdict(list)
    for test_id in range(test_trials):
        eval_videos_path = (
            RLMODEL_DIR / f"{rl_exp_name}/eval_videos/0_best_model_{test_id}/"
        )
        if not os.path.exists(eval_videos_path):
            os.makedirs(eval_videos_path)

        eval_seed = np.random.RandomState().randint(2**32)
        print("experiment eval random seed: ", eval_seed)
        env.seed(eval_seed)

        is_success = []
        exp_steps = []
        with torch.no_grad():
            # for num in range(test_num):
            for num in tqdm(
                range(test_num),
                desc=f"Processing best RL model",
                colour="green",
                leave=True,
            ):
                obs = env.reset(articulation_id=arti_id)
                if num < save_num:
                    frames = []
                    base_rgbd = env.render("cameras")
                    frames.append(base_rgbd)
                success_flag = False
                total_steps = 0
                for step in tqdm(range(MAX_STEPS), colour="red", leave=False):
                    action, _states = RL_model.predict(obs, deterministic=True)
                    obs, rewards, dones, info = env.step(action)
                    # print(f"{step} step: {info['is_success']}")
                    if num < save_num:
                        base_rgbd = env.render("cameras")
                        frames.append(base_rgbd)
                    if num == 0:
                        rgbdseg = env.get_handsensor_rgbdseg()
                        indices = env.sampler.sample_indices  # C*sample_num
                        vis_filename = os.path.join(
                            eval_videos_path, f"{num}_{step}.png"
                        )
                        vis_sample_pts(
                            rgbdseg,
                            indices,
                            num_classes,
                            pts_sample_num,
                            vis_model,
                            device,
                            vis_filename,
                        )
                    if not success_flag and info["is_success"] == 1.0:
                        success_flag = True
                        total_steps = step
                        break
                if num < save_num:
                    images_to_video(
                        images=frames,
                        output_dir=eval_videos_path,
                        video_name=f"{num}",
                        verbose=False,
                    )
                is_success.append(convert_np_bool_to_float(success_flag))
                exp_steps.append(total_steps)
        success_rate = np.sum(is_success) / test_num
        exp_steps = np.array(exp_steps)
        success_mask = exp_steps > 0
        if np.any(success_mask):
            mean_steps = np.mean(exp_steps[success_mask])
        else:
            mean_steps = MAX_STEPS
        print(f"{arti_id}, success_rate, mean_steps: {success_rate}, {mean_steps}")
        success_rates[arti_id].append(success_rate)
        success_steps[arti_id].append(mean_steps)
    print(f"{arti_id}, {test_trials} trials, success_rates: {success_rates[arti_id]}")
    print(f"{arti_id}, {test_trials} trials, success_steps: {success_steps[arti_id]}")
    sr_mean, sr_std = np.mean(success_rates[arti_id]), np.std(success_rates[arti_id])
    ss_mean, ss_std = np.mean(success_steps[arti_id]), np.std(success_steps[arti_id])
    print(f"{test_trials} trials, success_rates mean, std: {sr_mean}, {sr_std}")
    print(f"{test_trials} trials, success_steps mean, std: {ss_mean}, {ss_std}")
