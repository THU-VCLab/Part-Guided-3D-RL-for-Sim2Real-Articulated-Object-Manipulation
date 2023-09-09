import glob
import os
from collections import defaultdict

import cv2
import gym
import matplotlib.pyplot as plt
from arti_mani import RLMODEL_DIR, VISUALMODEL_DIR
from arti_mani.envs import *
from arti_mani.utils.common import convert_np_bool_to_float
from arti_mani.utils.visualization import images_to_video
from arti_mani.utils.wrappers import NormalizeActionWrapper, RenderInfoWrapper
from stable_baselines3.sac import SAC
from tqdm import tqdm


def vis_keypoints(rgb, uvz, uvz_pred, uvz_visable, save_filename):
    rgb_kptgt = rgb.copy()
    for idn in range(uvz.shape[0]):
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
    for idn in range(uvz_pred.shape[0]):
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
    plt.subplot(121)
    plt.imshow(rgb_kptgt)
    plt.title("GT")
    plt.axis("off")
    plt.subplot(122)
    plt.imshow(rgb_kptpred)
    plt.title("Pred")
    plt.axis("off")
    plt.savefig(save_filename, bbox_inches="tight", pad_inches=0.0)


if __name__ == "__main__":
    device = "cuda:0"
    mode = "door"  # "door", "drawer", "faucet"
    test_num = 10
    save_num = 10
    MAX_STEPS = 100
    rl_exp_name = "sac4_tr4eval1door_state_egostereo_keypoints3_bs128_tr16up16"
    smp_exp_name = "20230307_182821_D64H40W64_deconv3_kpts3norm01addvis_uvz_lr1e-3_mobilenetv2_dropout0.2_newdatafilter2drawer"
    eval_steps = []
    for file in glob.glob(str(RLMODEL_DIR / f"{rl_exp_name}/rl_model_*_steps.zip")):
        eval_steps.append(int(os.path.basename(file).split("_")[2]))
    eval_steps.sort()
    eval_steps = np.array(eval_steps)

    env_id = "ArtiMani-v0"
    # door: [0, 1006, 1030, 1047, 1081]
    # drawer: 1, [1045, 1054, 1063, 1067], [1004, 1005, 1016, 1024]
    # faucet: [5002, 5023, 5052, 5069]
    if mode == "door":
        arti_ids = [0, 1006, 1030, 1047, 1081]
    elif mode == "drawer":
        arti_ids = [1, 1004, 1005, 1016, 1024]
    elif mode == "faucet":
        arti_ids = [5052, 5004, 5007, 5023, 5069]
    else:
        raise NotImplementedError(mode)
    kptmodel_path = VISUALMODEL_DIR / f"kpt_model/{smp_exp_name}"
    other_handle_visible = False
    obs_mode = "state_egostereo_keypoints"
    control_mode = "pd_joint_delta_pos"

    env = gym.make(
        env_id,
        articulation_ids=arti_ids,
        segmodel_path=kptmodel_path,
        sample_mode=None,
        frame_num=0,
        sample_num=0,
        other_handle_visible=other_handle_visible,
        obs_mode=obs_mode,
        control_mode=control_mode,
        device=device,
        reward_mode="dense",  # "sparse", "dense"
    )
    env = NormalizeActionWrapper(env)
    env = RenderInfoWrapper(env)
    eval_seed = np.random.RandomState().randint(2**32)
    print("experiment eval random seed: ", eval_seed)
    env.seed(eval_seed)

    success_rates = defaultdict(list)
    success_steps = defaultdict(list)
    uvz_errors = defaultdict(list)
    for eval_step in eval_steps:
        model_path = RLMODEL_DIR / f"{rl_exp_name}/rl_model_{eval_step}_steps"
        RL_model = SAC.load(
            model_path,
            env=env,
            print_system_info=True,
        )
        for arti_id in arti_ids:
            eval_videos_path = (
                RLMODEL_DIR
                / f"{rl_exp_name}/eval_videos/{arti_id}/{eval_step / 1000000}M/"
            )
            if not os.path.exists(eval_videos_path):
                os.makedirs(eval_videos_path)

            is_success = []
            exp_steps = []
            uvz_error = []
            with torch.no_grad():
                for num in tqdm(
                    range(test_num),
                    desc=f"Processing {eval_step / 1000000}M",
                    colour="green",
                    leave=True,
                ):
                    obs = env.reset(articulation_id=arti_id)
                    if num < save_num:
                        frames = []
                        base_rgbd = env.render("cameras")
                        frames.append(base_rgbd)
                    success_flag = False
                    total_steps = MAX_STEPS
                    uvz_err = []
                    # for step in range(MAX_STEPS):
                    for step in tqdm(range(MAX_STEPS), colour="red", leave=False):
                        action, _states = RL_model.predict(obs, deterministic=True)
                        obs, rewards, dones, info = env.step(action)
                        # print(f"{step} step: {info['is_success']}")
                        if num < save_num:
                            base_rgbd = env.render("cameras")
                            frames.append(base_rgbd)
                        uvz = obs["uvz"]
                        uvz_pred = obs["uvz_pred"]
                        uvz_visable = obs["uvz_visable"]
                        uvz_err.append(np.abs(uvz_pred - uvz))
                        if num == 0:
                            rgb = env.get_handsensor_rgb()
                            vis_filename = os.path.join(
                                eval_videos_path, f"{num}_{step}.png"
                            )
                            vis_keypoints(rgb, uvz, uvz_pred, uvz_visable, vis_filename)
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
                    uvz_error.append(
                        np.array(uvz_err).mean(0)
                    )  # (steps, J, 3) -> (J, 3)
            success_rate = np.sum(is_success) / test_num
            mean_steps = np.sum(exp_steps) / test_num
            mean_kpt_err = np.array(uvz_error).mean(0)  # (test_num, J, 3) -> (J, 3)
            print(f"{arti_id}, success_rate, mean_steps: {success_rate}, {mean_steps}")
            print(
                f"{arti_id}, mean tip, center, bottom error: {mean_kpt_err[0]}, {mean_kpt_err[1]}, {mean_kpt_err[2]}"
            )
            success_rates[arti_id].append(success_rate)
            success_steps[arti_id].append(mean_steps)
            uvz_errors[arti_id].append(mean_kpt_err)

    fig, ax = plt.subplots()
    for arti_id in arti_ids:
        ax.plot(
            eval_steps / 1000000, success_rates[arti_id], label=arti_id, color="red"
        )
    ax.legend(loc="lower left")
    ax.set_xlabel("eval_model_steps(M)")
    ax.set_ylabel("success rate")
    ax2 = ax.twinx()
    for arti_id in arti_ids:
        ax2.plot(
            eval_steps / 1000000, success_steps[arti_id], label=arti_id, color="blue"
        )
    ax2.legend(loc="upper right")
    ax2.set_ylabel("mean steps")
    plt.savefig(RLMODEL_DIR / f"{rl_exp_name}/eval_videos/sr-evalsteps.png")
