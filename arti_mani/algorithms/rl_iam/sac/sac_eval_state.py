import glob
import os
from collections import defaultdict

import gym
import matplotlib.pyplot as plt
import torch
from arti_mani import RLMODEL_DIR
from arti_mani.envs import *
from arti_mani.utils.common import convert_np_bool_to_float
from arti_mani.utils.visualization import images_to_video
from arti_mani.utils.wrappers import NormalizeActionWrapper, RenderInfoWrapper
from stable_baselines3.sac import SAC

if __name__ == "__main__":
    device = "cuda:0"
    mode = "door"  # "door", "drawer", "faucet"
    test_num = 10
    save_num = 10
    success_nums = 0
    MAX_STEPS = 50
    rl_exp_name = "sac4_4faucet_state_30tip40finger_gripperclose_leftrew"
    eval_steps = []
    for file in glob.glob(str(RLMODEL_DIR / f"{rl_exp_name}/rl_model_*_steps.zip")):
        eval_steps.append(int(os.path.basename(file).split("_")[2]))
    eval_steps.sort()
    eval_steps = np.array(eval_steps)

    env_id = "ArtiMani-v0"
    other_handle_visible = False
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
    obs_mode = "state_dict"
    control_mode = "pd_joint_delta_pos"
    sample_mode = "random_sample"  # full_downsample, score_sample, random_sample, frameweight_sample
    frame_num = 1
    num_classes = 6
    state_repeat_times = 10
    pts_sample_num = 30

    env = gym.make(
        env_id,
        articulation_ids=arti_ids,
        segmodel_path=None,
        load_device="cuda:0",
        sample_mode=sample_mode,
        frame_num=frame_num,
        sample_num=pts_sample_num,
        other_handle_visible=False,
        obs_mode=obs_mode,
        control_mode=control_mode,
        reward_mode="dense",  # "sparse", "dense"
    )
    env = NormalizeActionWrapper(env)
    env = RenderInfoWrapper(env)
    eval_seed = np.random.RandomState().randint(2**32)
    print("experiment eval random seed: ", eval_seed)
    env.seed(eval_seed)

    success_rates = defaultdict(list)
    for eval_step in eval_steps:
        model_path = RLMODEL_DIR / f"{rl_exp_name}/rl_model_{eval_step}_steps"
        model = SAC.load(
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
            with torch.no_grad():
                for num in range(test_num):
                    obs = env.reset(articulation_id=arti_id)
                    if num < save_num:
                        frames = []
                        base_rgbd = env.render("cameras")
                        frames.append(base_rgbd)
                    success_flag = False
                    for i in range(MAX_STEPS):
                        action, _states = model.predict(obs, deterministic=True)
                        obs, rewards, dones, info = env.step(action)
                        # print(f"{i} step: {info['is_success']}")
                        if not success_flag and info["is_success"] == 1.0:
                            success_nums += 1
                            success_flag = True
                        if num < save_num:
                            base_rgbd = env.render("cameras")
                            frames.append(base_rgbd)
                    if num < save_num:
                        images_to_video(
                            images=frames,
                            output_dir=eval_videos_path,
                            video_name=f"{num}",
                        )
                    is_success.append(convert_np_bool_to_float(success_flag))
            sum_num = np.sum(is_success)
            print(f"success rate: {sum_num}/{test_num}")
            success_rates[arti_id].append(sum_num / test_num)

    plt.figure()
    for arti_id in arti_ids:
        plt.plot(eval_steps / 1000000, success_rates[arti_id], label=arti_id)
    plt.legend(loc="lower right")
    plt.xlabel("eval_model_steps(M)")
    plt.ylabel("success_rate")
    plt.title(f"Eval on {len(arti_ids)} {mode}(s)")
    plt.savefig(RLMODEL_DIR / f"{rl_exp_name}/eval_videos/sr-evalsteps.png")
