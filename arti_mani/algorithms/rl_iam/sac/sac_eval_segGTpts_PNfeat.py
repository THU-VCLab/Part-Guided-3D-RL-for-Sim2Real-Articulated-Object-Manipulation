import os
from collections import defaultdict

import gym
from arti_mani import RLMODEL_DIR
from arti_mani.envs import *
from arti_mani.utils.common import convert_np_bool_to_float
from arti_mani.utils.visualization import images_to_video
from arti_mani.utils.wrappers import NormalizeActionWrapper, RenderInfoWrapper
from stable_baselines3.sac import SAC

if __name__ == "__main__":
    device = "cuda:0"
    mode = "drawer"  # "door", "drawer", "faucet", "arti"
    test_num = 10
    save_num = 10
    success_nums = 0
    MAX_STEPS = 100
    if mode == "arti":
        rl_exp_names = sorted(
            [
                name
                for name in os.listdir(RLMODEL_DIR / "hybrid/Oracle-based RL")
                if mode in name
            ]
        )
    else:
        rl_exp_names = sorted(
            [
                name
                for name in os.listdir(RLMODEL_DIR / "Oracle-based RL")
                if mode in name
            ]
        )

    env_id = "ArtiMani-v0"
    if mode == "door":
        arti_ids = [0]
    elif mode == "drawer":
        arti_ids = [1]
    elif mode == "faucet":
        arti_ids = [5052]
    elif mode == "arti":
        arti_ids = [0, 1, 5052]
    else:
        raise NotImplementedError(mode)
    other_handle_visible = False
    obs_mode = "state_egostereo_segpoints_gt"
    control_mode = "pd_joint_delta_pos"
    sample_mode = "random_sample"  # full_downsample, score_sample, random_sample, frameweight_sample
    frame_num = 1
    num_classes = 6
    state_repeat_times = 10
    pts_sample_num = 32

    env = gym.make(
        env_id,
        articulation_ids=arti_ids,
        segmodel_path=None,
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
    eval_seed = np.random.RandomState().randint(2**32)
    print("experiment eval random seed: ", eval_seed)
    env.seed(eval_seed)

    success_rates = defaultdict(list)
    success_steps = defaultdict(list)
    for rl_exp_name in rl_exp_names:
        model_path = RLMODEL_DIR / f"{rl_exp_name}/best_model"
        RL_model = SAC.load(
            model_path,
            env=env,
            print_system_info=True,
        )
        for arti_id in arti_ids:
            eval_videos_path = (
                RLMODEL_DIR / f"{rl_exp_name}/eval_videos/{arti_id}_best_model/"
            )
            if not os.path.exists(eval_videos_path):
                os.makedirs(eval_videos_path)

            is_success = []
            exp_steps = []
            with torch.no_grad():
                for num in range(test_num):
                    obs = env.reset(articulation_id=arti_id)
                    if num < save_num:
                        frames = []
                        base_rgbd = env.render("cameras")
                        frames.append(base_rgbd)
                    success_flag = False
                    total_steps = 0
                    for step in range(MAX_STEPS):
                        action, _states = RL_model.predict(obs, deterministic=True)
                        obs, rewards, dones, info = env.step(action)
                        # print(f"{i} step: {info['is_success']}")
                        if not success_flag and info["is_success"] == 1.0:
                            success_nums += 1
                            total_steps = step
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

    for key in success_rates.keys():
        print("success rates: ", key, success_rates[key])
    for key in success_steps.keys():
        print("success steps: ", key, success_steps[key])
