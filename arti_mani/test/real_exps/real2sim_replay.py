import glob
import os

import gym
import matplotlib.pyplot as plt
import numpy as np
import sapien.core as sapien
import torch
from arti_mani import REALEXP_DIR
from arti_mani.envs.arti_mani import ArtiMani
from arti_mani.utils.sapien_utils import get_entity_by_name
from arti_mani.utils.visualization import images_to_video
from arti_mani.utils.wrappers import NormalizeActionWrapper


def setup_seed(seed=1029):
    os.environ["PYTHONHASHSEED"] = str(seed)  # 为了禁止hash随机化，使得实验可复现
    np.random.seed(seed)
    rng = np.random.RandomState(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    return rng


def scale_para(para, output_low, output_high):
    """
    para: norm to [-1, 1]
    output_low, output_high: reproject to (output_low, output_high)
    """
    bias = 0.5 * (output_high + output_low)
    weight = 0.5 * (output_high - output_low)
    output = weight * para + bias

    return output


if __name__ == "__main__":
    np.set_printoptions(suppress=True, precision=4)
    device = "cuda:0"
    mode = "faucet"  # door, drawer, faucet
    record_id = 3
    save_mode = True
    eval_seed = np.random.RandomState().randint(2**32)
    rng = setup_seed(eval_seed)
    print("experiment eval random seed: ", eval_seed)

    ## load real trajs
    real_traj_path = REALEXP_DIR / f"real_results/{mode}s/real2sim"
    real_traj_list = sorted(os.listdir(real_traj_path))
    real_traj_file = real_traj_list[record_id]
    print("real2sim record name: ", real_traj_file)
    traj_len = len(glob.glob(str(real_traj_path / real_traj_file) + "/test_*.png"))
    print("real2sim record length: ", traj_len)
    traj_qpos = []
    traj_eepos = []
    traj_action = []
    for ind in range(traj_len):
        qpos_eepos_action = np.load(
            real_traj_path / f"{real_traj_file}/{ind:02}_qpos_eepos_action.npy"
        )
        traj_qpos.append(qpos_eepos_action[:9])
        traj_eepos.append(qpos_eepos_action[9:12])
        traj_action.append(qpos_eepos_action[12:])

    # real2sim results
    real2sim_results_path = REALEXP_DIR / f"real_results/real2sim_videos"

    if mode == "door":
        arti_ids = [0]  # 0
    elif mode == "drawer":
        arti_ids = [1]  # 1
    elif mode == "faucet":
        arti_ids = [5052]
    elif mode == "arti":
        arti_ids = [0, 1, 5052]
    else:
        raise NotImplementedError
    obs_mode = "state_egostereo_rgbd"
    control_mode = "pd_joint_delta_pos"

    print("+++++ Build Real2Sim Env +++++++")
    env: ArtiMani = gym.make(
        "ArtiMani-v0",
        articulation_ids=arti_ids,
        segmodel_path=None,
        sample_mode="random_sample",
        frame_num=0,
        sample_num=32,
        other_handle_visible=False,
        obs_mode=obs_mode,
        control_mode=control_mode,
        reward_mode="dense",  # "sparse", "dense"
        device=device,
    )
    env = NormalizeActionWrapper(env)
    env.seed(eval_seed)
    print(f"Action space {env.action_space}")
    print(f"Control mode {env.control_mode}")
    action_range_low = env.unwrapped.action_space.low
    action_range_high = env.unwrapped.action_space.high
    print("Input Action space range (low):", action_range_low)
    print("Input Action space range (high):", action_range_high)

    obs = env.reset()
    robot_ee: sapien.Link = get_entity_by_name(
        env.agent._robot.get_links(), "xmate3_link7"
    )
    ### Set Cabinet0 Pose
    # z = env.unwrapped._articulation.get_root_pose().p[2]
    # # real init
    # x = -0.6 + 100 * 0.025  # 52
    # y = 0.4 - 16 * 0.025  # 16
    # rz = 106  # arctan(2/7) + 90
    # q4 = np.sin(0.5 * (rz / 180 * 3.1415926))
    # env.unwrapped._articulation.set_root_pose(
    #     sapien.Pose([x, y, z], [np.sqrt(1 - q4 ** 2), 0, 0, q4])
    # )
    # env.unwrapped._articulation.set_qpos([0, 0])
    env.unwrapped._articulation.set_root_pose(sapien.Pose([2, 0, 0], [1, 0, 0, 0]))

    ### Set Robot Pose
    ## real robot init
    if mode == "door" or mode == "drawer":
        qpos = np.array([1.4, -1.053, -2.394, 1.662, 1.217, 1.05, -0.8, 0.0, 0.0])
    elif mode == "faucet":
        qpos = np.array([-0.5, -0.143, 0, np.pi / 3, 0, 1.57, 1.57, 0.068, 0.068])
    else:
        raise NotImplementedError(mode)
    env.unwrapped._agent._robot.set_qpos(qpos)

    viewer = env.unwrapped.render()
    print("Press [b] to start")
    while True:
        if viewer.window.key_down("b"):
            break
        env.render()

    if save_mode:
        frames = []
        base_rgbd = env.render("cameras")
        frames.append(base_rgbd)
        qpos_err = []
    for step in range(traj_len):
        cur_qpos = traj_qpos[step]
        cur_eepos = traj_eepos[step]
        # env.unwrapped._agent._robot.set_qpos(cur_qpos)

        # task action
        cur_action = traj_action[step]
        print(step, cur_qpos, cur_eepos, cur_action)
        obs, rew, done, info = env.step(cur_action)
        # print(f"STEP: {step:5d}, rew: {rew}, done: {done}, info: {info}")
        base_rgbd = env.render(mode="cameras")
        if save_mode:
            frames.append(base_rgbd)
            qpos_err.append(env.unwrapped._agent._robot.get_qpos() - cur_qpos)
        viewer = env.unwrapped.render()
        # print("Press [c] to start")
        # while True:
        #     if viewer.window.key_down("c"):
        #         break
        #     env.render()

    if save_mode:
        images_to_video(
            images=frames,
            output_dir=str(real2sim_results_path),
            video_name=f"{record_id}_trajectory_{real_traj_file[:15]}",
        )
        qpos_err = np.array(qpos_err)
        labels = np.array([f"j{i}" for i in range(7)])
        plt.subplot(2, 1, 1)
        plt.plot(np.arange(traj_len), qpos_err[:, -2], label=labels)
        plt.title("qpos err (sim - real)")
        plt.legend()
        plt.subplot(2, 1, 2)
        plt.plot(np.arange(traj_len), qpos_err[:, -2:])
        plt.title("gripper pos err (sim - real)")
        plt.savefig(
            real2sim_results_path / f"{record_id}_{mode}_real2sim_err.png",
            dpi=300,
            pad_inches=0.2,
        )
