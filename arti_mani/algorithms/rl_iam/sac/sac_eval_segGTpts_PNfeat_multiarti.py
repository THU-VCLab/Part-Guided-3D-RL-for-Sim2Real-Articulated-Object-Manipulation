import faulthandler
import os
from collections import defaultdict

import gym
from arti_mani import RLMODEL_DIR
from arti_mani.envs import *
from arti_mani.utils.common import convert_np_bool_to_float
from arti_mani.utils.visualization import images_to_video
from arti_mani.utils.wrappers import NormalizeActionWrapper, RenderInfoWrapper
from stable_baselines3.sac import SAC

faulthandler.enable()


def draw_gripper_pts(env, max_pos=0.068, radius=0.005, color=(1, 0, 0), name=""):
    builder = env.unwrapped._scene.create_actor_builder()
    cur_gripper_pos = np.mean(env.unwrapped.agent._robot.get_qpos()[-2:])
    # gripper_pts_ee = sample_grasp_points_ee(max_pos - cur_gripper_pos, z_offset=0.03)
    gripper_pts_ee = sample_grasp_multipoints_ee(
        max_pos - cur_gripper_pos, num_points_perlink=10, x_offset=0.02
    )
    trans_ee2world = env.unwrapped.agent.grasp_site.pose.to_transformation_matrix()
    gripper_pts_world = transform_points(trans_ee2world, gripper_pts_ee)
    actors = []
    for pts in gripper_pts_world:
        builder.add_sphere_visual(radius=radius, color=color)
        gripper_pts = builder.build_static(name=name)
        gripper_pts.set_pose(sapien.Pose(pts))
        actors.append(gripper_pts)
    return actors


def draw_target_pts(env, gripper_pts_world, radius=0.005, color=(0, 0, 1), name=""):
    builder = env.unwrapped._scene.create_actor_builder()
    actors = []
    for pts in gripper_pts_world:
        builder.add_sphere_visual(radius=radius, color=color)
        gripper_pts = builder.build_static(name=name)
        gripper_pts.set_pose(sapien.Pose(pts))
        actors.append(gripper_pts)
    return actors


if __name__ == "__main__":
    device = "cuda:0"
    mode = "arti"  # "door", "drawer", "faucet", "laptop", "kitchen_pot"
    test_num = 50
    save_num = 5
    success_nums = 0
    MAX_STEPS = 100
    # rl_exp_name = "sac4_tr4eval1laptop_state_egostereo_segpoints_gt_random_samplef1s32_gt_0"
    # rl_exp_name = "hybrid/5arti/sac4_tr20eval5arti_state_ego_segpoints_gt_random_samplef1s32_multiarti_5class_graspP2turnP1_potaligndrawer"
    rl_exp_names = [
        "hybrid/5arti/sac4_tr20eval5arti_state_ego_segpoints_gt_random_samplef1s32_multiarti_5class_graspP2turnP1_3",
        "hybrid/5arti/sac4_tr20eval5arti_state_ego_segpoints_gt_random_samplef1s32_multiarti_5class_graspP2turnP1_4",
        "hybrid/5arti/sac4_tr20eval5arti_state_ego_segpoints_gt_random_samplef1s32_multiarti_5class_graspP2turnP1_5",
        "hybrid/5arti/sac4_tr20eval5arti_state_ego_segpoints_gt_random_samplef1s32_multiarti_5class_graspP2turnP1_6",
    ]
    env_id = "ArtiMani-v0"
    # door: [0, 1006, 1030, 1047, 1081]
    # drawer: 1, [1045, 1054, 1063, 1067], [1004, 1005, 1016, 1024]
    # faucet: [5002, 5023, 5052, 5069]
    arti_ids = {
        "door": [1006, 1030, 1047, 1081, 0],
        "drawer": [1005, 1016, 1024, 1076, 1],
        "faucet": [5004, 5007, 5023, 5069, 5052],
        "laptop": [9960, 9968, 9992, 9996, 9748],
        "kitchen_pot": [100051, 100054, 100055, 100060, 100015],
    }
    if mode == "door":
        arti_ids = [1006, 1030, 1047, 1081, 0]
    elif mode == "drawer":
        arti_ids = [1005, 1016, 1024, 1076, 1]
    elif mode == "faucet":
        arti_ids = [5004, 5007, 5023, 5069, 5052]
    elif mode == "laptop":
        arti_ids = [9960, 9968, 9992, 9996, 9748]
    elif mode == "kitchen_pot":
        arti_ids = [100051, 100054, 100055, 100060, 100015]
    elif mode == "arti":
        # arti_ids = [1006, 1030, 1047, 1081, 1005, 1016, 1024, 1076, 5004, 5007, 5023, 5069,
        #             9960, 9968, 9992, 9996, 100051, 100054, 100055, 100060, 0, 1, 5052, 9748, 100015]
        arti_ids = [0, 1, 5052, 9748, 100015]
    else:
        raise NotImplementedError(mode)
    other_handle_visible = False
    obs_mode = "state_ego_segpoints_gt"
    control_mode = "pd_joint_delta_pos"
    sample_mode = "random_sample"  # full_downsample, score_sample, random_sample, frameweight_sample
    frame_num = 1
    num_classes = 5
    state_repeat_times = 10
    pts_sample_num = 32
    # gripper_actors = None
    # target_grasp_actors = None

    env = gym.make(
        env_id,
        articulation_ids=arti_ids,
        segmodel_path=None,
        sample_mode=sample_mode,
        frame_num=frame_num,
        sample_num=pts_sample_num,
        other_handle_visible=other_handle_visible,
        num_classes=num_classes,
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
    for rl_exp_name in rl_exp_names:
        model_path = RLMODEL_DIR / f"{rl_exp_name}/best_model"
        RL_model = SAC.load(
            model_path,
            env=env,
            print_system_info=True,
        )
        for arti_id in arti_ids:
            eval_videos_path = (
                RLMODEL_DIR / f"{rl_exp_name}/eval_videos/{arti_id}/best_model/"
            )
            if not os.path.exists(eval_videos_path):
                os.makedirs(eval_videos_path)

            is_success = []
            # gripper_actors, target_grasp_actors = None, None
            with torch.no_grad():
                for num in range(test_num):
                    obs = env.reset(articulation_id=arti_id)
                    if num < save_num:
                        frames = []
                        base_rgbd = env.render("cameras")
                        frames.append(base_rgbd)
                    success_flag = False
                    for i in range(MAX_STEPS):
                        action, _states = RL_model.predict(obs, deterministic=True)
                        obs, rewards, dones, info = env.step(action)
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
            print(f"{arti_id}, success rate: {sum_num}/{test_num}")
            success_rates[arti_id].append(sum_num / test_num)
    print(success_rates)
