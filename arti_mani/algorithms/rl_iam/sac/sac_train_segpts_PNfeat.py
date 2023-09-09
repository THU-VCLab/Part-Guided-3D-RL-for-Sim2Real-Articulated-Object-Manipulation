import os
import random

from arti_mani import VISUALMODEL_DIR
from arti_mani.algorithms.rl_iam.feature_extract import CustomSegPNExtractor
from arti_mani.algorithms.rl_iam.rl_utils import sb3_make_env_multiarti
from arti_mani.envs import *
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.logger import configure
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor


def setup_seed(seed=1029):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)  # 为了禁止hash随机化，使得实验可复现
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


if __name__ == "__main__":
    np.set_printoptions(suppress=True, precision=4)
    device = "cuda:0"
    mode = "arti"  # "door", "drawer", "faucet", "arti"
    train_procs = 4
    eval_procs = 1
    save_eval_freq = 250000 // train_procs
    seed = np.random.RandomState().randint(2**32)
    print("experiment random seed: ", seed)
    setup_seed(seed)
    env_id = "ArtiMani-v0"
    # door: [0, 1006, 1030, 1047, 1081]
    # drawer: 1, [1045, 1054, 1063, 1067], [1004, 1005, 1016, 1024]
    # faucet: [5002, 5023, 5052, 5069], [5004, 5007, 5023, 5069]
    if mode == "door":
        arti_ids = [1006, 1030, 1047, 1081]
        eval_ids = [0]
    elif mode == "drawer":
        arti_ids = [1005, 1016, 1024, 1076]
        eval_ids = [1]
    elif mode == "faucet":
        arti_ids = [5004, 5007, 5023, 5069]
        eval_ids = [5052]
    elif mode == "arti":
        arti_ids = [
            1006,
            1030,
            1047,
            1081,
            1005,
            1016,
            1024,
            1076,
            5004,
            5007,
            5023,
            5069,
        ]
        eval_ids = [0, 1, 5052]
    else:
        raise NotImplementedError(mode)
    segmodel_path = (
        VISUALMODEL_DIR
        / "smp_model/20230219_000940_train52-val18_384_noDR_randombg_aug_stereo_bs16_focalloss_0.5step50lr0.001_RGBDunet-163264128_mobilenet_v2"
    )
    other_handle_visible = False
    obs_mode = "state_egostereo_segpoints"
    control_mode = "pd_joint_delta_pos"
    num_classes = 6
    state_repeat_times = 10
    sample_mode = "random_sample"  # full_downsample, score_sample, random_sample, fps_sample, frameweight_sample
    frame_num = 1
    pts_sample_num = 32
    # Create log dir
    exp_suffix = "smp0219"  # _largerange_adjustgripper_ptsaddnoise0.01
    log_dir = f"./logs/sac{train_procs}_tr{len(arti_ids)}eval{len(eval_ids)}{mode}_{obs_mode}_{sample_mode}f{frame_num}s{pts_sample_num}_{exp_suffix}/"
    os.makedirs(log_dir, exist_ok=True)

    vec_env = SubprocVecEnv(
        [
            sb3_make_env_multiarti(
                env_id=env_id,
                arti_ids=arti_ids,
                segmodel_path=segmodel_path,
                sample_mode=sample_mode,
                frame_num=frame_num,
                sample_num=pts_sample_num,
                other_handle_visible=other_handle_visible,
                obs_mode=obs_mode,
                control_mode=control_mode,
                device=device,
                rank=i,
                seed=seed,
            )
            for i in range(train_procs)
        ],
        start_method="spawn",
    )
    vec_env = VecMonitor(vec_env, log_dir)

    print("Observation Space: ", vec_env.observation_space)
    print("Action Space: ", vec_env.action_space)

    # setup callbacks
    checkpoint_callback = CheckpointCallback(
        save_freq=save_eval_freq, save_path=log_dir
    )
    eval_seed = np.random.RandomState().randint(2**32)
    print("experiment eval random seed: ", eval_seed)
    vec_eval_env = SubprocVecEnv(
        [
            sb3_make_env_multiarti(
                env_id=env_id,
                arti_ids=eval_ids,
                segmodel_path=segmodel_path,
                sample_mode=sample_mode,
                frame_num=frame_num,
                sample_num=pts_sample_num,
                other_handle_visible=other_handle_visible,
                obs_mode=obs_mode,
                control_mode=control_mode,
                device=device,
                rank=i,
                seed=seed,
            )
            for i in range(eval_procs)
        ],
        start_method="spawn",
    )
    vec_eval_env = VecMonitor(vec_eval_env, log_dir)
    eval_callback = EvalCallback(
        vec_eval_env,
        n_eval_episodes=50,
        best_model_save_path=log_dir,
        log_path=log_dir,
        eval_freq=save_eval_freq,
        deterministic=True,
        render=False,
    )

    # set up logger
    new_logger = configure(log_dir, ["stdout", "csv", "tensorboard"])

    # check_env(env)
    model = SAC(
        "MultiInputPolicy",
        vec_env,
        batch_size=256,  # 1024, 400
        ent_coef="auto_0.2",
        gamma=0.99,
        train_freq=16,  # 4, 64
        gradient_steps=16,  # 2, 4
        buffer_size=100000,
        learning_starts=800,
        use_sde=True,
        policy_kwargs=dict(
            log_std_init=-3.67,
            net_arch=[256, 256],
            features_extractor_class=CustomSegPNExtractor,
            features_extractor_kwargs=dict(
                num_classes=num_classes,
                state_repeat_times=state_repeat_times,
                sample_num=pts_sample_num,
            ),
            normalize_images=False,
            share_features_extractor=True,
        ),
        tensorboard_log=log_dir + "sac_opendoor_tb/",
        device=device,
        verbose=1,
    )
    # Set new logger
    model.set_logger(new_logger)
    model.learn(
        total_timesteps=2_000_000,
        callback=[checkpoint_callback, eval_callback],
    )
