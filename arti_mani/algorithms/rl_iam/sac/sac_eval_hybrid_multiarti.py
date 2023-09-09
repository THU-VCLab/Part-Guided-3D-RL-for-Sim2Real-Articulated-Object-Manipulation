import gym
from arti_mani import RLMODEL_DIR, VISUALMODEL_DIR
from arti_mani.envs import *
from arti_mani.utils.common import convert_np_bool_to_float
from arti_mani.utils.wrappers import NormalizeActionWrapper, RenderInfoWrapper
from stable_baselines3.sac import SAC
from tqdm import tqdm


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


if __name__ == "__main__":
    device = "cuda:0"
    # Ours, Oracle-based RL, Image-based RL, Keypoints-based RL, Dexpoints-based RL
    baseline_mode = "hybrid/Dexpoints-based RL"
    mode = "arti"  # "door", "drawer", "faucet", "arti"
    test_num = 50
    MAX_STEPS = 100
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

    if "Image" in baseline_mode or "Ours" == baseline_mode:
        smp_exp_name = "20230219_000940_train52-val18_384_noDR_randombg_aug_stereo_bs16_focalloss_0.5step50lr0.001_RGBDunet-163264128_mobilenet_v2"
        segmodel_path = VISUALMODEL_DIR / f"smp_model/{smp_exp_name}"
    elif "Keypoints" in baseline_mode:
        smp_exp_name = "20230307_182821_D64H40W64_deconv3_kpts3norm01addvis_uvz_lr1e-3_mobilenetv2_dropout0.2_newdatafilter2drawer"
        segmodel_path = VISUALMODEL_DIR / f"kpt_model/{smp_exp_name}"
    else:
        segmodel_path = None

    other_handle_visible = False
    # state_egostereo_segpoints, state_egostereo_dseg, state_egostereo_keypoints, state_egostereo_segpoints_gt, state_egostereo_dexpoints
    obs_mode = "state_egostereo_keypoints"
    control_mode = "pd_joint_delta_pos"
    sample_mode = "random_sample"  # full_downsample, score_sample, random_sample, frameweight_sample, fps_sample
    frame_num = 1
    num_classes = 6
    state_repeat_times = 10
    pts_sample_num = 32
    rl_exp_name = f"{baseline_mode}/sac4_tr12eval3arti_state_egostereo_keypoints3_bs256_tr16up16_5"

    env = gym.make(
        "ArtiMani-v0",
        articulation_ids=arti_ids,
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
    eval_seed = np.random.RandomState().randint(2**32)
    print("experiment eval random seed: ", eval_seed)
    env.seed(eval_seed)

    model_path = RLMODEL_DIR / f"{rl_exp_name}/best_model"
    RL_model = SAC.load(
        model_path,
        env=env,
        print_system_info=True,
    )
    for arti_id in arti_ids:
        is_success = []
        exp_steps = []
        with torch.no_grad():
            for num in tqdm(
                range(test_num),
                desc=f"Processing best model",
                colour="green",
                leave=True,
            ):
                obs = env.reset(articulation_id=arti_id)
                success_flag = False
                total_steps = 0
                # for step in range(MAX_STEPS):
                for step in tqdm(range(MAX_STEPS), colour="red", leave=False):
                    action, _states = RL_model.predict(obs, deterministic=True)
                    obs, rewards, dones, info = env.step(action)
                    # print(f"{step} step: {info['is_success']}")
                    if not success_flag and info["is_success"] == 1.0:
                        success_flag = True
                        total_steps = step
                        break
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
