import os

import cv2
import gym
import matplotlib.pyplot as plt
from arti_mani import VISUALMODEL_DIR
from arti_mani.envs import *


def main():
    np.set_printoptions(suppress=True, precision=4)
    log_name = "20230307_182821_D64H40W64_deconv3_kpts3norm01addvis_uvz_lr1e-3_mobilenetv2_dropout0.2_newdatafilter2drawer"
    vis_result_path = (
        VISUALMODEL_DIR / f"kpt_model/{log_name}/visual_results/test_kpts_new/"
    )
    if not os.path.exists(vis_result_path):
        os.makedirs(vis_result_path)

    visualization = False

    uvz_errors = []
    for mode in ["door", "drawer", "faucet"]:
        if mode == "door":
            arti_ids = [0, 1006, 1030, 1047, 1081]
        elif mode == "drawer":
            arti_ids = [1, 1004, 1005, 1016, 1024]
        elif mode == "faucet":
            arti_ids = [5052, 5004, 5007, 5023, 5069]
        else:
            raise NotImplementedError(mode)
        kpts_min = np.array([0, 0, 0.18])
        kpts_max = np.array([255, 143, 1.0])
        plt_x, plt_y = 2, 2

        # init env
        env = gym.make(
            "ArtiMani-v0",
            articulation_ids=arti_ids,
            segmodel_path=VISUALMODEL_DIR / f"kpt_model/{log_name}",
            device="cuda:0",
            obs_mode="state_egostereo_keypoints",
            control_mode="pd_joint_delta_pos",
        )
        uvz_error = []

        for arti_ind in arti_ids:
            _ = env.reset(articulation_id=arti_ind)
            obs_global = env.get_obs()
            rgbd = env.get_handsensor_rgbdseg()
            print(f"visualize result of {mode}-{arti_ind}")
            # viewer = env.render()
            # print("Press [e] to start")
            # while True:
            #     if viewer.window.key_down("e"):
            #         break
            #     env.render()

            uvz = obs_global["uvz"]
            uvz_pred = obs_global["uvz_pred"]
            uvz_visable = obs_global["uvz_visable"]
            uvz_norm_tensor = torch.from_numpy((uvz - kpts_min) / (kpts_max - kpts_min))
            uvz_pred_norm_tensor = torch.from_numpy(
                (uvz_pred - kpts_min) / (kpts_max - kpts_min)
            )
            loss = F.l1_loss(uvz_pred_norm_tensor, uvz_norm_tensor, reduction="mean")

            print("loss: ", loss.item())
            uvz_error.append(np.abs(uvz_pred - uvz))

            rgb = rgbd["rgb"]  # (H, W, 3)
            depth = rgbd["depth"].squeeze()  # (H, W)

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

            plt.subplot(plt_x, plt_y, 1)
            plt.imshow(rgb)
            plt.subplot(plt_x, plt_y, 2)
            plt.imshow(depth)
            plt.subplot(plt_x, plt_y, 3)
            plt.imshow(rgb_kptgt)
            plt.subplot(plt_x, plt_y, 4)
            plt.imshow(rgb_kptpred)
            plt.tight_layout()
            plt.savefig(vis_result_path / f"{mode}_{arti_ind}.jpg")
            if visualization:
                plt.show()
            plt.close()
        mean_arti_uvzerr = np.array(uvz_error).mean(0)
        print(f"{mode} mean uvz error: ", mean_arti_uvzerr)
        uvz_errors.append(mean_arti_uvzerr)
    print("mean uvz erros: ", np.array(uvz_errors).mean(0))


if __name__ == "__main__":
    main()
