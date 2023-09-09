import glob
import math
from collections import defaultdict
from copy import deepcopy
from datetime import datetime
from time import time

import cv2
import numpy as np
import segmentation_models_pytorch as smp
import torch
import yaml
from arti_mani.algorithms.config.config import Config
from arti_mani.algorithms.visual_net.Networks.Custom_Unet import (
    CustomUnet,
    SplitUnet,
    WCELoss,
)
from arti_mani.utils.cv_utils import visualize_depth, visualize_seg
from dataset import SegDataset, seed_everything
from matplotlib import pyplot as plt
from path import Path
from torch import optim
from torch.nn import functional as F
from torch.nn.utils.clip_grad import clip_grad_norm_
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter


def load_real_data(
    REAL_DIR="/ssd2/pwral_real_seg",
    PAD_SIZE=(256, 144),
    IM_SIZE=(256, 144),
    in_ch=4,
    only_vis=False,
):
    # load real data and preprocess
    REAL_DIR = Path(REAL_DIR)
    leftrightpad = int((PAD_SIZE[0] - IM_SIZE[0]) / 2)
    topbottompad = int((PAD_SIZE[1] - IM_SIZE[1]) / 2)
    real_img = {"rgb": [], "depth": [], "masks": []}
    for mode in ["door", "drawer", "faucet"]:
        for im_mode in ["rgb", "depth"]:
            files = glob.glob(str(REAL_DIR / f"{mode}/*{im_mode}.npy"))
            files.sort()
            for file in files:
                data = np.load(file)
                data = cv2.resize(
                    src=data, dsize=IM_SIZE, interpolation=cv2.INTER_NEAREST
                )
                if leftrightpad > 0 or topbottompad > 0:
                    data = cv2.copyMakeBorder(
                        data,
                        topbottompad,
                        topbottompad,
                        leftrightpad,
                        leftrightpad,
                        cv2.BORDER_DEFAULT,
                    )
                if im_mode == "rgb":
                    data = data[..., ::-1] / 255.0
                real_img[im_mode].append(data)
    # load real masks
    if not only_vis:
        mask_files = []
        for mode in ["door", "drawer", "faucet"]:
            files = glob.glob(str(REAL_DIR / f"masks/*{mode}.npy"))
            files.sort()
            mask_files.extend(files)
        for mf in mask_files:
            data = np.load(mf)
            data = cv2.resize(src=data, dsize=IM_SIZE, interpolation=cv2.INTER_NEAREST)
            if leftrightpad > 0 or topbottompad > 0:
                data = cv2.copyMakeBorder(
                    data,
                    topbottompad,
                    topbottompad,
                    leftrightpad,
                    leftrightpad,
                    cv2.BORDER_DEFAULT,
                )
            real_img["masks"].append(data)

    for key in real_img.keys():
        real_img[key] = np.array(real_img[key])

    if in_ch == 1:
        real_data = np.expand_dims(real_img["depth"], 3)
    elif in_ch == 3:
        real_data = np.array(real_img["rgb"])
    elif in_ch == 4:
        real_data = np.concatenate(
            [real_img["rgb"], np.expand_dims(real_img["depth"], 3)], axis=3
        )
    else:
        raise NotImplementedError
    # transfer to torch data
    real_rgbd = torch.from_numpy(real_data).permute(0, 3, 1, 2)  # (N, ch, H, W)
    return real_rgbd, real_img


class SMPTrainer(object):
    def __init__(self, cfg):
        # super().__init__(cfg)
        self.cfg = cfg
        smp_cfg = cfg.smp_config

        if smp_cfg["mode"] == "RGBD":
            in_ch = 4
        elif smp_cfg["mode"] == "RGD":
            in_ch = 3
        elif smp_cfg["mode"] == "RGB":
            in_ch = 3
        elif smp_cfg["mode"] == "D":
            in_ch = 1
        else:
            raise NotImplementedError

        if smp_cfg["encoder"] == "splitnet":
            self.model = SplitUnet(
                dropout_p=smp_cfg["dropout_p"],
                encoder_name=smp_cfg["encoder"],
                encoder_depth=smp_cfg["encoder_depth"],
                decoder_channels=smp_cfg["decoder_channels"],
                encoder_weights=smp_cfg["encoder_weights"],
                in_channels=in_ch,
                classes=cfg.num_classes,
            ).to(cfg.device)
        else:
            self.model = CustomUnet(
                dropout_p=smp_cfg["dropout_p"],
                encoder_name=smp_cfg["encoder"],
                encoder_depth=smp_cfg["encoder_depth"],
                decoder_channels=smp_cfg["decoder_channels"],
                encoder_weights=smp_cfg["encoder_weights"],
                in_channels=in_ch,
                classes=cfg.num_classes,
            ).to(cfg.device)

        self.classes = smp_cfg["classes"]
        self.num_classes = cfg.num_classes

        model_params = sum(
            p.numel() for p in self.model.parameters() if p.requires_grad
        )
        self.cfg.logger.info(f"* number of parameters: {model_params}")

        # init optimizer
        self.optimizer = optim.Adam(params=self.model.parameters(), lr=cfg.lr)
        self.val_loss = smp.losses.SoftCrossEntropyLoss(smooth_factor=0)
        if self.cfg.loss == "focal":
            self.loss = smp.losses.FocalLoss(mode="multiclass")
        elif self.cfg.loss == "wce":
            self.loss = WCELoss(smooth_factor=0.1)
        elif self.cfg.loss == "ce":
            self.loss = smp.losses.SoftCrossEntropyLoss(smooth_factor=0.1)
        elif self.cfg.loss == "dice":
            self.loss = smp.losses.DiceLoss(mode="multiclass")
        elif self.cfg.loss == "tver":
            self.loss = smp.losses.TverskyLoss(mode="multiclass")
        else:
            raise NotImplementedError
        ### StepLR
        self.scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimizer, step_size=40, gamma=0.3
        )

        training_set = SegDataset(
            data_mode="train",  # "train", "train_lowres"
            augmentation=smp_cfg["augmentation"],
            copy_paste=smp_cfg["copy_paste"],
            classes=smp_cfg["classes"],
            mode=smp_cfg["mode"],
        )

        val_set = SegDataset(
            data_mode="val",  # "val", "val_lowres"
            augmentation=False,
            copy_paste=False,
            classes=smp_cfg["classes"],
            mode=smp_cfg["mode"],
        )

        self.real_data, self.real_img = load_real_data(in_ch=in_ch)
        self.real_data = self.real_data.to(device=cfg.device, dtype=torch.float32)

        print("real data:", self.real_data.shape)

        self.train_loader = DataLoader(
            training_set,
            batch_size=cfg.batch_size,
            shuffle=True,
            pin_memory=True,
            num_workers=cfg.n_workers,
        )
        self.val_loader = DataLoader(
            val_set,
            batch_size=cfg.batch_size,
            shuffle=True,
            pin_memory=True,
            num_workers=cfg.n_workers,
        )

        self.cfg.logger.info(
            f"training_set size: {len(training_set)}, val_set size: {len(val_set)}"
        )
        # init logging stuff
        self.log_path = Path(cfg.exp_log_path)
        tb_logdir = self.log_path.abspath()
        # print(f'tensorboard --logdir={tb_logdir}\n')
        self.cfg.logger.info(f"tensorboard --logdir={tb_logdir}\n")
        self.sw = SummaryWriter(self.log_path)
        self.log_freq = len(self.train_loader)

        # starting values values
        self.epoch = 0
        self.train_losses = None
        self.val_losses = None
        self.best_val_epoch = 0
        self.best_val_loss = None
        self.best_val_items = None

    def load_ck(self):
        """load training checkpoint."""
        ck_path = self.log_path / "training.ck"
        if ck_path.exists():
            ck = torch.load(ck_path, map_location=torch.device("cpu"))
            # print('[loading checkpoint \'{}\']'.format(ck_path))
            self.cfg.logger.info("[loading checkpoint '{}']".format(ck_path))
            self.epoch = ck["epoch"]
            self.model.load_state_dict(ck["model"], strict=True)
            self.model.to(self.cfg.device)
            self.best_val_loss = ck["best_val_loss"]
            if ck.get("optimizer", None) is not None:
                self.optimizer.load_state_dict(ck["optimizer"])

    def save_ck(self):
        """save training checkpoint."""
        ck = {
            "epoch": self.epoch,
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "best_val_loss": self.best_val_loss,
        }
        torch.save(ck, self.log_path / "training.ck")

    def train(self):
        """train model for one epoch on the Training-Set."""
        self.model.train()

        start_time = time()
        times = []
        self.train_losses = []
        self.uncertaintys = []
        # train_items = dict.fromkeys(["iou", "f1", "acc", "prec", "recall"], [])
        train_items = defaultdict(list)

        t = time()
        for step, sample in enumerate(self.train_loader):
            self.optimizer.zero_grad()
            img = sample[2].to(device=self.cfg.device, dtype=torch.float32)
            seg_gt_onehot = (
                torch.stack(sample[3]).transpose(1, 0).to(self.cfg.device)
            )  # (N, C, H, W)
            seg_gt = torch.argmax(seg_gt_onehot, dim=1)  # (N, H, W)

            pred = self.model.forward(img)  # (N, C, H, W)

            # pred_mc = torch.argmax(pred, dim=1)  # (N, H, W)
            # first compute statistics for true positives, false positives, false negative and
            # true negative "pixels"
            # tp, fp, fn, tn = smp.metrics.get_stats(pred, seg_gt_onehot.long(), mode='multilabel', threshold=0.5)
            # tp, fp, fn, tn = smp.metrics.get_stats(pred_mc.long(),
            #                                        seg_gt.long(),
            #                                        mode='multiclass',
            #                                        num_classes=self.num_classes)

            # iou_metric = smp.metrics.iou_score(tp, fp, fn, tn, reduction='none')  # (bs, C)
            # f1_metric = smp.metrics.f1_score(tp, fp, fn, tn, reduction='none')  # (bs, C)
            # for ind, key in enumerate(self.classes):
            #     train_items[f'iou_{key}'].append(torch.mean(iou_metric, dim=0)[ind].item())
            #     train_items[f'f1_{key}'].append(torch.mean(f1_metric, dim=0)[ind].item())

            # then compute metrics with required reduction (see metric docs)
            # train_items['miou'].append(
            #     smp.metrics.iou_score(tp, fp, fn, tn, reduction='micro').item())
            # train_items['f1'].append(smp.metrics.f1_score(tp, fp, fn, tn, reduction='micro').item())
            # train_items["acc"].append(smp.metrics.accuracy(tp, fp, fn, tn, reduction="macro").item())
            # train_items["prec"].append(smp.metrics.precision(tp, fp, fn, tn, reduction="micro-imagewise").item())
            # train_items["recall"].append(smp.metrics.recall(tp, fp, fn, tn, reduction="micro-imagewise").item())

            # loss = softmax_focal_loss(seg_feat, seg_gt_onehot, alpha=0.75, reduction='mean')
            # loss = soft_dice_loss(seg_feat, seg_gt_onehot)
            # loss = log_cosh_dice_loss(seg_feat, seg_gt_onehot)
            # loss = self.wfocal_loss(seg_feat, seg_gt)
            # loss = F.cross_entropy(pred, seg_gt_onehot)
            if self.cfg.loss == "wce":
                # get uncertainty
                # get uncertainty map
                uncertain_pred = []
                with torch.no_grad():
                    for _ in range(self.cfg.smp_config["sample"]):
                        pred_seg_mtcsamp = self.model(
                            img, activate_dropout=True
                        )  # (N, 6, H, W)
                        uncertain_pred.append(F.softmax(pred_seg_mtcsamp, dim=1))
                    uncertain_mean = torch.mean(
                        torch.stack(uncertain_pred), dim=0
                    )  # (N, 6, H, W)
                    # print(torch.stack(uncertain_pred).shape, uncertain_mean.shape)
                    uncertain_map = -1.0 * torch.sum(
                        uncertain_mean * torch.log(uncertain_mean + 1e-6),
                        dim=1,
                        keepdim=True,
                    )  # (N, H, W)
                # print(uncertain_map.min(), uncertain_map.max())
                min_w = self.cfg.smp_config["min_weight"]
                norm_uncertainty = (uncertain_map - uncertain_map.min()) / (
                    uncertain_map.max() - uncertain_map.min()
                )
                weight = min_w + (1 - min_w) * norm_uncertainty**2
                loss = self.loss(pred, seg_gt, weight)
                self.uncertaintys.append(uncertain_map.mean().item())
            else:
                loss = self.loss(pred, seg_gt)
            loss.backward()
            self.train_losses.append(loss.item())
            clip_grad_norm_(self.model.parameters(), 10)
            self.optimizer.step()

            # print an incredible progress bar
            progress = (step + 1) / len(self.train_loader)
            bar_len = 40
            progress_bar = ("■" * int(bar_len * progress)) + (
                "┈" * (bar_len - int(bar_len * progress))
            )
            times.append(time() - t)
            t = time()
            if self.cfg.log_step or (not self.cfg.log_step and progress == 1):
                if step > 0 and step % self.cfg.log_step == 0:
                    self.cfg.logger.info(
                        "\r[{}] Epoch {:0{e}d}.{:0{s}d}: │{}│ {:6.2f}% │ Loss: {:.6f} │ ↯: {:5.2f} step/s".format(
                            datetime.now().strftime("%m-%d@%H:%M"),
                            self.epoch,
                            step,
                            progress_bar,
                            100 * progress,
                            np.mean(self.train_losses[-self.cfg.log_step :]),
                            1 / np.mean(times[-self.cfg.log_step :]),
                            e=math.ceil(math.log10(self.cfg.epochs)),
                            s=math.ceil(math.log10(self.log_freq)),
                        )
                    )

            # if step >= self.cfg.epoch_len - 1:
            #     break

        # log average loss of this epoch
        mean_uncertainty = np.mean(self.uncertaintys)
        self.sw.add_scalar(
            tag="uncertainty", scalar_value=mean_uncertainty, global_step=self.epoch
        )
        mean_epoch_loss = np.mean(self.train_losses)
        self.sw.add_scalar(
            tag="train/loss", scalar_value=mean_epoch_loss, global_step=self.epoch
        )
        for key in train_items.keys():
            mean_epoch_item = np.mean(train_items[key])
            self.sw.add_scalar(
                tag=f"train/iou/{key}",
                scalar_value=mean_epoch_item,
                global_step=self.epoch,
            )

        cur_lr = self.optimizer.state_dict()["param_groups"][0]["lr"]
        self.sw.add_scalar(tag="train_lr", scalar_value=cur_lr, global_step=self.epoch)

        # log epoch duration
        self.cfg.logger.info(f" │ T: {time() - start_time:.2f} s")

    def test(self):
        """test model on the Test-Set."""
        self.val_losses = []
        # val_items = dict.fromkeys(["iou", "f1", "acc", "prec", "recall"], [])
        val_items = defaultdict(list)

        self.model.eval()

        t = time()
        with torch.no_grad():
            for step, sample in enumerate(self.val_loader):
                img, masks = (
                    sample[2].to(device=self.cfg.device, dtype=torch.float32),
                    sample[3],
                )
                seg_gt_onehot = (
                    torch.stack(masks).transpose(1, 0).to(self.cfg.device)
                )  # (N, C, H, W)
                seg_gt = torch.argmax(seg_gt_onehot, dim=1)  # (N, H, W)

                pred = self.model(img)  # (N, C, H, W)
                pred_mc = torch.argmax(pred, dim=1)  # (N, H, W)

                # loss = softmax_focal_loss(seg_feat, seg_gt_onehot, alpha=0.75, reduction='mean')
                # loss = soft_dice_loss(seg_feat, seg_gt_onehot)
                # loss = log_cosh_dice_loss(seg_feat, seg_gt_onehot)
                # loss = self.wfocal_loss(seg_feat, seg_gt)
                # loss = F.cross_entropy(pred, seg_gt_onehot)

                loss = self.val_loss(pred, seg_gt)
                self.val_losses.append(loss.item())

                # first compute statistics for true positives, false positives, false negative and
                # true negative "pixels"
                # tp, fp, fn, tn = smp.metrics.get_stats(pred, seg_gt_onehot.long(), mode='multilabel', threshold=0.5)
                tp, fp, fn, tn = smp.metrics.get_stats(
                    pred_mc.long(),
                    seg_gt.long(),
                    mode="multiclass",
                    num_classes=self.num_classes,
                )

                iou_metric = smp.metrics.iou_score(
                    tp, fp, fn, tn, reduction="none"
                )  # (bs, C)
                # f1_metric = smp.metrics.f1_score(tp, fp, fn, tn, reduction='none')  # (bs, C)
                for ind, key in enumerate(self.classes):
                    val_items[f"iou_{key}"].append(
                        torch.mean(iou_metric, dim=0)[ind].item()
                    )
                    # val_items[f'f1_{key}'].append(torch.mean(f1_metric, dim=0)[ind].item())

                # then compute metrics with required reduction (see metric docs)
                val_items["miou"].append(
                    smp.metrics.iou_score(tp, fp, fn, tn, reduction="micro").item()
                )
                # val_items['f1'].append(smp.metrics.f1_score(tp, fp, fn, tn, reduction='micro').item())
                # val_items["acc"].append(smp.metrics.accuracy(tp, fp, fn, tn, reduction="macro").item())
                # val_items["prec"].append(smp.metrics.precision(tp, fp, fn, tn, reduction="micro-imagewise").item())
                # val_items["recall"].append(smp.metrics.recall(tp, fp, fn, tn, reduction="micro-imagewise").item())

        # log average loss on test set
        mean_val_loss = np.mean(self.val_losses)
        # print(f'\t● AVG Loss on VAL-set: {mean_val_loss:.6f} │ T: {time() - t:.2f} s')
        self.cfg.logger.info(
            f"\t● AVG Loss on VAL-set: {mean_val_loss:.8f} │ T: {time() - t:.2f} s"
        )

        self.sw.add_scalar(
            tag="val/loss", scalar_value=mean_val_loss, global_step=self.epoch
        )

        cur_mean_epoch_item = defaultdict(float)
        for key in val_items.keys():
            cur_mean_epoch_item[key] = float(np.mean(val_items[key]))
            self.sw.add_scalar(
                tag=f"val/iou/{key}",
                scalar_value=cur_mean_epoch_item[key],
                global_step=self.epoch,
            )

        # save best model
        if self.best_val_loss is None or mean_val_loss < self.best_val_loss:
            self.best_val_loss = mean_val_loss
            self.best_val_epoch = self.epoch
            torch.save(self.model.state_dict(), self.log_path / "best.pth")
            self.cfg.logger.info(f"\t● Save best model on epoch: {self.epoch}")
            self.cfg.logger.info(f"\t● best model metrics:")
            iou_str, f1_str = "", ""
            for k in cur_mean_epoch_item:
                if k.startswith("iou"):
                    iou_str += f"{k[4:]}: {cur_mean_epoch_item[k]:.3f} "
                # if 'f1' in k:
                #     obj = k if k == 'f1' else k[3:]
                #     f1_str += f'{obj}: {cur_mean_epoch_item[k]:.3f} '
            self.cfg.logger.info(f"test on val dataset:")
            self.cfg.logger.info(f'miou: {cur_mean_epoch_item["miou"]:.3f}')
            self.cfg.logger.info(f"{iou_str}")
            # self.cfg.logger.info('f1:')
            # self.cfg.logger.info(f'{f1_str}')
            self.best_val_items = cur_mean_epoch_item

    def test_real(self, vis=False):
        """test model on the Real-Test-Set."""
        self.model.eval()

        with torch.no_grad():
            # get uncertainty map
            uncertain_pred = []
            for ind in range(self.cfg.smp_config["sample"]):
                pred_seg_mtcsamp = self.model(
                    self.real_data, activate_dropout=True
                )  # (N, 6, H, W)
                uncertain_pred.append(pred_seg_mtcsamp)
            uncertain_mean = torch.mean(
                torch.stack(uncertain_pred), dim=0
            )  # (N, 6, H, W)
            uncertain_map = -1.0 * torch.sum(
                uncertain_mean * torch.log(uncertain_mean + 1e-6), dim=1
            )  # (N, H, W)
            # get segmentation prediction
            pred_seg = self.model(self.real_data)  # (N, 6, H, W)
            pred_mc = torch.argmax(pred_seg, dim=1).to(self.cfg.device)  # (N, H, W)
            real_seg = torch.from_numpy(self.real_img["masks"]).to(
                self.cfg.device
            )  # (N, H, W)

            # save qualified results
            if vis:
                subplot_num = [2, 3]
                # seg_pred_img = pred_seg.detach().cpu().numpy()
                pred_mc_img = pred_mc.cpu().numpy()  # (N, H, W)
                uncertain_map_img = uncertain_map.cpu().numpy()  # (N, H, W)
                uncertain_map_img = (uncertain_map_img - uncertain_map_img.min()) / (
                    uncertain_map_img.max() - uncertain_map_img.min()
                )
                kernel1 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
                kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
                for ind in range(self.real_data.shape[0]):
                    uncertain_fix = cv2.dilate(
                        uncertain_map_img[ind], kernel, iterations=1
                    )
                    # uncertain_fix = cv2.dilate(uncertain_fix, kernel2, iterations=1)
                    # uncertain_fix = cv2.erode(uncertain_fix, kernel, iterations=1)
                    # uncertain_fix = cv2.morphologyEx(uncertain_fix, cv2.MORPH_CLOSE, kernel1, iterations=1)
                    uncertain_fix = cv2.morphologyEx(
                        uncertain_fix, cv2.MORPH_OPEN, kernel1, iterations=1
                    )
                    plt.subplot(subplot_num[0], subplot_num[1], 1)
                    plt.imshow(self.real_img["rgb"][ind])
                    plt.subplot(subplot_num[0], subplot_num[1], 2)
                    plt.imshow(visualize_depth(self.real_img["depth"][ind])[..., ::-1])
                    plt.subplot(subplot_num[0], subplot_num[1], 3)
                    plt.imshow(
                        visualize_seg(self.real_img["masks"][ind], self.cfg.num_classes)
                    )
                    plt.subplot(subplot_num[0], subplot_num[1], 4)
                    plt.imshow(
                        visualize_seg(
                            pred_mc_img[ind], num_classes=self.cfg.num_classes
                        )
                    )
                    plt.subplot(subplot_num[0], subplot_num[1], 5)
                    plt.imshow(uncertain_map_img[ind], cmap="jet")
                    plt.subplot(subplot_num[0], subplot_num[1], 6)
                    plt.imshow(uncertain_fix, cmap="jet")
                    # for pid in range(num_classes):
                    #     plt.subplot(subplot_num[0], subplot_num[1], pid + 4)
                    #     plt.imshow(seg_pred_img[ind][pid])

                    plt.savefig(f"./vis_seg/{ind:03}.jpg", dpi=200)

            # real_seg_onehot = F.one_hot(
            #     real_seg.long(), num_classes=num_classes
            # ).long().permute(0, 3, 1, 2).to(device)  # (N, 6, H, W)
            tp, fp, fn, tn = smp.metrics.get_stats(
                pred_mc.long(),
                real_seg.long(),
                mode="multiclass",
                num_classes=self.num_classes,
            )
            iou_metric = smp.metrics.iou_score(
                tp, fp, fn, tn, reduction="none"
            )  # (bs, C)
            # f1_metric = smp.metrics.f1_score(tp, fp, fn, tn, reduction="none")  # (bs, C)
            miou = smp.metrics.iou_score(tp, fp, fn, tn, reduction="micro").item()
            self.cfg.logger.info(f"test on real dataset:")
            self.cfg.logger.info(f"miou: {miou:.3f}")
            self.sw.add_scalar(
                tag=f"test_real/iou/miou", scalar_value=miou, global_step=self.epoch
            )
            logstr = ""
            for ind, key in enumerate(self.classes):
                cur_iou = torch.mean(iou_metric, dim=0)[ind].item()
                logstr += f"{key}: {cur_iou:.3f} "
                self.sw.add_scalar(
                    tag=f"test_real/iou/{key}",
                    scalar_value=cur_iou,
                    global_step=self.epoch,
                )
            self.cfg.logger.info(logstr)

    def run(self):
        """start model training procedure (train > test > checkpoint >
        repeat)"""
        for e in range(self.epoch, self.cfg.epochs):
            # if self.epoch == 13:
            #     pass
            self.train()
            if e % 1 == 0:
                self.test()
                self.test_real()
            self.epoch += 1
            self.scheduler.step()
            self.save_ck()
            if e % 10 == 0 and e != 0:
                torch.save(self.model.state_dict(), self.log_path / f"{e}.pth")


def main(exp_name=None, seed=None):
    cfg = Config(seed=seed, exp_name=exp_name, mode="train", log=True)

    seed_everything(cfg.seed)

    ## save config file
    with open(f"{cfg.exp_log_path}/config.yaml", "w") as file:
        cfg_text = deepcopy(cfg)
        del cfg_text.logger
        yaml.emitter.Emitter.process_tag = lambda self, *args, **kw: None
        yaml.dump(cfg_text, file)

    cfg.logger.info(f"\n▶ Starting Experiment '{exp_name}' [seed: {cfg.seed}]")

    # trainer = Trainer(cfg=cfg)
    trainer = SMPTrainer(cfg=cfg)
    trainer.run()
    cfg.logger.info(f"\n▶ best val model: ")
    cfg.logger.info(
        f"\n  epoch, loss: {trainer.best_val_epoch}, {trainer.best_val_loss}"
    )
    cfg.logger.info(f"\n  metrics: {trainer.best_val_items}")


if __name__ == "__main__":
    main(exp_name="smp_model", seed=2)  # pretrain_unet, smp_model
