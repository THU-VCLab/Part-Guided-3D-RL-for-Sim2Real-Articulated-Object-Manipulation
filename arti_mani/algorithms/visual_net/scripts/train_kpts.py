import math
import os
import time
from datetime import datetime

import numpy as np
import torch
import torch.backends.cudnn as cudnn
from arti_mani.algorithms.config.config import Logger, set_seed
from arti_mani.algorithms.visual_net.Networks.keypoint_detection import (
    IntegralHumanPoseModel,
    JointLocationLoss,
)
from dataset import KeypointDataset
from path import Path
from torch import optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

cudnn.benchmark = True


class KPTSTrainer(object):
    def __init__(self):
        self.batch_size = 32
        self.lr = 1e-3
        self.epochs = 60
        self.mode = "RGBD"
        self.out_shape = (64, 40, 64)
        self.num_keypoints = 3
        self.num_deconv_layers = 3
        self.has_dropout = True
        self.device = torch.device("cuda:0")
        self.seed = set_seed()
        self.log_each_step = False
        exp_name = "kpt_model"
        exp_suffix = (
            f"kpts{self.num_keypoints}"
            f"_uvz_mode{self.mode}epoch{self.epochs}bs{self.batch_size}lr{self.lr}"
            f"_mobilenetv2_dropout0.2_newdatafilter2drawer_L2visable"
        )

        self.exp_log_path = f"log/{exp_name}/{time.strftime('%Y%m%d_%H%M%S', time.localtime())}_{exp_suffix}"
        if not os.path.exists(self.exp_log_path):
            os.makedirs(self.exp_log_path)
        self.logger = Logger(f"{self.exp_log_path}/log.txt")
        self.logger.info(
            f"\n▶ Starting Experiment '{exp_name}_{exp_suffix}' [seed: {self.seed}]"
        )

        self.model = IntegralHumanPoseModel(
            num_keypoints=self.num_keypoints,
            num_deconv_layers=self.num_deconv_layers,
            depth_dim=self.out_shape[0],
            has_dropout=self.has_dropout,
        ).to(self.device)

        model_params = sum(
            p.numel() for p in self.model.parameters() if p.requires_grad
        )
        self.logger.info(f"* number of parameters: {model_params}")

        # init optimizer
        self.optimizer = optim.Adam(params=self.model.parameters(), lr=self.lr)
        self.loss = JointLocationLoss()
        ### StepLR
        self.scheduler = torch.optim.lr_scheduler.MultiStepLR(
            self.optimizer,
            milestones=[30, 40, 50],
            gamma=0.5,
        )

        training_set = KeypointDataset(
            data_mode="train", augmentation=True, normalization=True, mode=self.mode
        )

        val_set = KeypointDataset(
            data_mode="val", augmentation=True, normalization=True, mode=self.mode
        )

        self.train_loader = DataLoader(
            training_set, batch_size=self.batch_size, shuffle=True, num_workers=8
        )
        self.val_loader = DataLoader(
            val_set, batch_size=self.batch_size, shuffle=True, num_workers=8
        )

        self.logger.info(
            f"training_set size: {len(training_set)}, val_set size: {len(val_set)}"
        )
        # init logging stuff
        self.log_path = Path(self.exp_log_path)
        tb_logdir = self.log_path.abspath()
        # print(f'tensorboard --logdir={tb_logdir}\n')
        self.logger.info(f"tensorboard --logdir={tb_logdir}\n")
        self.sw = SummaryWriter(self.log_path)
        self.log_freq = len(self.train_loader)

        # starting values values
        self.epoch = 0
        self.train_losses = None
        self.val_losses = None
        self.best_val_epoch = 0
        self.best_val_loss = None

    def load_ck(self):
        """
        load training checkpoint
        """
        ck_path = self.log_path / "training.ck"
        if ck_path.exists():
            ck = torch.load(ck_path, map_location=torch.device("cpu"))
            # print('[loading checkpoint \'{}\']'.format(ck_path))
            self.logger.info("[loading checkpoint '{}']".format(ck_path))
            self.epoch = ck["epoch"]
            self.model.load_state_dict(ck["model"], strict=True)
            self.model.to(self.device)
            self.best_val_loss = ck["best_val_loss"]
            if ck.get("optimizer", None) is not None:
                self.optimizer.load_state_dict(ck["optimizer"])

    def save_ck(self):
        """
        save training checkpoint
        """
        ck = {
            "epoch": self.epoch,
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "best_val_loss": self.best_val_loss,
        }
        torch.save(ck, self.log_path / "training.ck")

    def train(self):
        """
        train model for one epoch on the Training-Set.
        """
        self.model.train()

        start_time = time.time()
        times = []
        self.train_losses = []

        t = time.time()
        for step, sample in enumerate(self.train_loader):
            self.optimizer.zero_grad()
            img, pts = sample[2].float().to(self.device), sample[3].float().to(
                self.device
            )  # (N, 4, H, W)  (N, J, 3)
            pts_visable = sample[4].float().to(self.device)  # (N, J)

            pred = self.model.forward(img)  # (N, J*D, H', W')

            loss = self.loss(pred, pts, self.out_shape, pts_visable)
            loss.backward()
            self.train_losses.append(loss.item())

            self.optimizer.step(None)

            # print an incredible progress bar
            progress = (step + 1) / len(self.train_loader)
            progress_bar = ("█" * int(50 * progress)) + (
                "┈" * (50 - int(50 * progress))
            )
            times.append(time.time() - t)
            t = time.time()
            if self.log_each_step or (not self.log_each_step and progress == 1):
                self.logger.info(
                    "\r[{}] Epoch {:0{e}d}.{:0{s}d}: │{}│ {:6.2f}% │ Loss: {:.8f} │ ↯: {:5.2f} step/s".format(
                        datetime.now().strftime("%m-%d@%H:%M"),
                        self.epoch,
                        step + 1,
                        progress_bar,
                        100 * progress,
                        np.mean(self.train_losses),
                        1 / np.mean(times),
                        e=math.ceil(math.log10(self.epochs)),
                        s=math.ceil(math.log10(self.log_freq)),
                    )
                )

            # if step >= self.epochs - 1:
            #     break

        # log average loss of this epoch
        mean_epoch_loss = np.mean(self.train_losses)
        self.sw.add_scalar(
            tag="train_loss", scalar_value=mean_epoch_loss, global_step=self.epoch
        )

        cur_lr = self.optimizer.state_dict()["param_groups"][0]["lr"]
        self.sw.add_scalar(tag="train_lr", scalar_value=cur_lr, global_step=self.epoch)

        # log epoch duration
        self.logger.info(f" │ T: {time.time() - start_time:.2f} s")

    def test(self):
        """
        test model on the Test-Set
        """
        self.val_losses = []

        self.model.eval()

        t = time.time()
        for step, sample in enumerate(self.val_loader):
            img, pts = sample[2].float().to(self.device), sample[3].float().to(
                self.device
            )  # (N, 4, H, W)  (N, J, 3)
            pts_visable = sample[4].float().to(self.device)  # (N, J)

            pred = self.model.forward(img)  # (N, J*D, H', W')

            loss = self.loss(pred, pts, self.out_shape, pts_visable)
            self.val_losses.append(loss.item())

        # log average loss on test set
        mean_val_loss = np.mean(self.val_losses)
        # print(f'\t● AVG Loss on VAL-set: {mean_val_loss:.6f} │ T: {time() - t:.2f} s')
        self.logger.info(
            f"\t● AVG Loss on VAL-set: {mean_val_loss:.8f} │ T: {time.time() - t:.2f} s"
        )

        self.sw.add_scalar(
            tag="val_loss", scalar_value=mean_val_loss, global_step=self.epoch
        )

        # save best model
        if self.best_val_loss is None or mean_val_loss < self.best_val_loss:
            self.best_val_loss = mean_val_loss
            self.best_val_epoch = self.epoch
            torch.save(self.model.state_dict(), self.log_path / "best.pth")
            self.logger.info(f"\t● Save best model on epoch: {self.epoch}")

    def run(self):
        """
        start model training procedure (train > test > checkpoint > repeat)
        """
        for e in range(self.epoch, self.epochs):
            # if self.epoch == 13:
            #     pass
            self.train()
            if e % 1 == 0 and e != 0:
                self.test()
            self.epoch += 1
            self.scheduler.step()
            self.save_ck()
            if e % 50 == 0 and e != 0:
                torch.save(self.model.state_dict(), self.log_path / f"{e}.pth")


if __name__ == "__main__":
    trainer = KPTSTrainer()
    trainer.run()
    trainer.logger.info(f"\n▶ best val model: ")
    trainer.logger.info(
        f"\n  epoch, loss: {trainer.best_val_epoch}, {trainer.best_val_loss}"
    )
