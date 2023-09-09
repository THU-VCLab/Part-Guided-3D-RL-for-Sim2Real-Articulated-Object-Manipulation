import logging
import os
import random
import time
from typing import Optional

import numpy as np
import torch
import yaml
from arti_mani import ALG_DIR
from path import Path


def set_seed(seed=None):
    # type: (Optional[int]) -> int
    """
    set the random seed using the required value (`seed`)
    or a random value if `seed` is `None`
    :return: the newly set seed
    """
    if seed is None:
        seed = random.randint(1, 10000)
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)  # 为了禁止hash随机化，使得实验可复现
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    return seed


class Logger:
    def __init__(self, path, clevel=logging.DEBUG, Flevel=logging.INFO):
        self.logger = logging.getLogger(path)
        self.logger.setLevel(logging.DEBUG)
        fmt = logging.Formatter(
            "[%(asctime)s] [%(levelname)s] %(message)s", "%Y%m%d_%H:%M:%S"
        )
        # 设置CMD日志
        sh = logging.StreamHandler()
        sh.setFormatter(fmt)
        sh.setLevel(clevel)
        # 设置文件日志
        fh = logging.FileHandler(path)
        fh.setFormatter(fmt)
        fh.setLevel(Flevel)
        self.logger.addHandler(sh)
        self.logger.addHandler(fh)

    def debug(self, message):
        self.logger.debug(message)

    def info(self, message):
        self.logger.info(message)

    def war(self, message):
        self.logger.warning(message)

    def error(self, message):
        self.logger.error(message)

    def cri(self, message):
        self.logger.critical(message)


class Config(object):
    def __init__(
        self,
        conf_file_path=None,
        seed=None,
        exp_name="pretrain_unet",
        mode="train",
        log_name=None,
        log=True,
    ):
        assert mode in ["train", "test"]
        self.log_each_step = log

        # if the configuration file is not specified
        # try to load a configuration file based on the experiment name
        tmp = Path(__file__).parent / (exp_name + ".yaml")
        if conf_file_path is None and tmp.exists():
            conf_file_path = tmp

        # read the YAML configuation file
        if conf_file_path is None:
            cfg = {}
        else:
            conf_file = open(conf_file_path, "r")
            cfg = yaml.safe_load(conf_file)

        # read configuration parameters from YAML file
        # or set their default value
        self.seed = cfg.get("seed", set_seed(seed))  # type: int
        self.hmap_h = cfg.get("img_h", 72)  # type: int
        self.hmap_w = cfg.get("img_w", 128)  # type: int
        self.num_classes = cfg.get("num_classes", 2)  # type: int
        self.lr = cfg.get("lr", 0.0001)  # type: float
        self.epochs = cfg.get("epochs", 1000)  # type: int
        self.n_workers = cfg.get("n_workers", 8)  # type: int
        self.batch_size = cfg.get("batch_size", 1)  # type: int
        self.train_ratio = cfg.get("train_ratio", 0.7)  # type: float
        self.val_ratio = cfg.get("val_ratio", 0.1)  # type: float
        self.test_ratio = cfg.get("test_ratio", 0.2)  # type: float
        self.loss = cfg.get("loss", None)  # type: str
        self.exp_suffix = cfg.get("exp_suffix", None)  # type: str
        self.data_path = cfg.get("data_path", None)  # type: str
        self.segnet_config = cfg.get("segnet_config", {})  # type: dict
        self.smp_config = cfg.get("smp_config", {})  # type: dict

        if cfg.get("device", None) is not None and cfg["device"] != "cpu":
            os.environ["CUDA_VISIBLE_DEVICES"] = str(
                cfg.get("device").split(":")[1]
            )  # type: str
            self.device = "cuda:0"
        elif cfg.get("device", None) is not None and cfg["device"] == "cpu":
            self.device = "cpu"
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        assert self.exp_suffix is not None, "set exp_suffix in config yaml! "

        assert os.path.exists(
            ALG_DIR / self.data_path
        ), "the specified directory for the Dataset does not exists"

        # define and build output path
        if log_name is None:
            self.exp_log_path = f"log/{exp_name}/{time.strftime('%Y%m%d_%H%M%S', time.localtime())}_{self.exp_suffix}"
        else:
            self.exp_log_path = f"log/{exp_name}/{log_name}"
        if not os.path.exists(self.exp_log_path):
            os.makedirs(self.exp_log_path)

        # set logger
        self.logger = Logger(f"{self.exp_log_path}/{mode}_log.txt")

        self.logger.info(f"Config:\n{cfg}")
