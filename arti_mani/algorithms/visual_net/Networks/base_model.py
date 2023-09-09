from pathlib import Path
from typing import Union

import torch
import torch.nn as nn


class BaseModel(nn.Module):
    def __init__(self):
        super().__init__()

    @property
    def n_param(self):
        # type: (BaseModel) -> int
        """
        :return: number of parameters
        """
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    @property
    def is_cuda(self):
        # type: () -> bool
        """
        :return: `True` if the model is on Cuda; `False` otherwise
        """
        return next(self.parameters()).is_cuda

    def save_w(self, path):
        # type: (Union[str, Path]) -> None
        """
        save model weights in the specified path
        """
        torch.save(self.state_dict(), path)

    def load_w(self, path):
        # type: (Union[str, Path]) -> None
        """
        load model weights from the specified path
        """
        self.load_state_dict(torch.load(path))

    def requires_grad(self, flag):
        # type: (bool) -> None
        """
        param flag: True if the model requires gradient, False otherwise
        """

        for p in self.parameters():
            p.requires_grad = flag
