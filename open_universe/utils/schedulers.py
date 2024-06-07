# Copyright 2024 LY Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
A custom scheduler combining linear warmup with cosine annealing.

Author: Robin Scheibler (@fakufaku)
"""
import math
import warnings

import torch


class LinearWarmupCosineAnnealingLR(torch.optim.lr_scheduler.CosineAnnealingLR):
    r"""
    This scheduler has
    1) Linear learning rate warm-up from eta_min to the actual learning rate
       until T_warmup iterations
    2) Cosine Annealing Schedule starting at T_cosine gradually going down to
       eta_min at T_max

    Args:
        optimizer (Optimizer): Wrapped optimizer.
        T_warmup (int): Number of warmup iterations
        T_cosine (int): Start of cosing schedule
        T_max (int): Maximum number of iterations.
        eta_min (float): Minimum learning rate. Default: 0.
        last_epoch (int): The index of last epoch. Default: -1.
        verbose (bool): If ``True``, prints a message to stdout for
            each update. Default: ``False``.
    """

    def __init__(
        self,
        optimizer,
        T_warmup,
        T_cosine,
        T_max,
        eta_min=0,
        last_epoch=-1,
        verbose=False,
    ):
        self.T_cosine = T_cosine
        self.T_warmup = T_warmup
        super().__init__(optimizer, T_max, eta_min, last_epoch, verbose)
        assert self.T_warmup < self.T_cosine < self.T_max

    def get_lr(self):
        if not self._get_lr_called_within_step:
            warnings.warn(
                "To get the last learning rate computed by the scheduler, "
                "please use `get_last_lr()`.",
                UserWarning,
            )

        if self.last_epoch <= self.T_warmup:
            return [
                self.eta_min
                + (base_lr - self.eta_min) * self.last_epoch / self.T_warmup
                for base_lr, group in zip(self.base_lrs, self.optimizer.param_groups)
            ]

        denom = self.T_max - self.T_cosine
        num = self.last_epoch - self.T_cosine

        if self.last_epoch <= self.T_cosine:
            return [group["lr"] for group in self.optimizer.param_groups]
        elif self._step_count == 1 and self.last_epoch > self.T_cosine:
            return [
                self.eta_min
                + (base_lr - self.eta_min) * (1 + math.cos((num) * math.pi / denom)) / 2
                for base_lr, group in zip(self.base_lrs, self.optimizer.param_groups)
            ]
        elif (num - 1 - denom) % (2 * denom) == 0:
            return [
                group["lr"]
                + (base_lr - self.eta_min) * (1 - math.cos(math.pi / denom)) / 2
                for base_lr, group in zip(self.base_lrs, self.optimizer.param_groups)
            ]
        return [
            (1 + math.cos(math.pi * num / denom))
            / (1 + math.cos(math.pi * (num - 1) / denom))
            * (group["lr"] - self.eta_min)
            + self.eta_min
            for group in self.optimizer.param_groups
        ]
