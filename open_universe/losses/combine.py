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
# Combined Loss

A simple class that groups multiple loss functions into a single module.
"""
from hydra.utils import instantiate
import torch


class MultiLoss(torch.nn.Module):
    """
    A simple class that groups multiple loss functions into a single module.

    Parameters
    ----------
    losses : dict
        A dictionary with the loss names as keys and a tuple with the weight
        and the loss function as values.
    """
    def __init__(self, losses):
        super().__init__()
        losses = instantiate(losses, _recursive_=False)
        self.weights = {k: v[0] for k, v in losses.items()}
        self.losses = torch.nn.ModuleDict({k: v[1] for k, v in losses.items()})

    def forward(self, y_hat, y, with_dict=False):
        """
        Compute the loss.

        Parameters
        ----------
        y_hat : torch.Tensor
            The predicted values
        y : torch.Tensor
            The target values
        with_dict : bool, optional
            If True, alos return a dictionary with the individual losses

        Return
        ------
        loss : torch.Tensor
            The total loss
        loss_dict : dict
            A dictionary with the individual losses (only if `with_dict=True`)
        """
        loss = 0.0
        loss_dict = {}
        for name, loss_fn in self.losses.items():
            weight = self.weights[name]
            loss_dict[name] = loss_fn(y_hat, y)
            loss += weight * loss_dict[name]
        if with_dict:
            return loss, loss_dict
        else:
            return loss
