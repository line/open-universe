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
# LoRA Layers

Implementation of LoRA for linear and 1d convolutional layers.
"""
from typing import Optional, List, Union
import torch

import torch.nn.functional as F


class LoraConv1d(torch.nn.Module):
    """
    LoRA adapter class for a Conv1d layer

    Parameters
    ----------
    module : torch.nn.Module
        The original module to be adapted
    rank : int
        The rank of the low-rank approximation
    alpha : float, optional
        The scaling factor for the low-rank approximation
    """

    def __init__(
        self, module: torch.nn.Module, rank: int, alpha: Optional[float] = None
    ):
        super().__init__()

        if not isinstance(module, torch.nn.Conv1d):
            raise ValueError("module should be an instance of torch.nn.Conv1d")

        if module.padding_mode != "zeros":
            raise ValueError("LoRA only supports padding_mode='zeros' for Conv1d")

        self.rank = rank
        self.alpha = alpha if alpha is not None else rank

        # the original module
        self.conv = module

        # in the first part, we keep the size constant and do not apply any
        # striding or dilation
        w = self.conv.weight

        if w.shape[1] < rank or w.shape[0] < rank:
            raise ValueError(
                "The rank should be smaller than the input and output size"
            )

        self.lora_weight_a = torch.nn.Parameter(w.new_zeros(w.shape[0], rank))
        self.lora_weight_b = torch.nn.Parameter(
            w.new_zeros(rank, w.shape[1] * w.shape[2]).normal_()
        )

    def _get_weights(self):
        lora_w = torch.einsum("or,ri->oi", self.lora_weight_a, self.lora_weight_b)
        lora_w = lora_w.view(self.conv.weight.shape)
        return self.conv.weight + (self.alpha / self.rank) * lora_w

    def _get_kwargs(self):
        m = self.conv
        return dict(
            stride=m.stride,
            padding=m.padding,
            dilation=m.dilation,
            groups=m.groups,
            padding_mode=m.padding_mode,
        )

    def un_lora(self):
        # copy the original original parameters
        m = self.conv
        conv = torch.nn.Conv1d(
            in_channels=m.in_channels,
            out_channels=m.out_channels,
            bias=m.bias is not None,
            device=m.weight.device,
            dtype=m.weight.dtype,
            **self._get_kwargs(),
        )
        # copy updated parameters in the new module
        with torch.no_grad():
            conv.weight[:] = self._get_weights()
            conv.bias[:] = self.conv.bias
        return conv

    def forward(self, input):
        weight = self._get_weights()
        bias = self.conv.bias
        kwargs = self._get_kwargs()
        kwargs.pop("padding_mode", "zeros")
        return F.conv1d(input, weight, bias, **kwargs)


class LoraConvTranspose1d(torch.nn.Module):
    """
    LoRA adapter class for a Conv1d layer

    Parameters
    ----------
    module : torch.nn.Module
        The original module to be adapted
    rank : int
        The rank of the low-rank approximation
    alpha : float, optional
        The scaling factor for the low-rank approximation
    """

    def __init__(
        self, module: torch.nn.Module, rank: int, alpha: Optional[float] = None
    ):
        super().__init__()

        if not isinstance(module, torch.nn.ConvTranspose1d):
            raise ValueError("module should be an instance of torch.nn.Conv1d")

        self.rank = rank
        self.alpha = alpha if alpha is not None else rank

        # the original module
        self.conv = module

        # in the first part, we keep the size constant and do not apply any
        # striding or dilation
        w = self.conv.weight

        if w.shape[1] < rank or w.shape[0] < rank:
            raise ValueError(
                "The rank should be smaller than the input and output size"
            )

        self.lora_weight_a = torch.nn.Parameter(w.new_zeros(w.shape[0], rank))
        self.lora_weight_b = torch.nn.Parameter(
            w.new_zeros(rank, w.shape[1] * w.shape[2]).normal_()
        )

    def _get_weights(self):
        lora_w = torch.einsum("or,ri->oi", self.lora_weight_a, self.lora_weight_b)
        lora_w = lora_w.view(self.conv.weight.shape)
        return self.conv.weight + (self.alpha / self.rank) * lora_w

    def _get_kwargs(self):
        m = self.conv
        return dict(
            stride=m.stride,
            padding=m.padding,
            output_padding=m.output_padding,
            groups=m.groups,
            dilation=m.dilation,
        )

    def un_lora(self):
        # copy the original original parameters
        m = self.conv
        conv = torch.nn.Conv1d(
            in_channels=m.in_channels,
            out_channels=m.out_channels,
            bias=m.bias is not None,
            device=m.weight.device,
            dtype=m.weight.dtype,
            **self._get_kwargs(),
        )
        # copy updated parameters in the new module
        with torch.no_grad():
            conv.weight[:] = self._get_weights()
            if self.conv.bias is not None:
                conv.bias[:] = self.conv.bias
        return conv

    def forward(self, input):
        weight = self._get_weights()
        return F.conv_transpose1d(
            input, weight, bias=self.conv.bias, **self._get_kwargs()
        )


class LoraLinear(torch.nn.Module):
    """
    LoRA adapter class for a Linear layer

    Parameters
    ----------
    module : torch.nn.Module
        The original module to be adapted
    rank : int
        The rank of the low-rank approximation
    alpha : float, optional
        The scaling factor for the low-rank approximation
    """

    def __init__(
        self, module: torch.nn.Module, rank: int, alpha: Optional[float] = None
    ):
        super().__init__()

        if not isinstance(module, torch.nn.Linear):
            raise ValueError("module should be an instance of torch.nn.Linear")

        self.rank = rank
        self.alpha = alpha if alpha is not None else rank

        # the original module
        self.linear = module

        # the low-rank extension
        w = self.linear.weight

        if w.shape[1] < rank or w.shape[0] < rank:
            raise ValueError(
                "The rank should be smaller than the input and output size"
            )

        self.lora_linear_a = torch.nn.Parameter(w.new_zeros(w.shape[0], rank).normal_())
        self.lora_linear_b = torch.nn.Parameter(w.new_zeros(rank, w.shape[1]))

    def un_lora(self):
        # copy the original original parameters
        m = self.linear
        linear = torch.nn.Linear(
            in_features=m.in_features,
            out_features=m.out_features,
            bias=m.bias is not None,
            device=m.weight.device,
            dtype=m.weight.dtype,
        )
        # copy updated parameters in the new module
        with torch.no_grad():
            linear.weight[:] = self._get_weights()
            linear.bias[:] = self.conv.bias
        return linear

    def _get_weights(self):
        lora_w = torch.einsum("ir,ro->io", self.lora_linear_a, self.lora_linear_b)
        return self.linear.weight + (self.alpha / self.rank) * lora_w

    def forward(self, x):
        x = torch.einsum("...k,kl->...l", x, self._get_weights().T) + self.linear.bias
        return x
