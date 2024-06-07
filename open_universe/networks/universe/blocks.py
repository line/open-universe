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
Building blocks for the UNIVERSE network

Author: Robin Scheibler (@fakufaku)
"""
import math
from typing import Optional

import numpy as np
import scipy
import torch
import torch.nn.functional as F

from .. import bigvgan


def init_weights(m, mean=0.0, std=0.01):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        m.weight.data.normal_(mean, std)


def cond_weight_norm(x, use=False):
    if use:
        x = torch.nn.utils.weight_norm(x)
        x.apply(init_weights)
        return x
    else:
        return x


def remove_weight_norm(model):
    for name, child in model.named_children():
        try:
            torch.nn.utils.remove_weight_norm(child)
        except ValueError:
            remove_weight_norm(child)


def film(x, y):
    if y.shape[1] != 2 * x.shape[1]:
        raise ValueError("g should have 2 times more channels than y")
    y = y.view(y.shape + (1,) * (x.ndim - y.ndim))
    gamma = y[:, : x.shape[1], ...]
    beta = y[:, x.shape[1] :, ...]
    return gamma * x + beta


def get_binomial_filter(kernel_size):
    binomial = scipy.linalg.pascal(kernel_size, kind="lower", exact=True)
    norm = np.sqrt(np.mean(binomial**2))
    binomial = (binomial[kernel_size - 1, :] / norm).astype("float32")
    weights = torch.tensor(binomial, dtype=torch.float32)
    weights = weights / weights.square().mean().sqrt()
    return weights


def low_pass_filter(x, kernel_size, filter_type="binomial"):
    if filter_type == "binomial":
        weights = get_binomial_filter(kernel_size)
    else:
        raise NotImplementedError(f"Unknown filter type: {filter_type}")
    weights = weights.to(x.device)
    inch = x.shape[1]
    weights = torch.broadcast_to(weights[None, None, :], (inch, 1, weights.shape[0]))
    x = F.conv1d(x, weights, padding="same", groups=inch)
    return x


class IdentityProj:
    def __init__(self, *args, **kwargs):
        pass

    def __call__(self, x):
        return x


class LinearProj(torch.nn.Module):
    def __init__(self, in_dim, out_dim=None, use_weight_norm=False):
        super().__init__()
        if out_dim is None:
            out_dim = in_dim
        self.proj = cond_weight_norm(
            torch.nn.Conv1d(in_dim, out_dim, 1), use=use_weight_norm
        )

    def forward(self, x, c):
        return (self.proj(c) + x) / math.sqrt(2.0)


class GRUSeqModel(torch.nn.GRU):
    def __init__(self, oc):
        super().__init__(
            oc,  # number of channels after downsampling
            oc // 2,  # bi-directional double # of output channels
            num_layers=1,
            bidirectional=True,
            batch_first=True,
        )

    def forward(self, x):
        x, _ = super().forward(x.transpose(-2, -1).contiguous())
        return x.transpose(-2, -1).contiguous()


class BinomialAntiAlias(torch.nn.Module):
    def __init__(self, kernel_size):
        super().__init__()
        self.register_buffer("weights", get_binomial_filter(kernel_size))

    def forward(self, x):
        inch = x.shape[1]
        weights = torch.broadcast_to(
            self.weights[None, None, :], (inch, 1, self.weights.shape[0])
        )
        x = F.conv1d(x, weights, padding="same", groups=inch)
        return x


class PReLU_Conv(torch.nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bias=True,
        padding_mode="zeros",
        device=None,
        dtype=None,
        use_transpose=False,
        prelu_kwargs=None,
        act_type="prelu",
        use_weight_norm=False,
        use_antialiasing=False,
    ):
        super().__init__()

        if prelu_kwargs is None:
            prelu_kwargs = {}

        if use_transpose:
            ConvClass = torch.nn.ConvTranspose1d
        else:
            ConvClass = torch.nn.Conv1d

        self.stride = stride
        self.use_transpose = use_transpose

        self.antialiasing = use_antialiasing
        self.bias = None
        if self.antialiasing:
            if bias:
                self.bias = torch.nn.Parameter(torch.zeros(out_channels))
            else:
                self.bias = None
            self.low_pass_filter = BinomialAntiAlias(kernel_size=2 * kernel_size + 1)
            bias = False  # do not use bias in the convolution

        if act_type == "snake":
            self.prelu = bigvgan.AliasFreeSnake(in_channels, alpha_logscale=True)
        elif act_type == "snakebeta":
            self.prelu = bigvgan.AliasFreeSnake(
                in_channels, alpha_logscale=True, beta=True
            )
        elif act_type == "prelu":
            self.prelu = torch.nn.PReLU(device=device, dtype=dtype, **prelu_kwargs)
        elif act_type == "none":
            self.prelu = lambda x: x  # identity
        else:
            raise ValueError("'act_type' should be one of [prelu | snake]")

        self.conv = ConvClass(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
            padding_mode=padding_mode,
            device=device,
            dtype=dtype,
        )

        self.conv = cond_weight_norm(self.conv, use=use_weight_norm)

    def forward(self, x):
        r = x.shape[-1] % self.stride
        if not self.use_transpose and r != 0:
            pad = self.stride - r
            x = torch.nn.functional.pad(x, (0, pad))

        x = self.prelu(x)

        if self.antialiasing and not self.use_transpose:
            # for regular convolution (downsampling), low-pass before
            x = self.low_pass_filter(x)

        x = self.conv(x)

        if self.antialiasing and self.use_transpose:
            # for transposed convolution (upsampling), low-pass after
            x = self.low_pass_filter(x)

        if self.bias is not None:
            # manual bias are only added with antialiasing on
            x = x + self.bias.reshape((1, -1, 1))

        return x


class ConvBlock(torch.nn.Module):
    """
    Convolution block of UNVIVERSE speech enhancement network
    as described in the original paper Appendix D
    """

    def __init__(
        self,
        n_channels,
        rate_change=None,  # sampling rate change factor
        rate_change_dir="none",  # up/down
        act_type="prelu",
        antialiasing=False,
        use_weight_norm=False,
        signal_cond_type=None,
    ):
        super().__init__()

        if rate_change_dir not in ["up", "down", "none"]:
            raise ValueError(
                "The rate_change_dir value should be one of 'up' or 'down'"
            )

        if rate_change_dir in ["up", "down"] and rate_change is None:
            raise ValueError(
                "The rate_change should be specified when using for down/upsampling"
            )

        self.rate = rate_change
        self.rate_change_dir = rate_change_dir

        if self.rate_change_dir == "down":
            self.in_channels = n_channels
            self.out_channels = 2 * n_channels
            self.rate_change_conv = PReLU_Conv(
                n_channels,
                n_channels * 2,
                kernel_size=rate_change,
                stride=rate_change,
                use_weight_norm=use_weight_norm,
                use_antialiasing=antialiasing,
            )
        elif self.rate_change_dir == "up":
            self.in_channels = 2 * n_channels
            self.out_channels = n_channels
            self.rate_change_conv = PReLU_Conv(
                n_channels * 2,
                n_channels,
                kernel_size=rate_change,
                stride=rate_change,
                use_transpose=True,
                use_weight_norm=use_weight_norm,
                use_antialiasing=antialiasing,
            )
        else:
            self.in_channels = n_channels
            self.out_channels = n_channels
            self.rate_change_conv = None

        self.conv1 = PReLU_Conv(
            n_channels,
            n_channels,
            kernel_size=5,
            padding="same",
            act_type=act_type,
            use_weight_norm=use_weight_norm,
        )
        self.conv2 = PReLU_Conv(
            n_channels,
            n_channels,
            kernel_size=3,
            padding="same",
            act_type=act_type,
            use_weight_norm=use_weight_norm,
        )
        self.conv3 = PReLU_Conv(
            n_channels,
            n_channels,
            kernel_size=3,
            padding="same",
            act_type=act_type,
            use_weight_norm=use_weight_norm,
        )

        if signal_cond_type == "x-attention":
            self.signal_cond_proj = LocalCrossAttention(
                d_model=n_channels, num_memory=20, block_len=100
            )
        elif signal_cond_type == "linear":
            self.signal_cond_proj = LinearProj(
                n_channels, use_weight_norm=use_weight_norm
            )
        elif signal_cond_type == "none" or signal_cond_type is None:
            self.signal_cond_proj = None
        else:
            raise ValueError("Values for 'input_cond_proj_type' can be attention|none")

    def forward(
        self,
        h: torch.Tensor,
        noise_cond: Optional[torch.Tensor] = None,
        input_cond: Optional[torch.Tensor] = None,
        res: Optional[torch.Tensor] = None,
        length: Optional[int] = None,
    ):
        """
        Parameters
        ----------
        h: torch.Tensor
            input from previous stage
        noise_cond: torch.Tensor
            noise variance embedding vector
        input_cond: torch.Tensor
            optional conditioning input to use a condition
        res: torch.Tensor
            optional residual input from a skip connection, only for up-sampling block
        length: int
            optional target length for the upsampled signal, ignored for
            downsample blocks

        Returns
        -------
        h: torch.Tensor
            output to go to next stage
        res:
            output for the skip connection
        cond:
            signal conditioning output
        """

        norm_factor = 1.0 / math.sqrt(2)

        if self.rate_change_dir == "up":
            # do the upsampling here
            if length is not None and self.rate * h.shape[-1] < length:
                h = torch.nn.functional.pad(h, (0, 1))

            h = self.rate_change_conv(h)
            if h is None:
                breakpoint()

            if length is not None:
                h = torch.nn.functional.pad(h, (0, length - h.shape[-1]))

        if res is not None:
            if self.rate_change_dir != "down":
                h = (h + res) * norm_factor
            else:
                raise ValueError(
                    "The residual input is not allowed for downsampling blocks"
                )

        # input conditioning stage
        cond_out = self.conv1(h)
        if input_cond is not None:
            if self.signal_cond_proj is None:
                c = (cond_out + input_cond) * norm_factor
            if self.signal_cond_proj is not None:
                c = self.signal_cond_proj(cond_out, input_cond)
        else:
            c = cond_out

        # main stage
        if noise_cond is not None:
            c = film(c, noise_cond)
        c = self.conv2(c)
        c = self.conv3(c)

        # breakpoint()
        v_out = (h + c) * norm_factor  # residual

        if self.rate_change_dir == "down":
            # pad to make sure that we don't throw away samples
            r = h.shape[-1] % self.rate
            if r != 0:
                pad = self.rate - r
                v_pad = torch.nn.functional.pad(v_out, (0, pad))
            else:
                v_pad = v_out
            h = self.rate_change_conv(v_pad)
            return h, v_out, cond_out
        else:
            return v_out, v_out, cond_out
