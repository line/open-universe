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
The diffusion time embedding module

Author: Robin Scheibler (@fakufaku)
"""
import math

import torch


class Linear_PReLU(torch.nn.Module):
    def __init__(self, in_features, out_features, prelu_kwargs=None):
        super().__init__()
        if prelu_kwargs is None:
            prelu_kwargs = {}
        self.prelu = torch.nn.PReLU(**prelu_kwargs)
        self.lin = torch.nn.Linear(in_features, out_features)

    def forward(self, x):
        return self.prelu(self.lin(x))


class SigmaBlock(torch.nn.Module):
    """
    Implementation of the random Fourier features for the SigmaBlock of
    UNIVERSE speech enhancement as described in appendix D
    """

    def __init__(self, n_rff=32, n_dim=256, scale=16):
        super().__init__()

        self.register_buffer("freq", scale * torch.zeros(n_rff).normal_())
        self.layer1 = Linear_PReLU(2 * n_rff, 4 * n_rff)
        self.layer2 = Linear_PReLU(4 * n_rff, 8 * n_rff)
        self.layer3 = Linear_PReLU(8 * n_rff, n_dim)

    def forward(self, log10_sigma):
        """Output has n_rff ** (n_stages + 1) dimensions"""
        p = 2.0 * math.pi * self.freq[None, :] * log10_sigma[:, None]
        rff = torch.cat([torch.sin(p), torch.cos(p)], dim=-1)
        g = self.layer1(rff)
        g = self.layer2(g)
        g = self.layer3(g)
        return g


class SimpleTimeEmbedding(torch.nn.Module):
    """
    A simple time embedding linearly mapping the input to a sinusoid with
    continuous frequency
    """

    def __init__(self, n_dim=256):
        super().__init__()

        self.weight = torch.nn.Parameter(torch.zeros((1, 1)), requires_grad=True)
        self.bias = torch.nn.Parameter(torch.zeros((1, 1)), requires_grad=True)
        self.n_dim = n_dim

    def forward(self, log10_sigma):
        time = torch.arange(self.n_dim // 2, device=log10_sigma.device)
        f = 0.5 * torch.sigmoid(self.weight * log10_sigma[:, None] + self.bias)
        p = 2.0 * math.pi * f * time
        g = torch.cat([torch.sin(p), torch.cos(p)], dim=-1)
        return g
