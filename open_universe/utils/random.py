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
A few useful functions to create randomness.

Author: Robin Scheibler (@fakufaku)
"""
import math

import torch


def randn_like(y, generator=None):
    """
    Samples a vectory of normally distributed values of the same shape,
    type, and device as y.
    """
    return y.new_zeros(y.shape).normal_(generator=generator)


def center_truncated_normal(
    area=0.95, min=0.0, max=1.0, size=(1,), generator=None, device=None
):
    """
    Sample from a truncated normal distribution with a given area.
    The mode of the normal distribution is centered at (max + min) / 2.

    Parameters
    ----------
    area:
        central area of the normal distribution to preserve
    min:
        minimum value to sample
    max:
        value to sample
    """
    if isinstance(size, int):
        size = (size,)
    norm_dist = torch.distributions.normal.Normal(0.0, 1.0)
    if not isinstance(area, torch.Tensor):
        area = torch.tensor(area)
    q = norm_dist.icdf(area + 0.5 * (1 - area))
    mean = 0.5 * (max + min)
    std = 0.5 * (max - min) / q
    n = 0
    n_samples = math.prod(size)
    samples = []
    while n < n_samples:
        z = torch.randn(n_samples - n, generator=generator, device=device)
        new_s = mean + z * std
        sel = torch.logical_and(min <= new_s, new_s <= max)
        samples.append(new_s[sel])
        n += sel.sum()
    samples = torch.cat(samples)
    return samples.reshape(size)
