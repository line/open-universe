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
A few functions to count parameters or handle computing devices.

Author: Robin Scheibler (@fakufaku)
"""
from collections import defaultdict

import torch


def count_parameters(module):
    param_count = 0
    for p in module.parameters():
        param_count += p.numel()
    return param_count


def count_parameters_per_level(module, max_level=2):
    param_count = defaultdict(lambda: 0)
    for name, p in module.named_parameters():
        mods = name.split(".")
        c = p.numel()

        mod = ""
        param_count[mod] += c
        for idx in range(max_level):
            mod = ".".join(mods[: idx + 1])
            param_count[mod] += c

    return param_count


def to_device(data, device="cpu", to_numpy=False):
    """recursively transfers tensors to cpu"""
    if to_numpy and device != "cpu":
        raise ValueError("to_numpy and device=gpu is not compatible")

    if isinstance(data, list):
        return [to_device(d, device, to_numpy) for d in data]
    elif isinstance(data, dict):
        outdict = {}
        for key, val in data.items():
            outdict[key] = to_device(val, device, to_numpy)
        return outdict
    elif isinstance(data, torch.Tensor):
        data = data.to(device)
        if to_numpy:
            data = data.numpy()
        return data
    else:
        return data


def pad_dim_right(a, x):
    """Broadcasts a over all dimensions of x, except the batch dimension, which must match."""
    if a.shape != x.shape[: a.ndim]:
        raise ValueError("All left dimensions of a and x should be matching")
    pad_dim = (...,) + (None,) * (x.ndim - a.ndim)
    return a[pad_dim]
