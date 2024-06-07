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
Functions to normalize the signal amplitude.

Author: Robin Scheibler (@fakufaku)
"""
import torch


def _norm2(signal, eps=1e-5):
    return signal.std(dim=(1, 2), keepdim=True).clamp(min=eps)


def _norm_max(signal, eps=1e-5):
    std = abs(signal.view((signal.shape[0], -1))).max(dim=1).values
    return std[:, None, None].clamp(min=eps)


def _compute_gain(signal, norm, level, eps=1e-5):
    if norm == 2 or norm == "2":
        gain = level / _norm2(signal)
    elif norm == "max":
        gain = level / _norm_max(signal)
    elif norm == "2-max":
        norm_2 = _norm2(signal, eps=eps)
        norm_max = _norm_max(signal, eps=eps)
        gain = torch.minimum(level / norm_2, 1.0 / norm_max)
    else:
        raise NotImplementedError(
            f"Norm {norm} is not implemented for batch normalization"
        )
    return gain


def normalize_batch(batch, norm=2, level_db=0.0, ref="noisy", eps=1e-5, zero_mean=True):
    """
    Normalize the input by some norm

    Parameters
    ----------
    norm: int or str
        2 or 'max' for now
    level_db: float
        the target level to achieve in decibels
    ref:
        which signal to use when compute the gain, can be: noisy (default) | both.
        When choosing 'both', the signals are normalized separately
    """
    assert ref in ["noisy", "both"]
    level = 10 ** (level_db / 20.0)
    mix, *others = batch

    if zero_mean:
        mean = mix.mean(dim=(1, 2), keepdim=True)
        mix = mix - mean
    else:
        mean = 0.0

    gain = _compute_gain(mix, norm, level, eps=eps)
    mix = mix * gain

    out = [mix]

    for tgt in others:
        if tgt is not None:
            if ref == "both":
                if zero_mean:
                    mean_t = tgt.mean(dim=(1, 2), keepdim=True)
                    tgt = tgt - mean_t
                gain_t = _compute_gain(tgt, norm, level, eps=eps)
                tgt = tgt * gain_t
            else:
                tgt = (tgt - mean) * gain
        out.append(tgt)
    return out, mean, 1.0 / gain


def denormalize_batch(x, mean, std):
    return x * std + mean
