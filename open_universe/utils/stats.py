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
Median of vector values signals.

Author: Robin Scheibler (@fakufaku)
"""
import torch


def signal_median(signal):
    """Aggregated Median over all samples of a signal

    1. Sort the signal along the ensemble dimension
    2. For each sample, find the ensemble member that is closest to the median
    3. For each ensembler member, count the number of samples where it is the median
    4. Choose the ensemble member that is the median for the most samples
    5. Return that ensemble member as the median

    Args:
        signal (torch.Tensor): Signal to compute the median of.
            shape (ensemble, batch, ...)
    Returns:
        torch.Tensor: Median of the signal along the given dimension. shape (batch, ...)
    """
    shape = signal.shape
    signal = signal.flatten(start_dim=2)  # (ensemble, batch, samples)
    n = signal.shape[0]

    sorted_signal, sorted_indices = signal.sort(dim=0)

    _, min_indices = abs(sorted_indices - n / 2).min(dim=0)  # (batch, samples)

    # pad indices so that all possible bin values are represented.
    # we then reduce the counts by 1 to account for the padding.
    pad_bins = torch.broadcast_to(
        torch.arange(n, device=signal.device)[None, :], (min_indices.shape[0], n)
    )
    min_indices = torch.cat((min_indices, pad_bins), dim=1)

    # not efficient, because unique is not batched in torch
    # https://github.com/pytorch/pytorch/issues/103142
    counts = []
    for i in range(n):
        counts.append((min_indices == i).sum(dim=1, keepdim=True))
    counts = torch.cat(counts, dim=1) - 1
    select = counts.argmax(dim=1)  # (batch,)

    median_signal = torch.stack(
        [signal[select[i], i, :] for i in range(signal.shape[1])], dim=0
    )

    median_signal = median_signal.reshape(shape[1:])

    return median_signal
