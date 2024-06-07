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
# Perceptual Evaluation of Speech Quality (PESQ) Wrapper

A simple wrapper to measure PESQ.

Author: Robin Scheibler (@fakufaku)
"""
import torch
import torchaudio
from pesq import pesq


class PESQ(torch.nn.Module):
    """
    PESQ wrapper for PyTorch.

    Parameters
    ----------
    mode : str
        The mode of the PESQ computation. Can be 'wb' or 'nb'.
    audio_fs : int
        The sampling frequency of the audio signals to evaluate.
    pesq_fs : int
        The sampling frequency at which to compute PESQ.
    """
    def __init__(self, mode="wb", audio_fs=16000, pesq_fs=16000):
        super().__init__()
        self.mode = mode
        self.pesq_fs = pesq_fs
        self.audio_fs = audio_fs

        if pesq_fs != audio_fs:
            self.resampler = torchaudio.transforms.Resample(
                orig_freq=self.audio_fs, new_freq=self.pesq_fs
            )
        else:
            self.resampler = lambda x: x

    def forward(self, est, ref):
        """
        Compute the PESQ score between the estimated and reference signals.
        The input are assumed to be 3D tensors with batch, 1 channel, and time
        as dimensions.
        PESQ is computed sample-by-sample. All zero input samples are skipped.

        Parameters
        ----------
        est : torch.Tensor, shape = [batch, 1, time]
            The estimated signals
        ref : torch.Tensor, shape = [batch, 1, time]
            The reference signals
        """
        est = self.resampler(est)
        ref = self.resampler(ref)

        est = est.cpu().numpy()
        ref = ref.cpu().numpy()

        ave_pesq = list()
        for ii in range(est.shape[0]):
            try:
                ave_pesq.append(pesq(self.pesq_fs, ref[ii, 0], est[ii, 0], self.mode))
            except Exception:
                continue
        p_esq = torch.mean(torch.tensor(ave_pesq))

        return p_esq
