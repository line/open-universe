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
# Wrapper for Some Metrics

This can be used to monitor speech enhancement metrics of the validation set
during training
"""
from collections import defaultdict

import torch
import torchaudio

from .dnsmos import Compute_DNSMOS
from .pesq import pesq
from .lsd import log_spectral_distance
from .lps import LevenshteinPhonemeSimilarity


class EvalMetrics(torch.nn.Module):
    """
    Wrapper for some metrics that can be used to monitor the performance of
    speech enhancement models during training.

    PESQ, DNSMOS, and log-spectral distance are computed by default.

    Parameters
    ----------
    pesq_mode : str
        The mode for PESQ computation. Can be "wb" or "nb".
    lps : bool
        Whether to compute the Levenshtein phoneme similarity.
        This is off by default as it seemed to be unreasonably slow.
    audio_fs : int
    """
    def __init__(self, pesq_mode="wb", lps=False, audio_fs=16000):
        super().__init__()
        self.mode = pesq_mode
        self.eval_fs = 16000
        self.audio_fs = audio_fs

        self.dnsmos = Compute_DNSMOS()

        self.lps = LevenshteinPhonemeSimilarity() if lps else None

        if self.eval_fs != self.audio_fs:
            self.resampler = torchaudio.transforms.Resample(
                orig_freq=self.audio_fs, new_freq=self.eval_fs
            )
        else:
            self.resampler = lambda x: x

    def forward(self, est, ref):
        metrics = {}

        # log-spectral distance can be done at any sample rate
        metrics["lsd"] = log_spectral_distance(est, ref).mean()

        # resample at 16000 for metrics that need it
        est = self.resampler(est)
        ref = self.resampler(ref)

        est = est.cpu().numpy()
        ref = ref.cpu().numpy()

        metrics_lists = defaultdict(list)
        for ii in range(est.shape[0]):
            try:
                # PESQ
                metrics_lists["pesq"].append(
                    pesq(self.eval_fs, ref[ii, 0], est[ii, 0], self.mode)
                )

                # levenshtein phoneme similarity
                if self.lps is not None:
                    metrics_lists["lps"].append(self.lps(est[ii, 0], ref[ii, 0]))

                # DNSMOS
                dnsmos_met = self.dnsmos(est[ii, 0], self.eval_fs)
                for lbl in ["OVRL", "SIG", "BAK"]:
                    metrics_lists["dnsmos-" + lbl] = dnsmos_met[lbl]

            except Exception:
                continue

        # aggregate
        for met, L in metrics_lists.items():
            metrics[met] = torch.mean(torch.tensor(L))

        return metrics
