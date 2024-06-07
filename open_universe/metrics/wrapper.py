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
# Wrapper for All Metrics

A convenient Metrics class that unifies computation of all the metrics.
It will also cache resampled signals so they can be used for other
metrics with the same sampling frequency requirements.

Author: Robin Scheibler (@fakufaku)
"""
import inspect
from collections import defaultdict

import fast_bss_eval
import torch
import torchaudio
from pesq import pesq
from pystoi import stoi

from .dnsmos import Compute_DNSMOS
from .plcmos import PLCMOSEstimator
from .lsd import log_spectral_distance
from .lps import LevenshteinPhonemeSimilarity


def cached_resample(orig_fs, new_fs, cache={}, **signals):
    """
    Resample the signals to the new sampling frequency.
    The resampled signal is stored in the cache for later use.
    If the signal is already in the cache, it is not resampled again.
    """
    out = {}
    for label, signal in signals.items():
        if not (new_fs in cache and label in cache[new_fs]):
            cache[new_fs][label] = torchaudio.functional.resample(
                signal, orig_freq=orig_fs, new_freq=new_fs
            )
        out[label] = cache[new_fs][label]
    return out, cache


class Metrics:
    """
    Paramters
    ---------
    metrics : list of str
        List of metrics to compute. The available metrics are:
        pesq-wb, pesq-nb, stoi, stoi-ext, lsd, si-lsd, lps, dnsmos,
        plcmos, sdr, si-sdr
    """
    @classmethod
    def get_metric_names(cls):
        _available_metrics = []
        for key, value in inspect.getmembers(cls):
            if not key.startswith("_") and not key == "get_metric_names":
                _available_metrics.append(key.replace("_", "-"))
        return _available_metrics

    def __init__(
        self,
        metrics=None,
    ):
        self._available_metrics = Metrics.get_metric_names()

        if metrics is None:
            metrics = self._available_metrics
        else:
            self.metrics = []
            for met in metrics:
                if met not in self._available_metrics:
                    raise NotImplementedError(f"Metric {met} is not supported")
                self.metrics.append(met)

        if "dnsmos" in self.metrics:
            self._dnsmos = Compute_DNSMOS()

        if "plcmos" in self.metrics:
            self._plcmos = PLCMOSEstimator()

        if "lps" in self.metrics:
            self._lps = LevenshteinPhonemeSimilarity()

    def _pesq_base(self, ref, deg, fs, mode, cache={}):
        sig, cache = cached_resample(
            orig_fs=fs, new_fs=16000, cache=cache, ref=ref, deg=deg
        )
        return (
            pesq(16000, sig["ref"].cpu().numpy(), sig["deg"].cpu().numpy(), "wb"),
            cache,
        )

    def pesq_wb(self, ref, deg, fs, cache={}):
        if ref is None:
            return {}, cache
        val, cache = self._pesq_base(ref, deg, fs, "wb", cache=cache)
        return {"pesq-wb": val}, cache

    def pesq_nb(self, ref, deg, fs, cache={}):
        if ref is None:
            return {}, cache
        val, cache = self._pesq_base(ref, deg, fs, "nb", cache=cache)
        return {"pesq-nb": val}, cache

    def _stoi_base(self, ref, deg, fs, extended):
        val = stoi(ref.cpu().numpy(), deg.cpu().numpy(), fs, extended=extended)
        return val.tolist()

    def stoi(self, ref, deg, fs, cache={}):
        if ref is None:
            return {}, cache
        return {"stoi": self._stoi_base(ref, deg, fs, extended=False)}, cache

    def stoi_ext(self, ref, deg, fs, cache={}):
        if ref is None:
            return {}, cache
        return {"stoi-ext": self._stoi_base(ref, deg, fs, extended=True)}, cache

    def lsd(self, ref, deg, fs, cache={}, scale_invariant=False):
        if ref is None:
            return {}, cache

        # use standard default values, adapt for sampling frequency
        n_fft = int(0.025 * fs)  # 25 ms, 400 samples at 16 kHz
        hop_length = int(0.01 * fs)  # 10 ms, 160 samples at 16 kHz
        lsd = log_spectral_distance(
            deg,
            ref,
            n_fft=n_fft,
            hop_length=hop_length,
            scale_invariant=scale_invariant,
        )
        return {"lsd": float(lsd.mean())}, cache

    def si_lsd(self, ref, deg, fs, cache={}):
        """scale invariant LSD"""
        ret, cache = self.lsd(ref, deg, fs, cache=cache, scale_invariant=True)
        try:
            ret["si-lsd"] = ret.pop("lsd")
            return ret, cache
        except KeyError:
            return ret, cache

    def lps(self, ref, deg, fs, cache={}):
        if ref is None:
            return {}, cache

        sig, cache = cached_resample(
            orig_fs=fs, new_fs=self._lps.sr, cache=cache, ref=ref, deg=deg
        )
        val = self._lps(sig["deg"], sig["ref"])
        return {"lps": float(val)}, cache

    def dnsmos(self, ref, deg, fs, cache={}):
        sig, cache = cached_resample(orig_fs=fs, new_fs=16000, cache=cache, deg=deg)
        val = self._dnsmos(sig["deg"].cpu().numpy(), 16000)
        val = {
            m: float(val[m])
            for m in ["OVRL_raw", "SIG_raw", "BAK_raw", "OVRL", "SIG", "BAK"]
        }
        return val, cache

    def plcmos(self, ref, deg, fs, cache={}):
        sig, cache = cached_resample(orig_fs=fs, new_fs=16000, cache=cache, deg=deg)
        val = self._plcmos.run(sig["deg"].cpu().numpy(), 16000)
        return {"plcmos": float(val)}, cache

    def sdr(self, ref, deg, fs, cache={}):
        if ref is None:
            return {}, cache

        if ref.ndim == 1:
            ref = ref[None, None, :]
        elif ref.ndim == 2:
            ref = ref[None, :]
        if deg.ndim == 1:
            deg = deg[None, None, :]
        elif deg.ndim == 2:
            deg = deg[None, :]

        val = fast_bss_eval.sdr(
            ref, deg, zero_mean=False, return_perm=False, clamp_db=100
        )
        return {"sdr": float(val.mean())}, cache

    def si_sdr(self, ref, deg, fs, cache={}):
        if ref is None:
            return {}, cache

        if ref.ndim == 1:
            ref = ref[None, None, :]
        elif ref.ndim == 2:
            ref = ref[None, :]
        if deg.ndim == 1:
            deg = deg[None, None, :]
        elif deg.ndim == 2:
            deg = deg[None, :]

        val = fast_bss_eval.si_sdr(
            ref, deg, zero_mean=False, return_perm=False, clamp_db=100
        )
        return {"si-sdr": float(val.mean())}, cache

    def __call__(
        self, fs, degraded, reference=None, skip_list=None, skip_unknown_metrics=True
    ):
        """
        Parameters
        ----------
        fs : int
            The sampling frequency of the input signals
        degraded : torch.Tensor
            The degraded signal(s) to evaluate
        reference : torch.Tensor, optional
            The reference signal(s) to compare to
        skip_list : list of str, optional
            List of metrics to skip
        skip_unknown_metrics : bool, optional
            Whether to raise an error if an unknown metric is requested

        Returns
        -------
        dict or list of dict
            The computed metrics
        """
        # adjust lengths
        if reference is not None:
            M = max([reference.shape[-1], degraded.shape[-1]])
            degraded = torch.nn.functional.pad(degraded, (0, M - degraded.shape[-1]))
            reference = torch.nn.functional.pad(reference, (0, M - reference.shape[-1]))

            if reference.shape != degraded.shape:
                raise ValueError("The shapes of the inputs should match")

        if degraded.ndim > 2:
            raise ValueError("The input should have 1 or 2 dimensions")

        is_single_input = degraded.ndim == 1

        if is_single_input:
            degraded = degraded[None, :]
            if reference is not None:
                reference = reference[None, :]
        if reference is None:
            reference = [None] * degraded.shape[0]

        output = []
        for idx in range(degraded.shape[0]):
            cache = defaultdict(dict)
            metrics = {}
            for met in self.metrics:
                if skip_list is not None and met in skip_list:
                    continue

                # get the evaluation function
                func = getattr(self, met.replace("-", "_"), None)
                if func is None:
                    if skip_unknown_metrics:
                        continue
                    else:
                        raise ValueError(f"Metric {met} not supported")

                vals, cache = func(reference[idx], degraded[idx], fs, cache=cache)
                metrics.update(vals)
            output.append(metrics)

        if is_single_input:
            return output[0]
        else:
            return output
