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
# Log-spectral distance (LSD)

Functional implementation of the log-spectral distance and
scale-invariant log-spectral distance.
The latter just rescale the input by orthogonally projecting the target into
the estimate sub-space.
"""
import torch
import torchaudio


def log_spectral_distance(
    input,
    target,
    p=2,
    db=True,
    n_fft=400,
    hop_length=160,
    eps=1e-7,
    win_length=None,
    window=None,
    pad=0,
    scale_invariant=False,
    **stft_kwargs,
):
    """
    Implementation of the Log-Spectral Distance (LSD) metric.

    See:
    Gray and Markel, "Distance measures for speech processing," 1976.
    or
    https://en.wikipedia.org/wiki/Log-spectral_distance

    Parameters
    ----------
    input : torch.Tensor
        The input signal. Shape: [..., T]
    target : torch.Tensor
        The target signal. Shape: [..., T]
    p : float
        The norm to use. Default is 2.
    db : bool
        If True, the metric is computed in decibel units,
        i.e., `10.0 * torch.log10` is used instead of `torch.log`.
    n_fft : int
        The number of FFT points. Default is 400.
    hop_length : int
        The hop length for the STFT. Default is 160.
    eps : float
        A small value to avoid numerical instabilities. Default is 1e-7.
    win_length : int or None
        The window length. Default is None, which is equal to n_fft.
    window : torch.Tensor or None
        The window function. Default is None, which is a Hann window.
    pad : int
        The amount of padding to add to the input signal before computing the
        STFT. Default is 0.
    scale_invariant : bool
        If True, the input is rescaled by orthogonal projection of the target onto
        the input sub-space. Default is False.
    stft_kwargs : dict
        Additional arguments to pass to `torchaudio.functional.spectrogram`.

    Returns
    -------
    torch.Tensor
        The log-spectral distance. Shape: [...]
    """
    if win_length is None:
        win_length = n_fft

    if window is None:
        window = torch.hann_window(
            n_fft, periodic=True, dtype=input.dtype, device=input.device
        )

    if p is None or p <= 0:
        raise ValueError(f"p must be a positive number, but got p={p}")

    if scale_invariant:
        scaling_factor = torch.sum(input * target, -1, keepdim=True) / (
            torch.sum(input**2, -1, keepdim=True) + eps
        )
    else:
        scaling_factor = 1.0

    # input log-power spectrum
    # waveform, pad, window, n_fft, hop_length, win_length, power, normalized
    input = torchaudio.functional.spectrogram(
        input,
        pad=pad,
        win_length=win_length,
        window=window,
        n_fft=n_fft,
        hop_length=hop_length,
        power=2,  # power!
        normalized="window",  # keep the level consistent with time-domain level
        **stft_kwargs,
    )
    if db:
        input = 10 * torch.log10(input + eps)
    else:
        input = torch.log(input + eps)

    # target log-power spectrum
    target = torchaudio.functional.spectrogram(
        scaling_factor * target,
        pad=pad,
        win_length=win_length,
        window=window,
        n_fft=n_fft,
        hop_length=hop_length,
        power=2,
        normalized="window",  # keep the level consistent with time-domain level
        **stft_kwargs,
    )
    if db:
        target = 10.0 * torch.log10(target + eps)
    else:
        target = torch.log(target + eps)

    # norm
    denom = (target.shape[-1] * target.shape[-2]) ** (1 / p)  # to get an average value
    lsd = torch.norm(input - target, p=p, dim=(-2, -1)) / denom

    return lsd


class LogSpectralDistance(torch.nn.Module):
    """
    A torch module wrapper for the log-spectral distance.

    See `log_spectral_distance` for more details on the input arguments.
    """
    def __init__(
        self,
        p=2,
        db=True,
        n_fft=400,
        hop_length=160,
        win_length=None,
        window=None,
        pad=0,
        eps=1e-5,
        reduction="mean",
        scale_invariant=False,
        **stft_kwargs,
    ):
        super(LogSpectralDistance, self).__init__()
        self.p = p
        self.eps = eps
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length if win_length is not None else n_fft
        self.pad = pad
        self.stft_kwargs = stft_kwargs
        self.db = db
        self.reduction = reduction
        self.scale_invariant = scale_invariant
        if reduction not in ("mean", "sum", "none"):
            raise ValueError("Only reduction=mean|sum|none are supported")

        if window is None:
            window = torch.hann_window(self.win_length, periodic=True)
        self.register_buffer("window", window)

    def forward(self, input, target):
        dist = log_spectral_distance(
            input,
            target,
            p=self.p,
            db=self.db,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            pad=self.pad,
            window=self.window,
            win_length=self.win_length,
            eps=self.eps,
            scale_invariant=self.scale_invariant,
            **self.stft_kwargs,
        )

        if self.reduction == "mean":
            return dist.mean()
        elif self.reduction == "sum":
            return dist.sum()
        else:
            return dist
