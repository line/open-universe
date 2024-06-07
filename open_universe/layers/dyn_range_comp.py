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
# Transforms for Signal Pre-conditioning and Dynamic Range Compression

We standardize the architecture so that

input.ndim == 3, with dimensions (batch, channels, time)
output.ndim == 3, with dimensions (batch, channels, time)

Author: Robin Scheibler (@fakufaku)
"""
import torch
from omegaconf import OmegaConf


class IdentityTransform:
    """
    A dummy placeholders that does nothing.
    """

    def __call__(self, x, inv=None):
        return x

    def inv(self, x):
        return self(x)


def get_window(window_type, window_length):
    if window_type == "sqrthann":
        return torch.sqrt(torch.hann_window(window_length, periodic=True))
    elif window_type == "hann":
        return torch.hann_window(window_length, periodic=True)
    elif window_type == "hamming":
        return torch.hamming_window(window_length, periodic=True)
    else:
        raise NotImplementedError(f"Window type {window_type} not implemented!")


class CompressedMagSTFT(torch.nn.Module):
    def __init__(self, stft_kwargs, spec_kwargs, inv=False):
        super().__init__()
        # check input arguments
        assert all([k in stft_kwargs for k in ["n_fft", "hop_length", "window_name"]])
        assert all(
            [k in spec_kwargs for k in ["transform_type", "abs_exponent", "factor"]]
        )

        self._inv = inv

        self.stft_kwargs = OmegaConf.create(
            stft_kwargs
        )  # n_fft, hop_length, window_name
        self.spec_kwargs = OmegaConf.create(spec_kwargs)
        window = get_window(
            self.stft_kwargs.pop("window_name", "hann"), self.stft_kwargs.n_fft
        )
        self.register_buffer("stft_window", window)

    def forward(self, x, inv=False, length=None):
        if self._inv:
            inv = not inv

        if not inv:
            if x.shape[1] != 1:
                raise ValueError("Expects single channel input")
            if x.ndim != 3:
                raise ValueError("Expects a 3D input tensor (batch, channels, time)")

            x = x.squeeze(1)

            # forward transform
            x = self._stft(x)
            x = self._forward_transform(x)

            # make
            x = torch.view_as_real(x)  # (batch, freq, time, real/imag)
            # store real/imag in parallel channels
            x = x.moveaxis(3, 1)  # (batch, real/image, freq, time)
            x = x.flatten(start_dim=1, end_dim=2)

            return x  # (batch, real-imag-freq, time)

        else:
            if x.ndim != 3:
                raise ValueError(
                    "Expects a 3D input tensor (batch, freq/real/imag, time)"
                )
            # restore shape
            n_freq = x.shape[1] // 2
            x = x.reshape((x.shape[0], 2, n_freq, x.shape[2]))
            x = torch.view_as_complex(x.moveaxis(1, 3).contiguous())

            # inverse transform
            x = self._istft(self._backward_transform(x), length=length)

            # restore channel dim
            x = x.unsqueeze(1)
            return x

    def inv(self, x, length=None):
        return self(x, inv=True, length=length)

    def _forward_transform(self, spec):
        # forward spec
        if self.spec_kwargs.transform_type == "exponent":
            if self.spec_kwargs.abs_exponent != 1:
                # only do this calculation if spec_exponent != 1, otherwise it's quite
                # a bit of wasted computation and introduced numerical error
                e = self.spec_kwargs.abs_exponent
                mag = spec.abs()
                spec = (1e-7 + mag) ** (e - 1.0) * spec  # avoids the sign operation
            spec = spec * self.spec_kwargs.factor
        elif self.spec_kwargs.transform_type == "log":
            spec = torch.log(1 + spec.abs()) * torch.sgn(spec)
            spec = spec * self.spec_kwargs.factor
        elif self.spec_kwargs.transform_type == "none":
            spec = spec
        return spec

    def _backward_transform(self, spec):
        if self.spec_kwargs.transform_type == "exponent":
            spec = spec / self.spec_kwargs.factor
            if self.spec_kwargs.abs_exponent != 1:
                e = self.spec_kwargs.abs_exponent
                mag = spec.abs()
                spec = (1e-7 + mag) ** (1.0 / e - 1.0) * spec
        elif self.spec_kwargs.transform_type == "log":
            spec = spec / self.spec_kwargs.factor
            spec = (torch.exp(spec.abs()) - 1) * torch.sgn(spec)
        elif self.spec_kwargs.transform_type == "none":
            spec = spec
        return spec

    def _stft(self, sig):
        return torch.stft(
            sig,
            **{
                **self.stft_kwargs,
                "center": True,
                "return_complex": True,
                "window": self.stft_window,
                "pad_mode": "constant",
            },
        )

    def _istft(self, spec, length=None):
        return torch.istft(
            spec,
            **{
                **self.stft_kwargs,
                "center": True,
                "window": self.stft_window,
                "length": length,
            },
        )


class CompressedMagSTFTPadded(CompressedMagSTFT):
    def __init__(self, stft_kwargs, spec_kwargs, pad_block=None, inv=False):
        super().__init__(stft_kwargs, spec_kwargs, inv=inv)
        if pad_block is not None:
            if pad_block % self.stft_kwargs.hop_length != 0:
                raise ValueError("pad_block must be a multiple of hop_length")
            self.pad_block = pad_block
        else:
            self.pad_block = 0

    def _pad(self, x):
        # we are forced to use center=True, because custom padding causes an error
        # due to the NOLA condition in the torch stft. very frustrating.
        #
        # the stft with center=True produces one extra frame at the end
        # so we want to pad the signal to be M x pad_block + (pad_block - hop_length)
        #
        # to achieve that, we first pad to M x pad_block, then drop the last hop_length
        # samples

        if self.pad_block > 0:
            r = (x.shape[-1]) % self.pad_block
            if r > 0:
                pad = self.pad_block - r
                x = torch.nn.functional.pad(x, (0, pad), mode="constant", value=0.0)

        # now drop last frame
        x = x[..., : -self.stft_kwargs.hop_length]
        return x

    def _stft(self, sig):
        sig = self._pad(sig)
        return torch.stft(
            self._pad(sig),
            **{
                **self.stft_kwargs,
                "center": True,
                "return_complex": True,
                "window": self.stft_window,
                "pad_mode": "constant",
            },
        )

    def _istft(self, spec, length=None):
        if length is None:
            length = spec.shape[-1] * self.stft_kwargs.hop_length
        x = torch.istft(
            spec,
            **{
                **self.stft_kwargs,
                "center": True,
                "window": self.stft_window,
                "length": length,
            },
        )
        return x
