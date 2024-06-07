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
The mixture density network module.

Author: Robin Scheibler (@fakufaku)
"""
import math

import torch

from .blocks import PReLU_Conv


class ELU2(torch.nn.Module):
    def __init__(self, alpha=1.0, eps=1e-15):
        super().__init__()
        self.b = 1.0 + eps
        self.elu = torch.nn.ELU(alpha=alpha)

    def forward(self, x):
        return self.elu(x) + self.b


class MixtureDensityNetworkLoss(torch.nn.Module):
    def __init__(
        self,
        est_channels,
        tgt_channels,
        n_comp=3,
        eps=1e-5,
        sampling_rate=24000,
        sample_len_s=3.0,
        sigma_eps=1e-5,
        alpha_per_sample=False,
        reduction="mean",
    ):
        super().__init__()

        if reduction not in ["mean", "sum", "none"]:
            raise ValueError("reduction should be one of mean|sum|none")
        self.reduction = reduction
        self.alpha_per_sample = alpha_per_sample

        self.n_comp = n_comp
        self.tgt_channels = tgt_channels
        self.eps = eps
        self.layer_norm = torch.nn.LayerNorm(
            (est_channels, int(sampling_rate * sample_len_s))
        )
        self.conv = PReLU_Conv(
            est_channels, 3 * n_comp * tgt_channels, kernel_size=3, padding="same"
        )
        self.sigma_non_lin = ELU2(eps=sigma_eps)

    def _split(self, x):
        x = x.view((-1, 3, self.n_comp, self.tgt_channels, x.shape[-1]))

        mean = x[:, 0, ...]

        # the non-negative standard deviation
        sigma = x[:, 1, ...]
        sigma = self.sigma_non_lin(sigma)

        # make logits of non-negative weights summing to one
        alpha = x[:, 2, ...]  # (batch, comp, chan, time)
        if self.alpha_per_sample:
            # different alpha at each time step
            alpha = alpha.mean(dim=-2, keepdim=True)  # (batch, comp, 1, 1)
        else:
            # same alpha for all time steps
            alpha = alpha.mean(dim=(-2, -1), keepdim=True)  # (batch, comp, 1, 1)
        alpha = alpha.clamp(min=-10.0)  # limits minimum exp(alpha) ~ 1e-5
        # alpha = alpha.softmax(dim=1)  # replaced softmax by using log_softmax in _nll

        return mean, sigma, alpha

    @staticmethod
    @torch.jit.script
    def _nll(tgt, mean, sigma, alpha):
        """negative log-likelihood of mixture density"""
        tgt = tgt.unsqueeze(1)

        sqrt_2_pi = math.sqrt(2.0 * math.pi)
        log_p = -0.5 * ((tgt - mean) / sigma).square()
        log_p = log_p - torch.log(sqrt_2_pi * sigma)
        log_p = log_p + torch.log_softmax(alpha, dim=1)
        log_p = log_p.sum(dim=(-2, -1))  # (batch, n_comp)

        # normalize by the input shape
        nll = -torch.logsumexp(log_p, -1) / (tgt.shape[-1] * tgt.shape[-2])

        return nll

    def forward(self, est, tgt):
        if tgt.ndim > 3:
            tgt = tgt.flatten(start_dim=1, end_dim=-2)
        # normalization on channel dimension
        # we don't normalize over time since we don't know the
        # length in advance
        est = self.layer_norm(est)

        est = self.conv(est)

        mean, sigma, alpha = self._split(est)

        nll = self._nll(tgt, mean, sigma, alpha)

        if self.reduction == "mean":
            nll = nll.mean()
        elif self.reduction == "sum":
            nll = nll.sum()

        return nll

    @staticmethod
    def sample(logit, mean, std, random=True):
        """Sample from the mixture of gaussians

        logit: (B, n_components, T)
        mean: (B, n_components, out_dim, T)
        std: (B, n_components, out_dim, T)

        Return: (B, out_dim, T)
        """
        batch_size, n_components, out_dim = mean.shape[:3]
        if random:
            rng = torch.Generator(mean.device)
            prob = torch.softmax(logit, dim=1)
            prob = prob.transpose(1, 2).reshape(-1, n_components)  # (B*T, n_components)
            i_cmpnts = prob.multinomial(1, generator=rng).view(batch_size, -1)  # (B, T)
            i_cmpnts = i_cmpnts[:, None, None, :]  # (B, 1, 1, T)
            i_cmpnts = i_cmpnts.expand(-1, -1, out_dim, -1)  # (B, 1, out_dim, T)
            z = mean.new_empty(i_cmpnts.shape).normal_(
                generator=rng
            )  # (B, 1, out_dim, T)
            out = z * std.gather(1, i_cmpnts) + mean.gather(
                1, i_cmpnts
            )  # (B, 1, out_dim, T)
            out = out.squeeze(1)  # (B, out_dim, T)
        else:
            i_cmpnts = torch.argmax(logit, dim=1, keepdim=True)  # (B, 1, T)
            i_cmpnts = i_cmpnts.unsqueeze(2)  # (B, 1, 1, T)
            i_cmpnts = i_cmpnts.expand(-1, -1, out_dim, -1)  # (B, 1, out_dim, T)
            out = mean.gather(1, i_cmpnts).squeeze(1)  # (B, out_dim, T)
        return out
