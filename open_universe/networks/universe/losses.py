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
A wrapper to use the multiresolution loss with UNIVERSE

Author: Robin Scheibler (@fakufaku)
"""
import torch

from ...losses.multires_stft import MultiResL1SpecLoss


class UniverseMultiResL1SpecLoss(MultiResL1SpecLoss):
    def __init__(
        self,
        window_sz=[512],
        hop_sz=None,
        eps=1e-8,
        time_domain_weight=0.5,
        scale_invariant=False,
    ):
        super().__init__(
            window_sz=window_sz,
            hop_sz=hop_sz,
            eps=eps,
            time_domain_weight=time_domain_weight,
            scale_invariant=scale_invariant,
        )

    def forward(self, y_est, h_est, target, mix, y_est_trans, tgt_trans, mix_trans):
        M = min(y_est.shape[-1], target.shape[-1])
        target = torch.nn.functional.pad(target, (0, M - target.shape[-1]))
        y_est = torch.nn.functional.pad(y_est, (0, M - y_est.shape[-1]))
        l1_multires = super().forward(y_est, target)
        return l1_multires, {"l1_multires": l1_multires}
