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
The UNIVERSE score module

Author: Robin Scheibler (@fakufaku)
"""
import torch
from hydra.utils import instantiate

from .blocks import ConvBlock, PReLU_Conv, cond_weight_norm
from .sigma_block import SigmaBlock, SimpleTimeEmbedding


class ScoreEncoder(torch.nn.Module):
    def __init__(
        self,
        ds_factors,
        input_channels,
        noise_cond_dim,
        with_gru_conv_sandwich=False,
        with_extra_conv_block=False,
        act_type="prelu",
        use_weight_norm=False,
        seq_model="gru",
        use_antialiasing=False,
    ):
        super().__init__()

        c = input_channels
        self.extra_conv_block = with_extra_conv_block

        self.ds_modules = torch.nn.ModuleList(
            [
                ConvBlock(
                    c * 2**i,
                    r,
                    "down",
                    act_type=act_type,
                    use_weight_norm=use_weight_norm,
                    antialiasing=use_antialiasing,
                )
                for i, r in enumerate(ds_factors)
            ]
        )

        self.cond_proj = torch.nn.ModuleList(
            [
                cond_weight_norm(
                    torch.nn.Linear(noise_cond_dim, c * 2 ** (i + 1)),
                    use=use_weight_norm,
                )
                for i in range(len(ds_factors))
            ]
        )

        oc = input_channels * 2 ** len(ds_factors)  # num. channels bottleneck

        if self.extra_conv_block:
            self.ds_modules.append(
                ConvBlock(oc, act_type=act_type, use_weight_norm=use_weight_norm)
            )
            self.cond_proj.append(
                cond_weight_norm(
                    torch.nn.Linear(noise_cond_dim, 2 * oc),
                    use=use_weight_norm,
                )
            )

        self.seq_model = seq_model
        if seq_model == "gru":
            self.gru = torch.nn.GRU(
                oc,  # number of channels after downsampling
                oc // 2,  # bi-directional double # of output channels
                num_layers=1,
                bidirectional=True,
                batch_first=True,
            )

            self.gru_conv_sandwich = with_gru_conv_sandwich
            if self.gru_conv_sandwich:
                self.conv_block1 = ConvBlock(
                    oc, act_type=act_type, use_weight_norm=use_weight_norm
                )
                self.conv_block2 = ConvBlock(
                    oc, act_type=act_type, use_weight_norm=use_weight_norm
                )
        elif seq_model == "none":
            pass
        else:
            raise ValueError("Values for 'seq_model' can be gru|attention|none")

    def forward(self, x, noise_cond):
        residuals = []
        lengths = []
        for idx, (ds, lin) in enumerate(zip(self.ds_modules, self.cond_proj)):
            nc = lin(noise_cond)
            lengths.append(x.shape[-1])
            x, res, _ = ds(x, noise_cond=nc)
            residuals.append(res)

        if self.seq_model == "gru":
            if self.gru_conv_sandwich:
                x, *_ = self.conv_block1(x)
            x, _ = self.gru(x.transpose(-2, -1))
            x = x.transpose(-2, -1)
            if self.gru_conv_sandwich:
                x, *_ = self.conv_block2(x)
        elif self.seq_model == "attention":
            x = self.att(x)
        elif self.seq_model == "none":
            pass

        # return the residuals in reverse order to make it easy to use them in
        # the decoder
        return x, residuals[::-1], lengths[::-1]


class ScoreDecoder(torch.nn.Module):
    def __init__(
        self,
        up_factors,
        input_channels,
        noise_cond_dim,
        with_extra_conv_block=False,
        act_type="prelu",
        use_weight_norm=False,
        use_antialiasing=False,
    ):
        super().__init__()
        self.extra_conv_block = with_extra_conv_block

        n_channels = [
            input_channels * 2 ** (len(up_factors) - i - 1)
            for i in range(len(up_factors))
        ]

        self.up_modules = torch.nn.ModuleList()
        self.noise_cond_proj = torch.nn.ModuleList()
        self.signal_cond_proj = torch.nn.ModuleList()

        if self.extra_conv_block:
            # adds extra input block with constant channels
            oc = input_channels * 2 ** len(up_factors)
            self.up_modules.append(
                ConvBlock(oc, act_type=act_type, use_weight_norm=use_weight_norm)
            )
            self.noise_cond_proj.append(
                cond_weight_norm(
                    torch.nn.Linear(noise_cond_dim, 2 * oc),
                    use=use_weight_norm,
                )
            )
            self.signal_cond_proj.append(
                cond_weight_norm(
                    torch.nn.Conv1d(oc, oc, kernel_size=1),
                    use=use_weight_norm,
                )
            )

        for c, r in zip(n_channels, up_factors):
            self.up_modules.append(
                ConvBlock(
                    c,
                    r,
                    "up",
                    act_type=act_type,
                    use_weight_norm=use_weight_norm,
                    antialiasing=use_antialiasing,
                )
            )
            self.noise_cond_proj.append(
                cond_weight_norm(
                    torch.nn.Linear(noise_cond_dim, 2 * c),
                    use=use_weight_norm,
                )
            )
            self.signal_cond_proj.append(
                cond_weight_norm(
                    torch.nn.Conv1d(c, c, kernel_size=1),
                    use=use_weight_norm,
                )
            )

    def forward(self, x, noise_cond, input_cond, residuals, lengths):
        for lvl, (up, n_lin, s_lin, cond, res, length) in enumerate(
            zip(
                self.up_modules,
                self.noise_cond_proj,
                self.signal_cond_proj,
                input_cond,
                residuals,
                lengths,
            )
        ):
            nc = n_lin(noise_cond)
            sc = s_lin(cond)
            x, *_ = up(x, noise_cond=nc, input_cond=sc, res=res, length=length)
        return x


class ScoreNetwork(torch.nn.Module):
    def __init__(
        self,
        fb_kernel_size=3,
        rate_factors=[2, 4, 4, 5],
        n_channels=32,
        n_rff=32,
        noise_cond_dim=512,
        encoder_gru_conv_sandwich=False,
        extra_conv_block=False,
        encoder_act_type="prelu",
        decoder_act_type="prelu",
        precoding=None,
        input_channels=1,
        output_channels=1,
        use_weight_norm=False,
        seq_model="gru",
        use_antialiasing=False,
        time_embedding=None,
    ):
        super().__init__()

        if time_embedding == "simple":
            self.sigma_block = SimpleTimeEmbedding(n_dim=noise_cond_dim)
        else:
            self.sigma_block = SigmaBlock(n_rff, noise_cond_dim)

        self.input_channels = input_channels
        self.output_channels = output_channels

        self.input_conv = torch.nn.Conv1d(
            input_channels, n_channels, kernel_size=fb_kernel_size, padding="same"
        )
        self.encoder = ScoreEncoder(
            ds_factors=rate_factors,
            input_channels=n_channels,
            noise_cond_dim=noise_cond_dim,
            with_gru_conv_sandwich=encoder_gru_conv_sandwich,
            with_extra_conv_block=extra_conv_block,
            act_type=encoder_act_type,
            use_weight_norm=use_weight_norm,
            seq_model=seq_model,
            use_antialiasing=use_antialiasing,
        )
        self.decoder = ScoreDecoder(
            up_factors=rate_factors[::-1],
            input_channels=n_channels,
            noise_cond_dim=noise_cond_dim,
            with_extra_conv_block=extra_conv_block,
            act_type=decoder_act_type,
            use_weight_norm=use_weight_norm,
            use_antialiasing=use_antialiasing,
        )
        self.prelu = torch.nn.PReLU()
        self.output_conv = PReLU_Conv(
            n_channels,
            output_channels,
            kernel_size=fb_kernel_size,
            padding="same",
            use_weight_norm=use_weight_norm,
        )

        self.precoding = instantiate(precoding, _recursive_=True) if precoding else None

    def forward(self, x, sigma, cond):
        n_samples = x.shape[-1]

        if self.precoding:
            x = self.precoding(x)

        g = self.sigma_block(torch.log10(sigma))
        x = self.input_conv(x)
        h, residuals, lengths = self.encoder(x, noise_cond=g)
        s = self.decoder(
            h, noise_cond=g, input_cond=cond, residuals=residuals, lengths=lengths
        )
        s = self.output_conv(self.prelu(s))

        if self.precoding and hasattr(self.precoding, "inv"):
            s = self.precoding(s, inv=True)

        # adjust length and dimensions
        s = torch.nn.functional.pad(s, (0, n_samples - s.shape[-1]))

        return s
