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
The pytorch lightning model for LoRA fine-tuning
of the UNIVERSE networks.

Author: Robin Scheibler (@fakufaku)
"""
import logging
import math
from pathlib import Path
from typing import Optional

import torch
from hydra.utils import to_absolute_path
from omegaconf import DictConfig
from torch_ema import ExponentialMovingAverage

from ... import lora
from ...inference_utils import load_model
from .. import bigvgan as gan
from ..enhancement import EnhancementBaseModel
from .universe import randn

log = logging.getLogger(__name__)


class UniverseLoRA(EnhancementBaseModel):
    def __init__(
        self,
        model,
        fs,
        losses,
        training,
        validation,
        optimizer,
        scheduler,
        grad_clipper,
        diffusion=None,
        n_steps_backprop=1,
        use_lora=True,
        use_lora_score=True,
        use_lora_condition=True,
        lora_rank=16,
        lora_alpha=None,
        lora_train_biases=True,
        lora_train_names=[],
        use_hifigan_loss=False,
        use_partial_diffusion=False,
        partial_diffusion_random_steps=False,
        weight_hifigan_loss=1.0,
    ):
        if isinstance(model, (str, Path)):
            log.info(f"Loading pre-trained model from {model}")
            model = load_model(to_absolute_path(model))
        model.train()  # loader puts in eval mode

        super().__init__(
            model.fs,
            normalization_norm=model.normalization_norm,
            model=model,
            losses=losses,
            training=training,
            validation=validation,
            optimizer=optimizer,
            scheduler=scheduler,
            grad_clipper=grad_clipper,
            normalization_kwargs=model.normalization_kwargs,
        )

        self.n_steps_backprop = n_steps_backprop

        if n_steps_backprop < 1:
            raise ValueError("n_steps_backprop should be at least 1")

        if fs != self.model.fs:
            raise ValueError("The model fs should be the same as the input fs")

        if diffusion is None:
            self.diff_kwargs = DictConfig({"n_steps": 8, "epsilon": 1.3})
        elif isinstance(diffusion, dict):
            self.diff_kwargs = DictConfig(diffusion)
        else:
            self.diff_kwargs = diffusion

        # check if we want to use the hifi-gan loss
        self.use_hifigan_loss = use_hifigan_loss
        self.use_partial_diffusion = use_partial_diffusion
        self.partial_diffusion_random_steps = partial_diffusion_random_steps
        self.weight_hifigan_loss = weight_hifigan_loss

        # toggle the ema weights and remove unused modules in the original model
        self._fix_model(keep_gan_loss=self.use_hifigan_loss)

        # make it a lora module
        self.use_lora = use_lora
        self.lora_rank = lora_rank
        self.lora_alpha = lora_alpha
        if use_lora:
            if use_lora_score:
                if isinstance(self.model.score_model, torch.nn.Module):
                    lora.inject(self.model.score_model, self.lora_rank, self.lora_alpha)
                else:
                    lora.inject(self.model._edm_model, self.lora_rank, self.lora_alpha)
            if use_lora_condition:
                lora.inject(self.model.condition_model, self.lora_rank, self.lora_alpha)

            lora.freeze_parameters_except_lora_and_bias(
                self.model, train_biases=lora_train_biases, train_names=lora_train_names
            )

        # for moving average of weights
        # we exclude the loss parameters
        self.ema_decay = getattr(self.train_kwargs, "ema_decay", 0.0)
        log.info(f"Use EMA with decay {self.ema_decay}")
        if self.ema_decay > 0.0:
            self.ema = ExponentialMovingAverage(
                self.trainable_parameters(), decay=self.ema_decay
            )
            self._error_loading_ema = False
        else:
            self.ema = None
            self._error_loading_ema = False

    def trainable_parameters(self):
        for p in self.parameters():
            if p.requires_grad:
                yield p

    def _fix_model(self, keep_gan_loss=False):
        # toggle the ema weights
        if self.model.ema is not None:
            self.model.ema.copy_to(self.model.model_parameters())
            self.model.ema = None  # remove ema after that

        # we will only keep the score_model and condition_model
        # all the other loss related weights are dropped
        keep_modules = ["score_model", "condition_model", "_edm_model"]
        if keep_gan_loss:
            keep_modules += ["loss_mpd", "loss_mrd"]

        for name, module in list(self.model.named_children()):
            if name not in keep_modules:
                delattr(self.model, name)

        # remove weight norm
        self.model.remove_weight_norm()

    def hifi_gan_loss(self, y_est, target_original):
        # MPD loss
        y_df_hat_r, y_df_hat_g, fmap_f_r, fmap_f_g = self.model.loss_mpd(
            target_original, y_est
        )
        loss_fm_f = gan.feature_loss(fmap_f_r, fmap_f_g)
        loss_gen_f, losses_gen_f = gan.generator_loss(y_df_hat_g)

        # MRD loss
        y_ds_hat_r, y_ds_hat_g, fmap_s_r, fmap_s_g = self.model.loss_mrd(
            target_original, y_est
        )
        loss_fm_s = gan.feature_loss(fmap_s_r, fmap_s_g)
        loss_gen_s, losses_gen_s = gan.generator_loss(y_ds_hat_g)

        return loss_gen_f, loss_gen_s, loss_fm_f, loss_fm_s

    def compute_losses(self, enh, target, train=True):
        loss, losses = super().compute_losses(enh, target, train)

        if self.use_hifigan_loss:
            loss_gen_f, loss_gen_s, loss_fm_f, loss_fm_s = self.hifi_gan_loss(
                enh, target
            )
            w = self.weight_hifigan_loss
            loss += w * (loss_gen_f + loss_gen_s + loss_fm_f + loss_fm_s)
            losses["mpd"] = loss_gen_f
            losses["mrd"] = loss_gen_s
            losses["mpd_fm"] = loss_fm_f
            losses["mrd_fm"] = loss_fm_s

        return loss, losses

    def on_validation_epoch_start(self):
        super().on_validation_epoch_start()
        self.rng = torch.Generator(device=self.device)
        self.rng.manual_seed(682479040)

    def on_validation_epoch_end(self):
        self.rng = None

    def enhance(
        self,
        mix,
        n_steps: Optional[int] = None,
        epsilon: Optional[float] = None,
        target: Optional[torch.Tensor] = None,
        fake_score_snr: Optional[float] = None,
        rng: Optional[torch.Generator] = None,
        use_aux_signal: Optional[bool] = False,
        keep_rms: Optional[bool] = False,
        ensemble: Optional[int] = None,
        ensemble_stat: Optional[str] = "median",
        warm_start: Optional[int] = None,
    ) -> torch.Tensor:
        if rng is None and getattr(self, "rng", None) is not None:
            rng = self.rng
        return self.model.enhance(
            mix,
            n_steps,
            epsilon,
            target,
            fake_score_snr,
            rng,
            use_aux_signal,
            keep_rms,
            ensemble,
            ensemble_stat,
            warm_start,
        )

    def partial_diffusion(self, mix, rng=None):
        """
        this loss runs a few steps of diffusion and then backpropagagtes the
        model
        """
        n_steps = self.diff_kwargs.n_steps
        epsilon = self.diff_kwargs.epsilon

        if self.partial_diffusion_random_steps:
            n_steps = torch.randint(
                low=2, high=n_steps + 1, size=(1,), generator=rng
            ).item()

        # choose different final t for all batch samples
        t_final = mix.new_zeros(mix.shape[0]).uniform_(0, 1)
        delta_t = (1.0 - t_final) / (n_steps - 1)

        (mix, _), *denorm_args = self.model.normalize_batch((mix, None))
        mix_wav = mix
        mix = self.model.transform(mix)

        # compute parameters
        sigma_ratio = (
            self.model.diff_kwargs.sigma_max / self.model.diff_kwargs.sigma_min
        )
        gamma = sigma_ratio**-delta_t
        eta = 1 - gamma**epsilon
        beta = torch.sqrt(1 - gamma ** (2 * (epsilon - 1.0)))  # in terms of gamma only

        # discretize time
        time = mix.new_ones(mix.shape[0])  # diffusion time
        sigma = self.model.get_std_dev(time)

        # conditioning
        cond, aux_signal, aux_latent = self.model.condition_model(
            mix, x_wav=mix_wav, train=True
        )

        # initial value
        x = randn(mix, sigma, rng=rng)

        # diffusion steps
        for n in range(0, n_steps - 1):
            with torch.set_grad_enabled(n >= n_steps - self.n_steps_backprop):
                # the score now
                score = self.model.score_model(x, sigma, cond)

                # take one step
                time = time - delta_t
                sigma_next = self.model.get_std_dev(time)  # next time's sigma
                z = randn(x, sigma_next, rng=rng)
                x = (
                    x
                    + sigma[..., None, None] ** 2 * eta[..., None, None] * score
                    + beta[..., None, None] * z
                )

                sigma = sigma_next

        # last step always uses backprop
        with torch.set_grad_enabled(self.n_steps_backprop > 0):
            score = self.model.score_model(x, sigma, cond)
            x = x + sigma[:, None, None] ** 2 * score

        # inverse transform
        x = self.model.transform(x, inv=True)

        return x

    def forward(
        self,
        mix,
        n_steps: Optional[int] = None,
        epsilon: Optional[float] = None,
        rng: Optional[torch.Generator] = None,
        keep_rms: Optional[bool] = False,
    ) -> torch.Tensor:
        if epsilon is None:
            epsilon = self.diff_kwargs.epsilon

        if n_steps is None:
            n_steps = self.diff_kwargs.n_steps

        x_ndim = mix.ndim
        if x_ndim == 1:
            mix = mix[None, None, :]
        elif x_ndim == 2:
            mix = mix[:, None, :]
        elif x_ndim > 3:
            raise ValueError("The input should have at most 3 dimensions")

        if self.use_partial_diffusion:
            x = self.partial_diffusion(mix, rng=rng)

        else:
            mix_rms = mix.square().mean(dim=(-2, -1), keepdim=True).sqrt()

            # pad to multiple of total downsampling to remove border effects
            mix_len = mix.shape[-1]
            mix, pad = self.model.pad(mix)

            (mix, _), *denorm_args = self.model.normalize_batch((mix, None))
            mix_wav = mix
            mix = self.model.transform(mix)

            # compute parameters
            delta_t = 1.0 / (n_steps - 1)
            sigma_ratio = (
                self.model.diff_kwargs.sigma_max / self.model.diff_kwargs.sigma_min
            )
            gamma = sigma_ratio**-delta_t
            eta = 1 - gamma**epsilon
            # beta = math.sqrt(1 - ((1 - eta) / gamma) ** 2)  # paper original
            beta = math.sqrt(
                1 - gamma ** (2 * (epsilon - 1.0))
            )  # in terms of gamma only

            # discretize time
            time = torch.linspace(0, 1, n_steps).type_as(mix)
            time = time.flip(dims=[0])
            sigma = self.model.get_std_dev(time)
            sigma = torch.broadcast_to(sigma[None, :], (mix.shape[0], sigma.shape[0]))

            # conditioning
            cond, aux_signal, aux_latent = self.model.condition_model(
                mix, x_wav=mix_wav, train=True
            )

            # initial value
            x = randn(mix, sigma[:, 0], rng=rng)

            # diffusion steps
            for n in range(0, n_steps - 1):
                with torch.set_grad_enabled(n >= n_steps - self.n_steps_backprop):
                    s_now = sigma[:, n]
                    s_next = sigma[:, n + 1]
                    score = self.model.score_model(x, s_now, cond)
                    z = randn(x, s_next, rng=rng)
                    x = x + s_now[..., None, None] ** 2 * eta * score + beta * z

            # last step always uses backprop
            with torch.set_grad_enabled(self.n_steps_backprop > 0):
                score = self.model.score_model(x, sigma[:, -1], cond)
                x = x + sigma[:, -1, None, None] ** 2 * score

            # inverse transform
            x = self.model.transform(x, inv=True)

            # remove the padding and restore signal scale
            x = self.model.unpad(x, pad)
            x = torch.nn.functional.pad(x, (0, mix_len - x.shape[-1]))

            if keep_rms:
                x_rms = (
                    x.square().mean(dim=(-2, -1), keepdim=True).sqrt().clamp(min=1e-5)
                )
                x = x * (mix_rms / x_rms)

            scale = abs(x).max(dim=-1, keepdim=True).values
            x = torch.where(scale > 1.0, x / scale, x)

        if x_ndim == 1:
            x = x[0, 0]
        elif x_ndim == 2:
            x = x[:, 0, :]

        return x

    def optimizer_step(self, *args, **kwargs):
        # Method overridden so that the EMA params are updated after each optimizer step
        super().optimizer_step(*args, **kwargs)
        if self.ema is not None:
            self.ema.update(self.trainable_parameters())

    def on_load_checkpoint(self, checkpoint):
        ema = checkpoint.get("ema", None)
        if self.ema is not None:
            if ema is not None:
                self.ema.load_state_dict(checkpoint["ema"])
            else:
                self._error_loading_ema = True
                log.warn("EMA state_dict not found in checkpoint!")

    def train(self, mode=True, no_ema=False):
        res = super().train(
            mode
        )  # call the standard `train` method with the given mode

        if self.ema is None:
            return res

        if not self._error_loading_ema:
            if mode is False and not no_ema:
                # eval
                self.ema.store(
                    self.trainable_parameters()
                )  # store current params in EMA
                self.ema.copy_to(
                    self.trainable_parameters()
                )  # copy EMA parameters over current params for evaluation
            else:
                # train
                if self.ema.collected_params is not None:
                    self.ema.restore(
                        self.trainable_parameters()
                    )  # restore the EMA weights (if stored)
        return res

    def eval(self, no_ema=False):
        return self.train(False, no_ema=no_ema)

    def on_save_checkpoint(self, checkpoint):
        if self.ema is not None:
            checkpoint["ema"] = self.ema.state_dict()

    def to(self, *args, **kwargs):
        """Override PyTorch .to() to also transfer the EMA of the model weights"""
        if self.ema is not None:
            self.ema.to(*args, **kwargs)
        return super().to(*args, **kwargs)


if __name__ == "__main__":
    from omegaconf import OmegaConf

    config = OmegaConf.load("config/model/universe_lora.yaml")
    config.pop("_target_")
    config.training.audio_len = 2.0
    model = UniverseLoRA(use_partial_diffusion=True, **config)

    # count paramters and trainable parameters
    n_trainable = 0
    n_non_trainable = 0
    for p in model.model.parameters():
        if p.requires_grad:
            n_trainable += p.numel()
        else:
            n_non_trainable += p.numel()

    print(f"Model has {n_trainable + n_non_trainable} parameters")
    print(f"- {n_trainable} trainable parameters")
    print(f"- {n_non_trainable} non-trainable parameters")

    breakpoint()

    x = torch.randn(2, 1, model.fs * 2)
    y = model(x)

    breakpoint()
