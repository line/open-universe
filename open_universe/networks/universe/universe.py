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
The UNIVERSE(++) model with MDN loss

Author: Robin Scheibler (@fakufaku)
"""
import itertools
import logging
import math
from typing import Optional

import pytorch_lightning as pl
import torch
import torchaudio
from hydra.utils import instantiate
from omegaconf import OmegaConf
from torch_ema import ExponentialMovingAverage

from ... import utils
from ...layers.dyn_range_comp import IdentityTransform
from .blocks import remove_weight_norm
from .mdn import MixtureDensityNetworkLoss

log = logging.getLogger(__name__)


def randn(x, sigma, rng=None):
    noise = torch.randn(x.shape, dtype=x.dtype, device=x.device, generator=rng)
    return noise * sigma[:, None, None]


class Universe(pl.LightningModule):
    def __init__(
        self,
        fs,
        normalization_norm,
        score_model,
        condition_model,
        diffusion,
        losses,
        training,
        validation,
        optimizer,
        scheduler,
        grad_clipper,
        transform=None,
        normalization_kwargs={},
        with_noise_target=False,
        detach_cond=False,
        edm=None,
    ):
        # init superclass
        super().__init__()
        self.save_hyperparameters()

        self.fs = fs
        self.normalization_norm = normalization_norm
        self.normalization_kwargs = normalization_kwargs
        self.with_noise_target = with_noise_target
        self.detach_cond = detach_cond

        self.opt_kwargs = optimizer
        self.schedule_kwargs = scheduler
        self.grad_clip_kwargs = grad_clipper

        self.diff_kwargs = diffusion
        self.losses_kwargs = losses
        self.val_kwargs = validation
        self.train_kwargs = training

        # optional EDM loss function from paper
        # Elucidating the Design Space of Diffusion-Based Generative Models
        if edm is not None:
            log.info("Use EDM network parameterization")
            # store the parameters
            self.edm_kwargs = edm
            # wrap the original score function
            self._edm_model = instantiate(score_model, _recursive_=False)
            self.score_model = self._edm_score_wrapper
            self.with_edm = True
        else:
            self.score_model = instantiate(score_model, _recursive_=False)
            self.with_edm = False

        self.condition_model = instantiate(condition_model, _recursive_=False)
        self.n_channels = score_model.get("n_channels", 32)
        rate_factors = score_model.get("rate_factors", [2, 4, 4, 5])
        self.n_stages = len(rate_factors)
        self.latent_n_channels = 2**self.n_stages * self.n_channels
        self.tot_ds = math.prod(rate_factors)

        self.init_losses(score_model, condition_model, losses, training)

        self.enh_losses = torch.nn.ModuleDict()
        for name, loss_args in self.val_kwargs.enh_losses.items():
            self.enh_losses[name] = instantiate(loss_args)

        self.denormalize_batch = utils.denormalize_batch

        if transform is None:
            self.transform = IdentityTransform()
        else:
            self.transform = instantiate(transform, _recursive_=False)

        # for moving average of weights
        # we exclude the loss parameters
        self.ema_decay = getattr(self.train_kwargs, "ema_decay", 0.0)
        log.info(f"Use EMA with decay {self.ema_decay}")
        if self.ema_decay > 0.0:
            self.ema = ExponentialMovingAverage(
                self.model_parameters(), decay=self.ema_decay
            )
            self._error_loading_ema = False
        else:
            self.ema = None
            self._error_loading_ema = False

    def model_parameters(self):
        return itertools.chain(
            self.get_score_model().parameters(), self.condition_model.parameters()
        )

    def remove_weight_norm(self):
        remove_weight_norm(self)

    def init_losses(self, score_model, condition_model, losses, training):
        alpha_per_sample = losses.get("mdn_alpha_per_sample", False)
        log.info(f"Losses: Mixture density networks with {alpha_per_sample=}")
        """separate this init to allow to redefine in derived class"""

        cond_input_channels = getattr(condition_model, "input_channels", 1)
        num_targets = 2 if self.with_noise_target else 1

        if losses.weights.signal > 0.0:
            self.loss_signal = MixtureDensityNetworkLoss(
                est_channels=self.n_channels,
                tgt_channels=cond_input_channels * num_targets,
                n_comp=losses.mdn_n_comp,
                sampling_rate=self.fs // cond_input_channels,
                sample_len_s=training.audio_len,
                alpha_per_sample=alpha_per_sample,
            )
        else:
            self.loss_signal = None
        if losses.weights.latent > 0.0:
            self.loss_latent = MixtureDensityNetworkLoss(
                est_channels=self.latent_n_channels,
                tgt_channels=condition_model.n_mels * num_targets,
                n_comp=losses.mdn_n_comp,
                sampling_rate=self.fs // (cond_input_channels * self.tot_ds),
                sample_len_s=training.audio_len,
                alpha_per_sample=alpha_per_sample,
            )
        else:
            self.loss_latent = None
        self.loss_score = instantiate(losses.score_loss, _recursive_=False)

    def normalize_batch(self, batch, norm=None):
        if norm is None:
            norm = self.normalization_norm
        return utils.normalize_batch(batch, norm=norm, **self.normalization_kwargs)

    def _get_edm_weights(self, sigma):
        level_db = self.edm_kwargs.get(
            "data_level_db", self.normalization_kwargs.get("level_db", 0.0)
        )
        sigma_data = 10.0 ** (level_db / 20.0)
        sigma_norm = (sigma**2 + sigma_data**2) ** 0.5

        weights = {
            "skip": sigma_data**2 / (sigma**2 + sigma_data**2),
            "in": 1.0 / sigma_norm,
            "out": sigma * sigma_data / sigma_norm,
            "noise": self.edm_kwargs.noise,
        }

        return weights

    def get_score_model(self):
        if self.with_edm:
            return self._edm_model
        else:
            return self.score_model

    def _edm_score_wrapper(self, x, sigma, cond, with_speech_est=False):
        w = self._get_edm_weights(sigma)
        w_in = utils.pad_dim_right(w["in"], x)
        w_out = utils.pad_dim_right(w["out"], x)
        w_skip = utils.pad_dim_right(w["skip"], x)
        net_out = self._edm_model(w_in * x, w["noise"] * sigma, cond)  # speech estimate
        speech_est = w_skip * x + w_out * net_out
        score = (speech_est - x) / utils.pad_dim_right(sigma, x) ** 2

        if with_speech_est:
            return score, speech_est
        else:
            return score

    def print_count(self):
        num_params_score_and_cond = utils.count_parameters(
            self.condition_model
        ) + utils.count_parameters(self.score_model)
        print(f"UNIVERSE number of parameters: {num_params_score_and_cond}")
        self.condition_model.print_count(indent=2)
        self.score_model.print_count(indent=2)

    def pad(self, x, pad=None):
        if pad is None:
            pad = self.tot_ds - x.shape[-1] % self.tot_ds
        x = torch.nn.functional.pad(x, (pad // 2, pad - pad // 2))
        return x, pad

    def unpad(self, x, pad):
        return x[..., pad // 2 : -(pad - pad // 2)]

    def aux_to_wav(self, y_aux):
        return y_aux

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

        mix_rms = mix.square().mean(dim=(-2, -1), keepdim=True).sqrt()

        if ensemble is not None:
            mix_shape = mix.shape
            mix = torch.stack([mix] * ensemble, dim=0)
            mix = mix.view((-1,) + mix_shape[1:])

        # pad to multiple of total downsampling to remove border effects
        mix_len = mix.shape[-1]
        mix, pad = self.pad(mix)
        if target is not None:
            target, _ = self.pad(target, pad=pad)

        (mix, target), *denorm_args = self.normalize_batch((mix, target))
        mix_wav = mix
        mix = self.transform(mix)
        if target is not None:
            self.transform(target)

        # we set this up here to test the diffusion with a "perfect" score model
        if fake_score_snr is None:  # we can test we some degraded score too
            score_snr = 5.0  # db
        else:
            score_snr = fake_score_snr

        def score_wrapper(x, s, cond):
            if target is None:
                score = self.score_model(x, s, cond)
            else:
                true_score = -(x - target) / s[:, None, None] ** 2
                score_rms = (true_score**2).mean().sqrt()
                noise_rms = score_rms * 10 ** (-score_snr / 20.0)
                noise = torch.randn(
                    true_score.shape,
                    dtype=true_score.dtype,
                    device=true_score.device,
                    generator=rng,
                )
                score = true_score + noise * noise_rms
            return score

        # compute parameters
        delta_t = 1.0 / (n_steps - 1)
        gamma = (self.diff_kwargs.sigma_max / self.diff_kwargs.sigma_min) ** -delta_t
        eta = 1 - gamma**epsilon
        # beta = math.sqrt(1 - ((1 - eta) / gamma) ** 2)  # paper original
        beta = math.sqrt(1 - gamma ** (2 * (epsilon - 1.0)))  # in terms of gamma only

        # discretize time
        time = torch.linspace(0, 1, n_steps).type_as(mix)
        time = time.flip(dims=[0])
        sigma = self.get_std_dev(time)
        sigma = torch.broadcast_to(sigma[None, :], (mix.shape[0], sigma.shape[0]))

        # conditioning
        cond, aux_signal, aux_latent = self.condition_model(
            mix, x_wav=mix_wav, train=True
        )
        if use_aux_signal:
            # use the signal conditioner output
            x = self.aux_to_wav(aux_signal)

        else:
            # use diffusion

            # initial value
            if warm_start is None:
                x = randn(mix, sigma[:, 0], rng=rng)
                n_start = 0
            else:
                sig = self.aux_to_wav(aux_signal)
                x = sig + randn(sig, sigma[:, warm_start], rng=rng)
                n_start = warm_start

            # diffusion steps
            for n in range(n_start, n_steps - 1):
                s_now = sigma[:, n]
                s_next = sigma[:, n + 1]
                score = score_wrapper(x, s_now, cond)
                z = randn(x, s_next, rng=rng)
                x = x + s_now[..., None, None] ** 2 * eta * score + beta * z

            # last step
            score = score_wrapper(x, sigma[:, -1], cond)
            x = x + sigma[:, -1, None, None] ** 2 * score

        # inverse transform
        x = self.transform(x, inv=True)

        # remove the padding and restore signal scale
        x = self.unpad(x, pad)
        x = torch.nn.functional.pad(x, (0, mix_len - x.shape[-1]))

        if keep_rms:
            x_rms = x.square().mean(dim=(-2, -1), keepdim=True).sqrt().clamp(min=1e-5)
            x = x * (mix_rms / x_rms)

        scale = abs(x).max(dim=-1, keepdim=True).values
        x = torch.where(scale > 1.0, x / scale, x)

        if ensemble is not None:
            x = x.view((-1,) + mix_shape)
            if ensemble_stat == "mean":
                x = x.mean(dim=0)
            elif ensemble_stat == "median":
                x = x.median(dim=0).values
            elif ensemble_stat == "signal_median":
                x = utils.signal_median(x)
            else:
                raise NotImplementedError()

        if x_ndim == 1:
            x = x[0, 0]
        elif x_ndim == 2:
            x = x[:, 0, :]

        return x

    def forward(self, xt, sigma, cond):
        return self.score_model(xt, sigma, cond)

    def get_std_dev(self, time):
        if self.diff_kwargs.schedule == "geometric":
            s_min = self.diff_kwargs.sigma_min
            s_max = self.diff_kwargs.sigma_max
            return s_min * (s_max / s_min) ** time
        else:
            raise NotImplementedError()

    def on_train_epoch_start(self):
        pass

    def adapt_time_sampling(self, x):
        with torch.no_grad():
            if not hasattr(self, "pr_cum"):
                # default to time uniform in first epoch
                time = x.new_zeros(x.shape[0]).uniform_()
            else:
                pr_cum = torch.broadcast_to(
                    self.pr_cum, (x.shape[0], self.pr_cum.shape[0])
                )
                time = x.new_zeros(x.shape[0])
                dice_roll = x.new_zeros(x.shape[0]).uniform_()
                for i in range(self.val_score_values.shape[0]):
                    ts, te = self.val_score_bins[i], self.val_score_bins[i + 1]
                    cand = x.new_zeros(x.shape[0]).uniform_() * (te - ts) + ts
                    time = torch.where(
                        torch.logical_and(
                            dice_roll >= pr_cum[:, i], dice_roll < pr_cum[:, i + 1]
                        ),
                        cand,
                        time,
                    )
        return time

    def sample_sigma(
        self, x, time_sampling="time_uniform", t_min=0.0, t_max=1.0, rng=None
    ):
        # sample the variance
        u = torch.rand(x.shape[0], generator=rng, dtype=x.dtype, device=x.device)
        time = (t_max - t_min) * u + t_min
        s_min = self.diff_kwargs.sigma_min
        s_max = self.diff_kwargs.sigma_max
        if time_sampling == "time_uniform":
            # geometric variance schedule
            sigma = self.get_std_dev(time)
        elif time_sampling == "sigma_linear":
            sigma = (s_max - s_min) * time + s_min
        elif time_sampling == "diffsym":
            # differential symmetric sampling
            # explanation:
            # 1) sample sigma uniformly
            # 2) apply a tranformation to time that has the same
            #    derivative as the standard deviation progression,
            #    but symmetric with respect to time
            # 3) then, apply the geometric progression
            sigma = (s_max - s_min) * time + s_min
            sigma = s_max + s_min - sigma
            num = torch.log10((s_max + s_min - sigma) / s_min)
            denom = math.log10(s_max / s_min)
            time = 1.0 - num / denom
            sigma = self.get_std_dev(time)
        elif time_sampling == "adaptive":
            time = self.adapt_time_sampling(x)
            sigma = self.get_std_dev(time)
        elif time_sampling == "time_discrete":
            n_steps = self.diff_kwargs.get("n_steps", 32)
            steps = torch.linspace(0.0, 1.0, n_steps).to(x.device)
            idx = abs(steps[:, None] - time[None, :]).min(dim=0).indices
            time = steps[idx]
            sigma = self.get_std_dev(time)
        elif time_sampling.startswith("time_normal"):
            try:
                alpha = torch.tensor(float(time_sampling.split("_")[2]))
            except (IndexError, ValueError):
                alpha = torch.tensor(
                    0.95
                )  # we want to use 100 * alpha % of the distribution

            time = utils.random.center_truncated_normal(
                area=alpha,
                min=t_min,
                max=t_max,
                size=x.shape[0],
                generator=rng,
                device=x.device,
            )
            sigma = self.get_std_dev(time)
        else:
            raise NotImplementedError()

        return sigma, time

    def compute_losses(
        self,
        mix,
        target,
        train=True,
        time_sampling="time_uniform",
        t_min=0.0,
        t_max=1.0,
        rng=None,
    ):
        mix_trans = self.transform(mix)
        tgt_trans = self.transform(target)

        if self.with_noise_target:
            noise = mix - target
            target_aux = torch.cat((target, noise), dim=1)
            target_aux_trans = torch.cat((tgt_trans, self.transform(noise)), dim=1)
        else:
            target_aux = target
            target_aux_trans = tgt_trans

        sigma, _ = self.sample_sigma(mix_trans, time_sampling, t_min, t_max, rng=rng)

        # sample the noise and create the target
        z = target.new_zeros(tgt_trans.shape).normal_(generator=rng)
        x_t = tgt_trans + sigma[:, None, None] * z

        # run computations
        cond, y_est, h_est = self.condition_model(mix_trans, x_wav=mix, train=True)

        if self.detach_cond:
            cond = [c.detach() for c in cond]

        score = self.score_model(x_t, sigma, cond)

        # compute losses
        l_score = self.loss_score(sigma[..., None, None] * score, -z)

        if train:
            if self.losses_kwargs.weights.latent > 0.0 and h_est is not None:
                mel_target = self.condition_model.input_mel.compute_mel_spec(target_aux)
                mel_target = mel_target / torch.linalg.norm(
                    mel_target, dim=(-2, -1), keepdim=True
                ).clamp(min=1e-5)
                l_latent = self.loss_latent(h_est, mel_target)
            else:
                l_latent = l_score.new_zeros(1)

            if self.losses_kwargs.weights.signal > 0.0:
                l_signal = self.loss_signal(y_est, target_aux_trans)
            else:
                l_signal = l_score.new_zeros(1)

            loss = self.losses_kwargs.weights.score * l_score
            if torch.isnan(l_score):
                log.warn("Score loss is nan...")
                breakpoint()

            if not torch.isnan(l_signal):
                loss = loss + self.losses_kwargs.weights.signal * l_signal
            else:
                log.warn("Signal loss is nan, skip for total loss")

            if not torch.isnan(l_latent):
                loss = loss + self.losses_kwargs.weights.latent * l_latent
            else:
                log.warn("Latent loss is nan, skip for total loss")

            return loss, l_score, l_signal, l_latent
        else:
            return l_score

    def training_step(self, batch, batch_idx):
        batch = batch[:2]
        mix, target = batch

        if getattr(self.train_kwargs, "dynamic_mixing", False):
            noise = mix - target
            perm = torch.randperm(noise.shape[0])
            mix = target + noise[perm, ...]

        (mix, target), *stats = self.normalize_batch(
            (mix, target), norm=self.normalization_norm
        )

        loss, l_score, l_signal, l_latent = self.compute_losses(
            mix, target, train=True, time_sampling=self.train_kwargs.time_sampling
        )

        # every 10 steps, we log stuff
        self.log(
            "train/loss",
            loss,
            on_step=True,
            on_epoch=False,
            prog_bar=True,
            logger=True,
            sync_dist=False,
            batch_size=mix.shape[0],
        )
        kwargs = dict(
            batch_size=mix.shape[0],
            on_step=True,
            on_epoch=False,
            prog_bar=False,
            logger=True,
            sync_dist=False,
        )
        self.log("train/score", l_score, **kwargs)
        self.log("train/signal", l_signal, **kwargs)
        self.log("train/latent", l_latent, **kwargs)

        self.do_lr_warmup()

        return loss

    def on_train_epoch_end(self):
        pass

    def on_validation_epoch_start(self):
        self.n_batches_est_done = 0

        # save some samples to tensorboard
        self.n_tb_samples_saved = 0
        self.num_tb_samples = self.val_kwargs.get("num_tb_samples", 0)
        if not hasattr(self, "first_val_done"):
            self.first_val_done = False
        else:
            self.first_val_done = True

        # de-randomize the validation step
        self.rng = torch.Generator(device=self.device)
        self.rng.manual_seed(682479040)

    def validation_step(self, batch, batch_idx, dataset_i=0):
        batch = batch[:2]

        batch_scaled, *stats = self.normalize_batch(batch, norm=self.normalization_norm)
        mix, target = batch_scaled
        batch_size = mix.shape[0]

        tb = torch.linspace(0.0, 1.0, self.val_kwargs.n_bins + 1, device=mix.device)
        bin_scores = []
        for i in range(self.val_kwargs.n_bins):
            ls = self.compute_losses(
                self.pad(mix)[0],
                self.pad(target)[0],
                train=False,
                time_sampling="time_uniform",  # always sample uniformly for validation
                t_min=tb[i],
                t_max=tb[i + 1],
                rng=self.rng,
            )
            bin_scores.append(ls)
        self.val_score_bins = tb
        self.val_score_values = torch.tensor(bin_scores, device=mix.device)
        l_score = torch.mean(self.val_score_values)

        # compute the cumulative distribution
        # manual cumsum to be deterministic
        v = self.val_score_values.clamp(min=5e-4)
        pr_cum = v.new_zeros(v.shape[0] + 1)
        for idx, p in enumerate(v):
            pr_cum[idx + 1] = pr_cum[idx] + p
        pr_cum = pr_cum / pr_cum[-1]
        pr_cum[-1] = 1.0 + 1e-5  # to include the last bound
        self.pr_cum = pr_cum

        self.log(
            "val/score", l_score, on_epoch=True, sync_dist=True, batch_size=batch_size
        )
        for i in range(self.val_kwargs.n_bins):
            self.log(
                f"val/score_{tb[i]:.2f}-{tb[i+1]:.2f}",
                bin_scores[i],
                on_epoch=True,
                sync_dist=True,
                batch_size=batch_size,
            )

        # validation separation losses
        if (
            self.trainer.testing
            or self.n_batches_est_done < self.val_kwargs.max_enh_batches
        ):
            # use unscaled batch since the normalization is in the enhancement part
            mix, target = batch
            self.n_batches_est_done += 1
            est = self.enhance(mix, rng=self.rng)

            # Save some samples to tensorboard (only in the main process)
            if self.n_tb_samples_saved < self.num_tb_samples:
                num_save = min(
                    self.num_tb_samples - self.n_tb_samples_saved, batch_size
                )
                for idx in range(num_save):
                    sample_id = f"{self.global_rank}_{self.n_tb_samples_saved}"

                    mix_ = mix[idx] * 0.95 / torch.max(torch.abs(mix[idx]))
                    mix_loud = torchaudio.functional.loudness(mix[idx], self.fs)

                    if not self.first_val_done:
                        # save mix the first time
                        self.logger.experiment.add_audio(
                            f"mix/{sample_id}",
                            mix_.cpu(),
                            global_step=self.global_step,
                            sample_rate=self.fs,
                        )

                        # save target the first time, adjust to have same loudness
                        tgt_loud = torchaudio.functional.loudness(target[idx], self.fs)
                        tgt_gain = 10 ** ((mix_loud - tgt_loud) / 20)
                        self.logger.experiment.add_audio(
                            f"target/{sample_id}",
                            (target[idx] * tgt_gain).cpu(),
                            global_step=self.global_step,
                            sample_rate=self.fs,
                        )

                    est_loud = torchaudio.functional.loudness(est[idx], self.fs)
                    est_gain = 10 ** ((mix_loud - est_loud) / 20)
                    self.logger.experiment.add_audio(
                        f"enh/{sample_id}",
                        (est[idx] * est_gain).cpu(),
                        global_step=self.global_step,
                        sample_rate=self.fs,
                    )

                    # increment the number of samples saved, stop if we have enough
                    self.n_tb_samples_saved += 1
                    if self.n_tb_samples_saved >= self.num_tb_samples:
                        break

            for name, loss in self.enh_losses.items():
                val_metric = loss(est, target)

                # handle single value case
                if not isinstance(val_metric, dict):
                    val_metric = {"": val_metric}

                for sub_name, loss_val in val_metric.items():
                    self.log(
                        name + sub_name,
                        loss_val.to(mix.device),
                        on_epoch=True,
                        sync_dist=True,
                        batch_size=batch_size,
                    )

    def on_validation_epoch_end(self):
        pass

    def on_test_epoch_start(self):
        self.on_validation_epoch_start()

    def test_step(self, batch, batch_idx, dataset_i=None):
        return self.validation_step(batch, batch_idx, dataset_i=dataset_i)

    def test_epoch_end(self, outputs):
        """
        Called at the end of validation to aggregate outputs.
        :param outputs: list of individual outputs of each validation step.
        """
        self.validation_epoch_end(outputs)

    def configure_optimizers(self):
        opt_kwargs = OmegaConf.to_container(self.opt_kwargs, resolve=True)

        # the warm-up parameter is handled separately
        self.lr_warmup = opt_kwargs.pop("lr_warmup", None)
        self.lr_original = self.opt_kwargs.lr

        # We can have a list of keywords to exclude from weight decay
        weight_decay = opt_kwargs.pop("weight_decay", 0.0)
        wd_exclude_list = opt_kwargs.pop("weight_decay_exclude", [])

        def pick_excluded(name):
            return any([kw in name for kw in wd_exclude_list])

        excluded = [
            p
            for (name, p) in self.named_parameters()
            if pick_excluded(name) and p.requires_grad
        ]
        others = [
            p
            for (name, p) in self.named_parameters()
            if not pick_excluded(name) and p.requires_grad
        ]

        without_weight_decay = {"params": excluded}
        with_weight_decay = {"params": others, "weight_decay": weight_decay}

        # we may have some frozen layers, so we remove these parameters
        # from the optimization
        log.info(f"set optim with {self.opt_kwargs}")
        opt_kwargs = {
            **{"params": [without_weight_decay, with_weight_decay]},
            **opt_kwargs,
        }
        optimizer = instantiate(config=opt_kwargs, _recursive_=False, _convert_="all")

        if self.schedule_kwargs is not None:
            if "scheduler" not in self.schedule_kwargs:
                scheduler = instantiate(
                    {**self.schedule_kwargs, **{"optimizer": optimizer}}
                )
            else:
                scheduler = OmegaConf.to_container(self.schedule_kwargs, resolve=True)
                lr_sch_kwargs = scheduler.pop("scheduler")
                scheduler["scheduler"] = instantiate(
                    {**lr_sch_kwargs, **{"optimizer": optimizer}}, _recursive_=False
                )
        else:
            scheduler = None

        # this will be called in on_after_backward
        self.grad_clipper = instantiate(self.grad_clip_kwargs)

        if scheduler is None:
            return [optimizer]
        else:
            return {
                "optimizer": optimizer,
                "lr_scheduler": scheduler,
                "monitor": self.val_kwargs.main_loss,
            }

    def optimizer_step(self, *args, **kwargs):
        # Method overridden so that the EMA params are updated after each optimizer step
        super().optimizer_step(*args, **kwargs)
        if self.ema is not None:
            self.ema.update(self.model_parameters())

    def on_after_backward(self):
        if self.grad_clipper is not None:
            grad_norm, clipping_threshold = self.grad_clipper(self)
        else:
            # we still want to compute this for monitoring in tensorboard
            grad_norm = utils.grad_norm(self)
            clipped_norm = grad_norm

        # log every few iterations
        if self.trainer.global_step % 25 == 0:
            clipped_norm = min(grad_norm, clipping_threshold)

            # get the current learning reate
            opt = self.trainer.optimizers[0]
            current_lr = opt.state_dict()["param_groups"][0]["lr"]

            self.logger.log_metrics(
                {
                    "grad/norm": grad_norm,
                    "grad/clipped_norm": clipped_norm,
                    "grad/step_size": current_lr * clipped_norm,
                },
                step=self.trainer.global_step,
            )

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
                self.ema.store(self.model_parameters())  # store current params in EMA
                self.ema.copy_to(
                    self.model_parameters()
                )  # copy EMA parameters over current params for evaluation
            else:
                # train
                if self.ema.collected_params is not None:
                    self.ema.restore(
                        self.model_parameters()
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

    def do_lr_warmup(self):
        if not hasattr(self, "lr_warmup"):
            return

        if self.lr_warmup is not None and self.trainer.global_step < self.lr_warmup:
            lr_scale = min(1.0, float(self.trainer.global_step + 1) / self.lr_warmup)
            optimizer = self.trainer.optimizers[0]
            for pg in optimizer.param_groups:
                pg["lr"] = lr_scale * self.lr_original
