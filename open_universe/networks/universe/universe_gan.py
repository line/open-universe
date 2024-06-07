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
The UNIVERSE(++) network with HiFi-GAN loss

Author: Robin Scheibler (@fakufaku)
"""
import itertools
import logging

import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf

from .. import bigvgan as gan
from .blocks import PReLU_Conv
from .universe import Universe

log = logging.getLogger(__name__)


def instantiate_scheduler(config_scheduler, optimizer):
    if config_scheduler is None:
        return None

    scheduler = OmegaConf.to_container(config_scheduler, resolve=True)
    lr_sch_kwargs = scheduler.pop("scheduler")
    scheduler["scheduler"] = instantiate(
        {**lr_sch_kwargs, **{"optimizer": optimizer}}, _recursive_=False
    )
    return scheduler


def step_scheduler(
    scheduler, interval, frequency, epoch, step, is_last_batch, loss_val=None
):
    if scheduler is None:
        return

    on_epoch = interval == "epoch" and is_last_batch and (epoch + 1) % frequency == 0
    on_step = interval == "step" and (step + 1) % frequency == 0
    if on_epoch or on_step:
        if loss_val is not None:
            scheduler.step(loss_val)
        else:
            scheduler.step()


class UniverseGAN(Universe):
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
        detach_cond=False,
        edm=None,
    ):
        super().__init__(
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
            normalization_kwargs=normalization_kwargs,
            detach_cond=detach_cond,
            edm=edm,
        )

        # we have to do this for the GAN
        self.automatic_optimization = False

        self.log_kwargs = dict(
            on_step=True,
            on_epoch=False,
            logger=True,
            sync_dist=False,
        )

    def init_losses(self, score_model, condition_model, losses, training):
        """initialize the GAN losses here"""
        # define discriminators. MPD is used by default
        self.loss_mpd = gan.MultiPeriodDiscriminator(losses.multi_period_discriminator)

        # define additional discriminators. BigVGAN uses MRD as default
        self.loss_mrd = gan.MultiResolutionDiscriminator(
            losses.multi_resolution_discriminator
        )

        if losses.use_signal_decoupling:
            self.signal_decoupling_layer = PReLU_Conv(
                self.n_channels,
                1,
                kernel_size=3,
                padding="same",
                act_type=losses.get("signal_decoupling_act", None),
            )
        else:
            self.signal_decoupling_layer = None

        self.loss_score = instantiate(losses.score_loss, _recursive_=False)
        self.disc_freeze_step = losses.get("disc_freeze_step", 0)

        if losses.get("aux_loss", None) is not None:
            self.loss_aux = instantiate(losses.aux_loss, _recursive_=False)
        else:
            self.loss_aux = None

    def model_parameters(self):
        # exclude the loss parameters
        params = itertools.chain(
            self.get_score_model().parameters(), self.condition_model.parameters()
        )
        if self.signal_decoupling_layer is not None:
            params = itertools.chain(params, self.signal_decoupling_layer.parameters())
        return params

    def aux_to_wav(self, y_aux):
        if self.signal_decoupling_layer is not None:
            return self.signal_decoupling_layer(y_aux)
        else:
            return y_aux

    def training_step(self, batch, batch_idx):
        opt_score, opt_disc = self.optimizers()
        try:
            sch_score, sch_disc = self.lr_schedulers()
            has_schedulers = True
        except TypeError:
            has_schedulers = False

        batch = batch[:2]
        mix, target = batch

        if len(batch) == 4:
            # in this case we have a special target for the conditioner network
            target_original = batch[2]
        else:
            target_original = target

        if getattr(self.train_kwargs, "dynamic_mixing", False):
            noise = mix - target
            perm = torch.randperm(noise.shape[0])
            mix = target + noise[perm, ...]

        (mix, target, target_original), *stats = self.normalize_batch(
            (mix, target, target_original), norm=self.normalization_norm
        )

        # target_original = target
        mix = self.transform(mix)
        target = self.transform(target)

        sigma, _ = self.sample_sigma(mix, self.train_kwargs.time_sampling, 0.0, 1.0)

        # sample the noise and create the target
        z = target.new_zeros(target.shape).normal_()
        x_t = target + sigma[:, None, None] * z

        # run computations
        cond, y_est, _ = self.condition_model(mix, train=True)

        if self.detach_cond:
            cond = [c.detach() for c in cond]

        # compute score
        score = self.score_model(x_t, sigma, cond)

        # decouple the signal estimate
        if self.signal_decoupling_layer is not None:
            y_est = self.signal_decoupling_layer(y_est)
        y_est = self.transform(y_est, inv=True)

        # use a simple mel loss to regularize training
        mel_y_est = self.condition_model.input_mel.compute_mel_spec(y_est)
        mel_target = self.condition_model.input_mel.compute_mel_spec(target_original)

        # Discriminator optimization
        opt_disc.zero_grad()

        # 1. MPD
        y_df_hat_r, y_df_hat_g, _, _ = self.loss_mpd(target_original, y_est.detach())
        loss_disc_f, losses_disc_f_r, losses_disc_f_g = gan.discriminator_loss(
            y_df_hat_r, y_df_hat_g
        )
        # 2. MRD
        y_ds_hat_r, y_ds_hat_g, _, _ = self.loss_mrd(target_original, y_est.detach())
        loss_disc_s, losses_disc_s_r, losses_disc_s_g = gan.discriminator_loss(
            y_ds_hat_r, y_ds_hat_g
        )

        loss_disc = loss_disc_s + loss_disc_f

        if self.global_step >= self.disc_freeze_step:
            self.manual_backward(loss_disc)
            grad_norm_mpd = torch.nn.utils.clip_grad_norm_(
                self.loss_mpd.parameters(), self.grad_clip_vals.mpd
            )
            grad_norm_mrd = torch.nn.utils.clip_grad_norm_(
                self.loss_mrd.parameters(), self.grad_clip_vals.mrd
            )
            opt_disc.step()
            if has_schedulers:
                self.step_schedulers(sch_score, sch_disc)
        else:
            grad_norm_mpd = 0.0
            grad_norm_mrd = 0.0

        if torch.isnan(loss_disc):
            print(f"Found NaN!! Please investigate. {loss_disc_s=} {loss_disc_f=}")
            breakpoint()

        # Generator optimization
        opt_score.zero_grad()

        # 1. Score loss
        l_score = self.loss_score(sigma[..., None, None] * score, -z)

        # 2. L1 Mel-Spectrogram Loss
        loss_mel = torch.nn.functional.l1_loss(mel_y_est, mel_target)

        # 3. Auxiliary loss
        if self.loss_aux is not None:
            # optional linear warmup of the loss if it is introduced late
            aux_loss_schedule = self.losses_kwargs.get("aux_loss_schedule", {})
            start_step = aux_loss_schedule.get("start_step", 0)
            warmup_steps = aux_loss_schedule.get("warmup_steps", 0)
            if self.global_step >= start_step:
                w = (
                    min(1.0, (self.global_step - start_step) / warmup_steps)
                    if warmup_steps > 0
                    else 1.0
                )
                loss_aux = w * self.loss_aux(y_est, target_original)
            else:
                loss_aux = 0.0
        else:
            loss_aux = 0.0

        # 4. MPD loss
        y_df_hat_r, y_df_hat_g, fmap_f_r, fmap_f_g = self.loss_mpd(
            target_original, y_est
        )
        loss_fm_f = gan.feature_loss(fmap_f_r, fmap_f_g)
        loss_gen_f, losses_gen_f = gan.generator_loss(y_df_hat_g)

        # 5. MRD loss
        y_ds_hat_r, y_ds_hat_g, fmap_s_r, fmap_s_g = self.loss_mrd(
            target_original, y_est
        )
        loss_fm_s = gan.feature_loss(fmap_s_r, fmap_s_g)
        loss_gen_s, losses_gen_s = gan.generator_loss(y_ds_hat_g)

        loss_gen = (
            l_score * self.losses_kwargs.weights.score
            + loss_mel * self.losses_kwargs.weights.mel_l1
            + loss_aux * self.losses_kwargs.weights.get("aux", 1.0)
        )
        if self.global_step >= self.disc_freeze_step:
            loss_gen = loss_gen + loss_gen_s + loss_gen_f + loss_fm_s + loss_fm_f

        self.manual_backward(loss_gen)

        grad_norm_score = torch.nn.utils.clip_grad_norm_(
            self.get_score_model().parameters(), self.grad_clip_vals.score
        )

        grad_norm_cond = torch.nn.utils.clip_grad_norm_(
            self.condition_model.parameters(), self.grad_clip_vals.cond
        )
        opt_score.step()
        if self.ema is not None:
            self.ema.update(self.model_parameters())
        if has_schedulers:
            self.step_schedulers(sch_score, sch_disc)

        # every few steps, we log stuff
        self.log(
            "train/loss_disc",
            loss_disc,
            batch_size=mix.shape[0],
            prog_bar=True,
            **self.log_kwargs,
        )
        self.log(
            "train/loss_mpd", loss_disc_f, batch_size=mix.shape[0], **self.log_kwargs
        )
        self.log(
            "train/loss_mrd", loss_disc_s, batch_size=mix.shape[0], **self.log_kwargs
        )

        self.log(
            "train/loss_gen",
            loss_gen,
            prog_bar=True,
            batch_size=mix.shape[0],
            **self.log_kwargs,
        )
        self.log("train/score", l_score, batch_size=mix.shape[0], **self.log_kwargs)
        self.log(
            "train/signal_mel", loss_mel, batch_size=mix.shape[0], **self.log_kwargs
        )
        if self.loss_aux is not None:
            self.log(
                "train/signal_aux", loss_aux, batch_size=mix.shape[0], **self.log_kwargs
            )
        self.log("train/mrd_fm", loss_fm_s, batch_size=mix.shape[0], **self.log_kwargs)
        self.log("train/mpd_fm", loss_fm_f, batch_size=mix.shape[0], **self.log_kwargs)
        self.log(
            "train/mrd_gen", loss_gen_s, batch_size=mix.shape[0], **self.log_kwargs
        )
        self.log(
            "train/mpd_gen", loss_gen_f, batch_size=mix.shape[0], **self.log_kwargs
        )
        self.log("grad/score", grad_norm_score, **self.log_kwargs)
        self.log("grad/cond", grad_norm_cond, **self.log_kwargs)
        self.log("grad/mpd", grad_norm_mpd, **self.log_kwargs)
        self.log("grad/mrd", grad_norm_mrd, **self.log_kwargs)

        if torch.isnan(loss_gen):
            print(
                f"Found NaN!! Please investigate. {l_score=} {loss_mel=} "
                f"{loss_gen_s=} {loss_gen_f=} {loss_fm_s=} {loss_fm_f=}"
            )
            breakpoint()

    def step_schedulers(self, sch_score, sch_disc):
        # step the schedulers
        step_scheduler(
            sch_score,
            self.schedule_kwargs.generator.interval,
            self.schedule_kwargs.generator.frequency,
            self.current_epoch,
            self.global_step,
            self.trainer.is_last_batch,
            loss_val=None,
        )
        step_scheduler(
            sch_disc,
            self.schedule_kwargs.discriminator.interval,
            self.schedule_kwargs.discriminator.frequency,
            self.current_epoch,
            self.global_step,
            self.trainer.is_last_batch,
            loss_val=None,
        )

    def configure_optimizers(self):
        # generator
        opt_kwargs = OmegaConf.to_container(self.opt_kwargs.generator, resolve=True)

        # We can have a list of keywords to exclude from weight decay
        weight_decay = opt_kwargs.pop("weight_decay", 0.0)
        wd_exclude_list = opt_kwargs.pop("weight_decay_exclude", [])

        def pick_excluded(name):
            return any([kw in name for kw in wd_exclude_list])

        excluded = []
        others = []
        for submod in [self.get_score_model(), self.condition_model]:
            excluded += [
                p
                for (name, p) in submod.named_parameters()
                if pick_excluded(name) and p.requires_grad
            ]
            others += [
                p
                for (name, p) in submod.named_parameters()
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
        optimizer_gen = instantiate(
            config=opt_kwargs, _recursive_=False, _convert_="all"
        )

        # optimizer discriminator
        params_disc = list(self.loss_mrd.parameters()) + list(
            self.loss_mrd.parameters()
        )
        optimizer_disc = instantiate(
            self.opt_kwargs.discriminator, params=params_disc, _recursive_=False
        )

        self.grad_clip_vals = self.opt_kwargs.grad_clip_vals
        self.grad_clipper = None

        if self.schedule_kwargs is not None:
            scheduler_gen = instantiate_scheduler(
                self.schedule_kwargs.get("generator"), optimizer_gen
            )
            scheduler_disc = instantiate_scheduler(
                self.schedule_kwargs.get("discriminator"), optimizer_disc
            )
            return [optimizer_gen, optimizer_disc], [scheduler_gen, scheduler_disc]
        else:
            return [optimizer_gen, optimizer_disc]

    def on_after_backward(self):
        pass
