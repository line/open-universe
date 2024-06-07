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
Base classes for speech enhancement in pytorch lightning

Author: Robin Scheibler (@fakufaku)
"""
import logging

import pytorch_lightning as pl
import torch
import torchaudio
from hydra.utils import instantiate
from omegaconf import OmegaConf

from ... import utils

log = logging.getLogger(__name__)


class BaseModel(pl.LightningModule):
    """
    This is a base class for pytorch lightning models.

    Parameters
    ----------
    optimizer: dict or omegaconf.DictConfig
        The optimizer configuration
    scheduler: dict or omegaconf.DictConfig
        The scheduler configuration
    grad_clipper: dict or omegaconf.DictConfig
        The gradient clipping configuration
    """
    def __init__(
        self,
        optimizer,
        scheduler,
        grad_clipper,
    ):
        # init superclass
        super().__init__()

        self.opt_kwargs = optimizer
        self.schedule_kwargs = scheduler
        self.grad_clip_kwargs = grad_clipper

    def training_step(self, batch, batch_idx):
        raise NotImplementedError()
        return 0

    def on_train_epoch_end(self):
        pass

    def on_validation_epoch_start(self):
        pass

    def validation_step(self, batch, batch_idx, dataset_i=0):
        pass

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

    def on_after_backward(self):
        if self.grad_clipper is not None:
            grad_norm, clipping_threshold = self.grad_clipper(self)
        else:
            # we still want to compute this for monitoring in tensorboard
            grad_norm = utils.grad_norm(self)
            clipped_norm = grad_norm
            clipping_threshold = grad_norm

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


class EnhancementBaseModel(BaseModel):
    """
    A base class to train speech enhancement models with pytorch lightning.

    Parameters
    ----------
    fs: int
        The sampling frequency of the data
    normalization_norm: str
        The normalization to apply to the data (2 or max)
    model: dict or omegaconf.DictConfig
        The model configuration
    losses: dict or omegaconf.DictConfig
        The losses configuration
    training: dict or omegaconf.DictConfig
        The training configuration
    validation: dict or omegaconf.DictConfig
        The validation configuration
    optimizer: dict or omegaconf.DictConfig
        The optimizer configuration
    scheduler: dict or omegaconf.DictConfig
        The scheduler configuration
    grad_clipper: dict or omegaconf.DictConfig
        The gradient clipping configuration
    normalization_kwargs: dict or omegaconf.DictConfig, optional
        Extra arguments for the normalization
    """
    def __init__(
        self,
        fs,
        normalization_norm,
        model,
        losses,
        training,
        validation,
        optimizer,
        scheduler,
        grad_clipper,
        normalization_kwargs={},
    ):
        # init superclass
        super().__init__(optimizer, scheduler, grad_clipper)

        self.fs = fs
        self.normalization_norm = normalization_norm

        if normalization_kwargs is None:
            normalization_kwargs = {}
        self.normalization_kwargs = normalization_kwargs

        self.train_kwargs = training
        self.losses_kwargs = losses
        self.val_kwargs = validation
        self.loss_kwargs = losses

        if isinstance(model, torch.nn.Module):
            self.model = model
        else:
            self.model = instantiate(model, _recursive_=False)

        self.configure_losses()  # set self.losses_weights and self.losses_func

        self.denormalize_batch = utils.denormalize_batch

    def configure_losses(self):
        # training losses, allow for multi-target training
        if "_target_" in self.loss_kwargs:
            losses_weights = {"loss": 1.0}
            losses_func = {"loss": instantiate(self.loss_kwargs, _recursive_=False)}
        else:
            losses_weights = {}
            losses_func = torch.nn.ModuleDict()
            for loss_name, loss in self.loss_kwargs.items():
                if "_target_" in loss:
                    losses_func[loss_name] = instantiate(loss, _recursive_=False)
                    losses_weights[loss_name] = 1.0
                else:
                    losses_func[loss_name] = instantiate(loss.kwargs, _recursive_=False)
                    losses_weights[loss_name] = loss.get("weight", 1.0)
        self.losses_func = losses_func
        self.losses_weights = losses_weights

        # validation losses
        self.enh_losses = torch.nn.ModuleDict()
        for name, loss_args in self.val_kwargs.enh_losses.items():
            self.enh_losses[name] = instantiate(loss_args)

    def normalize_batch(self, batch, norm=None):
        if norm is None:
            norm = self.normalization_norm
        return utils.normalize_batch(batch, norm=norm, **self.normalization_kwargs)

    def forward(self, x):
        ret = self.model(x)
        ret = torch.nn.functional.pad(ret, (0, x.shape[-1] - ret.shape[-1]))
        if isinstance(ret, (list, tuple)):
            ret = ret[0]
        if not isinstance(ret, torch.Tensor):
            raise ValueError("The enhancement model should return a Tensor")
        return ret

    def enhance(self, mix, keep_rms=False):
        x_ndim = mix.ndim
        if x_ndim == 1:
            mix = mix[None, None, :]
        elif x_ndim == 2:
            mix = mix[:, None, :]
        elif x_ndim > 3:
            raise ValueError("The input should have at most 3 dimensions")

        mix_rms = mix.square().mean(dim=(-2, -1), keepdim=True).sqrt()

        (x, _), *stats = utils.normalize_batch((mix, None))

        x = self(x)

        x = self.denormalize_batch(x, *stats)

        if keep_rms:
            x_rms = x.square().mean(dim=(-2, -1), keepdim=True).sqrt().clamp(min=1e-5)
            x = x * (mix_rms / x_rms)

        if x_ndim == 1:
            x = x[0, 0]
        elif x_ndim == 2:
            x = x[:, 0, :]

        return x

    def on_train_epoch_start(self):
        pass

    def compute_losses(
        self,
        enh,
        target,
        train=True,
    ):
        loss = 0.0
        losses = {}
        for loss_name, loss_func in self.losses_func.items():
            losses[loss_name] = loss_func(enh, target)
            loss = loss + self.losses_weights[loss_name] * losses[loss_name]

        return loss, losses

    def training_step(self, batch, batch_idx):
        mix, target = batch[:2]

        if getattr(self.train_kwargs, "dynamic_mixing", False):
            noise = mix - target
            perm = torch.randperm(noise.shape[0])
            mix = target + noise[perm, ...]

        (mix, target), *stats = self.normalize_batch(
            (mix, target), norm=self.normalization_norm
        )

        enh = self(mix)

        train_loss, losses = self.compute_losses(enh, target)

        # every 10 steps, we log stuff
        self.log(
            "train/main_loss",
            train_loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            sync_dist=False,
            batch_size=mix.shape[0],
        )
        kwargs = dict(
            batch_size=mix.shape[0],
            on_step=True,
            on_epoch=True,
            prog_bar=False,
            logger=True,
            sync_dist=False,
        )
        for loss_name, loss_val in losses.items():
            self.log(f"train/{loss_name}", loss_val, **kwargs)

        if torch.isnan(train_loss):
            print("Found NaN!! Please investigate. Opening a prompt.")
            print("- is target nan ?", torch.isnan(target).any())
            print("- is mix nan ?", torch.isnan(mix).any())
            print("- is enh nan ?", torch.isnan(enh).any())
            for lkey, lval in losses.items():
                print(f"- is {lkey} nan ?", torch.isnan(lval).any())
            breakpoint()

        return train_loss

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

    def validation_step(self, batch, batch_idx, dataset_i=0):
        batch_scaled, *stats = self.normalize_batch(
            batch[:2], norm=self.normalization_norm
        )
        mix, target = batch_scaled
        batch_size = mix.shape[0]

        enh = self(mix)

        train_loss, losses = self.compute_losses(enh, target)

        self.log(
            "val/main_loss",
            train_loss,
            on_epoch=True,
            sync_dist=True,
            batch_size=batch_size,
        )
        for loss_name, loss_val in losses.items():
            self.log(
                f"val/{loss_name}",
                loss_val,
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
            mix, target = batch[:2]
            self.n_batches_est_done += 1
            est = self.enhance(mix)

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
                        loss_val.to(mix.device),  # keep on same device as the data
                        on_epoch=True,
                        sync_dist=True,
                        batch_size=batch_size,
                    )

    def on_validation_epoch_end(self):
        pass
