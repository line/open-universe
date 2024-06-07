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
# Main training script.

This script will read a hydra config and start the training of a model.
For each run, an experiment directory named `exp/<experiment_name>/<datetime>`
containing config, checkpoints, and tensorboard logs will be created.

Author: Robin Scheibler (@fakufaku)
"""
import logging
import os

import hydra
import pytorch_lightning as pl
import torch
from hydra.core.hydra_config import HydraConfig
from hydra.utils import instantiate, to_absolute_path
from omegaconf.omegaconf import open_dict
from pytorch_lightning import loggers as pl_loggers

from open_universe import utils

log = logging.getLogger(__name__)


@hydra.main(config_path="./config", config_name="config")
def main(cfg):
    if utils.ddp.is_rank_zero():
        hydra_conf = HydraConfig().get()
        exp_name = hydra_conf.run.dir
        log.info(f"Start experiment: {exp_name}")
    else:
        # when using DDP, if not rank zero, we are already in the run dir
        os.chdir(hydra.utils.get_original_cwd())

    # Some infos for DDP debug
    rank, world_size, worker, num_workers = utils.ddp.pytorch_worker_info()
    log.info(f"{rank=}/{world_size} {worker=}/{num_workers} PID={os.getpid()}")

    # seed all RNGs for deterministic behavior
    pl.seed_everything(cfg.seed)

    torch.autograd.set_detect_anomaly(True)
    torch.set_float32_matmul_precision("medium")

    callbacks = []
    # Use a fancy progress bar
    callbacks.append(pl.callbacks.RichProgressBar())
    # Monitor the learning rate
    callbacks.append(pl.callbacks.LearningRateMonitor(logging_interval="step"))
    # configure checkpointing to save all models
    # save_top_k == -1  <-- saves all models
    val_loss_name = f"{cfg.model.validation.main_loss}"
    loss_name = val_loss_name.split("/")[-1]  # avoid "/" in filenames
    modelcheckpoint_callback = pl.callbacks.ModelCheckpoint(
        monitor=val_loss_name,
        save_last=True,
        save_top_k=-1,
        mode=cfg.model.validation.main_loss_mode,
        filename="".join(["step-{step:08d}_", loss_name, "-{", val_loss_name, ":.4f}"]),
        auto_insert_metric_name=False,
    )
    callbacks.append(modelcheckpoint_callback)

    # the data module
    print("Using the DCASE2020 SELD original dataset")
    log.info("create datalogger")

    dm = instantiate(cfg.datamodule, _recursive_=False)
    log.info(f"Create datamodule with training set: {cfg.datamodule.train.dataset}")

    # init model
    log.info(f"Create new model {cfg.model._target_}")
    model = instantiate(cfg.model, _recursive_=False)

    # create a logger
    tb_logger = pl_loggers.TensorBoardLogger(save_dir=".", name="", version="")

    # if a ckpt path is provided, this means we want to resume
    ckpt_path = getattr(cfg, "ckpt_path", None)
    if ckpt_path is not None:
        ckpt_path = to_absolute_path(ckpt_path)
        with open_dict(cfg):
            cfg.trainer.num_sanity_val_steps = 0

    # most basic trainer, uses good defaults (auto-tensorboard, checkpoints,
    # logs, and more)
    trainer = instantiate(cfg.trainer, callbacks=callbacks, logger=tb_logger)

    if cfg.train:
        log.info("start training")
        trainer.fit(model, dm, ckpt_path=ckpt_path)

    if cfg.test:
        try:
            log.info("start testing")
            trainer.test(model, dm, ckpt_path="best")
        except pl.utilities.exceptions.MisconfigurationException:
            log.info(
                "test with current model value because no best model path is available"
            )
            trainer.validate(model, dm)
            trainer.test(model, dm)


if __name__ == "__main__":
    main()
