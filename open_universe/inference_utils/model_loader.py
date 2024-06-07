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
# Model Loader

This function can load a model from a checkpoint file or a Huggingface Model.

Author: Robin Scheibler (@fakufaku)
"""
from pathlib import Path

import torch
import yaml
from huggingface_hub import hf_hub_download
from hydra.utils import instantiate
from omegaconf import OmegaConf

# parse the models supported by lyse
supported_models = ["universe"]


def ckpt_to_config_path(ckpt_path):
    """
    Finds the config file path
    """
    ckpt_path = Path(ckpt_path)
    config_path_1 = ckpt_path.parent / "config.yaml"
    config_path_2 = ckpt_path.parents[1] / ".hydra/config.yaml"
    if config_path_1.exists():
        config_path = config_path_1
    elif config_path_2.exists():
        config_path = config_path_2
    else:
        raise ValueError(
            f"Could not find the configuration file for model {ckpt_path}."
        )
    return config_path


def open_update_config(path):
    """
    - open
    - create omegaconf
    """
    with open(path, "r") as f:
        config = yaml.safe_load(f)
    config = OmegaConf.create(config)
    return config


def load_model(ckpt_path, device=None, strict=True, return_config=False, hf_token=None):
    """
    Load a model from a checkpoint file or a Huggingface Model.

    Parameters
    ----------
    ckpt_path : str or Path
        Path to the checkpoint file or the Huggingface Model ID.
        If the string is not a local path, the function will look-up the model on HF.
        If this is a local path, then we assume this is a checkpoint file.
        The corresponding config file is assumed to be either in the same folder `./config.yaml`
        or in the parent folder `../.hydra/config.yaml`.
    device : str or torch.device
        The device on which to load the model.
    strict : bool
        If True, the model must have the same architecture as the checkpoint.
    return_config : bool
        If True, return the configuration file as well.
    """
    if not Path(ckpt_path).exists():
        try:
            # assume this must be an HF model
            colon_pos = ckpt_path.find(":")
            if colon_pos >= 0:
                repo_id = ckpt_path[:colon_pos]
                revision = ckpt_path[colon_pos + 1 :]
            else:
                repo_id = ckpt_path
                revision = None
            ckpt_path = hf_hub_download(
                repo_id=repo_id,
                filename="weights.ckpt",
                revision=revision,
                token=hf_token,
            )
            config_path = hf_hub_download(
                repo_id=repo_id,
                filename="config.yaml",
                revision=revision,
                token=hf_token,
            )
        except Exception as e:
            print(
                f"{ckpt_path} is not a local file and download from HF hub failed."
            )
            raise e
    else:
        ckpt_path = Path(ckpt_path)
        config_path = ckpt_to_config_path(ckpt_path)

    config = open_update_config(config_path)

    model = instantiate(config.model, _recursive_=False)
    model = model.to(device)

    data = torch.load(ckpt_path, map_location=device)

    if hasattr(model, "ema") and "ema" in data:
        model.ema.load_state_dict(data["ema"])
        # use False here since the EMA will always be used for inference
        model.load_state_dict(data["state_dict"], strict=False)

    elif hasattr(model, "ema") and "ema" not in data and model.ema is not None:
        model.load_state_dict(data["state_dict"], strict=strict)
        # restore ema from model_weights
        model.ema.store(model.model_parameters())

    else:
        model.load_state_dict(data["state_dict"], strict=strict)

    model.eval()

    if return_config:
        return model, config
    else:
        return model
