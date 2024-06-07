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
# LoRA Layers

Utilities to monkey-patch LoRA layers into a model, remove them,
or control which parameters get optimized.
"""
from .lora import LoraConv1d, LoraLinear, LoraConvTranspose1d
from typing import Union, Optional, List
import torch

lora_classes_map = {
    torch.nn.Conv1d: LoraConv1d,
    torch.nn.ConvTranspose1d: LoraConvTranspose1d,
    torch.nn.Linear: LoraLinear,
}


def get_adapter(
    module: torch.nn.Module, rank: int, alpha: Optional[float] = None
) -> Union[LoraConv1d, LoraLinear]:
    """
    Chooses the right LoRA adapter for the layer
    """
    if type(module) in lora_classes_map:
        try:
            return lora_classes_map[type(module)](module, rank, alpha)
        except ValueError:
            print(f"LoRA: Skip module of type {type(module)}")
            # this will skip modules that are not supported
            pass
    return None


def inject(model: torch.nn.Module, rank: int, alpha: Optional[float] = None):
    """
    Replaces the Linear and Conv1d modules by the LoRA counterparts.
    This method applies the replacement recursively to all sub-components.

    Parameters
    ----------
    model : nn.Module
        The PyTorch module or model to be modified.
    rank : int
        The rank of the low-rank approximation
    alpha : float, optional
        The scaling factor for the low-rank approximation
    """
    for name, module in model.named_children():
        # the adapter will return None if the module is not supported
        lora_module = get_adapter(module, rank, alpha)

        if lora_module is not None:
            setattr(model, name, lora_module)
        else:
            # Recursive call for child modules
            inject(module, rank, alpha)


def remove(model: torch.nn.Module):
    """
    Replaces the LoRA adapters by the updated Linear and Conv1d modules
    This method applies the replacement recursively to all sub-components.

    Parameters
    ----------
    model : nn.Module
        The PyTorch module or model to be modified.
    """
    for name, module in model.named_children():
        if isinstance(module, (LoraLinear, LoraConv1d)):
            # linear layer
            new_layer = module.un_lora()
            setattr(model, name, new_layer)
        else:
            # Recursive call for child modules
            remove(module)


def freeze_parameters_except_lora_and_bias(
    module: torch.nn.Module,
    train_biases: Optional[bool] = True,
    train_names: Optional[List[str]] = None,
):
    """
    Freezes all model parameters except for specific layers and types based on
    the configuration. Parameters in LoRA layers, the finetune head, bias
    parameters, embeddings, and layer norms can be set as trainable based on
    class settings.

    Parameters
    ----------
    module : nn.Module
        The PyTorch module or model to be modified.
    train_biases : bool
        If True, bias parameters are set as trainable.
    train_names : list of str
        List of parameter names that should be set as trainable.
    """
    if train_names is None:
        train_names = []
    for name, param in module.named_parameters():
        is_trainable = (
            "lora_" in name
            or any([s in name for s in train_names])
            or (train_biases and "bias" in name)
        )
        param.requires_grad = is_trainable
