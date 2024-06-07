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
# Function Signature to Arguments Parser

This is a utility to make available the model specific arguments
to the enhancement script

Author: Robin Scheibler (@fakufaku)
"""
import argparse
import typing


def add_enhance_arguments(model, parser):
    """
    Reads the signature of the `model.enhance` method and adds them
    to the parser.

    Parameters
    ----------
    model : torch.nn.Module
        A pytorch model that has an `enhance` method.
    parser : argparse.ArgumentParser
        The parser to which the arguments should be added.

    Returns
    -------
    argparse.ArgumentParser
        The parser with the arguments added.
    """
    if not (hasattr(model, "enhance") and callable(model.enhance)):
        raise ValueError("Model does not have an `enhance` method.")
    enhance_args = typing.get_type_hints(model.enhance)
    enhance_args.pop("return", None)
    default_kwargs = getattr(model, "diff_kwargs", {})

    type_casters = {}
    # and prepare the type casters to transform the strings
    # provided in the parser into their correct types
    for key, val in enhance_args.items():
        types = typing.get_args(enhance_args[key])
        if len(types) == 0:
            type_casters[key] = enhance_args[key]
        else:
            type_casters[key] = types[0]

    # drop default args that are not in the signature
    group = parser.add_argument_group("enhance", "Arguments of enhance function")
    for key, type_cast in type_casters.items():
        group.add_argument(
            f"--{key}", default=default_kwargs.get(key, None), type=type_cast
        )

    return parser
