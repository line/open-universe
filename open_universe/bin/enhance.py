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
# Inference script

This script will use a pre-trained model to enhance all the wav/mp3/flac files
in a folder and store the enhanced files in a new folder.
The folder structure is maintained.

Author: Robin Scheibler (@fakufaku)
"""
import argparse
import sys
from pathlib import Path

import torch
import torchaudio
from tqdm import tqdm

from open_universe import inference_utils

AUDIO_SUFFIXES = [".wav", ".mp3", ".flac"]


def handle_help():
    help_idx = -1
    try:
        help_idx = sys.argv.index("--help")
    except ValueError:
        pass
    try:
        help_idx = sys.argv.index("-h")
    except ValueError:
        pass

    # model may add new argument, so if the model is present, we want to defer
    # processing of help until after the model arguments are added
    if "--model" not in sys.argv:
        return False

    if help_idx >= 0:
        sys.argv.pop(help_idx)
        requires_help = True
    else:
        requires_help = False

    return requires_help


def find_files(path):
    # find all the files
    if not path.is_dir():
        files = [path]
        rel_path = path.parent
        dir_proc = False
    else:
        files = []
        rel_path = path
        dir_proc = True
        for p in path.rglob("*"):
            if p.suffix in AUDIO_SUFFIXES:
                files.append(p)
    return files, rel_path, dir_proc


def resample(audio, fs, target_fs):
    if fs != target_fs:
        audio = torchaudio.functional.resample(audio, fs, target_fs)
    return audio


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Enhance a file or a directory of audio files"
    )
    parser.add_argument(
        "input", type=Path, help="Path to an audio file or a folder of audio files"
    )
    parser.add_argument(
        "output",
        type=Path,
        help="Output path for the enhanced files. In the case of a folder, the orignal structure is retained.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="line-corporation/open-universe:plusplus",
        help="A model stored in the MLU model zoo",
    )
    parser.add_argument(
        "--hf-token",
        type=str,
        help="Huggingface access token",
    )
    parser.add_argument(
        "--model-strict",
        action="store_true",
        help="Use strict policy to load the model. Can help uncover problems.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1028282,
        help="Set a deterministic seed to get reproducible results",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help="The device to use, e.g. cuda:0, cpu. Default: cuda:0.",
    )


    # do a partial parsing to know what model to get
    requires_help = handle_help()
    args, _ = parser.parse_known_args()

    # choose device
    if torch.cuda.is_available():
        device = args.device
        if not (device == "cpu" or device.startswith("cuda")):
            raise ValueError(
                "Device name should be 'cpu' or 'cuda:X' where X is an integer. "
                f"Provided {device}"
            )
    else:
        print("Default to CPU inference because CUDA is not available.")
        device = "cpu"

    # load model
    model = inference_utils.load_model(
        args.model, device=device, strict=args.model_strict, hf_token=args.hf_token
    )

    # use a deterministic rng to get repeatable results
    rng = torch.Generator(device=device)
    rng.manual_seed(args.seed)

    # now add the arguments of the model to the parser
    inference_utils.add_enhance_arguments(model, parser)

    # finish the parsing
    if requires_help:
        sys.argv.append("--help")
    args = parser.parse_args()

    # get group arguments to use with enhance command
    arg_groups = {}
    for group in parser._action_groups:
        group_dict = {a.dest: getattr(args, a.dest, None) for a in group._group_actions}
        arg_groups[group.title] = argparse.Namespace(**group_dict)
    enhance_kwargs = vars(arg_groups["enhance"])

    # set the rng
    enhance_kwargs["rng"] = rng

    files, rel_path, dir_proc = find_files(args.input)

    if dir_proc:
        files = tqdm(files)

    for path in files:
        if dir_proc:
            output_path = args.output / path.relative_to(rel_path)
            output_path.parent.mkdir(exist_ok=True, parents=True)
        else:
            if args.output.is_dir():
                output_path = args.output / path.name
            else:
                output_path = args.output

        audio, fs = torchaudio.load(path)

        audio = audio.to(device)

        with torch.no_grad():
            audio = resample(audio, fs, model.fs)
            enh = model.enhance(audio, **enhance_kwargs)
            enh = resample(enh, model.fs, fs)

        torchaudio.save(output_path, enh.cpu(), fs)
