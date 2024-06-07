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
A script that will resample all the audio files in a target
folder to a different sampling frequency and store them
in a destination folder.

Author: Robin Scheibler (@fakufaku)
"""
import argparse
from pathlib import Path

import numpy as np
import soxr
from scipy.io import wavfile
from tqdm import tqdm

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Resamples a set of files to a different sampling frequency"
    )
    parser.add_argument("source", type=Path, help="Source folder")
    parser.add_argument("dest", type=Path, help="Destination folder")
    parser.add_argument("--fs", type=int, default=16000, help="New sampling frequency")
    args = parser.parse_args()

    # force cpu usage
    args.device = "cpu"
    args.batch_size = 1

    args.dest.mkdir(exist_ok=True, parents=True)

    if args.source == args.dest:
        raise ValueError("Source == Destination will result in overwrite. Abort.")

    resamplers = dict()

    batch = []
    for path in tqdm(args.source.rglob("**/*.wav")):
        fs, audio = wavfile.read(path)

        is_s16 = audio.dtype == np.int16

        if is_s16:
            audio = audio / (2**15)

        new_audio = soxr.resample(audio, fs, args.fs)

        if is_s16:
            new_audio = (new_audio * 2**15).astype(np.int16)

        new_path = args.dest / path.relative_to(args.source)
        new_path.parent.mkdir(exist_ok=True, parents=True)

        wavfile.write(str(new_path), args.fs, new_audio)
