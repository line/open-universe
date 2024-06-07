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
# NoisyDataset module

A simple dataset for speech enhancement where the clean/noisy samples are
stored in two different folders.

Author: Robin Scheibler (@fakufaku)
"""
import logging
import math
import os
import random
from pathlib import Path
from typing import Optional, Union

import torch
import torchaudio
from hydra.utils import instantiate, to_absolute_path
from omegaconf import DictConfig
from torch.utils.data import DataLoader, Dataset

log = logging.getLogger(__name__)


class NoisyDataset(Dataset):
    """
    A simple dataset for speech enhancement where the clean/noisy samples are
    stored in two different folders.

    Parameters
    ----------
    audio_path : str or Path
        Path to the root folder containing the dataset
    audio_len : int or float
        Length of the audio samples in seconds. If the samples are shorter,
        they are repeated, if they are longer, they are truncated.
    fs : int
        Sampling frequency of the audio samples.
    split : str
        The sub-folder where the target split (e.g., test, val, etc.) is stored.
        If `None`, the root folder is used.
    noisy_folder : str
        The sub-folder where the noisy samples are stored.
    clean_folder : str
        The sub-folder where the clean samples are stored. If `None`, the clean
        samples are not available and they will be replaced by a single 0.
    """

    def __init__(
        self,
        audio_path: Union[str, Path],
        audio_len: Union[int, float] = 4,
        fs: Optional[int] = 16000,
        split: Optional[str] = "train",
        noisy_folder: Optional[str] = "noisy",
        clean_folder: Optional[str] = "clean",
    ):
        # In case of Valentini dataset
        audio_path = Path(to_absolute_path(str(audio_path)))
        if split is not None:
            audio_path = audio_path / split
        self.noisy_path = audio_path / noisy_folder

        if clean_folder is None:
            self.clean_available = False
        else:
            self.clean_path = audio_path / clean_folder
            if not self.clean_path.exists():
                self.clean_available = False
            else:
                self.clean_available = True

        if not self.noisy_path.exists():
            raise FileNotFoundError(f"{self.noisy_path} does not exist")

        # get the noisy files first
        noisy_files = sorted(os.listdir(self.noisy_path))

        # find whichever clean files are available
        if self.clean_available:
            # only use files that are in both folders
            file_list = set(noisy_files) & set(os.listdir(self.clean_path))
            if len(file_list) == 0:
                # clean files are not available after all...
                self.clean_available = False
                self.file_list = noisy_files
            else:
                self.file_list = sorted(list(file_list))
        else:
            self.file_list = noisy_files

        if self.clean_available:
            log.info(
                f"{self.__class__}: path={audio_path} "
                f"{len(self.file_list)} noisy/clean pairs"
            )
        else:
            self.clean_path = None
            log.info(
                f"{self.__class__}: path={audio_path} "
                f"{len(self.file_list)} noisy files, no clean reference"
            )

        file_list = []

        self.file_list = sorted(os.listdir(self.noisy_path))

        if audio_len is not None:
            self.audio_len = int(audio_len * fs)
        else:
            self.audio_len = None
        self.fs = fs
        self.split = split

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        noisy_path = self.noisy_path / self.file_list[idx]
        key = noisy_path.stem

        noisy, sr = torchaudio.load(noisy_path)
        if self.clean_available:
            clean_path = self.clean_path / self.file_list[idx]
            clean, sr2 = torchaudio.load(clean_path)
            assert sr == sr2
        else:
            clean = 0

        if self.split == "test":
            return noisy, clean, key

        if self.audio_len is not None:
            ori_len = noisy.shape[-1]
            if ori_len < self.audio_len:
                rep = math.ceil(self.audio_len / ori_len)
                noisy = torch.tile(noisy, dims=(rep,))[..., : self.audio_len]
                if self.clean_available:
                    clean = torch.tile(clean, dims=(rep,))[..., : self.audio_len]
            else:
                st_idx = random.randint(0, ori_len - self.audio_len)
                noisy = noisy[..., st_idx : st_idx + self.audio_len]
                if self.clean_available:
                    clean = clean[..., st_idx : st_idx + self.audio_len]

        return noisy, clean, key
