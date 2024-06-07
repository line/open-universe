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
# Evaluation of Speech Enhancement Metrics

This script can run a bunch of speech enhancement quality metrics
on all the files in a folder. If clean reference files are not available
only the non-invasive metrics will be computed.

The supported metrics are:
- DNSMOS (OVRL, SIG, BAK)
- Levenshtein Phoneme Similarity (LPS)
- Log-Spectral Distance (LSD, also scale-invariant)
- PESQ (narrow-band and wideband)
- PLCMOS
- Signal-to-Distortion RAtio (SDR, SI-SDR)
- STOI (extended or not)

Author: Robin Scheibler (@fakufaku)
"""
import argparse
from collections import defaultdict
import numpy as np
import json
import os
import shutil
from pathlib import Path

import torchaudio
from hydra import compose, initialize
from tqdm import tqdm

from ..metrics import Metrics


def backup_file(path):
    if path.exists():
        bak_path = Path(str(path) + ".bak")
        i = 0
        while bak_path.exists():
            bak_path = Path(str(path) + f".bak{i}")
            i += 1
        shutil.copy2(path, bak_path)


def summarize(results, ignore_inf=True):
    metrics = set()
    summary = defaultdict(lambda: 0)
    denominator = defaultdict(lambda: 0)

    for res in results.values():
        for met, val in res.items():
            if isinstance(val, str):
                continue
            metrics.add(met)
            if ignore_inf or not np.isinf(val):
                summary[met] += val
                denominator[met] += 1
        summary["number"] += 1

    for met in metrics:
        summary[met] = summary[met] / denominator[met]

    return dict(summary)


def prepare(ref_path, deg_path, results_path):
    if results_path.exists():
        with open(results_path, "r") as f:
            results = json.load(f)
    else:
        results = {}

    # now collect all the wavs in the degraded path
    paths = {p.stem: {"deg": p, "ref": None} for p in deg_path.rglob("*.wav")}
    if ref_path is not None:
        for label in paths.keys():
            ref_p = ref_path / f"{label}.wav"
            if ref_p.exists():
                paths[label]["ref"] = ref_p

    # check consistency of existing results and file list
    n_results_not_in_path = sum([lbl not in paths for lbl in results.keys()])
    if n_results_not_in_path > 0:
        raise ValueError(
            "Some results do not have a corresponding file. Check consistency"
        )

    # creates some placeholders for files that do not exist in the current results
    for lbl in paths.keys():
        if lbl not in results:
            results[lbl] = {}

    return results, paths


def load_files(path_dict):
    deg, fs = torchaudio.load(path_dict["deg"])
    if deg.shape[0] > 1:
        raise ValueError("Expected mono data")
    deg = deg[0]
    if "ref" in path_dict and path_dict["ref"] is not None:
        ref, fs_ref = torchaudio.load(path_dict["ref"])
        if ref.shape[0] > 1:
            raise ValueError("Expected mono data")
        ref = ref[0]
        if fs != fs_ref:
            raise ValueError("ref and deg should have same sampling freq.")
    else:
        ref = None
    return fs, deg, ref


def save_results(results, results_path, summary_path):
    backup_file(results_path)
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)

    summary = summarize(results)
    backup_file(summary_path)
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)


if __name__ == "__main__":
    metric_choices = Metrics.get_metric_names()

    parser = argparse.ArgumentParser(
        description="Run evaluation on validation or test dataset"
    )
    parser.add_argument(
        "enhanced_path", type=Path, help="Path to enhanced speech folder"
    )
    parser.add_argument(
        "--ref_path",
        type=Path,
        help="Path to reference clean audio folder. If not provided, "
        "only non-intrusive metrics are computed.",
    )
    parser.add_argument(
        "--result_dir",
        type=Path,
        help="Path to directory where to store the results",
    )
    parser.add_argument(
        "--metrics",
        choices=metric_choices,
        nargs="+",
        help="Metrics to compute.",
    )
    args = parser.parse_args()

    ref_path = args.ref_path
    deg_path = args.enhanced_path
    ds_name = deg_path.stem

    if args.result_dir is None:
        result_dir = deg_path.parent
    else:
        result_dir = args.result_dir
        result_dir.mkdir(parents=True, exist_ok=True)

    # check output is writable
    if not os.access(result_dir, os.W_OK):
        raise PermissionError(f"The folder {result_dir} is not writable")

    results_path = result_dir / f"{ds_name}.json"
    summary_path = result_dir / f"{ds_name}_summary.json"

    results, paths = prepare(ref_path, deg_path, results_path)

    metrics_computer = Metrics(metrics=args.metrics)

    for label, _ in tqdm(results.items()):
        fs, deg, ref = load_files(paths[label])
        skip_list = set(results[label].keys())
        new_met = metrics_computer(fs, deg, ref, skip_list=skip_list)
        results[label].update(new_met)

    save_results(results, results_path, summary_path)
