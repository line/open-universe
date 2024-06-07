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
# Table plotting script

This script can gather the results from several model and print
them in a table.
Any format supported by the [tabulate](https://pypi.org/project/tabulate/) package
can be used.

This script was used to plot the example table in the README.

Author: Robin Scheibler (@fakufaku)
"""
import argparse
import json
from pathlib import Path
from tabulate import tabulate


def get_metric(dic, name):
    name_underscore = name.replace("-", "_")
    name_hyphen = name.replace("_", "-")
    if name_underscore in dic:
        return dic[name_underscore]
    elif name_hyphen in dic:
        return dic[name_hyphen]
    else:
        return None


def read_results(path, metrics):
    with open(path, "r") as f:
        data = json.load(f)

    return [get_metric(data, met) for met in metrics]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-f", "--format", type=str, default="plain", help="Format of the table"
    )
    parser.add_argument(
        "--metrics",
        "-m",
        nargs="+",
        default=["si-sdr", "pesq-wb", "stoi-ext", "lsd", "lps", "OVRL", "SIG", "BAK"],
        help="Metrics to include in the table",
    )
    parser.add_argument(
        "--results",
        "-r",
        nargs="+",
        type=Path,
        required=True,
        help="Paths to the results summary files",
    )
    parser.add_argument(
        "--labels", "-l", nargs="+", help="Labels for each of the results"
    )
    args = parser.parse_args()

    if args.labels is None:
        args.labels = [str(path.parent) for path in args.results]
    elif len(args.results) != len(args.labels):
        raise ValueError("Number of labels must match number of results")

    headers = ["model"] + args.metrics
    rows = []
    for label, path in zip(args.labels, args.results):
        rows.append([label] + read_results(path, args.metrics))

    print(tabulate(rows, headers=headers, floatfmt=".3f", tablefmt=args.format))
