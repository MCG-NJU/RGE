#!/usr/bin/env python3
"""Collect accuracy values from *_score.json files within experiment directories.
"""

from __future__ import annotations

import json
import os
import sys
from typing import Iterable, Mapping


DATASETS = [
    "ImageNet-1K",
    "N24News",
    "HatefulMemes",
    "VOC2007",
    "SUN397",
    "Place365",
    "ImageNet-A",
    "ImageNet-R",
    "ObjectNet",
    "Country211",
    "OK-VQA",
    "A-OKVQA",
    "DocVQA",
    "InfographicsVQA",
    "ChartQA",
    "Visual7W",
    "ScienceQA",
    "VizWiz",
    "GQA",
    "TextVQA",
    "VisDial",
    "CIRR",
    "VisualNews_t2i",
    "VisualNews_i2t",
    "MSCOCO_t2i",
    "MSCOCO_i2t",
    "NIGHTS",
    "WebQA",
    "FashionIQ",
    "Wiki-SS-NQ",
    "OVEN",
    "EDIS",
    "MSCOCO",
    "RefCOCO",
    "RefCOCO-Matching",
    "Visual7W-Pointing",
]


EXPERIMENT_DIRS = [
    "outputs/RGE-eval",
]


def load_json(path: str) -> Mapping[str, object] | None:
    try:
        with open(path, "r", encoding="utf-8") as file:
            return json.load(file)
    except FileNotFoundError:
        return None
    except json.JSONDecodeError:
        return None


def collect_acc_values(directory: str, datasets: Iterable[str]) -> list[str]:
    values: list[str] = []
    for dataset in datasets:
        filename = f"{dataset}_score.json"
        path = os.path.join(directory, filename)
        data = load_json(path)
        if data is None:
            values.append("NA")
            continue
        acc = data.get("acc")
        if acc is None:
            values.append("NA")
            continue
        values.append(str(acc))
    return values


def main(argv: list[str]) -> int:
    header = ["Experiment", *DATASETS]
    print("\t".join(header))

    directories = argv or EXPERIMENT_DIRS
    if not directories:
        print("(No experiment directories configured)", file=sys.stderr)
        return 1

    for directory in directories:
        acc_values = collect_acc_values(directory, DATASETS)
        experiment_name = os.path.basename(os.path.normpath(directory)) or directory
        row = [experiment_name, *acc_values]
        print("\t".join(row))

    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

