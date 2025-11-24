import os
from dataclasses import dataclass
from io import BytesIO
from typing import Any

import datasets
import numpy as np
import PIL.Image
import polars as pl
import torch
import torch.utils.data
import transformers
import sys
sys.path.extend(['.', '..'])

from src.utils import Truncation

IMAGE_EQUIV_TOKENS = 4.0  # NOTE: time: 1 image token = 4 text tokens
PATCH_SIZE = 28
IGNORE_INDEX = -100
PAD_TO_MULTIPLE_OF = 32


def get_n_tokens_in_image(bytes):
    try:
        img = PIL.Image.open(BytesIO(bytes))
        height, width = img.size
        n_tokens = ((height - 1) // PATCH_SIZE + 1) * ((width - 1) // PATCH_SIZE + 1)
        return n_tokens
    except Exception as e:
        print(f"Error in get_n_tokens_in_image: {e}, returning 0 tokens.")
        return 0


def get_sample_cost(sample, tokenizer):
    texts = sample["text"].split("<|vision_start|><|image_pad|><|vision_end|>")
    tokenization = tokenizer(texts)
    text_tokens = sum([len(tokens) for tokens in tokenization.input_ids])

    image_tokens = sum([get_n_tokens_in_image(image) for image in sample["images"]])
    n_images = len(sample["images"])

    return {
        "text_tokens": text_tokens,
        "image_tokens": image_tokens,
        "n_images": n_images,
    }


@dataclass
class Concat:
    data: list[datasets.Dataset]

    def __post_init__(self):
        self.cum_len = np.cumsum([0] + [len(d) for d in self.data])

    def __getitem__(self, idx):
        i = np.searchsorted(self.cum_len, idx, side="right") - 1
        return self.data[i][int(idx - self.cum_len[i])]

    def __len__(self):
        return self.cum_len[-1]


@dataclass
class Dataset(torch.utils.data.Dataset):
    datasets: Any
    stats_dir: str
    max_equiv_tokens: int = 1024
    micro_batch_size: int = 4096
    world_size: int = 4
    seed: int = 1
    tokenizer_id: str = "Qwen/Qwen2.5-VL-3B-Instruct"

    def __post_init__(self):
        self._epoch = None

        cost_threshold = (
            pl.DataFrame({"threshold": self.get_cost_threshold()})
            .with_row_index("group_index")
            .with_columns(
                micro_batch_size=(self.micro_batch_size // pl.col("threshold"))
            )
        )

        data, stats = self.load()
        self.data = data

        df = stats.select(
            cost=(
                pl.col("text_tokens")
                + pl.col("image_tokens") * IMAGE_EQUIV_TOKENS
                + pl.col("n_images") * 2
            ).clip(upper_bound=self.max_equiv_tokens),
            index=pl.col("index"),
        )
        groups = (
            df.select(
                index=pl.col("index"),
                group_index=cost_threshold["threshold"].search_sorted(df["cost"]),
            )
            .group_by("group_index", maintain_order=True)
            .agg(index=pl.col("index"), count=pl.col("index").len())
            .join(cost_threshold, on="group_index")
            .sort("group_index")
            .drop("group_index")
            .with_columns(
                n_global_batches=pl.col("count")
                // pl.col("micro_batch_size")
                // self.world_size,
            )
        )

        if groups.filter(pl.col("micro_batch_size") == 0)["count"].sum() > 0:
            raise ValueError("Some groups have micro_batch_size of 0")

        groups = groups.filter(pl.col("micro_batch_size") > 0)

        self.groups = groups

    def load(self):
        all_data = []
        all_stats = []
        tokenizer = transformers.AutoTokenizer.from_pretrained(self.tokenizer_id)

        offset = 0
        for d in self.datasets:
            print(f"Loading dataset: {d['name']} {d['split']} from {d['source']}")
            if d["source"] == "hf":
                data = datasets.load_dataset(
                    d["name"],
                    split=d["split"],
                    download_config=datasets.DownloadConfig(num_proc=10),
                )
            elif d["source"] == "local":
                dataset = datasets.load_from_disk(d["name"])
                if d["split"] in dataset:
                    data = dataset[d["split"]]
                else:
                    data = dataset
            else:
                assert False
            print(f"load {len(data)} samples for {d['name']}")

            # Extract only the last part of the path as dataset name
            dataset_name = d["name"].split("/")[-1]
            stats_dir = f"{self.stats_dir}/{dataset_name}_{d['split']}"
            if os.path.exists(stats_dir):
                stats = (
                    datasets.load_from_disk(stats_dir)[d["split"]]
                    if d["split"] in datasets.load_from_disk(stats_dir)
                    else datasets.load_from_disk(stats_dir)
                )
            else:
                stats = data.map(
                    lambda x: get_sample_cost(sample=x, tokenizer=tokenizer),
                    remove_columns=["text", "images"],
                    num_proc=16,  # NOTE: can be larger
                )
                stats.save_to_disk(stats_dir)
            stats: pl.DataFrame = stats.to_polars().with_columns(
                index=pl.arange(offset, offset + len(data)),
            )
            if "sample" in d:
                if d["sample"] == "-1" or d["sample"] > len(data):
                    stats = stats
                    print(f"Sample size {d['sample']}, using all data")
                else:
                    stats = stats.sample(
                        d["sample"], with_replacement=False, seed=self.seed
                    )
                    print(f"Sample {d['sample']} from {len(data)}")

            offset += len(data)
            all_data.append(data)
            all_stats.append(stats)

        data, stats = Concat(all_data), pl.concat(all_stats)

        return data, stats

    def get_cost_threshold(self):
        rate = 1.3  # NOTE: can be adjusted
        n_bins = 100
        return sorted(set(PAD_TO_MULTIPLE_OF * int(rate**i) for i in range(n_bins)))

    def __len__(self):
        return self.groups["n_global_batches"].sum() * self.world_size

    def set_epoch(self, epoch):
        if self._epoch == epoch:
            return

        self._epoch = epoch
        seed = self.seed + epoch

        batches = []
        for i in range(len(self.groups)):
            n_global_batches = self.groups["n_global_batches"][i]
            if n_global_batches == 0:
                continue

            micro_batch_size = self.groups["micro_batch_size"][i]
            global_batch_size = micro_batch_size * self.world_size
            n_samples_needed = n_global_batches * global_batch_size
            index = (
                self.groups["index"][i]
                .sample(
                    n_samples_needed, with_replacement=False, shuffle=True, seed=seed
                )
                .rename("index")
            )
            index = index.reshape([-1, global_batch_size]).cast(pl.List(int))
            batches.append(index)

        self.batches = pl.concat(batches).sample(fraction=1, shuffle=True, seed=seed)

    def __getitem__(self, index):
        global_batch_idx = index // self.world_size
        global_indices = self.batches[global_batch_idx]
        assert len(global_indices) % self.world_size == 0

        indices = global_indices[index % self.world_size :: self.world_size]
        return [self.data[i] for i in indices]
