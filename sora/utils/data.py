"""
Multi-resolution bucket DataLoader for SOAR training.

Reads a JSONL file with per-image metadata:
    {"md5": "...", "caption_en": "...", "bw": 512, "bh": 512}

Returns batches of {"pixel_values": [B,3,H,W], "prompts": list[str]}.
All images in a batch share the same (bw, bh) resolution.
"""

import json
import os
import random
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, Sampler
from torchvision import transforms


def _parse_caption(raw: str) -> str:
    if not raw or str(raw) in ("nan", "None", ""):
        return ""
    raw = str(raw).strip()
    if not raw.startswith("{"):
        return raw
    try:
        obj = json.loads(raw)
        for key in ("long_caption", "medium_caption", "short_caption", "text"):
            if key in obj and obj[key]:
                return str(obj[key])
        return raw
    except (json.JSONDecodeError, TypeError):
        return raw


class _Sample:
    __slots__ = ("md5", "bucket", "caption")

    def __init__(self, md5: str, bucket: Tuple[int, int], caption: str):
        self.md5 = md5
        self.bucket = bucket
        self.caption = caption


class BucketDataset(Dataset):
    def __init__(
        self,
        samples: List[_Sample],
        image_dir: str,
        bucket_to_indices: Dict[Tuple[int, int], List[int]],
        random_flip: bool = True,
    ):
        self.samples = samples
        self.image_dir = image_dir
        self.bucket_to_indices = bucket_to_indices
        self.random_flip = random_flip
        self._flip = transforms.RandomHorizontalFlip(p=1.0)
        self._to_tensor = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ])

    def __len__(self):
        return len(self.samples)

    @staticmethod
    def resize_and_crop(image: Image.Image, target_wh: Tuple[int, int]) -> Image.Image:
        tw, th = target_wh
        w, h = image.size
        tr = th / tw
        r = h / w
        if r < tr:
            rh, rw = th, int(round(th / h * w))
        else:
            rw, rh = tw, int(round(tw / w * h))
        image = image.resize((rw, rh), Image.LANCZOS)
        top = random.randint(0, max(0, rh - th))
        left = random.randint(0, max(0, rw - tw))
        return image.crop((left, top, left + tw, top + th))

    def __getitem__(self, idx):
        s = self.samples[idx]
        for _attempt in range(10):
            try:
                img_path = os.path.join(self.image_dir, f"{s.md5}.jpg")
                image = Image.open(img_path).convert("RGB")
                break
            except (FileNotFoundError, OSError):
                peers = self.bucket_to_indices.get(s.bucket, [idx])
                s = self.samples[random.choice(peers)]
        else:
            image = Image.new("RGB", s.bucket, (0, 0, 0))

        image = self.resize_and_crop(image, s.bucket)
        if self.random_flip and random.random() < 0.5:
            image = self._flip(image)
        return {
            "pixel_values": self._to_tensor(image),
            "prompt": _parse_caption(s.caption),
            "bucket": s.bucket,
        }


class BucketBatchSampler(Sampler):
    """Yields batches where every sample shares the same resolution bucket.
    Uniform per-sample probability across all buckets."""

    def __init__(
        self,
        samples: List[_Sample],
        bucket_to_indices: Dict[Tuple, List[int]],
        batch_size: int,
        rank: int = 0,
        world_size: int = 1,
        seed: int = 0,
        num_steps: Optional[int] = None,
    ):
        self.batch_size = batch_size
        self.rank = rank
        self.world_size = world_size
        self.seed = seed
        self.bucket_to_indices = bucket_to_indices
        self._bucket_list = list(bucket_to_indices.keys())
        counts = np.array(
            [len(bucket_to_indices[b]) for b in self._bucket_list], dtype=np.float64
        )
        self._bucket_weights = counts / counts.sum()
        total = sum(len(v) for v in bucket_to_indices.values())
        self._num_steps = num_steps or (total // batch_size // world_size)

    def __len__(self):
        return self._num_steps

    def __iter__(self):
        shared_rng = np.random.RandomState(self.seed)
        local_rng = np.random.RandomState(self.seed + self.rank + 1)

        pools: Dict[Tuple, List[int]] = {}
        for bucket, indices in self.bucket_to_indices.items():
            arr = indices.copy()
            local_rng.shuffle(arr)
            per_rank = len(arr) // self.world_size
            if per_rank < 1:
                if self.rank == 0:
                    pools[bucket] = arr
                continue
            start = self.rank * per_rank
            pools[bucket] = list(arr[start : start + per_rank])

        cursors: Dict[Tuple, int] = {k: 0 for k in pools}
        active_buckets = [b for b in self._bucket_list if b in pools and pools[b]]
        if not active_buckets:
            return
        weights = np.array(
            [len(pools[b]) for b in active_buckets], dtype=np.float64
        )
        weights /= weights.sum()

        for _ in range(self._num_steps):
            b_idx = shared_rng.choice(len(active_buckets), p=weights)
            bucket = active_buckets[b_idx]
            pool = pools[bucket]
            cursor = cursors[bucket]
            batch = []
            while len(batch) < self.batch_size:
                if cursor >= len(pool):
                    local_rng.shuffle(pool)
                    cursor = 0
                take = min(self.batch_size - len(batch), len(pool) - cursor)
                batch.extend(pool[cursor : cursor + take])
                cursor += take
            cursors[bucket] = cursor
            yield batch

    def set_epoch(self, epoch: int):
        self.seed = self.seed + epoch


def _collate_fn(examples):
    pixel_values = torch.stack([ex["pixel_values"] for ex in examples])
    pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()
    return {
        "pixel_values": pixel_values,
        "prompts": [ex["prompt"] for ex in examples],
        "target_size": examples[0]["bucket"],
    }


def build_bucket_dataloader(
    jsonl_path: str,
    image_dir: str,
    batch_size: int = 4,
    rank: int = 0,
    world_size: int = 1,
    num_workers: int = 8,
    random_flip: bool = True,
    seed: int = 0,
) -> DataLoader:
    """Build a DataLoader from JSONL with bucket-resolution batching."""
    print(f"[BucketDataLoader] Loading {jsonl_path}...", flush=True)
    samples = []
    bucket_to_indices: Dict[Tuple[int, int], List[int]] = defaultdict(list)

    with open(jsonl_path) as f:
        for line in f:
            d = json.loads(line)
            md5 = d["md5"]
            caption = d.get("caption_en", "")
            bw, bh = d["bw"], d["bh"]
            if not caption:
                continue
            idx = len(samples)
            samples.append(_Sample(md5, (bw, bh), caption))
            bucket_to_indices[(bw, bh)].append(idx)

    bucket_to_indices = dict(bucket_to_indices)
    print(
        f"[BucketDataLoader] {len(samples):,} samples, "
        f"{len(bucket_to_indices)} buckets",
        flush=True,
    )

    dataset = BucketDataset(
        samples=samples,
        image_dir=image_dir,
        bucket_to_indices=bucket_to_indices,
        random_flip=random_flip,
    )
    sampler = BucketBatchSampler(
        samples=samples,
        bucket_to_indices=bucket_to_indices,
        batch_size=batch_size,
        rank=rank,
        world_size=world_size,
        seed=seed,
    )
    loader = DataLoader(
        dataset,
        batch_sampler=sampler,
        num_workers=num_workers,
        collate_fn=_collate_fn,
        pin_memory=True,
        persistent_workers=num_workers > 0,
    )
    print(
        f"[BucketDataLoader] {len(sampler)} batches/epoch "
        f"(bs={batch_size}, rank={rank}/{world_size})",
        flush=True,
    )
    return loader
