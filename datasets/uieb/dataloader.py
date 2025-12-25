from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader

'''
Preprocessing script

'''
def _pil_to_tensor(img: Image.Image) -> torch.Tensor:
    """PIL RGB -> float tensor in [0,1], shape (3,H,W)."""
    arr = np.array(img).astype(np.float32) / 255.0
    #handles grayscale images
    if arr.ndim == 2:
        arr = np.stack([arr, arr, arr], axis=-1)
    
    #drops extra channels like alpha channels
    if arr.shape[2] == 4:
        arr = arr[:, :, :3]
    return torch.from_numpy(arr).permute(2, 0, 1).contiguous()


@dataclass
class UIEBConfig:
    root: str = "datasets/uieb"
    split: str = "train"  # "train" or "test"
    size: int = 256       # resize to (size, size)
    limit: int | None = None  # for quick tests


class UIEBDataset(Dataset):
    def __init__(self, cfg: UIEBConfig):
        self.cfg = cfg
        root = Path(cfg.root) / cfg.split
        self.raw_dir = root / "raw"
        self.gt_dir = root / "gt"

        if not self.raw_dir.exists() or not self.gt_dir.exists():
            raise FileNotFoundError(
                f"Expected folders:\n  {self.raw_dir}\n  {self.gt_dir}\n"
                "Please create them and put images inside."
            )

        #pairing raw and gt images by filename stem
        raw_paths = sorted([p for p in self.raw_dir.iterdir()
                            if p.suffix.lower() in {'.jpg', '.jpeg', '.png'}])

        gt_map = {p.stem: p for p in self.gt_dir.iterdir()
                  if p.suffix.lower() in {'.jpg', '.jpeg', '.png'}}

        pairs = []
        for rp in raw_paths:
            gp = gt_map.get(rp.stem)
            if gp is not None:
                pairs.append((rp, gp))

        if not pairs:
            raise RuntimeError(
                "No pairs found. Ensure raw and gt filenames match (same stem).\n"
                "Example: raw/0001.png and gt/0001.png"
            )

        if cfg.limit is not None:
            pairs = pairs[: cfg.limit]

        self.pairs = pairs

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, str]:
        raw_path, gt_path = self.pairs[idx]

        raw = Image.open(raw_path).convert("RGB")
        gt = Image.open(gt_path).convert("RGB")

        raw = raw.resize((self.cfg.size, self.cfg.size), Image.BICUBIC)
        gt = gt.resize((self.cfg.size, self.cfg.size), Image.BICUBIC)

        raw_t = _pil_to_tensor(raw)
        gt_t = _pil_to_tensor(gt)
        return raw_t, gt_t, raw_path.stem


def make_uieb_loader(
    cfg: UIEBConfig,
    batch_size: int = 8,
    num_workers: int = 0,
    shuffle: bool = True,
) -> DataLoader:
    ds = UIEBDataset(cfg)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle,
                      num_workers=num_workers, pin_memory=True)
