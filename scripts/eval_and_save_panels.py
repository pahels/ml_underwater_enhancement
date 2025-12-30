# scripts/eval_and_save_panels.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Tuple, List, Optional

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F

import cv2
from skimage.metrics import structural_similarity as ssim

from datasets.uieb.dataloader import UIEBConfig, make_uieb_loader
from models.unet import UNet


# -------------------- helpers --------------------

def tensor_to_uint8_rgb(x: torch.Tensor) -> np.ndarray:
    """
    x: (3,H,W) float in [0,1] -> uint8 RGB (H,W,3)
    """
    x = x.detach().clamp(0, 1).cpu().permute(1, 2, 0).numpy()
    return (x * 255.0).astype(np.uint8)


def uint8_rgb_to_tensor(img: np.ndarray) -> torch.Tensor:
    """
    img: uint8 RGB (H,W,3) -> (3,H,W) float in [0,1]
    """
    x = img.astype(np.float32) / 255.0
    return torch.from_numpy(x).permute(2, 0, 1).contiguous()


def apply_clahe_rgb(rgb: np.ndarray, clip_limit: float = 2.0, tile_grid: Tuple[int, int] = (8, 8)) -> np.ndarray:
    """
    CLAHE on luminance channel (LAB L).
    rgb: uint8 RGB (H,W,3)
    returns uint8 RGB
    """
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid)
    l2 = clahe.apply(l)
    lab2 = cv2.merge([l2, a, b])
    out = cv2.cvtColor(lab2, cv2.COLOR_LAB2RGB)
    return out


def save_quad(raw_rgb: np.ndarray, clahe_rgb: np.ndarray, pred_rgb: np.ndarray, gt_rgb: np.ndarray, out_path: Path) -> None:
    """
    Save: RAW | CLAHE | U-NET | GT (single row).
    """
    h, w, _ = raw_rgb.shape
    canvas = Image.new("RGB", (w * 4, h))
    canvas.paste(Image.fromarray(raw_rgb),   (0, 0))
    canvas.paste(Image.fromarray(clahe_rgb), (w, 0))
    canvas.paste(Image.fromarray(pred_rgb),  (w * 2, 0))
    canvas.paste(Image.fromarray(gt_rgb),    (w * 3, 0))
    canvas.save(out_path)


def psnr(pred: torch.Tensor, gt: torch.Tensor, eps: float = 1e-8) -> float:
    """
    pred, gt: (3,H,W) float in [0,1]
    """
    mse = torch.mean((pred - gt) ** 2).item()
    if mse < eps:
        return 99.0
    return float(10.0 * np.log10(1.0 / mse))


def ssim_rgb(pred_u8: np.ndarray, gt_u8: np.ndarray) -> float:
    """
    pred_u8, gt_u8: uint8 RGB (H,W,3)
    Compute SSIM on grayscale to keep it simple/robust.
    """
    pred_g = cv2.cvtColor(pred_u8, cv2.COLOR_RGB2GRAY)
    gt_g   = cv2.cvtColor(gt_u8, cv2.COLOR_RGB2GRAY)
    return float(ssim(pred_g, gt_g, data_range=255))


def find_latest_checkpoint(ckpt_dir: Path) -> Path:
    ckpts = sorted(ckpt_dir.glob("unet_epoch*.pt"))
    if not ckpts:
        raise FileNotFoundError(f"No checkpoints found in {ckpt_dir}")
    # sort by epoch number if possible
    def epoch_num(p: Path) -> int:
        s = p.stem.replace("unet_epoch", "")
        try:
            return int(s)
        except:
            return -1
    ckpts = sorted(ckpts, key=epoch_num)
    return ckpts[-1]


# -------------------- main --------------------

def main():
    device = (
        "mps" if torch.backends.mps.is_available()
        else "cuda" if torch.cuda.is_available()
        else "cpu"
    )
    print("Using device:", device)

    # ---- paths ----
    ckpt_dir = Path("results/train_runs/run1/checkpoints")
    ckpt_path = Path("results/train_runs/run1/checkpoints/unet_epoch58.pt")
    print("Loading checkpoint:", ckpt_path)

    out_root = Path("results/eval/epoch58")
    panels_dir = out_root / "panels"
    panels_dir.mkdir(parents=True, exist_ok=True)

    # ---- data (test split) ----
    cfg = UIEBConfig(split="test", size=256, limit=None)
    loader = make_uieb_loader(cfg, batch_size=1, shuffle=False, num_workers=0)

    # ---- model ----
    model = UNet().to(device)
    ckpt = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    # ---- eval loop ----
    psnr_u, ssim_u = [], []
    psnr_c, ssim_c = [], []

    max_save = 50  # number of panels to save (set to None for all)
    saved = 0

    with torch.no_grad():
        for raw, gt, name in loader:
            raw = raw.to(device)  # (1,3,H,W)
            gt  = gt.to(device)

            pred = model(raw)      # (1,3,H,W)

            raw_ = raw[0]
            gt_  = gt[0]
            pred_ = pred[0]

            # convert to uint8 RGB
            raw_u8  = tensor_to_uint8_rgb(raw_)
            gt_u8   = tensor_to_uint8_rgb(gt_)
            pred_u8 = tensor_to_uint8_rgb(pred_)

            # CLAHE baseline from RAW
            clahe_u8 = apply_clahe_rgb(raw_u8)

            # metrics
            psnr_u.append(psnr(pred_, gt_))
            ssim_u.append(ssim_rgb(pred_u8, gt_u8))

            # CLAHE metrics (convert clahe -> tensor)
            clahe_t = uint8_rgb_to_tensor(clahe_u8).to(gt_.device)
            psnr_c.append(psnr(clahe_t, gt_))
            ssim_c.append(ssim_rgb(clahe_u8, gt_u8))


            # save panel
            if max_save is None or saved < max_save:
                out_path = panels_dir / f"{name[0]}_RAW_CLAHE_UNET_GT.png"
                save_quad(raw_u8, clahe_u8, pred_u8, gt_u8, out_path)
                saved += 1

    # ---- summary ----
    def mean(xs): return float(np.mean(xs)) if xs else float("nan")

    print("\n=== TEST METRICS (mean) ===")
    print(f"U-NET  PSNR: {mean(psnr_u):.3f}   SSIM: {mean(ssim_u):.4f}")
    print(f"CLAHE  PSNR: {mean(psnr_c):.3f}   SSIM: {mean(ssim_c):.4f}")
    print("\nSaved panels to:", panels_dir)


if __name__ == "__main__":
    main()
