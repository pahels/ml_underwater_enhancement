# training/train.py
from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image
import torch
import torch.nn as nn
from torch.optim import Adam
from tqdm import tqdm

from datasets.uieb.dataloader import UIEBConfig, make_uieb_loader
from models.unet import UNet
from models.losses import PerceptualLoss, color_consistency_loss


def tensor_to_pil(x: torch.Tensor) -> Image.Image:
    """
    Convert a (3,H,W) float tensor in [0,1] to a PIL RGB image.
    """
    x = x.detach().clamp(0, 1).cpu().permute(1, 2, 0).numpy()
    x = (x * 255.0).astype(np.uint8)
    return Image.fromarray(x)


def save_triplet(raw: torch.Tensor, pred: torch.Tensor, gt: torch.Tensor, out_path: Path) -> None:
    """
    Save a side-by-side panel: RAW | PRED | GT.
    Each input is (3,H,W) float tensor in [0,1].
    """
    raw_im = tensor_to_pil(raw)
    pred_im = tensor_to_pil(pred)
    gt_im = tensor_to_pil(gt)

    w, h = raw_im.size
    canvas = Image.new("RGB", (w * 3, h))
    canvas.paste(raw_im, (0, 0))
    canvas.paste(pred_im, (w, 0))
    canvas.paste(gt_im, (w * 2, 0))
    canvas.save(out_path)


def main():
    # ---------------- device ----------------
    device = (
        "mps" if torch.backends.mps.is_available()
        else "cuda" if torch.cuda.is_available()
        else "cpu"
    )
    print("Using device:", device)

    # ---------------- output dirs ----------------
    run_dir = Path("results/train_runs/run1")
    ckpt_dir = run_dir / "checkpoints"
    samples_root = run_dir / "samples"

    ckpt_dir.mkdir(parents=True, exist_ok=True)
    samples_root.mkdir(parents=True, exist_ok=True)

    # ---------------- data ----------------
    cfg = UIEBConfig(
        root="datasets/uieb",
        split="train",
        size=256,
        limit=200,  # set to None for full dataset
    )

    loader = make_uieb_loader(
        cfg,
        batch_size=4,
        shuffle=True,
        num_workers=0,
    )

    # ---------------- model ----------------
    model = UNet().to(device)

    # ---------------- losses ----------------
    l1_loss = nn.L1Loss()
    perceptual = PerceptualLoss().to(device)

    # ---------------- optimizer ----------------
    optimizer = Adam(model.parameters(), lr=1e-4)

    # ---------------- resume config ----------------
    # Keep epoch 39 results, but resume training to 60 epochs.
    resume = True
    resume_ckpt = Path("results/train_runs/run1/checkpoints/unet_epoch39.pt")
    start_epoch = 60  # next epoch after the checkpoint
    epochs = 75       # train up to this epoch

    if resume:
        if not resume_ckpt.exists():
            raise FileNotFoundError(f"Resume checkpoint not found: {resume_ckpt}")
        print(f"Resuming from checkpoint: {resume_ckpt}")
        ckpt = torch.load(resume_ckpt, map_location="cpu")
        model.load_state_dict(ckpt["model_state"])
        optimizer.load_state_dict(ckpt["optimizer_state"])
    else:
        start_epoch = 1

    # ---------------- training ----------------
    for epoch in range(start_epoch, epochs + 1):
        model.train()
        running_loss = 0.0

        for raw, gt, _ in tqdm(loader, desc=f"Epoch {epoch}/{epochs}"):
            raw = raw.to(device)
            gt = gt.to(device)

            pred = model(raw)

            loss_l1 = l1_loss(pred, gt)
            loss_perc = perceptual(pred, gt)
            loss_color = color_consistency_loss(pred)

            # tuned weights (avoid gray collapse)
            loss = loss_l1 + 0.3 * loss_perc + 0.2 * loss_color

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        avg_loss = running_loss / max(1, len(loader))
        print(f"Epoch {epoch} avg loss: {avg_loss:.4f}")

        # ---------------- save checkpoint ----------------
        torch.save(
            {
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "avg_loss": avg_loss,
                "config": cfg.__dict__,
            },
            ckpt_dir / f"unet_epoch{epoch}.pt",
        )

        # ---------------- save samples ----------------
        epoch_dir = samples_root / f"epoch_{epoch:02d}"
        epoch_dir.mkdir(parents=True, exist_ok=True)

        model.eval()
        with torch.no_grad():
            raw_b, gt_b, names = next(iter(loader))
            raw_b = raw_b.to(device)
            gt_b = gt_b.to(device)
            pred_b = model(raw_b)

            for i in range(min(4, raw_b.size(0))):
                save_triplet(
                    raw_b[i],
                    pred_b[i],
                    gt_b[i],
                    epoch_dir / f"{names[i]}_raw_pred_gt.png",
                )

    print("Training finished.")
    print(f"Checkpoints saved to: {ckpt_dir}")
    print(f"Samples saved to:     {samples_root}")


if __name__ == "__main__":
    main()
