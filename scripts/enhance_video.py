# scripts/enhance_video.py
from __future__ import annotations

from pathlib import Path
import subprocess
import shutil

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from models.unet import UNet

# =========================
# 🔧 USER SETTINGS (EDIT THESE)
# =========================
INPUT_VIDEO = "videos/underwater_v2.mov"   # <-- CHANGE THIS
OUTPUT_DIR  = "results/eval/video_demo"
CHECKPOINT  = "results/train_runs/run1/checkpoints/unet_best.pt"

MAX_FRAMES = -1        # -1 = all frames
PAD_MULT   = 16        # safe for U-Net
SHARPEN_AMOUNT = 0.6
SHARPEN_RADIUS = 1.2
CRF = 16               # ffmpeg quality (lower = better)
PRESET = "slow"        # ffmpeg preset

# =========================


def load_model(ckpt_path: Path, device: str) -> UNet:
    model = UNet().to(device)
    ckpt = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model


def unsharp_mask(bgr: np.ndarray, amount=0.6, radius=1.2) -> np.ndarray:
    if amount <= 0:
        return bgr
    blur = cv2.GaussianBlur(bgr, (0, 0), radius)
    sharp = cv2.addWeighted(bgr, 1 + amount, blur, -amount, 0)
    return np.clip(sharp, 0, 255).astype(np.uint8)


@torch.no_grad()
def enhance_frame(frame_bgr: np.ndarray, model: UNet, device: str) -> np.ndarray:
    h, w = frame_bgr.shape[:2]

    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    x = torch.from_numpy(rgb).float().div(255.0)
    x = x.permute(2, 0, 1).unsqueeze(0).to(device)

    pad_h = (PAD_MULT - h % PAD_MULT) % PAD_MULT
    pad_w = (PAD_MULT - w % PAD_MULT) % PAD_MULT
    if pad_h or pad_w:
        x = F.pad(x, (0, pad_w, 0, pad_h), mode="reflect")

    y = model(x).clamp(0, 1)
    y = y[:, :, :h, :w][0]

    out = (y.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
    out = cv2.cvtColor(out, cv2.COLOR_RGB2BGR)
    return unsharp_mask(out, SHARPEN_AMOUNT, SHARPEN_RADIUS)


def encode_video(frames_dir: Path, out_path: Path, fps: float):
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg not found. Install with: brew install ffmpeg")

    cmd = [
        "ffmpeg", "-y",
        "-framerate", str(fps),
        "-i", str(frames_dir / "%06d.png"),
        "-c:v", "libx264",
        "-crf", str(CRF),
        "-preset", PRESET,
        "-pix_fmt", "yuv420p",
        str(out_path),
    ]
    subprocess.run(cmd, check=True)

@torch.no_grad()
def enhance_frame_tiled(
    frame_bgr: np.ndarray,
    model: UNet,
    device: str,
    tile_size: int = 512,
    overlap: int = 64,
) -> np.ndarray:
    """
    Enhance a frame using overlapping tiles to preserve detail.
    """
    h, w = frame_bgr.shape[:2]
    out = np.zeros_like(frame_bgr, dtype=np.float32)
    weight = np.zeros((h, w, 1), dtype=np.float32)

    step = tile_size - overlap

    for y in range(0, h, step):
        for x in range(0, w, step):
            y0 = y
            x0 = x
            y1 = min(y0 + tile_size, h)
            x1 = min(x0 + tile_size, w)

            tile = frame_bgr[y0:y1, x0:x1]

            # pad tile if near border
            pad_h = tile_size - (y1 - y0)
            pad_w = tile_size - (x1 - x0)
            if pad_h > 0 or pad_w > 0:
                tile = cv2.copyMakeBorder(
                    tile, 0, pad_h, 0, pad_w, cv2.BORDER_REFLECT
                )

            # --- run model ---
            rgb = cv2.cvtColor(tile, cv2.COLOR_BGR2RGB)
            x_t = torch.from_numpy(rgb).float().div(255.0)
            x_t = x_t.permute(2, 0, 1).unsqueeze(0).to(device)

            y_t = model(x_t).clamp(0, 1)[0]
            out_tile = (y_t.permute(1, 2, 0).cpu().numpy() * 255.0).astype(np.float32)
            out_tile = cv2.cvtColor(out_tile, cv2.COLOR_RGB2BGR)

            out_tile = out_tile[: (y1 - y0), : (x1 - x0)]

            out[y0:y1, x0:x1] += out_tile
            weight[y0:y1, x0:x1] += 1.0

    out /= np.maximum(weight, 1e-6)
    return out.astype(np.uint8)


def main():
    device = (
        "mps" if torch.backends.mps.is_available()
        else "cuda" if torch.cuda.is_available()
        else "cpu"
    )
    print("Using device:", device)

    in_path = Path(INPUT_VIDEO)
    if not in_path.exists():
        raise FileNotFoundError(f"Input video not found: {in_path}")

    out_dir = Path(OUTPUT_DIR)
    frames_dir = out_dir / "frames"
    sbs_dir = out_dir / "frames_sbs"

    frames_dir.mkdir(parents=True, exist_ok=True)
    sbs_dir.mkdir(parents=True, exist_ok=True)

    model = load_model(Path(CHECKPOINT), device)

    cap = cv2.VideoCapture(str(in_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        enhanced = enhance_frame_tiled(frame, model, device)

        cv2.imwrite(str(frames_dir / f"{idx:06d}.png"), enhanced)
        cv2.imwrite(
            str(sbs_dir / f"{idx:06d}.png"),
            np.concatenate([frame, enhanced], axis=1)
        )

        idx += 1
        if idx % 30 == 0:
            print(f"Processed {idx} frames")
        if MAX_FRAMES > 0 and idx >= MAX_FRAMES:
            break

    cap.release()
    print("Encoding videos...")

    encode_video(frames_dir, out_dir / "enhanced_hq.mp4", fps)
    encode_video(sbs_dir, out_dir / "compare_raw_vs_unet_hq.mp4", fps)

    print("Done.")
    print("Saved to:", out_dir)


if __name__ == "__main__":
    main()
