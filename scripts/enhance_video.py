# scripts/enhance_video.py
from __future__ import annotations

import argparse
import shutil
import subprocess
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from models.unet import UNet


def load_model(ckpt_path: Path, device: str) -> UNet:
    model = UNet().to(device)
    ckpt = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model


def unsharp_mask(bgr: np.ndarray, amount: float = 0.6, radius: float = 1.2) -> np.ndarray:
    """Light sharpening to counter 'soft' neural output."""
    if amount <= 0:
        return bgr
    blur = cv2.GaussianBlur(bgr, (0, 0), radius)
    sharp = cv2.addWeighted(bgr, 1.0 + amount, blur, -amount, 0)
    return np.clip(sharp, 0, 255).astype(np.uint8)


@torch.no_grad()
def enhance_frame_bgr_fullres(frame_bgr: np.ndarray, model: UNet, device: str, pad_mult: int = 16) -> np.ndarray:
    """
    Full-resolution inference:
      - Convert frame to tensor
      - Reflect-pad to multiple of pad_mult
      - Run model
      - Crop back to original size
    """
    h, w = frame_bgr.shape[:2]

    # BGR -> RGB
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

    # uint8 -> float tensor in [0,1], (1,3,H,W)
    x = torch.from_numpy(rgb).float().div(255.0)
    x = x.permute(2, 0, 1).unsqueeze(0).contiguous().to(device)

    # pad to multiple of pad_mult
    pad_h = (pad_mult - (h % pad_mult)) % pad_mult
    pad_w = (pad_mult - (w % pad_mult)) % pad_mult
    if pad_h or pad_w:
        x = F.pad(x, (0, pad_w, 0, pad_h), mode="reflect")

    # forward
    y = model(x).clamp(0, 1)

    # crop back
    y = y[:, :, :h, :w][0]

    # tensor -> uint8 RGB
    out_rgb = (y.permute(1, 2, 0).cpu().numpy() * 255.0).astype(np.uint8)

    # RGB -> BGR
    out_bgr = cv2.cvtColor(out_rgb, cv2.COLOR_RGB2BGR)
    return out_bgr


def run_ffmpeg_encode(frames_dir: Path, out_path: Path, fps: float, crf: int, preset: str) -> None:
    """
    Encode %06d.png frames to H.264 mp4 using ffmpeg at high quality.
    """
    if shutil.which("ffmpeg") is None:
        print("\nffmpeg not found. Install with: brew install ffmpeg\n")
        print("Then run:\n"
              f"ffmpeg -framerate {fps} -i {frames_dir}/%06d.png "
              f"-c:v libx264 -preset {preset} -crf {crf} -pix_fmt yuv420p {out_path}\n")
        return

    cmd = [
        "ffmpeg", "-y",
        "-framerate", str(fps),
        "-i", str(frames_dir / "%06d.png"),
        "-c:v", "libx264",
        "-preset", preset,
        "-crf", str(crf),
        "-pix_fmt", "yuv420p",
        str(out_path),
    ]
    print("\nEncoding with ffmpeg:\n", " ".join(cmd), "\n")
    subprocess.run(cmd, check=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="Input video (.mov/.mp4/etc.)")
    ap.add_argument("--ckpt", default="results/train_runs/run1/checkpoints/unet_best.pt", help="Checkpoint path")
    ap.add_argument("--out_dir", default="results/eval/video_run", help="Output directory for frames/videos")

    ap.add_argument("--max_frames", type=int, default=-1, help="Process first N frames (-1 = all)")
    ap.add_argument("--pad_mult", type=int, default=16, help="Pad multiple (usually 16 or 32)")

    # Optional: downscale for speed while keeping aspect ratio (0 disables)
    ap.add_argument("--max_dim", type=int, default=0,
                    help="If >0, resize so max(H,W)=max_dim for faster processing (keeps aspect ratio)")

    # Sharpen
    ap.add_argument("--sharpen_amount", type=float, default=0.6, help="Unsharp mask amount (0 disables)")
    ap.add_argument("--sharpen_radius", type=float, default=1.2, help="Unsharp mask radius")

    # Outputs
    ap.add_argument("--write_frames", action="store_true", help="Write enhanced frames as PNGs")
    ap.add_argument("--write_sbs_frames", action="store_true", help="Write side-by-side PNGs (raw|enh)")
    ap.add_argument("--encode_mp4", action="store_true", help="Encode enhanced frames to mp4 via ffmpeg")
    ap.add_argument("--encode_sbs_mp4", action="store_true", help="Encode side-by-side frames to mp4 via ffmpeg")

    ap.add_argument("--fps", type=float, default=0.0, help="Override FPS (0 = read from video)")
    ap.add_argument("--crf", type=int, default=16, help="ffmpeg quality (lower is better; 16-20 good)")
    ap.add_argument("--preset", default="slow", help="ffmpeg preset: ultrafast/fast/medium/slow")

    args = ap.parse_args()

    device = (
        "mps" if torch.backends.mps.is_available()
        else "cuda" if torch.cuda.is_available()
        else "cpu"
    )
    print("Using device:", device)

    ckpt_path = Path(args.ckpt)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    in_path = Path(args.input)
    if not in_path.exists():
        raise FileNotFoundError(f"Input video not found: {in_path}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    frames_dir = out_dir / "frames_enhanced"
    sbs_frames_dir = out_dir / "frames_sbs"

    if args.write_frames:
        frames_dir.mkdir(parents=True, exist_ok=True)
    if args.write_sbs_frames:
        sbs_frames_dir.mkdir(parents=True, exist_ok=True)

    model = load_model(ckpt_path, device)

    cap = cv2.VideoCapture(str(in_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {in_path}")

    src_fps = cap.get(cv2.CAP_PROP_FPS)
    fps = args.fps if args.fps and args.fps > 0 else (src_fps if src_fps and src_fps > 0 else 30.0)

    w0 = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h0 = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Input resolution: {w0}x{h0}  fps={fps}")

    idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        # Optional resize for speed
        frame_proc = frame
        if args.max_dim and args.max_dim > 0:
            h, w = frame.shape[:2]
            scale = args.max_dim / max(h, w)
            if scale < 1.0:  # only downscale
                new_w = int(round(w * scale))
                new_h = int(round(h * scale))
                frame_proc = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)

        enhanced = enhance_frame_bgr_fullres(frame_proc, model, device, pad_mult=args.pad_mult)
        enhanced = unsharp_mask(enhanced, amount=args.sharpen_amount, radius=args.sharpen_radius)

        # If we processed a resized frame, upscale back to original for side-by-side consistency if desired
        if frame_proc.shape[:2] != frame.shape[:2]:
            enhanced_full = cv2.resize(enhanced, (w0, h0), interpolation=cv2.INTER_CUBIC)
        else:
            enhanced_full = enhanced

        if args.write_frames:
            cv2.imwrite(str(frames_dir / f"{idx:06d}.png"), enhanced_full)

        if args.write_sbs_frames:
            sbs = np.concatenate([frame, enhanced_full], axis=1)
            cv2.imwrite(str(sbs_frames_dir / f"{idx:06d}.png"), sbs)

        idx += 1
        if idx % 30 == 0:
            print(f"Processed {idx} frames...")
        if args.max_frames > 0 and idx >= args.max_frames:
            break

    cap.release()
    print("Done processing frames:", idx)

    # Encode videos (best quality) using ffmpeg
    if args.encode_mp4 and args.write_frames:
        out_mp4 = out_dir / "enhanced_hq.mp4"
        run_ffmpeg_encode(frames_dir, out_mp4, fps=fps, crf=args.crf, preset=args.preset)

    if args.encode_sbs_mp4 and args.write_sbs_frames:
        out_sbs_mp4 = out_dir / "compare_raw_vs_unet_hq.mp4"
        run_ffmpeg_encode(sbs_frames_dir, out_sbs_mp4, fps=fps, crf=args.crf, preset=args.preset)

    print("\nOutputs:")
    if args.write_frames:
        print("  Enhanced frames:", frames_dir)
    if args.write_sbs_frames:
        print("  Side-by-side frames:", sbs_frames_dir)
    if args.encode_mp4 and args.write_frames:
        print("  Enhanced video:", out_dir / "enhanced_hq.mp4")
    if args.encode_sbs_mp4 and args.write_sbs_frames:
        print("  Side-by-side video:", out_dir / "compare_raw_vs_unet_hq.mp4")


if __name__ == "__main__":
    main()
