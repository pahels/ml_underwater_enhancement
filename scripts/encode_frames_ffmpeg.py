# scripts/encode_frames_ffmpeg.py
from __future__ import annotations

import argparse
import shutil
import subprocess
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames_dir", required=True, help="Folder containing %06d.png frames")
    ap.add_argument("--output", required=True, help="Output mp4 path")
    ap.add_argument("--fps", type=float, default=30.0)
    ap.add_argument("--crf", type=int, default=16)
    ap.add_argument("--preset", default="slow")
    args = ap.parse_args()

    frames_dir = Path(args.frames_dir)
    if not frames_dir.exists():
        raise FileNotFoundError(frames_dir)

    if shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg not found. Install: brew install ffmpeg")

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        "ffmpeg", "-y",
        "-framerate", str(args.fps),
        "-i", str(frames_dir / "%06d.png"),
        "-c:v", "libx264",
        "-preset", args.preset,
        "-crf", str(args.crf),
        "-pix_fmt", "yuv420p",
        str(out_path),
    ]
    print(" ".join(cmd))
    subprocess.run(cmd, check=True)
    print("Saved:", out_path)


if __name__ == "__main__":
    main()
