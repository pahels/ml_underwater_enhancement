# scripts/run_baselines.py
from pathlib import Path
import cv2
from tqdm import tqdm

from baselines.gray import gray_world_bgr
from baselines.clahe import clahe_lab_bgr

RAW_DIR = Path("datasets/uieb/train/raw")
OUT_DIR = Path("results/baselines/train")
OUT_DIR.mkdir(parents=True, exist_ok=True)

def side_by_side(*imgs_bgr):
    
    return cv2.hconcat(imgs_bgr)

def main(limit: int = 60):
    paths = sorted(
        p for p in RAW_DIR.iterdir()
        if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
    )[:limit]

    for p in tqdm(paths, desc="Running baselines"):
        img = cv2.imread(str(p), cv2.IMREAD_COLOR)
        if img is None:
            continue

        gw = gray_world_bgr(img)
        cl = clahe_lab_bgr(img)

        panel = side_by_side(img, gw, cl)
        out_path = OUT_DIR / f"{p.stem}_raw_gw_clahe.png"
        cv2.imwrite(str(out_path), panel)

if __name__ == "__main__":
    main()
