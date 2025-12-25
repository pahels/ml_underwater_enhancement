# baselines/clahe.py
import numpy as np
import cv2

def clahe_lab_bgr(
    img_bgr: np.ndarray,
    clip_limit: float = 2.0,
    tile_grid_size=(8, 8),
) -> np.ndarray:
    """
    CLAHE applied to L channel in LAB color space. - Lightness (black-white), red-green, blue-yellow
    Input/Output: uint8 BGR image.

    Enhances local contrast, which can help bring out details in underwater images.
    Split image into tiles and apply histogram equalization to each tile.

    Makes edges more visible
    Recovers details in murky regions
    Improves feature detection

    Can amplify noise 
    """
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    clahe = cv2.createCLAHE(
        clipLimit=clip_limit,
        tileGridSize=tile_grid_size,
    )
    l2 = clahe.apply(l)

    lab2 = cv2.merge([l2, a, b])
    return cv2.cvtColor(lab2, cv2.COLOR_LAB2BGR)

