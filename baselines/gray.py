# baselines/gray_world.py
import numpy as np
import cv2

def gray_world_bgr(img_bgr: np.ndarray) -> np.ndarray:
    """
    Gray-world white balance.
    Input/Output: uint8 BGR image.

    So basically all the pixels shuold average out to a gray color
    Underwater images lose red wavelengths first, so we boost red channel while decreasing blue and green
    
    """
    img = img_bgr.astype(np.float32)
    b, g, r = cv2.split(img)

    b_mean, g_mean, r_mean = b.mean(), g.mean(), r.mean()
    gray = (b_mean + g_mean + r_mean) / 3.0

    b *= gray / (b_mean + 1e-6)
    g *= gray / (g_mean + 1e-6)
    r *= gray / (r_mean + 1e-6)

    out = cv2.merge([b, g, r])
    return np.clip(out, 0, 255).astype(np.uint8)
