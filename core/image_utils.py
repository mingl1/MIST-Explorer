import cv2
import numpy as np
from numpy.typing import NDArray


def scale_adjust(arr: np.ndarray) -> NDArray[np.uint8]:
    if arr.dtype == np.uint16:
        return cv2.convertScaleAbs(arr, alpha=(255.0 / 65535.0))
    elif arr.dtype == np.uint8:
        return np.clip(arr, 0, 255)
    elif arr.dtype == np.uint32:
        max_val = arr.max()
        if max_val == 0:
            return np.zeros_like(arr, dtype=np.uint8)
        return ((arr / max_val) * 255).astype(np.uint8)
    elif arr.dtype in [np.float32, np.float64]:
        arr_clipped = np.clip(arr, 0, None)  # remove negative values
        max_val = arr_clipped.max()
        if max_val > 0:
            arr_scaled = (arr_clipped / max_val) * 255
        else:
            arr_scaled = arr_clipped
        return np.clip(arr_scaled, 0, 255).astype(np.uint8)

    else:
        raise ValueError(f"Unsupported array type: {arr.dtype}")


def create_lut(new_min, new_max):
    lut = np.zeros(256, dtype=np.uint8)  # uint8 for display
    lut[new_min: new_max + 1] = np.linspace(
        start=0,
        stop=255,
        num=(new_max - new_min + 1),
        endpoint=True,
        dtype=np.uint8,
    )
    lut[:new_min] = 0  # clip between 0 and 255
    lut[new_max + 1:] = 255

    return lut


def auto_contrast_helper(
        img,
        lower=1.0,
        upper=99.0,
        zero_eps=None,
        min_span=5):
    """
    Auto-contrast using robust percentiles on foreground (non-zero) pixels.
    lower/upper are *percentiles* (e.g., 1.0, 99.0).
    """
    if img.size == 0:
        return 0, 255

    img = scale_adjust(img)

    vals = np.asarray(img, dtype=np.float32).ravel()
    if vals.size < 16:
        return 0, 255

    vmax_all = np.nanmax(vals)
    if not np.isfinite(vmax_all) or vmax_all <= 0:
        return 0, 255

    # Decide what "black" means based on the image scale
    # If img in 0..1 -> ignore values <= 1e-6; if 0..255 -> ignore <= 0.5
    if zero_eps is None:
        zero_eps = 1e-6 if vmax_all <= 1.5 else 0.5

    # Foreground mask: ignore zeros/near-zeros
    fg = vals[vals > zero_eps]
    if fg.size < vals.size * 0.01:
        # Not enough foreground — fall back to full range
        vmin, vmax = 0.0, vmax_all
    else:
        vmin, vmax = np.percentile(fg, [lower, upper])

        # Guard against degenerate spans (e.g., almost all zeros + a few
        # same-valued pixels)
        if vmax - vmin < min_span:
            # widen to something usable without blowing out the image
            vmin = max(0.0, vmin - 0.5 * min_span)
            vmax = min(vmax_all, vmin + min_span)

    return vmin, vmax


def adjustContrast(img, alpha=5, beta=15):
    alpha = 5  # Contrast control
    beta = 15  # Brightness control
    return cv2.convertScaleAbs(img, alpha=alpha, beta=beta)
