import cv2
import numpy as np
from matplotlib import colormaps
from matplotlib.colors import Colormap
from numpy.typing import NDArray
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QIcon, QImage, QPixmap


def scale_adjust(arr: np.ndarray) -> NDArray[np.uint8]:
    if arr.dtype == np.uint16:
        return cv2.convertScaleAbs(arr, alpha=(255.0 / 65535.0))
    elif arr.dtype == np.uint8:
        return np.clip(arr, 0, 255)
    elif arr.dtype in [np.uint32, np.int32, np.int64, np.uint64]:
        arr_clipped = np.clip(arr, 0, None)
        max_val = arr_clipped.max()
        if max_val == 0:
            return np.zeros_like(arr, dtype=np.uint8)
        return ((arr_clipped / max_val) * 255).astype(np.uint8)
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


def generate_lut(cmap: str):
    """Generate an 8-bit lookup table for a matplotlib colormap (BGR output)."""
    color_map: Colormap = colormaps.get_cmap(cmap)
    label_range = np.linspace(0, 1, 256)
    temp = color_map(label_range)
    temp = temp[:, 2::-1]  # drop alpha and convert RGBA → BGR
    return np.uint8(temp * 256).reshape(256, 1, 3)


def label2rgb(labels, lut):
    """Apply a BGR colormap LUT to a grayscale (or already-3ch) image."""
    if len(labels.shape) == 3 and labels.shape[2] == 3:
        r, g, b = cv2.split(labels)
        return cv2.LUT(cv2.merge((r, g, b)), lut)
    if len(labels.shape) > 2:
        labels = labels[:, :, 0]
    return cv2.LUT(cv2.merge((labels, labels, labels)), lut)


def create_thumbnail(wrapper, size: int = 50) -> QIcon:
    """Create a QIcon thumbnail from an ImageWrapper.

    Expensive work (scale_adjust + downscale to render_size) is cached on
    ``wrapper._thumb_cache`` so that repeated calls caused by contrast/cmap
    changes only execute cheap LUT operations on the already-small array.
    """
    render_size = size * 2  # 2× for Retina sharpness

    # --- Stage 1: build/retrieve the pre-scaled base array (cached) ---
    if render_size not in wrapper._thumb_cache:
        arr = scale_adjust(wrapper.data.copy())
        h, w = arr.shape[:2]
        if h > 0 and w > 0 and max(h, w) > render_size:
            scale = render_size / max(h, w)
            new_h = max(1, int(h * scale))
            new_w = max(1, int(w * scale))
            arr = cv2.resize(arr, (new_w, new_h), interpolation=cv2.INTER_AREA)
        wrapper._thumb_cache[render_size] = arr
    arr = wrapper._thumb_cache[render_size].copy()

    # --- Stage 2: apply contrast LUT (cheap on small array) ---
    if arr.size > 0:
        vmin_i = int(wrapper.contrast_min)
        vmax_i = int(wrapper.contrast_max)
        if vmax_i > vmin_i:
            arr = create_lut(vmin_i, vmax_i)[arr]

    # --- Stage 3: apply colormap (cheap on small array) ---
    cmap = wrapper.cmap
    if cmap is not None and cmap not in ("gray", "label_image"):
        arr = label2rgb(arr, generate_lut(cmap))
        arr = cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)

    qimage = numpy_to_qimage(arr)
    thumbnail = QPixmap(qimage).scaled(
        render_size, render_size,
        Qt.AspectRatioMode.KeepAspectRatio,
        Qt.TransformationMode.SmoothTransformation,
    )
    return QIcon(thumbnail)


def adjustContrast(img, alpha=5, beta=15):
    alpha = 5  # Contrast control
    beta = 15  # Brightness control
    return cv2.convertScaleAbs(img, alpha=alpha, beta=beta)

def auto_contrast(img):
    return adjustContrast(scale_adjust(img))
def to_pixmap(data: QPixmap | np.ndarray | QImage):
    """Sends a pixmap to the canvas for display"""
    # convert pixmap to pixmapItem
    pixmap = None
    if isinstance(data, QPixmap):
        pixmap = data
    elif isinstance(data, QImage):
        pixmap = QPixmap(data)
    elif isinstance(data, np.ndarray):
        pixmap = QPixmap(numpy_to_qimage(data))
    assert pixmap is not None
    return pixmap
def numpy_to_qimage(array: np.ndarray) -> QImage:
    if not array.data.contiguous:
        array = np.ascontiguousarray(array)

    qimage = None
    if len(array.shape) == 2:
        # Grayscale image
        height, width = array.shape
        format = (
            QImage.Format.Format_Grayscale16
            if array.dtype == np.uint16
            else QImage.Format.Format_Grayscale8
        )
        bytes_per_pixel = 2 if array.dtype == np.uint16 else 1
        bytes_per_line = width * bytes_per_pixel  # uint8
        qimage = QImage(array.data, width, height, bytes_per_line, format)
    elif len(array.shape) == 3:
        height, width, channels = array.shape
        if channels == 3:
            # RGB image
            qimage = QImage(
                array.data, width, height, width * channels, QImage.Format.Format_RGB888
            )
        elif channels == 4:
            # RGBA image
            qimage = QImage(
                array.data,
                width,
                height,
                width * channels,
                QImage.Format.Format_RGBA8888,
            )
    else:
        raise ValueError("Unsupported array shape: {}".format(array.shape))
    if qimage is None:
        raise ValueError("Failed to create QImage from numpy array")
    return qimage if qimage is not None else QImage()