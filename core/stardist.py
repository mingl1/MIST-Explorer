import os

import cv2 as cv
import numpy as np
from matplotlib import colormaps
from PIL import Image
from pyclesperanto import dilate_labels
from PyQt6.QtCore import QThread, pyqtSignal
from PyQt6.QtWidgets import QFileDialog
from stardist.models import StarDist2D

from core import ImageWrapper
from core.image_utils import create_lut, scale_adjust
from core.project_naming import STARDIST_LABEL_BASE_NAME, prefix_with_project_name

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"


def normalize(x, pmin=3, pmax=99.8, axis=None, clip=False, eps=1e-20, dtype=np.float32):
    """Percentile-based image normalization."""

    mi = np.percentile(x, pmin, axis=axis, keepdims=True)
    ma = np.percentile(x, pmax, axis=axis, keepdims=True)
    return normalize_mi_ma(x, mi, ma, clip=clip, eps=eps, dtype=dtype)


def normalize_mi_ma(x, mi, ma, clip=False, eps=1e-20, dtype=np.float32):
    if dtype is not None:
        x = x.astype(dtype, copy=False)
        mi = dtype(mi) if np.isscalar(mi) else mi.astype(dtype, copy=False)
        ma = dtype(ma) if np.isscalar(ma) else ma.astype(dtype, copy=False)
        eps = dtype(eps)

    try:
        import numexpr

        x = numexpr.evaluate("(x - mi) / ( ma - mi + eps )")
    except ImportError:
        x = (x - mi) / (ma - mi + eps)

    if clip:
        x = np.clip(x, 0, 1)

    return x


class StarDist(QThread):
    stardist_done = pyqtSignal(ImageWrapper, bool, str)
    cell_image_set = pyqtSignal(str, str)
    progress = pyqtSignal(int, str)
    error_signal = pyqtSignal(str)
    cell_channel = pyqtSignal(str)

    def __init__(self):
        super().__init__()
        self.protein_channels = None
        self.np_image = None
        self.project_name = None
        self.is_temp_project = False
        self.params = {
            "channel": "Channel 1",
            "model": "2D_versatile_fluo",
            "percentile_low": 3,
            "percentile_high": 99.80,
            "prob_threshold": 0.48,
            "nms_threshold": 0.3,
            "scale": 1.0,
            "n_tiles": 0,
            "radius": 5,
            "use_contrasted_image": False,
        }
        self.aligned = False
        self.current_model = ""

    def load_cell_image(self, arr):
        self.cell_image = arr
        self.aligned = True

    def __get_cell_image(self):
        if self.protein_channels is None and self.np_image is not None:
            return self.np_image
        if self.protein_channels and self.np_image is None:
            wrapper = self.protein_channels.get(self.params["channel"])
            if wrapper is not None:
                return wrapper.data
        return None

    def _window_image_by_contrast(self, image: np.ndarray, contrast_min, contrast_max):
        image_uint8 = scale_adjust(image)
        cmin = int(np.clip(int(contrast_min), 0, 255))
        cmax = int(np.clip(int(contrast_max), 0, 255))
        if cmax < cmin:
            cmin, cmax = cmax, cmin
        if cmin == cmax:
            if cmax < 255:
                cmax += 1
            elif cmin > 0:
                cmin -= 1
            else:
                return image_uint8.astype(np.uint16) * 257
        lut = create_lut(cmin, cmax)
        contrasted_uint8 = np.clip(cv.LUT(image_uint8, lut), 0, 254, dtype=np.uint8)
        # Keep segmentation input high-bit depth while preserving display-window semantics.
        return contrasted_uint8.astype(np.uint16) * 257

    def _resolve_segmentation_input(self):
        wrapper = None
        image = None

        if self.protein_channels is not None and self.np_image is None:
            wrapper = self.protein_channels.get(self.params["channel"])
            if wrapper is not None:
                image = wrapper.data
        elif self.protein_channels is None and self.np_image is not None:
            image = self.np_image

        if image is None:
            return None

        if not self.params.get("use_contrasted_image", False):
            return image

        if wrapper is None:
            return image

        contrast_min = getattr(wrapper, "contrast_min", None)
        contrast_max = getattr(wrapper, "contrast_max", None)
        if contrast_min is None or contrast_max is None:
            return image

        return self._window_image_by_contrast(image, contrast_min, contrast_max)

    def run(self):
        self._cancel_requested = False  # reset each run

        cell_image = self._resolve_segmentation_input()
        if cell_image is None:
            self._fatal_error_message("No cell image available for processing")
            return
        assert isinstance(cell_image, np.ndarray), "cell_image must be a numpy array"

        self.progress.emit(0, "Starting StarDist")
        if self.current_model != str(self.params["model"]):
            try:
                self.model = StarDist2D.from_pretrained(str(self.params["model"]))
            except Exception as e:
                self._fatal_error_message(f"Model load failed: {e}")
                return
            self.current_model = str(self.params["model"])
        model = self.model

        self.progress.emit(10, "Model loaded")

        try:
            model_input = self._prepare_model_input(cell_image, model)
        except ValueError as e:
            self._fatal_error_message(str(e))
            return

        # normalize input
        norm_img = normalize(
            model_input,
            self.params["percentile_low"],
            self.params["percentile_high"],
        )

        guess_tiles = self.params["n_tiles"]
        if guess_tiles == 0:
            guess_tiles = model._guess_n_tiles(model_input)

        # total number of tiles
        total_tiles = int(guess_tiles[0] * guess_tiles[1])

        # run tile generator
        try:
            labels_gen = model._predict_instances_generator(
                norm_img,
                prob_thresh=self.params["prob_threshold"],
                nms_thresh=self.params["nms_threshold"],
                scale=self.params["scale"],
                n_tiles=guess_tiles,
            )
        except Exception as e:
            self._fatal_error_message(f"Prediction error: {e}")
            return

        # accumulate with progress
        stardist_labels = None
        for i, (labels, *_) in enumerate(labels_gen):
            if self._cancel_requested:
                self.progress.emit(100, "Cancelled")
                return
            stardist_labels = labels  # last one is full image
            pct = 10 + int(70 * max(0, (i - 2) / total_tiles))
            self.progress.emit(pct, f"Processing tile {max(0, i - 2)}/{total_tiles}")

        if stardist_labels is None:
            self._fatal_error_message("No labels produced")
            return

        if self._cancel_requested:
            self.progress.emit(100, "Cancelled")
            return

        # post-processing
        self.progress.emit(95, "Dilating")
        try:
            self.stardist_labels_grayscale = np.array(
                dilate_labels(stardist_labels, radius=self.params["radius"]),
                dtype=np.uint16,
            )
        except Exception as e:
            self._fatal_error_message(
                f"Error during dilation: {e}. You may need to install pocl-opencl-icd (WSL2 users)."
            )
            return

        if self._cancel_requested:
            self.progress.emit(100, "Cancelled")
            return

        # done
        self.progress.emit(100, "StarDist Done")
        result = ImageWrapper(
            self.stardist_labels_grayscale, name="Channel 1", cmap="gray"
        )
        label_name = prefix_with_project_name(
            STARDIST_LABEL_BASE_NAME,
            self.project_name,
            is_temp_project=self.is_temp_project,
        )
        self.stardist_done.emit(result, True, label_name)

    def cancel(self):
        self._cancel_requested = True
        self.progress.emit(99, "Cancelling...")

    def save_image(self):
        file_name, _ = QFileDialog.getSaveFileName(
            None, "Save File", "image.png", "*.png;;*.jpg;;*.tif;; All Files(*)"
        )
        if self.stardist_labels_grayscale is not None:
            Image.fromarray(self.stardist_labels_grayscale).save(file_name)
        else:
            self.error_signal.emit("Cannot save. No stardist labels available")

    def change_cmap(self):
        pass

    def generate_lut(self, cmap: str):
        label_range = np.linspace(0, 1, 256)
        return np.uint8(colormaps[cmap](label_range)[:, 2::-1] * 256).reshape(256, 1, 3)

    def label2rgb(self, labels, lut):
        return cv.LUT(cv.merge((labels, labels, labels)), lut)

    def update_channels(self, protein_channels, _):
        self.np_image = None
        self.protein_channels = protein_channels

    def set_protein_image(self, protein_channels, channel="Channel 1", name=None):
        self.protein_channels = protein_channels
        self.params["channel"] = channel
        self.cell_image_set.emit(name, channel)
        self.np_image = None

    def set_image_to_process(self, np_image):
        self.protein_channels = None
        self.np_image = np_image

    def set_project_context(self, project_name, is_temp_project=False):
        self.project_name = project_name
        self.is_temp_project = bool(is_temp_project)

    def set_channel(self, channel):
        self.params["channel"] = channel

    def set_model(self, model):
        self.params["model"] = model

    def set_percentile_low(self, value):
        self.params["percentile_low"] = value

    def set_percentile_high(self, value):
        self.params["percentile_high"] = value

    def set_prob_thresh(self, value):
        self.params["prob_threshold"] = value

    def set_number_tiles(self, value):
        self.params["n_tiles"] = value

    def set_dilation_radius(self, value):
        self.params["radius"] = value

    def set_nms_thresh(self, value):
        self.params["nms_threshold"] = value

    def set_scale(self, value):
        scale = float(value)
        self.params["scale"] = scale if scale > 0 else 1.0

    def set_num_tiles(self, value):
        self.params["n_tiles"] = value

    def set_dialation_radisu(self, value):
        self.params["radius"] = value

    def set_use_contrasted_image(self, enabled):
        self.params["use_contrasted_image"] = bool(enabled)

    def _fatal_error_message(self, msg):
        self.error_signal.emit(msg)
        self.progress.emit(100, "")

    def _prepare_model_input(self, image: np.ndarray, model):
        axes = model.config.axes
        if "C" not in axes:
            return image

        expected_ndim = len(axes)
        channel_axis = axes.index("C")
        expected_channels = int(model.config.n_channel_in)

        img = image
        if img.ndim == expected_ndim - 1:
            if expected_channels == 1:
                return img
            img = np.expand_dims(img, axis=channel_axis)
            return np.repeat(img, expected_channels, axis=channel_axis)

        if img.ndim != expected_ndim:
            raise ValueError(
                f"Input image has {img.ndim} dims, but model expects {expected_ndim - 1} or {expected_ndim} dims "
                f"(axes={axes})."
            )

        in_channels = img.shape[channel_axis]
        if in_channels == expected_channels:
            return img
        if in_channels == 1 and expected_channels > 1:
            return np.repeat(img, expected_channels, axis=channel_axis)
        if expected_channels == 1 and in_channels > 1:
            return np.mean(img, axis=channel_axis)

        raise ValueError(
            f"Input channels ({in_channels}) are incompatible with model expected channels ({expected_channels}) "
            f"for axes {axes}."
        )
