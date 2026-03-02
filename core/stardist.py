import os

import cv2 as cv
import numpy as np
import scipy.ndimage
import skimage.filters
import skimage.segmentation
import centrosome.cpmorphology
import centrosome.smooth
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

SEGMENTATION_METHOD_STARDIST = "StarDist"
SEGMENTATION_METHOD_PRIMARY_OBJECTS = "CellProfiler-like"

PRIMARY_OBJECT_SETTINGS = {
    "threshold_smoothing_scale": 1.3488,
    "threshold_correction_factor": 1.0,
    "threshold_range": (0.0, 1.0),
    "fill_holes_after_thresholding": True,
    "fill_holes_after_declumping": True,
    "automatic_smoothing": True,
    "automatic_maxima_suppression": True,
    "low_res_maxima": True,
    "exclude_border_objects": True,
}


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


def _global_minimum_cross_entropy_threshold(image, mask, correction, tmin, tmax):
    pixels = image[mask]
    if pixels.size == 0:
        threshold = 0.0
    elif np.all(pixels == pixels.ravel()[0]):
        threshold = float(pixels.ravel()[0])
    else:
        unique = np.unique(pixels)
        if unique.size > 1:
            tol = max(float(np.min(np.diff(unique))) / 2.0, 0.5 / 65536.0)
        else:
            tol = 0.5 / 65536.0
        threshold = float(skimage.filters.threshold_li(pixels, tolerance=tol))

    threshold *= float(correction)
    return min(max(threshold, float(tmin)), float(tmax))


def _apply_threshold(image, threshold, mask, smoothing_scale):
    if smoothing_scale == 0:
        return (image >= threshold) & mask, 0.0

    sigma = float(smoothing_scale) / 0.6744 / 2.0
    blurred = centrosome.smooth.smooth_with_function_and_mask(
        image,
        lambda x: scipy.ndimage.gaussian_filter(x, sigma, mode="constant", cval=0),
        mask,
    )
    return (blurred >= threshold) & mask, sigma


def _calc_declump_smoothing_filter_size(min_diameter):
    return 2.35 * float(min_diameter) / 3.5


def _smooth_for_declumping(image, mask, filter_size):
    if filter_size == 0:
        return image

    sigma = filter_size / 2.35
    radius = max(int(float(filter_size) / 2.0), 1)
    filt = (
        1.0
        / np.sqrt(2.0 * np.pi)
        / sigma
        * np.exp(-0.5 * np.arange(-radius, radius + 1) ** 2 / sigma**2)
    )

    def fgaussian(arr):
        out = scipy.ndimage.convolve1d(arr, filt, axis=0, mode="constant")
        return scipy.ndimage.convolve1d(out, filt, axis=1, mode="constant")

    edge = fgaussian(mask.astype(float))
    masked = image.copy()
    masked[~mask] = 0
    smoothed = fgaussian(masked)
    masked[mask] = smoothed[mask] / edge[mask]
    return masked


def _get_maxima(image, labeled_image, maxima_mask, image_resize_factor):
    if image_resize_factor < 1.0:
        shape = np.array(image.shape) * image_resize_factor
        ij = np.mgrid[0 : shape[0], 0 : shape[1]].astype(float) / image_resize_factor
        resized_image = scipy.ndimage.map_coordinates(image, ij)
        resized_labels = scipy.ndimage.map_coordinates(labeled_image, ij, order=0).astype(
            labeled_image.dtype
        )
    else:
        resized_image = image
        resized_labels = labeled_image

    binary_maxima = centrosome.cpmorphology.is_local_maximum(
        resized_image, resized_labels, maxima_mask
    )
    binary_maxima[resized_image <= 0] = 0

    if image_resize_factor < 1.0:
        inv_factor = float(image.shape[0]) / float(binary_maxima.shape[0])
        ij = np.mgrid[0 : image.shape[0], 0 : image.shape[1]].astype(float) / inv_factor
        binary_maxima = scipy.ndimage.map_coordinates(binary_maxima.astype(float), ij) > 0.5

    return centrosome.cpmorphology.binary_shrink(binary_maxima)


def _filter_on_border(labels):
    if labels.max() <= 0:
        return labels

    border_labels = np.concatenate(
        [
            labels[0, :],
            labels[:, 0],
            labels[-1, :],
            labels[:, -1],
        ]
    )
    border_labels = np.unique(border_labels[border_labels > 0])
    if border_labels.size == 0:
        return labels
    labels[np.isin(labels, border_labels)] = 0
    return labels


def _filter_on_size(labels, min_diameter, max_diameter):
    object_count = int(labels.max())
    if object_count <= 0:
        return labels

    areas = scipy.ndimage.sum(
        np.ones(labels.shape),
        labels,
        np.arange(0, object_count + 1, dtype=np.int32),
    )
    areas = np.asarray(areas, dtype=float)

    min_allowed_area = np.pi * (float(min_diameter) ** 2) / 4.0
    max_allowed_area = np.pi * (float(max_diameter) ** 2) / 4.0

    area_image = areas[labels]
    labels[(area_image < min_allowed_area) | (area_image > max_allowed_area)] = 0
    return labels


def _identify_primary_objects_like_cellprofiler(image_2d, min_size, max_size):
    image = np.asarray(image_2d, dtype=np.float32)
    if image.ndim != 2:
        raise ValueError(f"Expected 2D image, got shape {image.shape}")

    if np.nanmax(image) > 1.0 or np.nanmin(image) < 0.0:
        lo, hi = np.percentile(image, [0.5, 99.5])
        if hi > lo:
            image = np.clip((image - lo) / (hi - lo), 0.0, 1.0)
        else:
            image = np.clip(image, 0.0, 1.0)

    settings = PRIMARY_OBJECT_SETTINGS
    mask = np.ones(image.shape, dtype=bool)

    final_threshold = _global_minimum_cross_entropy_threshold(
        image=image,
        mask=mask,
        correction=settings["threshold_correction_factor"],
        tmin=settings["threshold_range"][0],
        tmax=settings["threshold_range"][1],
    )

    binary, _ = _apply_threshold(
        image=image,
        threshold=final_threshold,
        mask=mask,
        smoothing_scale=settings["threshold_smoothing_scale"],
    )

    if settings["fill_holes_after_thresholding"]:
        size_fn = lambda size, _: size < (float(max_size) * float(max_size))
        binary = centrosome.cpmorphology.fill_labeled_holes(binary, size_fn=size_fn)

    labeled, _ = scipy.ndimage.label(binary, np.ones((3, 3), dtype=bool))

    if labeled.max() > 0:
        if settings["automatic_smoothing"]:
            declump_filter_size = _calc_declump_smoothing_filter_size(min_size)
        else:
            declump_filter_size = 10.0

        blurred = _smooth_for_declumping(image, mask, declump_filter_size)

        if min_size > 10 and settings["low_res_maxima"]:
            image_resize_factor = 10.0 / float(min_size)
            maxima_suppression_size = (
                7.0 if settings["automatic_maxima_suppression"] else 7.5
            )
        else:
            image_resize_factor = 1.0
            maxima_suppression_size = (
                float(min_size) / 1.5 if settings["automatic_maxima_suppression"] else 7.0
            )

        maxima_mask = centrosome.cpmorphology.strel_disk(
            max(1, maxima_suppression_size - 0.5)
        )
        maxima = _get_maxima(blurred, labeled, maxima_mask, image_resize_factor)
        labeled_maxima, object_count = scipy.ndimage.label(
            maxima, np.ones((3, 3), dtype=bool)
        )

        if object_count > 0:
            watershed_image = 1.0 - image
            markers_dtype = (
                np.int16 if object_count < np.iinfo(np.int16).max else np.int32
            )
            markers = np.zeros(watershed_image.shape, markers_dtype)
            markers[labeled_maxima > 0] = -labeled_maxima[labeled_maxima > 0]
            watershed_boundaries = skimage.segmentation.watershed(
                connectivity=np.ones((3, 3), dtype=bool),
                image=watershed_image,
                markers=markers,
                mask=labeled != 0,
            )
            labeled = -watershed_boundaries

    if settings["exclude_border_objects"] and labeled.max() > 0:
        labeled = _filter_on_border(labeled)
    if labeled.max() > 0:
        labeled = _filter_on_size(labeled, min_size, max_size)

    if settings["fill_holes_after_declumping"] and labeled.max() > 0:
        labeled = centrosome.cpmorphology.fill_labeled_holes(labeled)

    labeled, _ = centrosome.cpmorphology.relabel(labeled)
    return np.asarray(labeled, dtype=np.uint16)


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
            "segmentation_method": SEGMENTATION_METHOD_STARDIST,
            "model": "2D_versatile_fluo",
            "percentile_low": 3,
            "percentile_high": 99.80,
            "prob_threshold": 0.48,
            "nms_threshold": 0.3,
            "scale": 1.0,
            "n_tiles": 0,
            "radius": 5,
            "use_contrasted_image": False,
            "min_size": 60,
            "max_size": 180,
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

        segmentation_method = self.params.get(
            "segmentation_method", SEGMENTATION_METHOD_STARDIST
        )
        if segmentation_method == SEGMENTATION_METHOD_PRIMARY_OBJECTS:
            self._run_primary_objects(cell_image)
            return

        self._run_stardist(cell_image)

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

    def set_segmentation_method(self, method):
        normalized = str(method).strip()
        if normalized == SEGMENTATION_METHOD_PRIMARY_OBJECTS:
            self.params["segmentation_method"] = SEGMENTATION_METHOD_PRIMARY_OBJECTS
            return
        self.params["segmentation_method"] = SEGMENTATION_METHOD_STARDIST

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

    def set_min_size(self, value):
        parsed = int(value)
        parsed = max(1, parsed)
        self.params["min_size"] = parsed
        if self.params["max_size"] < parsed:
            self.params["max_size"] = parsed

    def set_max_size(self, value):
        parsed = int(value)
        parsed = max(1, parsed)
        self.params["max_size"] = parsed
        if self.params["min_size"] > parsed:
            self.params["min_size"] = parsed

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

    def _emit_segmentation_result(self):
        result = ImageWrapper(self.stardist_labels_grayscale, name="Channel 1", cmap="gray")
        label_name = prefix_with_project_name(
            STARDIST_LABEL_BASE_NAME,
            self.project_name,
            is_temp_project=self.is_temp_project,
        )
        self.stardist_done.emit(result, True, label_name)

    def _run_primary_objects(self, cell_image: np.ndarray):
        self.progress.emit(0, "Starting CellProfiler-like segmentation")
        min_size = int(self.params.get("min_size", 60))
        max_size = int(self.params.get("max_size", 180))
        if max_size < min_size:
            min_size, max_size = max_size, min_size
            self.params["min_size"] = min_size
            self.params["max_size"] = max_size

        try:
            self.stardist_labels_grayscale = _identify_primary_objects_like_cellprofiler(
                cell_image, min_size=min_size, max_size=max_size
            )
        except Exception as exc:
            self._fatal_error_message(f"CellProfiler-like segmentation failed: {exc}")
            return

        if self._cancel_requested:
            self.progress.emit(100, "Cancelled")
            return

        self.progress.emit(100, "CellProfiler-like segmentation done")
        self._emit_segmentation_result()

    def _run_stardist(self, cell_image: np.ndarray):
        self.progress.emit(0, "Starting StarDist")
        if self.current_model != str(self.params["model"]):
            try:
                self.model = StarDist2D.from_pretrained(str(self.params["model"]))
            except Exception as exc:
                self._fatal_error_message(f"Model load failed: {exc}")
                return
            self.current_model = str(self.params["model"])
        model = self.model

        self.progress.emit(10, "Model loaded")

        try:
            model_input = self._prepare_model_input(cell_image, model)
        except ValueError as exc:
            self._fatal_error_message(str(exc))
            return

        norm_img = normalize(
            model_input,
            self.params["percentile_low"],
            self.params["percentile_high"],
        )

        guess_tiles = self.params["n_tiles"]
        if guess_tiles == 0:
            guess_tiles = model._guess_n_tiles(model_input)

        total_tiles = int(guess_tiles[0] * guess_tiles[1])

        try:
            labels_gen = model._predict_instances_generator(
                norm_img,
                prob_thresh=self.params["prob_threshold"],
                nms_thresh=self.params["nms_threshold"],
                scale=self.params["scale"],
                n_tiles=guess_tiles,
            )
        except Exception as exc:
            self._fatal_error_message(f"Prediction error: {exc}")
            return

        stardist_labels = None
        for i, (labels, *_) in enumerate(labels_gen):
            if self._cancel_requested:
                self.progress.emit(100, "Cancelled")
                return
            stardist_labels = labels
            pct = 10 + int(70 * max(0, (i - 2) / total_tiles))
            self.progress.emit(pct, f"Processing tile {max(0, i - 2)}/{total_tiles}")

        if stardist_labels is None:
            self._fatal_error_message("No labels produced")
            return

        if self._cancel_requested:
            self.progress.emit(100, "Cancelled")
            return

        self.progress.emit(95, "Dilating")
        try:
            self.stardist_labels_grayscale = np.array(
                dilate_labels(stardist_labels, radius=self.params["radius"]),
                dtype=np.uint16,
            )
        except Exception as exc:
            self._fatal_error_message(
                f"Error during dilation: {exc}. You may need to install pocl-opencl-icd (WSL2 users)."
            )
            return

        if self._cancel_requested:
            self.progress.emit(100, "Cancelled")
            return

        self.progress.emit(100, "StarDist Done")
        self._emit_segmentation_result()
