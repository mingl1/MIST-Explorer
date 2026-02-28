import cv2 as cv
import numpy as np

from core.canvas import ImageWrapper
from core.image_utils import create_lut, scale_adjust
from core.stardist import StarDist


def test_resolve_segmentation_input_raw_mode():
    model = StarDist()
    wrapper = ImageWrapper(
        np.array([[10, 20], [30, 40]], dtype=np.uint16), name="Channel 1"
    )
    wrapper.contrast_min = 50
    wrapper.contrast_max = 150
    model.update_channels({"Channel 1": wrapper}, False)
    model.set_channel("Channel 1")
    model.set_use_contrasted_image(False)

    resolved = model._resolve_segmentation_input()

    assert resolved is wrapper.data


def test_resolve_segmentation_input_contrasted_mode():
    model = StarDist()
    raw = np.array([[0, 64], [128, 255]], dtype=np.uint8)
    wrapper = ImageWrapper(raw, name="Channel 1")
    wrapper.contrast_min = 64
    wrapper.contrast_max = 192
    model.update_channels({"Channel 1": wrapper}, False)
    model.set_channel("Channel 1")
    model.set_use_contrasted_image(True)

    resolved = model._resolve_segmentation_input()

    expected = (
        np.clip(cv.LUT(scale_adjust(raw), create_lut(64, 192)), 0, 254, dtype=np.uint8)
        .astype(np.uint16)
        * 257
    )
    assert resolved.dtype == np.uint16
    assert np.array_equal(resolved, expected)


def test_resolve_segmentation_input_contrasted_mode_falls_back_for_np_image():
    model = StarDist()
    raw = np.array([[100, 200], [300, 400]], dtype=np.uint16)
    model.set_image_to_process(raw)
    model.set_use_contrasted_image(True)

    resolved = model._resolve_segmentation_input()

    assert resolved is raw


def test_resolve_segmentation_input_uses_selected_channel_contrast():
    model = StarDist()
    ch1 = ImageWrapper(np.array([[0, 32], [64, 96]], dtype=np.uint8), name="Channel 1")
    ch1.contrast_min = 0
    ch1.contrast_max = 64

    ch2_raw = np.array([[0, 128], [192, 255]], dtype=np.uint8)
    ch2 = ImageWrapper(ch2_raw, name="Channel 2")
    ch2.contrast_min = 128
    ch2.contrast_max = 255

    model.update_channels({"Channel 1": ch1, "Channel 2": ch2}, False)
    model.set_channel("Channel 2")
    model.set_use_contrasted_image(True)

    resolved = model._resolve_segmentation_input()

    expected = (
        np.clip(
            cv.LUT(scale_adjust(ch2_raw), create_lut(128, 255)), 0, 254, dtype=np.uint8
        ).astype(np.uint16)
        * 257
    )
    assert np.array_equal(resolved, expected)
