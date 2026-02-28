import uuid

import numpy as np
import pytest

from core.canvas import ImageStorage, ImageWrapper
from ui.image_manager import ImageManager


@pytest.fixture(autouse=True)
def _clear_image_storage():
    storage = ImageStorage()
    storage.clear_data()
    yield
    storage.clear_data()


def _add_image(manager: ImageManager, name: str, channels: dict[str, ImageWrapper]):
    image_uuid = uuid.uuid4()
    manager.storage.add_data(str(image_uuid), {"name": name, "data": channels})
    manager.add_item(image_uuid)
    return image_uuid


def _multi_channels():
    return {
        "Channel 1": ImageWrapper(
            np.array([[1, 2], [3, 4]], dtype=np.uint16), name="Channel 1"
        ),
        "Channel 2": ImageWrapper(
            np.array([[5, 6], [7, 8]], dtype=np.uint16), name="Channel 2"
        ),
    }


def test_allowed_exports_for_child_layer_is_tif_and_png(qapp):
    manager = ImageManager()
    _add_image(manager, "multi_layer", _multi_channels())
    root_index = manager.image_tree_model.index(0, 0)
    child_index = manager.image_tree_model.index(0, 0, root_index)

    assert manager.image_tree_view._allowed_export_formats(child_index) == [
        "tif",
        "png",
    ]


def test_allowed_exports_for_stardist_child_is_tif_and_png(qapp):
    manager = ImageManager()
    channels = _multi_channels()
    channels["Channel 3"] = ImageWrapper(
        np.array([[0, 1], [1, 2]], dtype=np.uint16),
        name="StarDist Labels",
        cmap="label_image",
    )
    _add_image(manager, "multi_with_stardist", channels)
    root_index = manager.image_tree_model.index(0, 0)
    stardist_child_index = manager.image_tree_model.index(2, 0, root_index)

    assert manager.image_tree_view._allowed_export_formats(stardist_child_index) == [
        "tif",
        "png",
    ]


def test_allowed_exports_for_single_layer_root_is_tif_and_png(qapp):
    manager = ImageManager()
    _add_image(
        manager,
        "single_layer",
        {
            "Channel 1": ImageWrapper(
                np.array([[10, 20], [30, 40]], dtype=np.uint16), name="Channel 1"
            )
        },
    )
    root_index = manager.image_tree_model.index(0, 0)

    assert manager.image_tree_view._allowed_export_formats(root_index) == ["tif", "png"]


def test_allowed_exports_for_multi_layer_root_is_tif_only(qapp):
    manager = ImageManager()
    _add_image(manager, "multi_layer", _multi_channels())
    root_index = manager.image_tree_model.index(0, 0)

    assert manager.image_tree_view._allowed_export_formats(root_index) == ["tif"]


def test_save_as_png_uses_selected_single_channel(qapp, monkeypatch, tmp_path):
    manager = ImageManager()
    channel_1 = np.array([[1, 2], [3, 4]], dtype=np.uint8)
    channel_2 = np.array([[11, 12], [13, 14]], dtype=np.uint8)
    _add_image(
        manager,
        "multi_layer",
        {
            "Channel 1": ImageWrapper(channel_1, name="Channel 1"),
            "Channel 2": ImageWrapper(channel_2, name="Channel 2"),
        },
    )
    root_index = manager.image_tree_model.index(0, 0)
    child_index = manager.image_tree_model.index(1, 0, root_index)

    monkeypatch.setattr(
        "ui.image_manager.QFileDialog.getExistingDirectory",
        lambda *_args, **_kwargs: str(tmp_path),
    )

    saved = {}

    class _FakeImage:
        def save(self, path):
            saved["path"] = path

    def _fake_fromarray(array):
        saved["array"] = np.array(array, copy=True)
        return _FakeImage()

    monkeypatch.setattr("ui.image_manager.Image.fromarray", _fake_fromarray)

    manager.image_tree_view.save_as(child_index, "png", single_channel="Channel 2")

    assert np.array_equal(saved["array"], channel_2)
    assert not np.array_equal(saved["array"], channel_1)
    assert saved["path"].endswith("Channel 2.png")
