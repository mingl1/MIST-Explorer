import numpy as np
import pytest
from unittest.mock import MagicMock, patch
from core.canvas import ImageStorage, ImageWrapper, MemoryEfficientImageCache, ImageGraphicsView


def test_image_storage():
    storage = ImageStorage()
    storage.add_data("1", {"data": np.zeros((10, 10))})
    assert len(storage) == 1
    retrieved_data = storage.get_data("1")
    assert retrieved_data is not None
    assert "data" in retrieved_data
    storage.remove_data("1")
    assert len(storage) == 0

def test_image_wrapper():
    data = np.zeros((10, 10), dtype=np.uint16)
    wrapper = ImageWrapper(data, name="test", cmap="viridis")
    assert wrapper.name == "test"
    assert wrapper.cmap == "viridis"
    assert wrapper.data.dtype == np.uint16
    uint8_data = wrapper.get_uint8_data()
    assert uint8_data.dtype == np.uint8
    uint16_data = wrapper.get_uint16_data()
    assert uint16_data.dtype == np.uint16

def test_memory_efficient_image_cache():
    cache = MemoryEfficientImageCache(max_cache_size_mb=1)
    data1 = np.zeros((100, 100), dtype=np.uint8)
    data2 = np.zeros((200, 200), dtype=np.uint8)
    cache.put("uuid1", "channel1", "key1", data1)
    assert cache.get("uuid1", "channel1", "key1") is not None
    cache.put("uuid1", "channel1", "key2", data2)
    # This should evict key1
    assert cache.get("uuid1", "channel1", "key1") is not None

@pytest.fixture
def image_graphics_view(qtbot):
    with patch("core.canvas.ImageStorage") as mock_storage, patch("core.canvas.Worker") as mock_worker:
        controller = MagicMock()
        view = ImageGraphicsView(controller)
        qtbot.addWidget(view)
        return view

def test_set_uuid(image_graphics_view):
    image_graphics_view.set_uuid("test_uuid")
    assert image_graphics_view.uuid == "test_uuid"

def test_clear_canvas(image_graphics_view):
    image_graphics_view.clear_canvas()
    assert image_graphics_view.uuid is None
    assert len(image_graphics_view.working_channels) == 0

def test_swap_channel(image_graphics_view):
    image_graphics_view.working_channels = {"Channel 1": ImageWrapper(np.zeros((10, 10)), "Channel 1")}
    image_graphics_view.swap_channel(0)
    assert image_graphics_view.current_channel == 0

def test_update_contrast(image_graphics_view):
    image_graphics_view.image_wrapper = ImageWrapper(np.zeros((10, 10)), "test")
    with patch.object(image_graphics_view, "update_image") as mock_update_image:
        image_graphics_view.update_contrast((0, 255))
        mock_update_image.assert_called_once()

def test_add_to_canvas(image_graphics_view):
    with patch.object(image_graphics_view, "_process_new_file") as mock_process_new_file:
        image_graphics_view.add_to_canvas("test.png")
        mock_process_new_file.assert_called_once_with("test.png")

def test_reset_image(image_graphics_view):
    image_graphics_view.reset_working_channels = {"Channel 1": ImageWrapper(np.zeros((10, 10)), "Channel 1")}
    with patch.object(image_graphics_view, "update_image") as mock_update_image:
        image_graphics_view.reset_image()
        mock_update_image.assert_called_once()

def test_rotate_image(image_graphics_view):
    image_graphics_view.working_channels = {"Channel 1": ImageWrapper(np.zeros((10, 10)), "Channel 1")}
    with patch("core.canvas.Worker") as mock_worker:
        image_graphics_view.rotate_image("90")
        mock_worker.assert_called_once()

def test_auto_contrast(image_graphics_view):
    image_graphics_view.image_wrapper = ImageWrapper(np.zeros((10, 10)), "test")
    with patch.object(image_graphics_view, "update_contrast") as mock_update_contrast:
        image_graphics_view.auto_contrast()
        mock_update_contrast.assert_called_once()

def test_blur_layer(image_graphics_view):
    image_graphics_view.working_channels = {"Channel 1": ImageWrapper(np.zeros((10, 10)), "Channel 1")}
    with patch("core.canvas.Worker") as mock_worker:
        image_graphics_view.blur_layer(0.5)
        mock_worker.assert_called_once()

def test_crop(image_graphics_view):
    image_graphics_view.image_wrapper = ImageWrapper(np.zeros((10, 10)), "test")
    rect = MagicMock()
    rect.left.return_value = 0
    rect.top.return_value = 0
    rect.right.return_value = 5
    rect.bottom.return_value = 5
    with patch("core.canvas.ImageDialog") as mock_dialog:
         patch("core.canvas.Worker") as mock_worker:
        mock_dialog.return_value.confirm_crop = True
        image_graphics_view.crop(rect)
        mock_worker.assert_called_once()

def test_flip_horizontal(image_graphics_view):
    image_graphics_view.working_channels = {"Channel 1": ImageWrapper(np.zeros((10, 10)), "Channel 1")}
    with patch.object(image_graphics_view, "set_pixmap") as mock_set_pixmap:
        image_graphics_view.flip_horizontal()
        mock_set_pixmap.assert_called_once()

def test_flip_vertical(image_graphics_view):
    image_graphics_view.working_channels = {"Channel 1": ImageWrapper(np.zeros((10, 10)), "Channel 1")}
    with patch.object(image_graphics_view, "set_pixmap") as mock_set_pixmap:
        image_graphics_view.flip_vertical()
        mock_set_pixmap.assert_called_once()