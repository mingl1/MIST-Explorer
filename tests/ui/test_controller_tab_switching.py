from types import SimpleNamespace
from unittest.mock import Mock

from controller import Controller


def _make_controller(prev_tab_index: int, uuid_value):
    controller = Controller.__new__(Controller)
    controller.prev_tab_index = prev_tab_index
    controller.view = SimpleNamespace(
        canvas=SimpleNamespace(
            show_images_tab_image=Mock(),
            show_view_tab_image=Mock(),
            set_buttons_visible=Mock(),
        ),
        view_tab=SimpleNamespace(process_images=Mock()),
    )
    controller.model_canvas = SimpleNamespace(
        uuid=uuid_value,
        current_channel="ch-1",
        swap_channel=Mock(),
        clear_canvas=Mock(),
    )
    return controller


def test_need_canvas_change_preserves_camera_view_to_analysis():
    controller = _make_controller(prev_tab_index=1, uuid_value="abc")

    controller.need_canvas_change(2)

    controller.view.canvas.show_view_tab_image.assert_not_called()
    controller.view.view_tab.process_images.assert_not_called()
    controller.view.canvas.show_images_tab_image.assert_not_called()
    controller.model_canvas.swap_channel.assert_not_called()
    controller.model_canvas.clear_canvas.assert_not_called()


def test_need_canvas_change_preserves_camera_analysis_to_view():
    controller = _make_controller(prev_tab_index=2, uuid_value="abc")

    controller.need_canvas_change(1)

    controller.view.canvas.show_view_tab_image.assert_not_called()
    controller.view.view_tab.process_images.assert_not_called()
    controller.view.canvas.show_images_tab_image.assert_not_called()
    controller.model_canvas.swap_channel.assert_not_called()
    controller.model_canvas.clear_canvas.assert_not_called()


def test_need_canvas_change_extract_to_view_refreshes_view():
    controller = _make_controller(prev_tab_index=0, uuid_value="abc")

    controller.need_canvas_change(1)

    controller.view.canvas.show_view_tab_image.assert_called_once_with()
    controller.view.view_tab.process_images.assert_called_once_with()
    controller.view.canvas.show_images_tab_image.assert_not_called()
    controller.model_canvas.swap_channel.assert_not_called()
    controller.model_canvas.clear_canvas.assert_not_called()


def test_need_canvas_change_view_to_extract_swaps_channel_when_uuid_exists():
    controller = _make_controller(prev_tab_index=1, uuid_value="abc")

    controller.need_canvas_change(0)

    controller.view.canvas.show_images_tab_image.assert_called_once_with()
    controller.model_canvas.swap_channel.assert_called_once_with("ch-1")
    controller.model_canvas.clear_canvas.assert_not_called()
    controller.view.canvas.show_view_tab_image.assert_not_called()
    controller.view.view_tab.process_images.assert_not_called()


def test_need_canvas_change_view_to_extract_clears_canvas_when_uuid_missing():
    controller = _make_controller(prev_tab_index=1, uuid_value=None)

    controller.need_canvas_change(0)

    controller.view.canvas.show_images_tab_image.assert_called_once_with()
    controller.model_canvas.clear_canvas.assert_called_once_with()
    controller.model_canvas.swap_channel.assert_not_called()
    controller.view.canvas.show_view_tab_image.assert_not_called()
    controller.view.view_tab.process_images.assert_not_called()
