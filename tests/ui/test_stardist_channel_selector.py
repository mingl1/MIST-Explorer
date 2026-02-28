from types import SimpleNamespace

from PyQt6.QtWidgets import QVBoxLayout, QWidget

from ui.stardist.stardist_ui import StarDistUI


def _make_stardist_ui(qtbot):
    parent = QWidget()
    layout = QVBoxLayout(parent)
    stardist_ui = StarDistUI(parent, layout)
    stardist_ui._test_parent = parent
    qtbot.addWidget(parent)
    return stardist_ui


def test_stardist_channel_selector_keeps_previous_selection_on_rebuild(qapp, qtbot):
    stardist_ui = _make_stardist_ui(qtbot)
    channels = {
        "Channel 1": SimpleNamespace(name="Channel 1"),
        "Channel 2": SimpleNamespace(name="Channel 2"),
    }
    stardist_ui.updateChannelSelector(channels)
    stardist_ui.stardist_channel_selector.setCurrentText("Channel 2")

    channels_with_virtual_label = {
        "Channel 1": SimpleNamespace(name="Channel 1"),
        "Channel 2": SimpleNamespace(name="Channel 2"),
        "Channel 3": SimpleNamespace(name="StarDist Labels"),
    }
    stardist_ui.updateChannelSelector(channels_with_virtual_label)

    assert stardist_ui.stardist_channel_selector.currentText() == "Channel 2"


def test_stardist_channel_selector_falls_back_when_previous_selection_missing(qapp, qtbot):
    stardist_ui = _make_stardist_ui(qtbot)
    stardist_ui.updateChannelSelector(
        {
            "Channel 1": SimpleNamespace(name="Channel 1"),
            "Channel 2": SimpleNamespace(name="Channel 2"),
        }
    )
    stardist_ui.stardist_channel_selector.setCurrentText("Channel 2")

    stardist_ui.updateChannelSelector({"Channel 1": SimpleNamespace(name="Channel 1")})

    assert stardist_ui.stardist_channel_selector.currentText() == "Channel 1"
