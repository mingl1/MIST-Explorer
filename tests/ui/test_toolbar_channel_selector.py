from types import SimpleNamespace

from PyQt6.QtWidgets import QWidget

from ui.toolbar.toolbar_ui import ToolBarUI


def _build_channels(count: int):
    return {f"Channel {i}": SimpleNamespace(name=f"Channel {i}") for i in range(1, count + 1)}


def _make_toolbar(qtbot):
    parent = QWidget()
    toolbar = ToolBarUI(parent)
    # Keep parent alive for the toolbar's lifetime in this test.
    toolbar._test_parent = parent
    qtbot.addWidget(parent)
    return toolbar


def test_programmatic_update_does_not_emit_channel_changed(qapp, qtbot):
    toolbar = _make_toolbar(qtbot)
    emitted = []
    toolbar.channelChanged.connect(emitted.append)

    toolbar.updateChannelSelector(_build_channels(3))

    assert emitted == []


def test_pending_index_applies_after_repopulation(qapp, qtbot):
    toolbar = _make_toolbar(qtbot)
    emitted = []
    toolbar.channelChanged.connect(emitted.append)

    toolbar.setChannelSelector(2)
    toolbar.updateChannelSelector(_build_channels(4))

    assert toolbar.channelSelector.currentIndex() == 2
    assert emitted == []


def test_channel_selector_repopulation_has_no_duplicates(qapp, qtbot):
    toolbar = _make_toolbar(qtbot)

    channels = _build_channels(3)
    toolbar.updateChannelSelector(channels)
    toolbar.updateChannelSelector(channels)

    item_texts = [toolbar.channelSelector.itemText(i) for i in range(toolbar.channelSelector.count())]
    assert toolbar.channelSelector.count() == len(channels)
    assert len(set(item_texts)) == len(channels)
