from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import QComboBox, QHBoxLayout, QLabel, QWidget

from core.project_naming import is_segmentation_channel


class SegmentationImageSelector(QWidget):
    """Shared image + channel selector for the Segmentation tab.

    Both Gaussian Blur and StarDist use the image/channel chosen here instead
    of implicitly operating on whatever is currently loaded on the canvas.
    """

    image_changed = pyqtSignal(str)   # emits UUID string
    channel_changed = pyqtSignal(str) # emits channel key, e.g. "Channel 1"

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 4, 0, 4)

        layout.addWidget(QLabel("Image"))
        self.image_combo = QComboBox()
        self.image_combo.setMinimumWidth(120)
        layout.addWidget(self.image_combo)

        self.channel_label = QLabel("Channel")
        self.channel_combo = QComboBox()
        layout.addWidget(self.channel_label)
        layout.addWidget(self.channel_combo)

        self._updating = False
        self._manually_set = False
        self._set_channel_visible(False)

        self.image_combo.currentIndexChanged.connect(self._on_image_changed)
        self.channel_combo.currentIndexChanged.connect(self._on_channel_changed)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def refresh_images(self, image_tree_model):
        """Rebuild image list from *image_tree_model* (top-level rows only).

        Preserves the current UUID selection if the image is still present.
        Defaults to "(None)" placeholder when no prior selection exists.
        Does NOT emit signals — caller decides whether to fire image_changed.
        """
        self._updating = True
        prev_uuid = self.current_uuid()
        self.image_combo.clear()

        # Placeholder — keeps default state as "no selection"
        self.image_combo.addItem("(None)", userData=None)

        for i in range(image_tree_model.rowCount()):
            item = image_tree_model.item(i)
            if item is None:
                continue
            uuid = str(item.data(Qt.ItemDataRole.UserRole))
            name = item.text()
            self.image_combo.addItem(name, userData=uuid)

        # Restore previous selection if still present
        if prev_uuid:
            for i in range(self.image_combo.count()):
                if self.image_combo.itemData(i) == prev_uuid:
                    self.image_combo.setCurrentIndex(i)
                    break

        self._updating = False

    def set_channels(self, channels: dict):
        """Populate channel dropdown from *channels* dict (ImageWrapper values).

        Filters out segmentation channels. Hides the channel row when only
        one non-segmentation channel exists (single-channel image).
        """
        self._updating = True
        prev = self.channel_combo.currentText()
        self.channel_combo.clear()

        non_seg = [
            k for k, w in channels.items()
            if not is_segmentation_channel(w)
        ]
        non_seg.sort(key=lambda x: int(x.replace("Channel ", "")))
        self.channel_combo.addItems(non_seg)

        if prev in non_seg:
            self.channel_combo.setCurrentText(prev)

        self._set_channel_visible(len(non_seg) > 1)
        self._updating = False

    def current_uuid(self) -> str | None:
        """Return the UUID of the currently selected image, or None."""
        return self.image_combo.currentData()  # None for "(None)" placeholder or empty combo

    def set_selected_uuid_silent(self, uuid: str) -> bool:
        """Select image by uuid without emitting image_changed. Returns True if found.

        Also marks the selector as manually set so canvas auto-follow is disabled.
        """
        self._manually_set = True
        self._updating = True
        found = False
        for i in range(self.image_combo.count()):
            if self.image_combo.itemData(i) == uuid:
                self.image_combo.setCurrentIndex(i)
                found = True
                break
        self._updating = False
        return found

    def follow_canvas_uuid(self, uuid: str) -> None:
        """Update selector to uuid if the user has not manually overridden it.

        Emits image_changed so downstream handlers refresh channels etc.
        No-op if the user has already made a manual selection.
        """
        if self._manually_set:
            return
        self._updating = True
        found = False
        for i in range(self.image_combo.count()):
            if self.image_combo.itemData(i) == uuid:
                self.image_combo.setCurrentIndex(i)
                found = True
                break
        self._updating = False
        if found:
            self.image_changed.emit(uuid)

    def set_current_channel_silent(self, channel_key: str):
        """Select channel by key without emitting channel_changed."""
        self._updating = True
        self.channel_combo.setCurrentText(channel_key)
        self._updating = False

    def current_channel(self) -> str:
        """Return the currently selected channel key (defaults to 'Channel 1')."""
        return self.channel_combo.currentText() or "Channel 1"

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def hide_channel_selector(self):
        """Permanently hide the channel selector (image-only mode)."""
        self._set_channel_visible(False)
        # Prevent set_channels from ever showing it again
        self._channel_hidden = True

    def _set_channel_visible(self, visible: bool):
        if getattr(self, "_channel_hidden", False):
            return
        self.channel_label.setVisible(visible)
        self.channel_combo.setVisible(visible)

    def _on_image_changed(self, _index: int):
        if self._updating:
            return
        uuid = self.image_combo.currentData()
        if uuid:
            self._manually_set = True
            self.image_changed.emit(str(uuid))

    def _on_channel_changed(self, _index: int):
        if self._updating:
            return
        channel = self.channel_combo.currentText()
        if channel:
            self.channel_changed.emit(channel)
