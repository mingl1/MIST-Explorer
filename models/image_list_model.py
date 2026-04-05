import logging

from PyQt6.QtCore import QSize, Qt
from PyQt6.QtGui import QStandardItem, QStandardItemModel

from core import ImageStorage
from core.image_utils import create_thumbnail

logger = logging.getLogger(__name__)


class ImageTreeItem(QStandardItem):
    def __init__(
        self,
        uuid,
        channel="Channel 1",
        useItemName=False,
        image_ready=False,
        display_text=None,
    ):
        super().__init__()
        self.storage = ImageStorage()
        image_dict = self.storage.get_data(uuid)
        if not image_dict:
            raise ValueError(f"No image data found for UUID: {uuid}, type: {type(uuid)}")
        name = image_dict.get("name", "")
        text = name if useItemName else (display_text or channel)
        data = image_dict.get("data", {})
        self.useItemName = useItemName
        self.channel = channel
        if not data:
            image_ready = False
        else:
            image_ready = True
        if image_ready:
            thumb_size = 50 if useItemName else 30
            row_height = 60 if useItemName else 40
            wrapper = data[channel]
            icon = create_thumbnail(wrapper.data, size=thumb_size,
                                    cmap=wrapper.cmap,
                                    contrast_min=wrapper.contrast_min,
                                    contrast_max=wrapper.contrast_max)
            self.setIcon(icon)
            self.setData(QSize(0, row_height), Qt.ItemDataRole.SizeHintRole)
            if useItemName:
                self.setEditable(True)
                self.setFlags(
                    Qt.ItemFlag.ItemIsEnabled
                    | Qt.ItemFlag.ItemIsSelectable
                    | Qt.ItemFlag.ItemIsEditable
                )
                metadata_text = image_dict.get("metadata", None)
                if metadata_text is not None:
                    self.setData(metadata_text, Qt.ItemDataRole.ToolTipRole)
            else:
                self.setFlags(Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsSelectable)
        self.setText(text)
        self.setData(uuid, Qt.ItemDataRole.UserRole)
        self.setData(channel, Qt.ItemDataRole.WhatsThisRole)

    def set_icon(self, data=None):
        """Set the icon for the item."""
        logger.debug(
            f"Setting icon for {self.data(Qt.ItemDataRole.UserRole)} - {self.channel}"
        )
        if data is None:
            image_dict = self.storage.get_data(self.data(Qt.ItemDataRole.UserRole))
            if image_dict is None:
                raise ValueError(
                    f"No image data found for UUID: {self.data(Qt.ItemDataRole.UserRole)}"
                )
            data = image_dict.get("data", {})
        if not data:
            raise ValueError(
                f"No image data found for UUID: {self.data(Qt.ItemDataRole.UserRole)}"
            )
        channel_data = data.get(self.channel, None)
        if channel_data is None:
            raise ValueError(f"No data found for channel: {self.channel}")
        thumb_size = 50 if self.useItemName else 30
        row_height = 60 if self.useItemName else 40
        icon = create_thumbnail(channel_data.data, size=thumb_size,
                                cmap=channel_data.cmap,
                                contrast_min=channel_data.contrast_min,
                                contrast_max=channel_data.contrast_max)
        self.setIcon(icon)
        self.setData(QSize(0, row_height), Qt.ItemDataRole.SizeHintRole)

    def onTextEdited(self, new_text):
        self.storage.update_name(self.data(Qt.ItemDataRole.UserRole), new_text)

    def setData(self, value, role=Qt.ItemDataRole.UserRole):
        super().setData(value, role)
        if role == Qt.ItemDataRole.EditRole:
            self.onTextEdited(value)


class ImageTreeModel(QStandardItemModel):
    def __init__(self, images=None):
        super().__init__()
