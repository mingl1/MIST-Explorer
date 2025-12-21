from PyQt6.QtCore import QSize, Qt, pyqtSignal
from PyQt6.QtGui import QIcon, QPixmap, QStandardItem, QStandardItemModel

from core import ImageStorage
from utils import numpy_to_qimage


class ImageTreeItem(QStandardItem):
    def __init__(self, uuid, channel="Channel 1", useItemName=False, image_ready=False):
        super().__init__()
        self.storage = ImageStorage()
        image_dict = self.storage.get_data(uuid)
        if not image_dict:
            raise ValueError(f"No image data found for UUID: {uuid}, type: {type(uuid)}")
        name = image_dict.get("name", "")
        text = name if useItemName else channel
        data = image_dict.get("data", {})
        self.useItemName = useItemName
        self.channel = channel
        if not data:
            image_ready = False
        else:
            image_ready = True
        if image_ready:
            icon = numpy_to_qimage(data[channel].data)

            if useItemName:
                thumbnail = QPixmap(icon).scaled(
                    50, 50, Qt.AspectRatioMode.KeepAspectRatio
                )
                self.setData(QSize(0, 60), Qt.ItemDataRole.SizeHintRole)
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
                thumbnail = QPixmap(icon).scaled(
                    30, 30, Qt.AspectRatioMode.KeepAspectRatio
                )
                self.setData(QSize(0, 40), Qt.ItemDataRole.SizeHintRole)
                self.setFlags(Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsSelectable)

            icon = QIcon(thumbnail)
            self.setIcon(icon)
        self.setText(text)
        self.setData(uuid, Qt.ItemDataRole.UserRole)
        self.setData(channel, Qt.ItemDataRole.WhatsThisRole)

    def set_icon(self, data=None):
        """Set the icon for the item. Always a child"""
        print(
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
        icon = numpy_to_qimage(channel_data.data)
        if self.useItemName:
            thumbnail = QPixmap(icon).scaled(50, 50, Qt.AspectRatioMode.KeepAspectRatio)
            self.setData(QSize(0, 60), Qt.ItemDataRole.SizeHintRole)
        else:
            thumbnail = QPixmap(icon).scaled(30, 30, Qt.AspectRatioMode.KeepAspectRatio)
            self.setData(QSize(0, 40), Qt.ItemDataRole.SizeHintRole)
        if thumbnail is None:
            raise ValueError("Failed to create thumbnail from image data")
        icon = QIcon(thumbnail)
        self.setIcon(icon)

    def onTextEdited(self, new_text):
        self.storage.update_name(self.data(Qt.ItemDataRole.UserRole), new_text)

    def setData(self, value, role=Qt.ItemDataRole.UserRole):
        super().setData(value, role)
        if role == Qt.ItemDataRole.EditRole:
            self.onTextEdited(value)


class ImageTreeModel(QStandardItemModel):
    def __init__(self, images=None):
        super().__init__()
