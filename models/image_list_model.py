from PyQt6.QtCore import Qt
from PyQt6.QtGui import QPixmap, QIcon

from core import ImageStorage
from utils import numpy_to_qimage
from PyQt6.QtGui import QStandardItemModel, QStandardItem
from PyQt6.QtCore import QSize


class ImageTreeItem(QStandardItem):
    def __init__(self, uuid, channel="Channel 1", useItemName=False):
        super().__init__()
        self.storage = ImageStorage()
        image_dict = self.storage.get_data(uuid)
        if not image_dict:
            raise ValueError(f"No image data found for UUID: {uuid}")
        name = image_dict.get("name", "")
        text = name if useItemName else channel
        data = image_dict.get("data", {})
        icon = numpy_to_qimage(data[channel].data)

        if useItemName:
            thumbnail = QPixmap(icon).scaled(50, 50, Qt.AspectRatioMode.KeepAspectRatio)
            self.setData(QSize(0, 60), Qt.ItemDataRole.SizeHintRole)
            self.setEditable(True)
            self.setFlags(
                Qt.ItemFlag.ItemIsEnabled
                | Qt.ItemFlag.ItemIsSelectable
                | Qt.ItemFlag.ItemIsEditable
            )
        else:
            thumbnail = QPixmap(icon).scaled(30, 30, Qt.AspectRatioMode.KeepAspectRatio)
            self.setData(QSize(0, 40), Qt.ItemDataRole.SizeHintRole)
            self.setFlags(Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsSelectable)

        icon = QIcon(thumbnail)
        self.setIcon(icon)
        self.setText(text)
        self.setData(uuid, Qt.ItemDataRole.UserRole)
        self.setData(channel, Qt.ItemDataRole.ToolTipRole)

    def onTextEdited(self, new_text):
        self.storage.update_name(self.data(Qt.ItemDataRole.UserRole), new_text)

    def setData(self, value, role=Qt.ItemDataRole.UserRole):
        super().setData(value, role)
        if role == Qt.ItemDataRole.EditRole:
            self.onTextEdited(value)


class ImageTreeModel(QStandardItemModel):
    def __init__(self, images=None):
        super().__init__()
