from PyQt6.QtCore import QModelIndex, Qt
from PyQt6.QtGui import QPixmap, QIcon

from core.canvas import ImageStorage
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
            self.setData(QSize(0, 60), Qt.ItemDataRole.SizeHintRole)
            thumbnail = QPixmap(icon).scaled(50, 50, Qt.AspectRatioMode.KeepAspectRatio)

        else:
            thumbnail = QPixmap(icon).scaled(30, 30, Qt.AspectRatioMode.KeepAspectRatio)

            self.setData(QSize(0, 40), Qt.ItemDataRole.SizeHintRole)
        icon = QIcon(thumbnail)
        self.setIcon(icon)
        self.setEditable(True)
        self.setText(text)
        self.setData(uuid, Qt.ItemDataRole.UserRole)
        self.setData(channel, Qt.ItemDataRole.ToolTipRole)
        self.setFlags(Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsSelectable)


class ImageTreeModel(QStandardItemModel):
    def __init__(self, images=None):
        super().__init__()

    #     self.images = images or []
    #     self.storage = ImageStorage()

    # def data(self, index: QModelIndex, role: int = Qt.ItemDataRole.DisplayRole):
    #     if role == Qt.ItemDataRole.DisplayRole:
    #         return self.images[index.row()]["name"]
    #     if role == Qt.ItemDataRole.DecorationRole:
    #         data = self.images[index.row()]["data"]["Channel 1"].data
    #         qimage = numpy_to_qimage(data)
    #         pixmap = QPixmap.fromImage(qimage)
    #         icon = QIcon(pixmap)
    #         return icon

    # def rowCount(self, parent=QModelIndex()) -> int:
    #     return len(self.images)
