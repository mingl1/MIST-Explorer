import os
from pyexpat import model
from PyQt6.QtGui import *
from PyQt6.QtCore import *
from PyQt6.QtWidgets import *
import numpy as np
from core.canvas import ImageStorage
from utils import numpy_to_qimage
from PyQt6.QtCore import pyqtSignal
from models.image_list_model import ImageTreeModel, ImageTreeItem


class Manager(QWidget):
    tissue_target_selected = pyqtSignal(str)
    tissue_unaligned_selected = pyqtSignal(str)

    def __init__(self, parent, model_canvas=None, model_stardist=None):

        super().__init__(parent)

        self.setWindowTitle("Image List")
        self.__layout = QVBoxLayout(self)
        self.image_tree_view = ImageTreeWidget(self, model_canvas, model_stardist)
        self.image_tree_model = ImageTreeModel()
        self.image_tree_view.setModel(self.image_tree_model)
        self.root_node = self.image_tree_model.invisibleRootItem()
        self.image_tree_view.setIconSize(QSize(50, 50))
        self.image_tree_view.setHeaderHidden(True)
        self.storage = ImageStorage()
        self.model_canvas = model_canvas
        self.model_stardist = model_stardist
        self.__layout.addWidget(self.image_tree_view)

    def set_model_canvas(self, model):
        self.model_canvas = model
        self.image_tree_view.model_canvas = model

    def set_model_stardist(self, model):
        self.model_stardist = model
        self.image_tree_view.model_stardist = model

    def on_text_edited(self):
        sender = self.sender()  # This is the QLineEdit that was edited
        if not sender:
            return
        self.storage.update_name(sender.objectName(), sender.text())  # type: ignore

    def add_item(self, uuid):
        print("adding item")
        assert self.root_node is not None, "Root node is not initialized"
        main_item = ImageTreeItem(uuid, channel="Channel 1", useItemName=True)
        item = self.storage.get_data(uuid)
        assert item is not None, f"No data found for UUID: {uuid}"
        item_data = item["data"]
        channels = item_data.keys()
        if len(channels) > 1:
            for channel in channels:
                channel_item = ImageTreeItem(uuid, channel=channel, useItemName=False)
                main_item.appendRow(channel_item)
        self.root_node.appendRow(main_item)
        # self.image_tree_view.setExpanded(
        #     self.image_tree_model.indexFromItem(main_item), True
        # )

        # h_layout = QHBoxLayout()
        # thumbnail_label = QLabel(self)
        # thumbnail_pixmap = QPixmap(numpy_to_qimage(data))
        # thumbnail_label.setPixmap(
        #     thumbnail_pixmap.scaled(50, 50, Qt.AspectRatioMode.KeepAspectRatio)
        # )
        # h_layout.addWidget(thumbnail_label)

        # text_label = QLineEdit(name, self)
        # text_label.setObjectName(uuid)
        # text_label.editingFinished.connect(self.on_text_edited)

        # text_label.setStyleSheet("QLineEdit { border: none; background: transparent; }")

        # h_layout.addWidget(text_label)

        # item_widget = QWidget()
        # item_widget.setLayout(h_layout)

        # self.image_tree_view.setItemWidget(item, item_widget)

        # # Store the image data in the item's user role
        # item.setData(Qt.ItemDataRole.UserRole, uuid)
        # item.setSizeHint(item_widget.sizeHint())

    def add_to_storage(self, uuid, obj):
        self.storage.add_data(uuid, obj)

    def update_item_layer(self, uuid, new_data, layer_name):
        self.storage.update_data(uuid, new_data, layer_name)

    # !TODO: move elsewhere
    def set_tissue_target_image(self, uuid):
        """Handle setting an image as tissue target image"""

        self.tissue_target_selected.emit(uuid)

    def set_tissue_unaligned_image(self, uuid):
        """Handle setting an image as tissue unaligned image"""
        # item = self.storage.get_data(uuid)
        self.tissue_unaligned_selected.emit(uuid)


class ImageTreeWidget(QTreeView):
    def __init__(self, parent, model_canvas, model_stardist=None):
        super().__init__(parent)
        # self.setDragDropMode(QListWidget.DragDropMode.InternalMove)
        self.storage = ImageStorage()
        self.model_canvas = model_canvas
        self.model_stardist = model_stardist
        self.doubleClicked.connect(self.on_item_selected)

    def on_item_selected(self, item):
        channel, uuid = self._name_and_uuid_from_item(item, tooltip=True)
        is_new_image = self.model_canvas.uuid != uuid
        if not is_new_image:
            item = self.storage.get_data(uuid)
            assert item is not None, f"No data found for UUID: {uuid}"
            data = item["data"]
            assert data is not None, f"No data found for UUID: {uuid}"
            image_wrapper = data[channel]
            self.model_canvas.add_to_canvas(image_wrapper, as_new_image=False)
        else:
            self.model_canvas.add_to_canvas(
                uuid, as_new_image=False, target_channel=channel
            )

    def contextMenuEvent(self, event):
        menu = QMenu(self)
        item = self.indexAt(event.pos())
        if item:
            set_menu = QMenu("Set as...", self)
            set_reference = QAction("Reference")
            set_target = QAction("Target")
            set_cell_image = QAction("Cell Image")

            set_tissue_target_image = QAction("Tissue Target Image")
            set_tissue_unaligned_image = QAction("Tissue Unaligned Image")

            # rename = QAction("Rename")
            save_as_tiff = QAction("Save as TIF")
            set_menu.addAction(set_reference)
            set_menu.addAction(set_target)
            set_menu.addAction(set_cell_image)
            set_menu.addAction(set_tissue_target_image)
            set_menu.addAction(set_tissue_unaligned_image)

            set_reference.triggered.connect(
                lambda: self.show_message("reference selected")
            )
            set_target.triggered.connect(lambda: self.show_message("target selected"))
            set_cell_image.triggered.connect(lambda: self.set_for_stardist(item))
            set_tissue_target_image.triggered.connect(
                lambda: self.set_as_tissue_target(item)
            )
            set_tissue_unaligned_image.triggered.connect(
                lambda: self.set_as_tissue_unaligned(item)
            )

            save_as_tiff.triggered.connect(lambda: self.save_as(item, "tif"))
            delete = QAction("Delete", self)
            delete.triggered.connect(lambda: self.delete_item(item))
            # rename.triggered.connect(lambda: self.rename(item))

            menu.addMenu(set_menu)
            menu.addAction(save_as_tiff)
            menu.addAction(delete)

            menu.exec(event.globalPos())

    def set_for_stardist(self, item):
        _, uuid = self._name_and_uuid_from_item(item)
        data = self.parent().storage.get_data(uuid)["data"]
        self.parent().model_stardist.setProteinImage(data)

    def show_message(self, message):
        QMessageBox.information(self, "Selection", message)

    def delete_item(self, item: QListWidgetItem):
        row = self.row(item)
        self.takeItem(row)

    def _name_and_uuid_from_item(self, item, tooltip=False):
        item = self.model().itemFromIndex(item)  # type: ignore
        if tooltip is False:
            name = item.text()
        else:
            name = item.data(Qt.ItemDataRole.ToolTipRole)
        uuid = item.data(Qt.ItemDataRole.UserRole)
        if not uuid:
            raise ValueError("Item does not have a valid UUID.")
        return name, uuid

    def save_as(self, item, type):
        name, uuid = self._name_and_uuid_from_item(item)
        channel_dict = self.parent().storage.get_data(uuid)["data"]
        print(channel_dict)
        if type == "tif":
            import tifffile

            print("inside")
            folder_path = QFileDialog.getExistingDirectory(
                self, "Select Folder to Save TIFF"
            )
            if folder_path:
                file_path = os.path.join(folder_path, f"{name}.tif")
                arrays = [
                    channel_obj.data
                    for key, channel_obj in sorted(channel_dict.items())
                ]
                stacked = np.stack(arrays, axis=0)  # Shape: (channels, H, W)
                tifffile.imwrite(
                    file_path, stacked, photometric="minisblack", imagej=True
                )

    def set_as_tissue_target(self, item: QListWidgetItem):
        """Set the selected image as the tissue target image for alignment"""
        name, uuid = self._name_and_uuid_from_item(item)
        # Emit a signal with the image data and name
        self.parent().set_tissue_target_image(uuid)

    def set_as_tissue_unaligned(self, item: QListWidgetItem):
        """Set the selected image as the tissue unaligned image for alignment"""
        name, uuid = self._name_and_uuid_from_item(item)
        # Emit a signal with the image data and name
        self.parent().set_tissue_unaligned_image(uuid)
