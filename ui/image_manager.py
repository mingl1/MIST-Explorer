"""
Image Manager module.
"""
import logging
import os
import uuid
from pathlib import Path
from typing import Optional
from uuid import UUID

import numpy as np
import tifffile  # pylint: disable=import-error
# pylint: disable=no-name-in-module
from PyQt6.QtCore import QSize, Qt, pyqtSignal
from PyQt6.QtGui import QAction, QActionGroup
from PyQt6.QtWidgets import (QFileDialog, QMenu, QMessageBox, QTreeView,
                             QVBoxLayout, QWidget)

from core import ImageGraphicsView, ImageStorage, StarDist
from core.canvas import ReferenceGraphicsView
from core.project_manager import ProjectManager
from models.image_list_model import ImageTreeItem, ImageTreeModel

logger = logging.getLogger(__name__)


class ImageManager(QWidget):
    """
    Widget to manage the list of loaded images.
    """
    # pylint: disable=too-many-instance-attributes
    def __init__(self, parent=None):
        super().__init__(parent)

        self.setWindowTitle("Image List")
        self.__layout = QVBoxLayout(self)
        self.image_tree_view = ImageTreeWidget(self)
        self.image_tree_model = ImageTreeModel()
        self.image_tree_view.setModel(self.image_tree_model)
        self.root_node = self.image_tree_model.invisibleRootItem()
        # max icon size
        self.image_tree_view.setIconSize(QSize(50, 50))
        self.image_tree_view.setHeaderHidden(True)
        self.storage = ImageStorage()
        self.__layout.addWidget(self.image_tree_view)

        # Initialize attributes
        self.model_canvas: Optional[ImageGraphicsView] = None
        self.model_stardist: Optional[StarDist] = None
        self.model_reference_canvas: Optional[ReferenceGraphicsView] = None
        self.current_project_path: Optional[Path] = None

        # Connect deletion signal to backend cleanup
        self.image_tree_view.item_deleted.connect(self._handle_item_deletion)

    def set_project_path(self, project_path: Path):
        """Set the current project path for auto-saving."""
        self.current_project_path = project_path

    def set_model_canvas(self, model):
        """Set the model canvas."""
        self.model_canvas = model
        self.image_tree_view.model_canvas = model

    def set_model_stardist(self, model):
        """Set the stardist model."""
        self.model_stardist = model
        self.image_tree_view.model_stardist = model

    def set_model_reference_canvas(self, model):
        """Set the reference canvas model."""
        self.model_reference_canvas = model
        self.image_tree_view.model_reference_canvas = model

    def add_item(self, item_uuid):
        """Add an item to the tree."""
        logger.debug("adding item")
        assert self.root_node is not None, "Root node is not initialized"
        main_item = ImageTreeItem(item_uuid, channel="Channel 1", useItemName=True)
        item = self.storage.get_data(item_uuid)
        assert item is not None, f"No data found for UUID: {item_uuid}"
        item_data = item["data"]
        channels = item_data.keys()
        channels = sorted(channels, key=lambda x: int(x.replace("Channel ", "")))
        if len(channels) > 1:
            for channel in channels:
                channel_item = ImageTreeItem(item_uuid, channel=channel, useItemName=False)
                main_item.appendRow(channel_item)
        self.root_node.appendRow(main_item)

        if self.current_project_path:
            self._save_image_to_project(item_uuid, item)

    def _save_image_to_project(self, item_uuid, item):
        """Save image data to the project folder."""
        if item is None:
            return

        item_data = item.get("data", {})
        if not item_data:
            return

        image_name = item.get("name", f"Image_{item_uuid}")
        channel_count = len(item_data)
        original_filename = item.get("original_filename", "")

        contrast_settings = {}
        for channel_name, wrapper in item_data.items():
            contrast_settings[channel_name] = (wrapper.contrast_min, wrapper.contrast_max)

        ProjectManager.save_image(
            project_path=self.current_project_path,
            image_uuid=str(item_uuid),
            channel_data=item_data,
            image_name=image_name,
            channel_count=channel_count,
            original_filename=original_filename,
            contrast_settings=contrast_settings,
        )

    def set_channel_icon(self, item_uuid, channel):
        """Set the icon for the channel item"""
        assert self.root_node is not None, "Root node is not initialized"
        main_item = None
        image_data = self.storage.get_data(item_uuid)
        if image_data is None:
            raise ValueError(f"No image data found for UUID: {item_uuid}")
        image_data = image_data.get("data", {})
        if channel not in image_data:
            raise ValueError(
                f"Channel {channel} not found in image data for UUID: {item_uuid}"
            )
        model = self.image_tree_model
        for i in range(model.rowCount()):
            item = model.item(i)
            if item is None:
                continue
            if item.data(Qt.ItemDataRole.UserRole) == item_uuid:
                main_item = item
                break
        if main_item is None:
            raise ValueError(f"No main item found for UUID: {item_uuid}")
        channel_item = None
        if main_item.data(Qt.ItemDataRole.WhatsThisRole) == channel:
            if isinstance(main_item, ImageTreeItem):
                main_item.set_icon(image_data)
        for i in range(main_item.rowCount()):
            child = main_item.child(i)
            if child is None:
                continue
            if child.data(Qt.ItemDataRole.WhatsThisRole) == channel:
                channel_item = child
                break
        if isinstance(channel_item, ImageTreeItem):
            channel_item.set_icon(image_data)

    def add_to_storage(self, item_uuid, obj):
        """Add data to storage."""
        logger.debug(f"adding {item_uuid} to storage")
        self.storage.add_data(item_uuid, obj)

    def _handle_item_deletion(self, item_uuid: UUID):
        """Handle backend cleanup when an item is deleted."""
        # Remove from in-memory storage
        self.storage.remove_data(item_uuid)

        # Remove from project files and metadata if project is open
        if self.current_project_path:
            ProjectManager.delete_image(self.current_project_path, str(item_uuid))


class ImageTreeWidget(QTreeView):
    """
    Tree view widget for displaying images.
    """
    tissue_target_selected = pyqtSignal(UUID, bool, int)
    tissue_unaligned_selected = pyqtSignal(UUID, bool, int)
    protein_data = pyqtSignal(UUID, int)
    stardist_label = pyqtSignal(UUID, int)
    item_deleted = pyqtSignal(UUID)

    def __init__(self, parent):
        super().__init__(parent)
        self.storage = ImageStorage()
        self._model_canvas = None
        self._model_stardist = None
        self._model_reference_canvas = None
        self.doubleClicked.connect(self.show_on_canvas)

    @property
    def model_canvas(self):
        """Get the model canvas."""
        return self._model_canvas

    @model_canvas.setter
    def model_canvas(self, value: ImageGraphicsView):
        self._model_canvas = value

    @property
    def model_reference_canvas(self):
        """Get the reference canvas model."""
        return self._model_reference_canvas

    @model_reference_canvas.setter
    def model_reference_canvas(self, value: ReferenceGraphicsView):
        self._model_reference_canvas = value

    @property
    def model_stardist(self):
        """Get the stardist model."""
        return self._model_stardist

    @model_stardist.setter
    def model_stardist(self, value: StarDist):
        self._model_stardist = value

    def show_on_canvas(self, item):
        """Show selected item on canvas."""
        assert self.model_canvas is not None, "model_canvas is not set"
        logger.debug("show on canvas")
        model_canvas = self.model_canvas
        _, my_uuid = self._name_and_uuid_from_item(item, tooltip=True)
        different = str(my_uuid) != str(model_canvas.uuid)
        if not different:
            channel = self._get_channel_from_item(item)
            model_canvas.swap_channel(channel)
        else:
            channel = self._get_channel_from_item(item, as_int=False)
            model_canvas.add_to_canvas(
                my_uuid, as_new_image=False, target_channel=str(channel)
            )

    def show_on_reference_canvas(self, item_uuid: UUID, channel: int):
        """Show selected item on reference canvas."""
        assert (
            self.model_reference_canvas is not None
        ), "model_reference_canvas is not set"
        cname = f"Channel {channel + 1}"
        self.model_reference_canvas.add_to_canvas(item_uuid, cname)

    # pylint: disable=invalid-name, too-many-locals, too-many-statements
    def contextMenuEvent(self, event):  # type: ignore
        """Handle context menu event."""
        menu = QMenu(self)
        item = self.indexAt(event.pos())
        if item and item.isValid():
            _, item_uuid = self._name_and_uuid_from_item(item)
            channel = int(self._get_channel_from_item(item))
            is_leaf = self._is_leaf(item)
            set_reference = QAction("Reference")
            set_cell_image = QAction("Cell Image (Stardist)")

            set_tissue_target_image = QAction("Tissue Target Image")
            set_tissue_unaligned_image = QAction("Tissue Unaligned Image")
            set_stardist_label = QAction("StarDist Label")

            save_as_tiff = QAction("Save as TIF")

            set_reference.triggered.connect(
                lambda: self.show_on_reference_canvas(item_uuid, channel)
            )
            set_cell_image.triggered.connect(lambda: self.set_for_stardist(item))
            set_tissue_target_image.triggered.connect(
                lambda: self.set_as_tissue_target(item_uuid, is_leaf, channel)
            )
            set_tissue_unaligned_image.triggered.connect(
                lambda: self.set_as_tissue_unaligned(item_uuid, is_leaf, channel)
            )
            set_stardist_label.triggered.connect(
                lambda: self.set_as_stardist_label(item_uuid, channel)
            )

            save_as_tiff.triggered.connect(lambda: self.save_as(item, "tif"))
            delete = QAction("Delete", self)
            delete.triggered.connect(lambda: self.delete_item(item))
            model = self.model()
            assert isinstance(model, ImageTreeModel), "Model is not set"

            if not is_leaf:
                set_menu = QMenu("Set as...", self)
                menu.addMenu(set_menu)
                set_menu.addAction(set_reference)

                stardist_menu = set_menu.addMenu("Stardist")
                stardist_menu.addAction(set_cell_image)

                tissue_menu = set_menu.addMenu("Tissue")
                tissue_menu.addAction(set_tissue_target_image)
                tissue_menu.addAction(set_tissue_unaligned_image)

                set_default_channel = QMenu("Set Default Channel as", self)
                channel_group = QActionGroup(self)
                channel_group.setExclusive(True)

                current_default = model.itemFromIndex(item)
                assert current_default is not None, "Current default item is None"
                current_default = current_default.data(Qt.ItemDataRole.WhatsThisRole)
                if current_default is None:
                    current_default = "Channel 1"
                children = model.itemFromIndex(item)
                assert children is not None, "Children item is None"
                for i in range(children.rowCount()):
                    child_item = children.child(i)
                    assert child_item is not None, "Child item is None"
                    channel_name = child_item.data(Qt.ItemDataRole.WhatsThisRole)
                    action = QAction(channel_name, self)
                    action.setCheckable(True)
                    action.setChecked(channel_name == current_default)
                    action.triggered.connect(
                        lambda _, channel_name=channel_name: (
                            model.setData(
                                item, channel_name, Qt.ItemDataRole.WhatsThisRole
                            )
                        )
                    )
                    channel_group.addAction(action)
                    set_default_channel.addAction(action)
                menu.addMenu(set_default_channel)
            else:
                set_reference.setText("Set as Reference")
                set_cell_image.setText("Set as Cell Image")
                set_tissue_target_image.setText("Set as Tissue Target Image")
                set_tissue_unaligned_image.setText("Set as Tissue Unaligned Image")
                set_stardist_label.setText("Set as StarDist Label")
                menu.addAction(set_reference)

                stardist_menu = menu.addMenu("Stardist")
                stardist_menu.addAction(set_cell_image)
                stardist_menu.addAction(set_stardist_label)

                tissue_menu = menu.addMenu("Tissue")
                tissue_menu.addAction(set_tissue_target_image)
                tissue_menu.addAction(set_tissue_unaligned_image)
            model_item = model.itemFromIndex(item)
            assert model_item is not None, "Item is None"
            # only add delete & save as tiff action if the item is a root item
            if model_item.parent() is None:
                menu.addAction(delete)
                menu.addAction(save_as_tiff)
            menu.exec(event.globalPos())

    def set_for_stardist(self, item):
        """Set the image for stardist model."""
        assert isinstance(self.model_stardist, StarDist), "model_stardist is not set"
        name, item_uuid = self._name_and_uuid_from_item(item)
        item = self.storage.get_data(item_uuid)
        assert item is not None, f"No data found for UUID: {item_uuid}"
        data = item["data"]
        self.model_stardist.set_protein_image(data, name=name)

    def show_message(self, message):
        """Show a message box."""
        QMessageBox.information(self, "Selection", message)

    def delete_item(self, index):
        """Delete an item from the tree."""
        model = self.model()
        assert isinstance(model, ImageTreeModel), "Model is not set"
        item = model.itemFromIndex(index)
        _, item_uuid = self._name_and_uuid_from_item(index)
        assert item is not None, "Item is None"
        row = model.indexFromItem(item).row()
        self.item_deleted.emit(item_uuid)
        model.removeRow(row)

    def _name_and_uuid_from_item(self, item, tooltip=False) -> tuple[str, uuid.UUID]:
        item = self.model().itemFromIndex(item)  # type: ignore
        if tooltip is False:
            name = item.text()
        else:
            name = item.data(Qt.ItemDataRole.WhatsThisRole)
        user_role_data = item.data(Qt.ItemDataRole.UserRole)
        if isinstance(user_role_data, uuid.UUID):
            item_uuid = user_role_data
        else:
            item_uuid = uuid.UUID(user_role_data)
        if not item_uuid:
            raise ValueError("Item does not have a valid UUID.")

        return name, item_uuid

    def _get_channel_from_item(self, item, as_int=True):
        model = self.model()
        assert isinstance(model, ImageTreeModel), "Model is not set"
        item = model.itemFromIndex(item)
        assert item is not None, "Item is None"
        channel = item.data(Qt.ItemDataRole.WhatsThisRole)
        if as_int is False:
            return str(channel)
        try:
            channel = int(channel.replace("Channel ", "")) - 1
            logger.debug(channel)
        except ValueError as exc:
            raise ValueError("Invalid default channel format.") from exc
        return channel

    def _is_leaf(self, item):
        model = self.model()
        assert isinstance(model, ImageTreeModel), "Model is not set"
        item = model.itemFromIndex(item)
        assert item is not None, "Item is None"
        return not item.hasChildren()

    def save_as(self, item, file_type):
        """Save the item to file."""
        name, item_uuid = self._name_and_uuid_from_item(item)
        name = os.path.splitext(name)[0]
        item = self.storage.get_data(item_uuid)
        assert item is not None, f"No data found for UUID: {item_uuid}"
        channel_dict = item["data"]
        if file_type == "tif":
            logger.debug("inside")
            folder_path = QFileDialog.getExistingDirectory(
                self, "Select Folder to Save TIFF"
            )
            if folder_path:
                file_path = os.path.join(folder_path, f"{name}.tif")
                arrays = [
                    channel_obj.data for _, channel_obj in sorted(channel_dict.items())
                ]
                stacked = np.stack(arrays, axis=0)  # Shape: (channels, H, W)
                tifffile.imwrite(
                    file_path, stacked, photometric="minisblack", imagej=True
                )

    def set_as_tissue_target(self, i_uuid: UUID, is_leaf: bool, channel: int):
        """Set the selected image as the tissue target image for alignment"""

        self.tissue_target_selected.emit(i_uuid, is_leaf, channel)

    def set_as_tissue_unaligned(self, i_uuid: UUID, is_leaf: bool, channel: int):
        """Set the selected image as the tissue unaligned image for alignment"""
        self.tissue_unaligned_selected.emit(i_uuid, is_leaf, channel)

    def set_as_stardist_label(self, i_uuid: UUID, channel: int):
        """Set the selected image as the StarDist label image for alignment"""
        self.stardist_label.emit(i_uuid, channel)
