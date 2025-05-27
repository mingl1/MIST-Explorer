from PyQt6.QtGui import *
from PyQt6.QtCore import *
from PyQt6.QtWidgets import *
from core.canvas import ImageStorage
from utils import numpy_to_qimage
from PyQt6.QtCore import pyqtSignal

class Manager(QWidget):
    # Add signals for tissue image selection
    tissue_target_selected = pyqtSignal(dict, str)
    tissue_unaligned_selected = pyqtSignal(object, str)
    
    def __init__(self, parent=None):
        super().__init__(parent)
        
        self.setWindowTitle("Image List")
        self.__layout = QVBoxLayout(self)
        self.list_widget = ListWidget(self)
        self.storage = ImageStorage()
        self.__layout.addWidget(self.list_widget)

    def add_item(self, uuid):
        print("adding item")
        item = QListWidgetItem(self.list_widget)
        h_layout = QHBoxLayout()
        name = self.storage.get_data(uuid)['name']
        data = self.storage.get_data(uuid)['data']['Channel 1'].data
        thumbnail_label = QLabel(self)
        thumbnail_pixmap = QPixmap(numpy_to_qimage(data))
        thumbnail_label.setPixmap(thumbnail_pixmap.scaled(50, 50, Qt.AspectRatioMode.KeepAspectRatio))
        h_layout.addWidget(thumbnail_label)

        text_label = QLineEdit(name, self)

        text_label.setStyleSheet("QLineEdit { border: none; background: transparent; }")

        h_layout.addWidget(text_label)

        item_widget = QWidget()
        item_widget.setLayout(h_layout)
        
        self.list_widget.setItemWidget(item, item_widget)
        
        # Store the image data in the item's user role
        item.setData(Qt.ItemDataRole.UserRole, uuid)
        item.setSizeHint(item_widget.sizeHint())
    
    def add_to_storage(self, uuid, obj):
        self.storage.add_data(uuid, obj)
        print(uuid)
    
    def set_tissue_target_image(self, uuid):
        """Handle setting an image as tissue target image"""
        item = self.storage.get_data(uuid)
        self.tissue_target_selected.emit(item["data"], item['name'])
        
    def set_tissue_unaligned_image(self, uuid):
        """Handle setting an image as tissue unaligned image"""
        item = self.storage.get_data(uuid)
        self.tissue_unaligned_selected.emit(item["data"], item['name'])

class ListWidget(QListWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setDragDropMode(QListWidget.DragDropMode.InternalMove)

    def contextMenuEvent(self, event):
        menu = QMenu(self)

        item = self.itemAt(event.pos())
        if item:
            set_menu = QMenu("Set as...", self)
            set_reference = QAction("Reference")
            set_target = QAction("Target")
            set_cell_image = QAction("Cell Image")

            set_tissue_target_image = QAction("Tissue Target Image")
            set_tissue_unaligned_image = QAction("Tissue Unaligned Image")

            # rename = QAction("Rename")

            set_menu.addAction(set_reference)
            set_menu.addAction(set_target)
            set_menu.addAction(set_cell_image)
            set_menu.addAction(set_tissue_target_image)
            set_menu.addAction(set_tissue_unaligned_image)

            set_reference.triggered.connect(lambda: self.show_message("reference selected"))
            set_target.triggered.connect(lambda: self.show_message("target selected"))
            set_cell_image.triggered.connect(lambda: self.show_message("cell image selected"))
            set_tissue_target_image.triggered.connect(lambda: self.set_as_tissue_target(item))
            set_tissue_unaligned_image.triggered.connect(lambda: self.set_as_tissue_unaligned(item))
            
            delete = QAction("Delete", self)
            delete.triggered.connect(lambda: self.delete_item(item))
            # rename.triggered.connect(lambda: self.rename(item))

            menu.addMenu(set_menu)
            # menu.addAction(rename)
            menu.addAction(delete)

            menu.exec(event.globalPos())

    def show_message(self, message):
        QMessageBox.information(self, "Selection", message)
        
    def delete_item(self, item:QListWidgetItem):
        row = self.row(item)
        self.takeItem(row)
        
    def set_as_tissue_target(self, item:QListWidgetItem):
        """Set the selected image as the tissue target image for alignment"""
        # Get the item widget and access the text label (QLineEdit)
        item_widget = self.itemWidget(item)
        layout = item_widget.layout()
        # The text label is the second widget in the layout
        text_label = layout.itemAt(1).widget() 
        name = text_label.text()
        
        uuid = item.data(Qt.ItemDataRole.UserRole)
        # Emit a signal with the image data and name
        self.parent().set_tissue_target_image(uuid)
        
    def set_as_tissue_unaligned(self, item:QListWidgetItem):
        """Set the selected image as the tissue unaligned image for alignment"""
        # Get the item widget and access the text label (QLineEdit)
        item_widget = self.itemWidget(item)
        layout = item_widget.layout()
        # The text label is the second widget in the layout
        text_label = layout.itemAt(1).widget() 
        name = text_label.text()
        
        uuid = item.data(Qt.ItemDataRole.UserRole)
        # Emit a signal with the image data and name
        self.parent().set_tissue_unaligned_image(uuid)
