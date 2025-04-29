import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import cv2
import numpy as np
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                            QHBoxLayout, QPushButton, QLabel, QFileDialog, 
                            QSpinBox, QComboBox, QMessageBox)
from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtGui import QImage, QPixmap
from core.register import TileMap

# Add the current directory to the Python path
sys.path.append(str(Path(__file__).parent))
from tile_align import load_image, align_tiles_with_sift

class AlignmentWorker(QThread):
    finished = pyqtSignal(np.ndarray)
    progress = pyqtSignal(int)
    error = pyqtSignal(str)

    def __init__(self, fixed_img, moving_img, tile_size, overlap):
        super().__init__()
        self.fixed_img = fixed_img
        self.moving_img = moving_img
        self.tile_size = tile_size
        self.overlap = overlap

    def run(self):
        try:
            result = align_tiles_with_sift(
                self.fixed_img, 
                self.moving_img, 
                self.tile_size, 
                self.overlap
            )
            self.finished.emit(result)
        except Exception as e:
            self.error.emit(str(e))

class ImageViewer(QLabel):
    def __init__(self):
        super().__init__()
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setMinimumSize(400, 400)
        self.setStyleSheet("border: 1px solid black;")

    def set_image(self, image):
        if image is None:
            return
        
        # Convert numpy array to QImage
        height, width = image.shape
        bytes_per_line = width
        q_img = QImage(image.data, width, height, bytes_per_line, QImage.Format.Format_Grayscale8)
        
        # Scale image to fit the label while maintaining aspect ratio
        pixmap = QPixmap.fromImage(q_img)
        scaled_pixmap = pixmap.scaled(self.size(), Qt.AspectRatioMode.KeepAspectRatio)
        self.setPixmap(scaled_pixmap)

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Tile Alignment Tool")
        self.setMinimumSize(1200, 800)

        # Initialize variables
        self.fixed_img = None
        self.moving_img = None
        self.aligned_img = None
        self.current_view = "original"  # or "aligned"

        # Create main widget and layout
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        layout = QHBoxLayout(main_widget)

        # Left panel for controls
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        
        # Image loading controls
        load_group = QWidget()
        load_layout = QVBoxLayout(load_group)
        
        self.load_fixed_btn = QPushButton("Load Fixed Image")
        self.load_moving_btn = QPushButton("Load Moving Image")
        self.load_fixed_btn.clicked.connect(lambda: self.load_image("fixed"))
        self.load_moving_btn.clicked.connect(lambda: self.load_image("moving"))
        
        load_layout.addWidget(self.load_fixed_btn)
        load_layout.addWidget(self.load_moving_btn)
        
        # Alignment parameters
        param_group = QWidget()
        param_layout = QVBoxLayout(param_group)
        
        self.tile_size_spin = QSpinBox()
        self.tile_size_spin.setRange(64, 1024)
        self.tile_size_spin.setValue(512)
        self.tile_size_spin.setSingleStep(64)
        
        self.overlap_spin = QSpinBox()
        self.overlap_spin.setRange(0, 256)
        self.overlap_spin.setValue(100)
        self.overlap_spin.setSingleStep(16)
        
        param_layout.addWidget(QLabel("Tile Size:"))
        param_layout.addWidget(self.tile_size_spin)
        param_layout.addWidget(QLabel("Overlap:"))
        param_layout.addWidget(self.overlap_spin)
        
        # Alignment button
        self.align_btn = QPushButton("Align Images")
        self.align_btn.clicked.connect(self.start_alignment)
        self.align_btn.setEnabled(False)
        
        # View controls
        self.view_combo = QComboBox()
        self.view_combo.addItems(["Original", "Aligned"])
        self.view_combo.currentTextChanged.connect(self.update_view)
        
        # Add all controls to left panel
        left_layout.addWidget(load_group)
        left_layout.addWidget(param_group)
        left_layout.addWidget(self.align_btn)
        left_layout.addWidget(QLabel("View:"))
        left_layout.addWidget(self.view_combo)
        left_layout.addStretch()
        
        # Right panel for image display
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        
        # Create image viewers
        self.fixed_viewer = ImageViewer()
        self.moving_viewer = ImageViewer()
        
        right_layout.addWidget(QLabel("Fixed Image:"))
        right_layout.addWidget(self.fixed_viewer)
        right_layout.addWidget(QLabel("Moving Image:"))
        right_layout.addWidget(self.moving_viewer)
        
        # Add panels to main layout
        layout.addWidget(left_panel, 1)
        layout.addWidget(right_panel, 3)

    def load_image(self, image_type):
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            f"Select {image_type.title()} Image",
            "",
            "Image Files (*.png *.jpg *.jpeg *.tif *.tiff)"
        )
        
        if file_path:
            try:
                img = load_image(file_path)
                if image_type == "fixed":
                    self.fixed_img = img
                    self.fixed_viewer.set_image(img)
                else:
                    self.moving_img = img
                    self.moving_viewer.set_image(img)
                
                # Enable align button if both images are loaded
                self.align_btn.setEnabled(
                    self.fixed_img is not None and 
                    self.moving_img is not None
                )
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to load image: {str(e)}")

    def start_alignment(self):
        if self.fixed_img is None or self.moving_img is None:
            return
        
        # Disable controls during alignment
        self.align_btn.setEnabled(False)
        self.load_fixed_btn.setEnabled(False)
        self.load_moving_btn.setEnabled(False)
        
        # Create and start worker thread
        self.worker = AlignmentWorker(
            self.fixed_img,
            self.moving_img,
            self.tile_size_spin.value(),
            self.overlap_spin.value()
        )
        self.worker.finished.connect(self.alignment_finished)
        self.worker.error.connect(self.alignment_error)
        self.worker.start()

    def alignment_finished(self, result):
        self.aligned_img = result
        self.view_combo.setCurrentText("Aligned")
        self.update_view("Aligned")
        
        # Re-enable controls
        self.align_btn.setEnabled(True)
        self.load_fixed_btn.setEnabled(True)
        self.load_moving_btn.setEnabled(True)

    def alignment_error(self, error_msg):
        QMessageBox.critical(self, "Alignment Error", error_msg)
        
        # Re-enable controls
        self.align_btn.setEnabled(True)
        self.load_fixed_btn.setEnabled(True)
        self.load_moving_btn.setEnabled(True)

    def update_view(self, view):
        if view == "Original":
            self.fixed_viewer.set_image(self.fixed_img)
            self.moving_viewer.set_image(self.moving_img)
        else:  # Aligned
            if self.aligned_img is not None:
                self.fixed_viewer.set_image(self.fixed_img)
                self.moving_viewer.set_image(self.aligned_img)
            else:
                QMessageBox.warning(self, "Warning", "No aligned image available")

def main():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main() 