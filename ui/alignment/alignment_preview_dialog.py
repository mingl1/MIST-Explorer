from PyQt6.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QPushButton,
                             QLabel, QGraphicsView, QGraphicsScene)
from PyQt6.QtCore import Qt, QSize
from PyQt6.QtGui import QPixmap, QImage, QColor
import numpy as np
import cv2

class AlignmentPreviewDialog(QDialog):
    """Dialog to preview the alignment before confirming or canceling"""
    
    def __init__(self, target_image, aligned_image, parent=None):
        super().__init__(parent)
        self.target_image = target_image
        self.aligned_image = aligned_image
        self.result_accepted = False
        
        self.setWindowTitle("Alignment Preview")
        self.resize(800, 600)
        
        # Create layout
        self.layout = QVBoxLayout(self)
        
        # Create preview image
        self.preview_label = QLabel("Red = Target, Green = Aligned")
        self.preview_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        
        # Create graphics view for the overlay image
        self.graphics_view = QGraphicsView()
        self.graphics_scene = QGraphicsScene()
        self.graphics_view.setScene(self.graphics_scene)
        
        # Create buttons
        self.button_layout = QHBoxLayout()
        
        self.confirm_button = QPushButton("Confirm")
        self.confirm_button.setStyleSheet("""
            QPushButton {
                background-color: #4CAF50;
                color: white;
                font-weight: bold;
                min-width: 100px;
                min-height: 30px;
                border-radius: 4px;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
        """)
        self.confirm_button.clicked.connect(self.accept_alignment)
        
        self.cancel_button = QPushButton("Cancel")
        self.cancel_button.setStyleSheet("""
            QPushButton {
                background-color: #f44336;
                color: white;
                font-weight: bold;
                min-width: 100px;
                min-height: 30px;
                border-radius: 4px;
            }
            QPushButton:hover {
                background-color: #d32f2f;
            }
        """)
        self.cancel_button.clicked.connect(self.reject)
        
        self.button_layout.addWidget(self.confirm_button)
        self.button_layout.addWidget(self.cancel_button)
        
        # Add widgets to layout
        self.layout.addWidget(self.preview_label)
        self.layout.addWidget(self.graphics_view)
        self.layout.addLayout(self.button_layout)
        
        # Create the overlay image
        self.create_overlay()
    
    def create_overlay(self):
        """Create the red/green overlay image for preview"""
        try:
            # Ensure both images are of the same size
            h, w = self.target_image.shape[:2]
            target_resized = self.target_image
            aligned_resized = self.aligned_image
            
            # Convert to 8-bit if needed
            if target_resized.dtype != np.uint8:
                target_resized = self.to_uint8(target_resized)
            if aligned_resized.dtype != np.uint8:
                aligned_resized = self.to_uint8(aligned_resized)
            
            # Create an RGB image with the red and green channels
            overlay = np.zeros((h, w, 3), dtype=np.uint8)
            
            # Make sure we handle both grayscale and color images
            if len(target_resized.shape) == 2:
                overlay[:, :, 0] = target_resized  # Target is red
            else:
                overlay[:, :, 0] = cv2.cvtColor(target_resized, cv2.COLOR_BGR2GRAY)
                
            if len(aligned_resized.shape) == 2:
                overlay[:, :, 1] = aligned_resized  # Aligned is green
            else:
                overlay[:, :, 1] = cv2.cvtColor(aligned_resized, cv2.COLOR_BGR2GRAY)
            
            # Convert to QImage and display
            height, width, channels = overlay.shape
            bytes_per_line = channels * width
            q_image = QImage(overlay.data, width, height, bytes_per_line, QImage.Format.Format_RGB888)
            pixmap = QPixmap.fromImage(q_image)
            
            # Add the pixmap to the scene
            self.graphics_scene.clear()
            self.graphics_scene.addPixmap(pixmap)
            self.graphics_view.fitInView(self.graphics_scene.sceneRect(), 
                                         Qt.AspectRatioMode.KeepAspectRatio)
            
        except Exception as e:
            print(f"Error creating overlay: {str(e)}")
    
    def to_uint8(self, image):
        """Convert image to uint8 with proper scaling"""
        # Check if image is already uint8
        if image.dtype == np.uint8:
            return image
            
        # Convert to float and scale to 0-255
        img_float = image.astype(np.float32)
        if img_float.max() > img_float.min():  # Check to avoid division by zero
            img_norm = ((img_float - img_float.min()) * (255.0 / (img_float.max() - img_float.min())))
            return img_norm.astype(np.uint8)
        else:
            return np.zeros_like(image, dtype=np.uint8)
    
    def resizeEvent(self, event):
        """Handle resize to keep the image properly scaled"""
        super().resizeEvent(event)
        self.graphics_view.fitInView(self.graphics_scene.sceneRect(), 
                                    Qt.AspectRatioMode.KeepAspectRatio)
    
    def accept_alignment(self):
        """Set result as accepted and close dialog"""
        self.result_accepted = True
        self.accept() 