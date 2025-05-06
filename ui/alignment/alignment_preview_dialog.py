from PyQt6.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QPushButton,
                             QLabel)
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QPixmap, QImage
import numpy as np
import sys
import os
import tempfile
import cv2

# Add path to import microfilm
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'lib')))

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
        self.preview_label.setStyleSheet("font-weight: bold; font-size: 14px;")
        
        # Create image display label
        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image_label.setMinimumSize(600, 400)
        
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
        self.layout.addWidget(self.image_label)
        self.layout.addLayout(self.button_layout)
        
        # Create the overlay image directly
        self.create_direct_overlay()
    
    def create_direct_overlay(self):
        """Create and display the overlay directly without using external libraries"""
        try:
            print("Creating direct overlay visualization...")
            # Ensure the images are available
            if self.target_image is None or self.aligned_image is None:
                raise ValueError("Missing one or both images for alignment preview")
            
            # Crop to same dimensions
            target_img, aligned_img = self._ensure_same_size(self.target_image, self.aligned_image)
            print(f"After cropping: target={target_img.shape}, aligned={aligned_img.shape}")
            
            # Convert to grayscale if color
            if len(target_img.shape) > 2 and target_img.shape[2] > 1:
                target_gray = cv2.cvtColor(target_img, cv2.COLOR_BGR2GRAY)
            else:
                target_gray = target_img
                
            if len(aligned_img.shape) > 2 and aligned_img.shape[2] > 1:
                aligned_gray = cv2.cvtColor(aligned_img, cv2.COLOR_BGR2GRAY)
            else:
                aligned_gray = aligned_img
            
            # Convert to uint8 if needed
            if target_gray.dtype != np.uint8:
                target_gray = self.to_uint8(target_gray)
            if aligned_gray.dtype != np.uint8:
                aligned_gray = self.to_uint8(aligned_gray)
            
            # Adjust contrast
            target_gray = self.adjust_contrast(target_gray)
            aligned_gray = self.adjust_contrast(aligned_gray)
            
            # Create RGB overlay
            h, w = target_gray.shape
            overlay = np.zeros((h, w, 3), dtype=np.uint8)
            overlay[:, :, 0] = target_gray   # Red channel = target
            overlay[:, :, 1] = aligned_gray  # Green channel = aligned
            
            # Save to debug file
            try:
                print("Saving debug file...")
                cv2.imwrite("micro.png", overlay)
                print("Saved overlay to micro.png")
            except Exception as save_error:
                print(f"Warning: Could not save debug file: {save_error}")
            
            # Convert to QImage
            print("Converting to QImage...")
            height, width, channels = overlay.shape
            bytes_per_line = channels * width
            q_image = QImage(overlay.data, width, height, bytes_per_line, QImage.Format.Format_RGB888)
            
            # Convert to QPixmap and display
            print("Setting pixmap...")
            pixmap = QPixmap.fromImage(q_image)
            self.image_label.setPixmap(pixmap.scaled(
                self.image_label.width(), self.image_label.height(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation
            ))
            print("Overlay displayed successfully!")
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"Error creating overlay: {str(e)}")
            self.image_label.setText(f"Error creating overlay: {str(e)}")
    
    def _ensure_same_size(self, img1, img2):
        """Ensure both images have the same dimensions by cropping to the smallest common size"""
        print(f"Original shapes: img1={img1.shape}, img2={img2.shape}")
        
        # Get the minimum width and height
        min_height = min(img1.shape[0], img2.shape[0])
        min_width = min(img1.shape[1], img2.shape[1])
        
        # Crop both images to these dimensions
        img1_cropped = img1[:min_height, :min_width]
        img2_cropped = img2[:min_height, :min_width]
        
        # If images are RGB (3 channels), make sure to keep the channel dimension
        if len(img1.shape) > 2:
            img1_cropped = img1_cropped[:min_height, :min_width, :img1.shape[2]]
        if len(img2.shape) > 2:
            img2_cropped = img2_cropped[:min_height, :min_width, :img2.shape[2]]
        
        print(f"Cropped shapes: img1={img1_cropped.shape}, img2={img2_cropped.shape}")
        
        return img1_cropped, img2_cropped
    
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
    
    def adjust_contrast(self, img, min_percentile=2, max_percentile=98):
        """Adjust image contrast using percentile-based clipping"""
        # Calculate percentiles
        minval = np.percentile(img, min_percentile)
        maxval = np.percentile(img, max_percentile)
        
        # Clip and rescale
        img_adjusted = np.clip(img, minval, maxval)
        if maxval > minval:
            img_adjusted = ((img_adjusted - minval) / (maxval - minval)) * 255
        return img_adjusted.astype(np.uint8)
    
    def resizeEvent(self, event):
        """Handle resize events to resize the displayed image"""
        super().resizeEvent(event)
        # Recreate the overlay on resize
        self.create_direct_overlay()
    
    def accept_alignment(self):
        """Set result as accepted and close dialog"""
        self.result_accepted = True
        self.accept() 