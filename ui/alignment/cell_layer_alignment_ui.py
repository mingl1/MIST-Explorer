from PyQt6.QtWidgets import QVBoxLayout, QHBoxLayout, QGroupBox, QLabel, QPushButton, QWidget, QMessageBox
from PyQt6.QtCore import QCoreApplication, QMetaObject, pyqtSignal, pyqtSlot
import numpy as np
from core.cell_layer_alignment import CellLayerAligner
from ui.alignment.alignment_preview_dialog import AlignmentPreviewDialog
from utils import numpy_to_qimage, scale_adjust
from PyQt6.QtGui import QPixmap
from PyQt6.QtWidgets import QGraphicsPixmapItem,QComboBox
class CellLayerAlignmentUI(QWidget):
    errorSignal = pyqtSignal(str)
    alignmentCompleteSignal = pyqtSignal(object, str)
    replaceLayerSignal = pyqtSignal(object)  
    loadOnCanvasSignal = pyqtSignal(object)  
    channelChanged = pyqtSignal(int)
    def __init__(self,  containing_layout:QVBoxLayout, parent=None):
        super().__init__()
        self.image_channels = [1,1]
        self.target_image = None
        self.unaligned_image = None
        self.target_name = "not loaded"
        self.unaligned_name = "not loaded"
        self.aligner = CellLayerAligner()
        self._setup_ui(parent, containing_layout)
        self._setup_connections()

    def _setup_ui(self, parent, containing_layout:QVBoxLayout):
        self.alignment_groupbox = QGroupBox(parent)
        
        # Main layout for the groupbox
        self.main_layout = QVBoxLayout(self.alignment_groupbox)
        
        # Title and labels
        self.title_label = QLabel("Cell Layer Alignment")
        self.title_label.setStyleSheet("font-weight: bold; font-size: 14px;")
        
        # Image 1 layout
        self.image1_layout = QHBoxLayout()
        self.image1_label = QLabel("Target Image:")
        self.image1_label.setMinimumWidth(120)
        self.image1_status = QLabel("not loaded")
        self.image1_status.setStyleSheet("font-weight: bold; color: #555;")
        self.image1_layout.addWidget(self.image1_label)
        self.image1_layout.addWidget(self.image1_status)
        self.image1_layout.setStretch(1, 1)  # Make the status label expand
        self.image_1_channel_selector = QComboBox(self)
        self.image1_layout.addWidget(self.image_1_channel_selector)
        self.image_1_channel_selector.setVisible(False)
        
        # Image 2 layout
        self.image2_layout = QHBoxLayout()
        self.image2_label = QLabel("Unaligned Image:")
        self.image2_label.setMinimumWidth(120)
        self.image2_status = QLabel("not loaded")
        self.image2_status.setStyleSheet("font-weight: bold; color: #555;")
        self.image2_layout.addWidget(self.image2_label)
        self.image2_layout.addWidget(self.image2_status)
        self.image_2_channel_selector = QComboBox(self)
        self.image2_layout.addWidget(self.image_2_channel_selector)
        self.image_2_channel_selector.setVisible(False)
        self.image2_layout.setStretch(1, 1)  # Make the status label expand
        
        # Register button
        self.register_button = QPushButton("Register Images")
        self.register_button.setEnabled(False)  # Initially disabled until both images are loaded
        self.register_button.clicked.connect(self.register_images)
        self.register_button.setMinimumHeight(30)
        self.register_button.setStyleSheet("""
            QPushButton {
                background-color: #4CAF50;
                color: white;
                font-weight: bold;
                border-radius: 4px;
            }
            QPushButton:disabled {
                background-color: #cccccc;
                color: #666666;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
        """)
        
        # Add all elements to the main layout
        self.main_layout.addWidget(self.title_label)
        self.main_layout.addSpacing(5)
        self.main_layout.addLayout(self.image1_layout)
        self.main_layout.addLayout(self.image2_layout)
        self.main_layout.addSpacing(10)
        self.main_layout.addWidget(self.register_button)
        
        # Add the groupbox to the containing layout
        if containing_layout:
            containing_layout.addWidget(self.alignment_groupbox)
        
        self.__retranslate_UI()
        QMetaObject.connectSlotsByName(self)
    
    def _setup_connections(self):
        """Set up the connections to the aligner thread"""
        self.aligner.progress.connect(self._handle_progress)
        self.aligner.error.connect(self._handle_error)
        self.aligner.aligned_image_signal.connect(self._handle_aligned_image)
        self.aligner.finished.connect(self._handle_finished)
        
        self.image_1_channel_selector.currentIndexChanged.connect(self.change_image1_channel)
        self.image_2_channel_selector.currentIndexChanged.connect(self.change_image2_channel)
        
        
        # Also connect our errorSignal to pass errors to the parent
        self.aligner.error.connect(self.errorSignal)
    
    @pyqtSlot(int)
    def change_image1_channel(self, index):
        self.image_channels[0] = index
    
    @pyqtSlot(int)
    def change_image2_channel(self, index):
        self.image_channels[1] = index
    
    def __retranslate_UI(self):
        _translate = QCoreApplication.translate
        self.alignment_groupbox.setTitle(_translate("MainWindow", "Cell Layer Alignment"))
        self.title_label.setText(_translate("MainWindow", "Cell Layer Alignment"))
        self.image1_label.setText(_translate("MainWindow", "Target Image:"))
        self.image2_label.setText(_translate("MainWindow", "Unaligned Image:"))
        self.register_button.setText(_translate("MainWindow", "Register Images"))
    
    def set_target_image(self, obj, name):
        """Set the target image for alignment"""
        self.target_image = obj
        
        self.target_name = name
        self.image1_status.setText(name)
        self.image1_status.setStyleSheet("font-weight: bold; color: #007700;")  # Green to indicate it's loaded
        self.image_1_channel_selector.setVisible(True)
        self.image_1_channel_selector.clear()
        self.image_1_channel_selector.addItems(obj.keys())
        self._check_can_register()
    
    def set_unaligned_image(self, obj, name):
        """Set the unaligned image that will be registered to the target"""
        self.unaligned_image = obj
        self.unaligned_name = name
        self.image2_status.setText(name)
        self.image2_status.setStyleSheet("font-weight: bold; color: #007700;")  # Green to indicate it's loaded
        self.image_2_channel_selector.setVisible(True)
        self.image_2_channel_selector.addItems(obj.keys())

        self._check_can_register()
    
    def _check_can_register(self):
        """Check if both images are loaded and enable/disable register button"""
        if self.target_image is not None and self.unaligned_image is not None:
            self.register_button.setEnabled(True)
        else:
            self.register_button.setEnabled(False)
    
    def register_images(self):
        """Start the image registration process"""
        if not self.target_image is None and not self.unaligned_image is None:
            # Disable the register button during processing
            self.register_button.setEnabled(False)
            self.register_button.setText("Processing...")
            self.aligner.progress.emit(10, "Creating Pyramids")
            self.aligner.set_target_image(target_image=self.target_image[f"Channel {self.image_channels[0]+1}"].data)
            self.aligner.set_unaligned_image(unaligned_image=self.unaligned_image[f"Channel {self.image_channels[1]+1}"].data)
            # Start the alignment process in a separate thread
            self.aligner.start()
    
    def _handle_progress(self, value, message):
        """Handle progress updates from the aligner thread"""
        # You could add a progress bar to the UI if needed
        # For now, we'll just update the button text
        self.register_button.setText(f"{message} ({value}%)")
    
    def _handle_error(self, error_message):
        """Handle error messages from the aligner thread"""
        QMessageBox.critical(self, "Alignment Error", error_message)
        self.register_button.setEnabled(True)
        self.register_button.setText("Register Images")
        
        # Reset the image status colors to indicate failure
        if self.target_image is not None:
            self.image1_status.setStyleSheet("font-weight: bold; color: #FF0000;")  # Red to indicate error
        if self.unaligned_image is not None:
            self.image2_status.setStyleSheet("font-weight: bold; color: #FF0000;")  # Red to indicate error
    
    def _handle_aligned_image(self, aligned_image, target_small, aligned_small):
        """Handle the aligned image result"""
        # Store the aligned image
        self.aligned_image = aligned_image
        self.target_small = target_small
        self.aligned_small = aligned_small
        
        # Show the preview dialog
        self._show_preview_dialog(aligned_image, target_small, aligned_small)
    
    def _show_preview_dialog(self, aligned_image, target_small, aligned_small):
        """Show the preview dialog with red/green overlay"""
        # Use the downscaled images for the preview dialog
        preview_dialog = AlignmentPreviewDialog(target_small, aligned_small, self)
        result = preview_dialog.exec()
        
        if result == 1 and preview_dialog.result_accepted:  # User clicked Confirm
            # Generate a name for the aligned image
            aligned_name = f"Aligned_{self.unaligned_name}_to_{self.target_name}"
            
            # First, load the aligned image directly onto the canvas
            self.replaceLayerSignal.emit(aligned_image)
            
            # Then, emit the signal to replace the layer
            
            if aligned_image.dtype != np.uint8:
                aligned_image = scale_adjust(aligned_image)

            qimage = numpy_to_qimage(aligned_image)
            pixmap = QPixmap(qimage)
            pixmap = pixmap
            pixmapItem = QGraphicsPixmapItem(pixmap)
            self.loadOnCanvasSignal.emit(pixmapItem)
            # Also emit the signal to add the aligned image to the manager
            self.alignmentCompleteSignal.emit(aligned_image, aligned_name)
            
            # Show success message
            QMessageBox.information(self, "Alignment Complete", 
                                   f"Successfully aligned and loaded image onto canvas. Layer 2 in {self.target_name} was also replaced.")
        else:
            # User canceled, but still load onto canvas and add to manager
            aligned_name = f"Aligned_{self.unaligned_name}_to_{self.target_name}"
            
            # Load directly onto the canvas
            self.loadOnCanvasSignal.emit(aligned_image)
            
            # Add to the manager
            self.alignmentCompleteSignal.emit(aligned_image, aligned_name)
            
            QMessageBox.information(self, "Alignment Complete", 
                                   f"Alignment successful. The aligned image has been loaded onto the canvas and added to your workspace.")
            
    def _handle_finished(self):
        """Handle when the alignment thread finishes"""
        self.register_button.setEnabled(True)
        self.register_button.setText("Register Images")
        
        # Show a success message if there was no error
        if hasattr(self.aligner, 'result') and self.aligner.result is not None:
            # Set colors to indicate success
            self.image1_status.setStyleSheet("font-weight: bold; color: #007700;")  # Green to indicate success
            self.image2_status.setStyleSheet("font-weight: bold; color: #007700;")  # Green to indicate success
            
            QMessageBox.information(self, "Alignment Complete", 
                                   f"Successfully aligned {self.unaligned_name} to {self.target_name}") 