from PyQt6.QtGui import *
from PyQt6.QtCore import *
from PyQt6.QtWidgets import *
import os
import argparse
from ui.toolbar.menubar_ui import MenuBarUI
from ui.toolbar.toolbar_ui import ToolBarUI
from ui.stardist.stardist_ui import StarDistUI
from ui.alignment.cell_intensity_ui import CellIntensityUI
from ui.processing.crop_ui import CropUI
from ui.processing.rotation_ui import RotateUI
from ui.canvas_ui import ImageGraphicsViewUI, ReferenceGraphicsViewUI
from ui.alignment.register_ui import RegisterUI
from ui.view_tab import ImageOverlay
from ui.analysis.AnalysisTab import AnalysisTab
from ui.processing.gaussian_blur import GaussianBlur
from core.canvas import MetaData
from ui.ImageManager import Manager
from ui.alignment.cell_layer_alignment_ui import CellLayerAlignmentUI
import numpy as np


class Ui_MainWindow(QMainWindow):
    
    saveSignal = pyqtSignal()
    
    def __init__(self, parent=None):
        QImageReader.setAllocationLimit(0)
        super().__init__()
        
        self.args = self._parse_arguments() # Enables passing in image & reference as cli arguments
        self._setup_main_window()
        self._add_shortcuts()
        
        self._setup_central_widget()
        self._setup_menubar_and_toolbar()
        self._setup_side_panel()
        self._setup_canvas()
        
        # Tab setup!
        self._setup_images_tab()
        self._setup_preprocessing_tab()
        self._setup_view_tab()
        self._setup_analysis_tab()
        self._setup_metadata_tab()
        
        self._setup_status_bar()
        self._setup_layout()
    
        self._connect_signals()
        self._retranslate_UI()
        QMetaObject.connectSlotsByName(self)
    
    def _parse_arguments(self):
        """Parse command line arguments"""
        parser = argparse.ArgumentParser(
            prog='MIST-Explorer',
            description='Working on it...',
            epilog='Intended for testing'
        )
        parser.add_argument('-i', '--image')  # image path
        parser.add_argument('-r', '--reference')  # reference path
        return parser.parse_args()
    
    def _setup_main_window(self):
        """Setup main window properties"""
        self.resize(1280, 800)
        self.setMinimumSize(QSize(1024, 768))
    
    def _add_shortcuts(self):
        """Add keyboard shortcuts"""
        
        # Can add more shortcuts by adding tuple in form: (key_press_string, function)
        shortcuts = [
            ("Ctrl+R", self.select),
            ("Ctrl+C", self.circle_select),
            ("Ctrl+P", self.poly_select),
            ("Ctrl+S", self.save),
            # A for analysis?, was previously a comment but was outdated?
        ]
        
        for key_sequence, slot in shortcuts:
            shortcut = QShortcut(QKeySequence(key_sequence), self)
            shortcut.activated.connect(slot)
    
    def _setup_central_widget(self):
        self.centralwidget = QWidget(self)
        self.central_widget_layout = QHBoxLayout(self.centralwidget)
        self.main_layout = QHBoxLayout()
        self.setCentralWidget(self.centralwidget)
    
    def _setup_menubar_and_toolbar(self):
        self.menubar = MenuBarUI(self, enc=self)
        self.toolBar = ToolBarUI(self)
    
    def _setup_side_panel(self):
        """Setup the collapsible side panel"""
        # Create side panel container
        self.sidePanel = QWidget(self.centralwidget)
        self.sidePanelLayout = QVBoxLayout(self.sidePanel)
        self.sidePanelLayout.setContentsMargins(0, 0, 0, 0)
        self.sidePanelLayout.setSpacing(0)
        self.sidePanel.setMaximumWidth(500)
        
        # Create toggle button
        self.toggleButton = QPushButton("◀", self.sidePanel)
        self.toggleButton.setFixedSize(20, 60)
        self.toggleButton.clicked.connect(self.toggleSidePanel)
        self.toggleButton.setStyleSheet("""
            QPushButton:hover {
                background-color: #e0e0e0;
            }
        """)
        
        # Create side panel container layout
        self.sidePanelContainer = QHBoxLayout()
        self.sidePanelContainer.setContentsMargins(0, 0, 0, 0)
        self.sidePanelContainer.setSpacing(0)
        
        # Create stacked widget for tabs
        self.stackedWidget = QStackedWidget(self.sidePanel)
        self.sidePanelLayout.addWidget(self.stackedWidget)
        
        # Add to container
        self.sidePanelContainer.addWidget(self.sidePanel)
        self.sidePanelContainer.addWidget(self.toggleButton)
    
    def _setup_canvas(self):
        """Setup the main canvas and reference view"""
        self.canvas = ImageGraphicsViewUI(self.centralwidget, enc=self)
        self.canvas.setMinimumSize(QSize(800, 500))
        
        self.small_view = ReferenceGraphicsViewUI(self.centralwidget)
        self.small_view.setParent(self.canvas)
        self.small_view.hide()
    
    def _setup_images_tab(self):
        """Setup the images workspace tab"""
        images_scroll = self._create_scroll_area()
        
        self.images_tab = Manager(self.canvas)
        self.images_tab.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Preferred)
        
        images_scroll.setWidget(self.images_tab)
        self.stackedWidget.addWidget(images_scroll)
    
    def _setup_preprocessing_tab(self):
        """Setup the preprocessing tab with all its components"""
        preprocess_scroll = self._create_scroll_area()
        
        # Create main preprocessing widget
        self.preprocessing_tab = QWidget()
        self.horizontalLayout = QHBoxLayout(self.preprocessing_tab)
        self.preprocessing_dockwidget_main_vlayout = QVBoxLayout()
        self.horizontalLayout.addLayout(self.preprocessing_dockwidget_main_vlayout)
        
        # Setup preprocessing components
        self._setup_preprocessing_components()
        
        # Set layout properties
        self.preprocessing_dockwidget_main_vlayout.setSpacing(5)
        self.preprocessing_dockwidget_main_vlayout.setContentsMargins(0, 0, 0, 0)
        
        preprocess_scroll.setWidget(self.preprocessing_tab)
        self.stackedWidget.addWidget(preprocess_scroll)
    
    def _setup_preprocessing_components(self):
        """Setup individual preprocessing components"""
        # Save button
        self.save_button = QPushButton('Save Canvas')
        self.save_button.clicked.connect(self.save_canvas)
        
        # Crop and rotate components
        self._setup_crop_rotate_components()
        
        # Flip components
        self._setup_flip_components()
        
        # Other processing components
        self.register_groupbox = RegisterUI(self.preprocessing_tab, self.preprocessing_dockwidget_main_vlayout)
        self.gaussian_blur = GaussianBlur(self.preprocessing_tab, self.preprocessing_dockwidget_main_vlayout)
        
        # Cell layer alignment
        self._setup_cell_layer_alignment()
        
        # StarDist and Cell Intensity
        self.stardist_groupbox = StarDistUI(self.preprocessing_tab, self.preprocessing_dockwidget_main_vlayout)
        self.cellIntensity_groupbox = CellIntensityUI(self.preprocessing_tab, self.preprocessing_dockwidget_main_vlayout)
        
        # Add save button at the end
        self.preprocessing_dockwidget_main_vlayout.addWidget(self.save_button)
    
    def _setup_crop_rotate_components(self):
        """Setup crop and rotate components side by side"""
        self.crop_groupbox = CropUI(self.preprocessing_tab)
        self.rotate_groupbox = RotateUI(self.preprocessing_tab)
        
        # Layout crop and rotate next to each other
        self.rotate_crop_hlayout = QHBoxLayout()
        self.rotate_crop_hlayout.addWidget(self.crop_groupbox.crop_groupbox)
        self.rotate_crop_hlayout.addWidget(self.rotate_groupbox.rotate_groupbox)
        self.rotate_crop_hlayout.setSpacing(3)
        
        self.preprocessing_dockwidget_main_vlayout.addLayout(self.rotate_crop_hlayout)
    
    def _setup_flip_components(self):
        """Setup flip buttons"""
        self.flip_groupbox = QGroupBox("Flip Image")
        self.flip_layout = QHBoxLayout()
        
        self.flip_horizontal_btn = QPushButton("Flip Horizontal")
        self.flip_vertical_btn = QPushButton("Flip Vertical")
        
        # Connect to canvas flip methods
        self.flip_horizontal_btn.clicked.connect(self.canvas.flip_horizontal)
        self.flip_vertical_btn.clicked.connect(self.canvas.flip_vertical)
        
        self.flip_layout.addWidget(self.flip_horizontal_btn)
        self.flip_layout.addWidget(self.flip_vertical_btn)
        self.flip_groupbox.setLayout(self.flip_layout)
        
        self.preprocessing_dockwidget_main_vlayout.addWidget(self.flip_groupbox)
    
    def _setup_cell_layer_alignment(self):
        """Setup cell layer alignment component and its connections"""
        self.cell_layer_alignment = CellLayerAlignmentUI(
            self.preprocessing_tab, 
            self.preprocessing_dockwidget_main_vlayout
        )
        
        # Connect signals
        self.images_tab.tissue_target_selected.connect(self.cell_layer_alignment.set_target_image)
        self.images_tab.tissue_unaligned_selected.connect(self.cell_layer_alignment.set_unaligned_image)
        self.cell_layer_alignment.alignmentCompleteSignal.connect(self.add_item_to_manager)
        self.cell_layer_alignment.replaceLayerSignal.connect(self.replace_layer_in_canvas)
        self.cell_layer_alignment.loadOnCanvasSignal.connect(self.load_image_on_canvas)
        self.cell_layer_alignment.aligner.progress.connect(self.update_progress_bar)
    
    def _setup_view_tab(self):
        """Setup the view tab"""
        view_scroll = self._create_scroll_area()
        
        self.view_tab = ImageOverlay(self.canvas, enc=self)
        self.view_tab.setObjectName("view_tab")
        self.view_tab.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Preferred)
        
        view_scroll.setWidget(self.view_tab)
        self.stackedWidget.addWidget(view_scroll)
    
    def _setup_analysis_tab(self):
        """Setup the analysis tab"""
        analysis_scroll = self._create_scroll_area()
        
        self.analysis_tab = AnalysisTab(self.canvas, self)
        self.analysis_tab.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Preferred)
        self.analysis_tab.setObjectName("analysis_tab")
        
        analysis_scroll.setWidget(self.analysis_tab)
        self.stackedWidget.addWidget(analysis_scroll)
    
    def _setup_metadata_tab(self):
        """Setup the metadata tab"""
        metadata_scroll = self._create_scroll_area()
        
        self.metadata_tab = MetaData(self.canvas)
        self.metadata_tab.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Preferred)
        
        metadata_scroll.setWidget(self.metadata_tab)
        self.stackedWidget.addWidget(metadata_scroll)
    
    def _create_scroll_area(self):
        """Create a standardized scroll area for tabs"""
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        return scroll_area
    
    def _setup_status_bar(self):
        """Setup status bar with progress bar"""
        container = QWidget()
        self.statusbar = QStatusBar(self)
        self.setStatusBar(self.statusbar)
        
        self.progressBarLabel = QLabel("")
        self.progressBar = QProgressBar()
        self.progressBar.setMaximum(100)
        
        progressBarLayout = QHBoxLayout()
        progressBarLayout.addWidget(self.progressBarLabel)
        progressBarLayout.addWidget(self.progressBar)
        container.setLayout(progressBarLayout)
        
        self.statusbar.addPermanentWidget(container)
        
        # Style the progress bar
        progressBarStyle = """
            QProgressBar {
                border: 2px solid grey;
                border-radius: 2px;
                text-align: right;
                height: 5px;
                margin-right: 30px;
            }
            QProgressBar::chunk {
                background-color: green;
                width: 20px;
            }
        """
        self.progressBar.setStyleSheet(progressBarStyle)
    
    def _setup_layout(self):
        """Setup the final layout structure"""
        # Add components to main layout
        self.main_layout.addLayout(self.sidePanelContainer)
        self.main_layout.addWidget(self.canvas)
        
        # Add main layout to central widget
        self.central_widget_layout.addLayout(self.main_layout)
    
    def _connect_signals(self):
        """Connect remaining signals"""
        # Connect toolbar tab change signal
        self.toolBar.tabChanged.connect(self.stackedWidget.setCurrentIndex)
        # Start with Images tab
        self.stackedWidget.setCurrentIndex(0)
    
    def _retranslate_UI(self):
        """Set UI text and translations"""
        _translate = QCoreApplication.translate
        self.setWindowTitle(_translate("MainWindow", "MIST-Explorer"))
    
    # Event handlers and utility methods
    def updateMousePositionLabel(self, text):
        """Update mouse position in toolbar"""
        self.toolBar.statusLine.setText(text)
    
    def save_canvas(self):
        """Save canvas signal emission"""
        print("saving")
        self.saveSignal.emit()
    
    def update_progress_bar(self, value, text):
        """Update progress bar with value and text"""
        if self.progressBar.value() == 100:
            self.progressBar.reset()
        self.progressBar.setValue(value)
        self.progressBarLabel.setText(text + " ...")
    
    def select(self):
        """Set rectangle selection mode"""
        print("selecting")
        self.canvas.select = "rect"
    
    def circle_select(self):
        """Set circle selection mode"""
        print("selecting")
        self.canvas.select = "circle"
    
    def poly_select(self):
        """Toggle polygon selection mode"""
        if self.canvas.select == "poly":
            self.canvas.select = False
            return
        self.canvas.select = "poly"
    
    def save(self):
        """Save current canvas to file"""
        from PIL import Image
        from PyQt6.QtWidgets import QFileDialog
        import numpy as np
        
        file_name, _ = QFileDialog.getSaveFileName(
            None, "Save File", "image.png", 
            "*.png;;*.jpg;;*.tif;; All Files(*)"
        )
        
        if file_name:
            # Use pixmapItem directly if available
            if hasattr(self.canvas, 'pixmapItem') and self.canvas.pixmapItem:
                pixmap = self.canvas.pixmapItem.pixmap()
            else:
                pixmap = self.canvas.grab()
            
            # Convert to numpy array and save
            qimage = pixmap.toImage()
            buffer = qimage.bits().asstring(qimage.width() * qimage.height() * qimage.depth() // 8)
            image = np.frombuffer(buffer, dtype=np.uint8).reshape(
                (qimage.height(), qimage.width(), qimage.depth() // 8)
            )
            
            if image.shape[2] == 4:  # Remove alpha channel if present
                image = image[:, :, :3]
            
            image = image[:, :, ::-1]  # Convert BGR to RGB
            Image.fromarray(image).save(file_name)
    
    def toggleSidePanel(self):
        """Toggle side panel visibility"""
        if self.sidePanel.isVisible():
            self.sidePanel.hide()
            self.toggleButton.setText("▶")
        else:
            self.sidePanel.show()
            self.toggleButton.setText("◀")
    
    def get_metadata(self, metadata: dict):
        """Update metadata tab with new metadata"""
        self.metadata = metadata
        self.metadata_tab.populate_table(self.metadata)
    
    def add_item_to_manager(self, data, name):
        """Add item to image manager"""
        self.images_tab.add_item(data, name)
    
    def replace_layer_in_canvas(self, target_image, aligned_image):
        """Replace the second layer in the target image with the aligned image"""
        try:
            # Debug prints
            print("Canvas attributes:", dir(self.canvas))
            print("Has np_channels:", hasattr(self.canvas, 'np_channels'))
            if hasattr(self.canvas, 'np_channels'):
                print("np_channels keys:", self.canvas.np_channels.keys())
            
            channel_to_replace = "Channel 2"
            
            # Initialize channels if they don't exist
            if not hasattr(self.canvas, 'np_channels') or not self.canvas.np_channels:
                self._initialize_canvas_channels(aligned_image, channel_to_replace)
                return True
            
            # Replace or add Channel 2
            self._update_canvas_channel(aligned_image, channel_to_replace)
            return True
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Error replacing layer: {str(e)}")
            import traceback
            traceback.print_exc()
            return False
    
    def _initialize_canvas_channels(self, aligned_image, channel_name):
        """Initialize canvas channels when they don't exist"""
        from dataclasses import dataclass
        
        @dataclass
        class ChannelInfo:
            data: np.ndarray
            cmap: str = 'gray'
        
        print("Initializing channels")
        self.canvas.np_channels = {}
        
        # Add existing pixmap as Channel 1 if available
        if hasattr(self.canvas, 'pixmapItem') and self.canvas.pixmapItem:
            print("Using existing pixmap for Channel 1")
            pixmap = self.canvas.pixmapItem.pixmap()
            image = pixmap.toImage()
            width, height = image.width(), image.height()
            
            ptr = image.bits()
            ptr.setsize(height * width * 4)  # 4 for RGBAa
            arr = np.frombuffer(ptr, np.uint8).reshape((height, width, 4))
            
            self.canvas.np_channels["Channel 1"] = ChannelInfo(arr[:,:,:3])  # RGB only
        
        # Add aligned image as Channel 2
        self.canvas.np_channels[channel_name] = ChannelInfo(aligned_image)
        self.canvas.currentChannelNum = 1  # Index 1 corresponds to Channel 2
        
        # Update canvas
        if hasattr(self.canvas, 'update_image'):
            self.canvas.update_image('gray')
        else:
            self.load_image_on_canvas(aligned_image)
        
        self.update_progress_bar(100, f"Added {channel_name} as a new layer")
    
    def _update_canvas_channel(self, aligned_image, channel_name):
        """Update existing canvas channel"""
        from dataclasses import dataclass
        
        @dataclass
        class ChannelInfo:
            data: np.ndarray
            cmap: str = 'gray'
        
        current_active_channel = getattr(self.canvas, 'currentChannelNum', 0)
        print("Current active channel:", current_active_channel)
        
        # Replace or add Channel 2
        if channel_name in self.canvas.np_channels:
            print(f"Replacing {channel_name} with aligned image")
            self.canvas.np_channels[channel_name].data = aligned_image
        else:
            print(f"Adding {channel_name} as a new channel")
            self.canvas.np_channels[channel_name] = ChannelInfo(aligned_image)
        
        # Clear image cache if it exists
        if hasattr(self.canvas, 'image_cache'):
            self.canvas.image_cache.clear()
        
        # Update Channel 2
        self.canvas.currentChannelNum = 1  # Index 1 corresponds to Channel 2
        
        if hasattr(self.canvas, 'update_image'):
            cmap = getattr(self.canvas.np_channels[channel_name], 'cmap', 'gray')
            self.canvas.update_image(cmap)
            self.canvas.loadChannels(self.canvas.np_channels)
            
            # Restore original active channel if different
            if current_active_channel != 1 and hasattr(self.canvas, 'swap_channel'):
                self.canvas.swap_channel(current_active_channel)
        else:
            self.load_image_on_canvas(aligned_image)
        
        self.update_progress_bar(100, f"Replaced {channel_name} with aligned image")
    
    def load_image_on_canvas(self, image):
        """Load the given image directly onto the canvas"""
        try:
            if image is None:
                return

            from utils import numpy_to_qimage
            from PyQt6.QtGui import QPixmap
            
            # Prepare image for display
            image_for_display = self._prepare_image_for_display(image)
            
            # Convert to QImage and QPixmap
            q_image = numpy_to_qimage(image_for_display)
            pixmap = QPixmap(q_image)
            
            # Update or create pixmapItem
            self._update_canvas_pixmap(pixmap)
            
            self.update_progress_bar(100, "Loaded aligned image onto canvas")
            return True
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Error loading image onto canvas: {str(e)}")
            import traceback
            traceback.print_exc()
            return False
    
    def _prepare_image_for_display(self, image):
        """Prepare numpy image for display"""
        if image.dtype != np.uint8:
            if image.max() > 255:
                return ((image / image.max()) * 255).astype(np.uint8)
            else:
                return image.astype(np.uint8)
        return image
    
    def _update_canvas_pixmap(self, pixmap):
        """Update canvas with new pixmap"""
        if hasattr(self.canvas, 'pixmapItem') and self.canvas.pixmapItem:
            # Update existing pixmap
            self.canvas.pixmapItem.setPixmap(pixmap)
            self.canvas.pixmap = pixmap
        else:
            # Create new pixmapItem
            pixmap_item = QGraphicsPixmapItem(pixmap)
            self.canvas.pixmap = pixmap
            self.canvas.pixmapItem = pixmap_item
            self.canvas.scene().addItem(pixmap_item)
            
            # Center and fit in view
            item_rect = pixmap_item.boundingRect()
            self.canvas.setSceneRect(item_rect)
            self.canvas.fitInView(item_rect, Qt.AspectRatioMode.KeepAspectRatio)
        
        # Signal canvas update
        self.canvas.canvasUpdated.emit(pixmap)