from PyQt6.QtGui import *
from PyQt6.QtCore import *
from PyQt6.QtWidgets import *
import os
from ui.toolbar.menubar_ui import MenuBarUI; from ui.toolbar.toolbar_ui import ToolBarUI; from ui.stardist.stardist_ui import StarDistUI; from ui.alignment.cell_intensity_ui import CellIntensityUI
from ui.processing.crop_ui import CropUI; from ui.processing.rotation_ui import RotateUI; from ui.canvas_ui import ImageGraphicsViewUI, ReferenceGraphicsViewUI
from ui.alignment.register_ui import RegisterUI
from ui.view_tab import ImageOverlay
from ui.analysis.AnalysisTab import AnalysisTab
from ui.processing.gaussian_blur import GaussianBlur
from core.canvas import MetaData
from ui.ImageManager import Manager
from ui.alignment.cell_layer_alignment_ui import CellLayerAlignmentUI
import numpy as np


class Ui_MainWindow(QMainWindow):
    def __init__(self, parent=None):
        QImageReader.setAllocationLimit(0)
        super().__init__()
        
        self.analysis_shortcut = QShortcut(QKeySequence("Ctrl+R"), self) # A for analysis
        self.analysis_shortcut.activated.connect(self.select)

        self.analysis_shortcut = QShortcut(QKeySequence("Ctrl+C"), self) # C for circle select
        self.analysis_shortcut.activated.connect(self.circle_select)

        self.analysis_shortcut = QShortcut(QKeySequence("Ctrl+P"), self) # C for circle select
        self.analysis_shortcut.activated.connect(self.poly_select)
        
        self.save_shortcut = QShortcut(QKeySequence("Ctrl+S"), self)
        self.save_shortcut.activated.connect(self.save)
        
        ### Formerly set up ui

        self.resize(1280, 800)
        self.setMinimumSize(QSize(1024, 768))
        self.centralwidget = QWidget(self) # central widget inside main window
        self.central_widget_layout = QHBoxLayout(self.centralwidget) # add a layout to centralwidget so window can resize properly
        self.main_layout = QHBoxLayout() # main layout to add align canvas and the tab

        # add a menubar
        self.menubar = MenuBarUI(self, enc=self)

        # add a toolbar
        self.toolBar = ToolBarUI(self)
        
        # Create a container widget for the side panel
        self.sidePanel = QWidget(self.centralwidget)
        self.sidePanelLayout = QVBoxLayout(self.sidePanel)
        self.sidePanelLayout.setContentsMargins(0, 0, 0, 0)
        self.sidePanelLayout.setSpacing(0)
        self.sidePanel.setMaximumWidth(500)
        
        # Add collapse/expand button
        self.toggleButton = QPushButton("◀", self.sidePanel)
        self.toggleButton.setFixedSize(20, 60)
        self.toggleButton.clicked.connect(self.toggleSidePanel)
        self.toggleButton.setStyleSheet("""
            QPushButton:hover {
                background-color: #e0e0e0;
            }
        """)
        
        # Create a horizontal layout for the side panel and toggle button
        self.sidePanelContainer = QHBoxLayout()
        self.sidePanelContainer.setContentsMargins(0, 0, 0, 0)
        self.sidePanelContainer.setSpacing(0)
        
        # Create the stacked widget directly in the side panel
        self.stackedWidget = QStackedWidget(self.sidePanel)
        
        # canvas
        self.canvas = ImageGraphicsViewUI(self.centralwidget, enc=self)
        self.canvas.setMinimumSize(QSize(800, 500))

        self.small_view = ReferenceGraphicsViewUI(self.centralwidget)
        self.small_view.setParent(self.canvas)
        self.small_view.hide()

        ## images workspace#####
        images_scroll = QScrollArea()
        images_scroll.setWidgetResizable(True)
        images_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        images_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        
        self.images_tab = Manager(self.canvas)
        self.images_tab.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Preferred)
        
        images_scroll.setWidget(self.images_tab)
        self.stackedWidget.addWidget(images_scroll)

        ####### preprocess tab ###################################
        # Create scroll area for preprocessing tab
        preprocess_scroll = QScrollArea()
        preprocess_scroll.setWidgetResizable(True)
        preprocess_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        preprocess_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        
        self.preprocessing_tab = QWidget()
        self.horizontalLayout = QHBoxLayout(self.preprocessing_tab)
        self.preprocessing_dockwidget_main_vlayout = QVBoxLayout()
        self.horizontalLayout.addLayout(self.preprocessing_dockwidget_main_vlayout)
        
        # reference view
        self.small_view = ReferenceGraphicsViewUI(self.centralwidget)
        self.small_view.setParent(self.canvas)
        self.small_view.hide()

        self.save_button = QPushButton('Save Canvas')
        self.save_button.clicked.connect(self.save_canvas)
        
        # crop
        self.crop_groupbox = CropUI(self.preprocessing_tab)
        
        # rotate
        self.rotate_groupbox = RotateUI(self.preprocessing_tab)
        
        
        # layout crop and rotate next to each other
        self.rotate_crop_hlayout = QHBoxLayout()
        
        self.rotate_crop_hlayout.addWidget(self.crop_groupbox.crop_groupbox)
        self.rotate_crop_hlayout.addWidget(self.rotate_groupbox.rotate_groupbox)
        # self.rotate_crop_hlayout.addWidget(self.gaussian_blur.gaussian_blur)
        self.rotate_crop_hlayout.setSpacing(3)
        
        self.preprocessing_dockwidget_main_vlayout.addLayout(self.rotate_crop_hlayout)

        # Add flip buttons
        self.flip_groupbox = QGroupBox("Flip Image")
        self.flip_layout = QHBoxLayout()
        
        self.flip_horizontal_btn = QPushButton("Flip Horizontal")
        self.flip_vertical_btn = QPushButton("Flip Vertical")
        
        # Use the canvas methods directly - this will update the UI pixmap and emit the signals
        self.flip_horizontal_btn.clicked.connect(self.canvas.flip_horizontal)
        self.flip_vertical_btn.clicked.connect(self.canvas.flip_vertical)
        
        self.flip_layout.addWidget(self.flip_horizontal_btn)
        self.flip_layout.addWidget(self.flip_vertical_btn)
        self.flip_groupbox.setLayout(self.flip_layout)
        
        self.preprocessing_dockwidget_main_vlayout.addWidget(self.flip_groupbox)

        self.register_groupbox = RegisterUI(self.preprocessing_tab, self.preprocessing_dockwidget_main_vlayout)
        self.gaussian_blur = GaussianBlur(self.preprocessing_tab, self.preprocessing_dockwidget_main_vlayout)

        # Add the cell layer alignment UI
        self.cell_layer_alignment = CellLayerAlignmentUI(self.preprocessing_tab, self.preprocessing_dockwidget_main_vlayout)
        
        # Now that cell_layer_alignment exists, connect the signals
        self.images_tab.tissue_target_selected.connect(self.cell_layer_alignment.set_target_image)
        self.images_tab.tissue_unaligned_selected.connect(self.cell_layer_alignment.set_unaligned_image)
        self.cell_layer_alignment.alignmentCompleteSignal.connect(self.add_item_to_manager)
        self.cell_layer_alignment.replaceLayerSignal.connect(self.replace_layer_in_canvas)
        self.cell_layer_alignment.loadOnCanvasSignal.connect(self.load_image_on_canvas)
        
        # Connect to progress bar
        self.cell_layer_alignment.aligner.progress.connect(self.update_progress_bar)
        
        # stardist UI
        self.stardist_groupbox = StarDistUI(self.preprocessing_tab, self.preprocessing_dockwidget_main_vlayout)

        self.cellIntensity_groupbox = CellIntensityUI(self.preprocessing_tab, self.preprocessing_dockwidget_main_vlayout)
        self.preprocessing_dockwidget_main_vlayout.addWidget(self.save_button)

        self.preprocessing_dockwidget_main_vlayout.setSpacing(5)
        self.preprocessing_dockwidget_main_vlayout.setContentsMargins(0, 0, 0, 0)

        preprocess_scroll.setWidget(self.preprocessing_tab)
        self.stackedWidget.addWidget(preprocess_scroll)

        ####### view tab #######################################
        # Create scroll area for view tab
        view_scroll = QScrollArea()
        view_scroll.setWidgetResizable(True)
        view_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        view_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        
        self.view_tab = ImageOverlay(self.canvas, enc=self)
        self.view_tab.setObjectName("view_tab")
        self.view_tab.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Preferred)
        view_scroll.setWidget(self.view_tab)
        self.stackedWidget.addWidget(view_scroll)
        
        ####### analysis tab #######################################
        # Create scroll area for analysis tab
        analysis_scroll = QScrollArea()
        analysis_scroll.setWidgetResizable(True)
        analysis_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        analysis_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        
        self.analysis_tab = AnalysisTab(self.canvas, self)
        self.analysis_tab.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Preferred)
        self.analysis_tab.setObjectName("analysis_tab")
        analysis_scroll.setWidget(self.analysis_tab)
        self.stackedWidget.addWidget(analysis_scroll)


        metadata_scroll = QScrollArea()
        metadata_scroll.setWidgetResizable(True)
        metadata_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        metadata_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.metadata_tab = MetaData(self.canvas)
        self.metadata_tab.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Preferred)

        metadata_scroll.setWidget(self.metadata_tab)
        self.stackedWidget.addWidget(metadata_scroll)

        # Set up the side panel layout
        self.sidePanelLayout.addWidget(self.stackedWidget)
        
        # Add the side panel and toggle button to the container
        self.sidePanelContainer.addWidget(self.sidePanel)
        self.sidePanelContainer.addWidget(self.toggleButton)
        
        # Add the container to the main layout
        self.main_layout.addLayout(self.sidePanelContainer)
        self.main_layout.addWidget(self.canvas)
        
        self.central_widget_layout.addLayout(self.main_layout)
        self.setCentralWidget(self.centralwidget)
        
        # status bar
        container = QWidget()
        self.statusbar = QStatusBar(self)
        self.setStatusBar(self.statusbar)
        self.progressBarLabel = QLabel("")
        self.progressBar = QProgressBar()
        progressBarLayout = QHBoxLayout()
        self.progressBar.setMaximum(100)
        progressBarLayout.addWidget(self.progressBarLabel)
        progressBarLayout.addWidget(self.progressBar)
        container.setLayout(progressBarLayout)

        self.statusbar.addPermanentWidget(container)
        
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
  
        self.progressBar.text
        self.__retranslateUI()

        # Connect toolbar tab change signal
        self.toolBar.tabChanged.connect(self.stackedWidget.setCurrentIndex)
        # Start with Images tab
        self.stackedWidget.setCurrentIndex(0)

        QMetaObject.connectSlotsByName(self)
        

    def updateMousePositionLabel(self, text):
        self.toolBar.statusLine.setText(text)
    
    def __retranslateUI(self):
        _translate = QCoreApplication.translate
        self.setWindowTitle(_translate("MainWindow", "MIST-Explorer"))

    saveSignal = pyqtSignal()
    def save_canvas(self):
        print("saving")
        self.saveSignal.emit()
        
    def update_progress_bar(self, value, str):
        if self.progressBar.value() == 100:
            self.progressBar.reset()
        self.progressBar.setValue(value)
        self.progressBarLabel.setText(str + " ...")

    def select(self):
        print("selecting")
        self.canvas.select = "rect"

    def circle_select(self):
        print("selecting")
        self.canvas.select = "circle"

    def poly_select(self):
        if self.canvas.select == "poly":
            self.canvas.select = False
            return
        
        self.canvas.select = "poly"
    
    def save(self):
        from PIL import Image
        from PyQt6.QtWidgets import QFileDialog
        import numpy as np
        
        file_name, _ = QFileDialog.getSaveFileName(None, "Save File", "image.png", "*.png;;*.jpg;;*.tif;; All Files(*)")
        
        if file_name:
            # Instead of calling grab(), use pixmapItem directly if available
            if hasattr(self.canvas, 'pixmapItem') and self.canvas.pixmapItem:
                pixmap = self.canvas.pixmapItem.pixmap()
            else:
                pixmap = self.canvas.grab()
            
            qimage = pixmap.toImage()
            buffer = qimage.bits().asstring(qimage.width() * qimage.height() * qimage.depth() // 8)
            image = np.frombuffer(buffer, dtype=np.uint8).reshape((qimage.height(), qimage.width(), qimage.depth() // 8))
            
            if image.shape[2] == 4:  # If the image has an alpha channel
                image = image[:, :, :3]  # Remove the alpha channel
            
            image = image[:, :, ::-1]  # Convert BGR to RGB
            Image.fromarray(image).save(file_name)

    def toggleSidePanel(self):
        if self.sidePanel.isVisible():
            self.sidePanel.hide()
            self.toggleButton.setText("▶")
        else:
            self.sidePanel.show()
            self.toggleButton.setText("◀")

    def get_metadata(self, metadata: dict):
        self.metadata = metadata
        self.metadata_tab.populate_table(self.metadata)

    def add_item_to_manager(self, data, name):
        self.images_tab.add_item(data, name)

    def replace_layer_in_canvas(self, target_image, aligned_image):
        """Replace the second layer in the target image with the aligned image"""
        # Access the correct attributes from the image_tab which holds the channels
        try:
            # Debug prints
            print("Canvas attributes:", dir(self.canvas))
            print("Has np_channels:", hasattr(self.canvas, 'np_channels'))
            if hasattr(self.canvas, 'np_channels'):
                print("np_channels keys:", self.canvas.np_channels.keys())
            
            # If we don't have channels yet, we'll need to create them
            channel_to_replace = "Channel 2"
            
            # Check if we have image channels (layered image)
            if not hasattr(self.canvas, 'np_channels') or not self.canvas.np_channels:
                # Initialize channels if they don't exist
                print("Initializing channels")
                from dataclasses import dataclass
                
                @dataclass
                class ChannelInfo:
                    data: np.ndarray
                    cmap: str = 'gray'
                
                # Initialize np_channels if it doesn't exist
                self.canvas.np_channels = {}
                
                # If we have a pixmap in the canvas, use it as Channel 1
                if hasattr(self.canvas, 'pixmapItem') and self.canvas.pixmapItem:
                    print("Using existing pixmap for Channel 1")
                    # Convert pixmap to numpy array
                    pixmap = self.canvas.pixmapItem.pixmap()
                    image = pixmap.toImage()
                    width = image.width()
                    height = image.height()
                    
                    # Convert QImage to numpy array
                    ptr = image.bits()
                    ptr.setsize(height * width * 4)  # 4 for RGBA
                    arr = np.frombuffer(ptr, np.uint8).reshape((height, width, 4))
                    
                    # Add as Channel 1
                    self.canvas.np_channels["Channel 1"] = ChannelInfo(arr[:,:,:3])  # RGB only
                
                # Add aligned image as Channel 2
                self.canvas.np_channels[channel_to_replace] = ChannelInfo(aligned_image)
                
                # Set current channel number
                self.canvas.currentChannelNum = 1  # Index 1 corresponds to Channel 2
                
                # Make sure the canvas has the methods it needs
                if not hasattr(self.canvas, 'update_image'):
                    print("Warning: Canvas doesn't have update_image method")
                    # Fall back to using our load_image_on_canvas function instead
                    self.load_image_on_canvas(aligned_image)
                    return True
                
                # Update the canvas
                self.canvas.update_image('gray')
                
                # Update progress bar
                self.update_progress_bar(100, f"Added {channel_to_replace} as a new layer")
                return True
            
            # We have channels, so try to replace or add Channel 2
            # Get current active channel (or default to 0 if not found)
            current_active_channel = 0
            if hasattr(self.canvas, 'currentChannelNum'):
                current_active_channel = self.canvas.currentChannelNum
                print("Current active channel:", current_active_channel)
            
            # Check if Channel 2 exists already
            if channel_to_replace in self.canvas.np_channels:
                print(f"Replacing {channel_to_replace} with aligned image")
                # Replace the data in Channel 2
                self.canvas.np_channels[channel_to_replace].data = aligned_image
            else:
                print(f"Adding {channel_to_replace} as a new channel")
                # Add Channel 2 if it doesn't exist
                from dataclasses import dataclass
                
                @dataclass
                class ChannelInfo:
                    data: np.ndarray
                    cmap: str = 'gray'
                
                self.canvas.np_channels[channel_to_replace] = ChannelInfo(aligned_image)
            
            # Clear any image caches
            if hasattr(self.canvas, 'image_cache'):
                self.canvas.image_cache.clear()
            
            # Temporarily switch to Channel 2 and update it
            self.canvas.currentChannelNum = 1  # Index 1 corresponds to Channel 2
            
            # Check if update_image method exists
            if not hasattr(self.canvas, 'update_image'):
                print("Warning: Canvas doesn't have update_image method")
                # Fall back to using our load_image_on_canvas function instead
                self.load_image_on_canvas(aligned_image)
                return True
            
            # Get colormap from channel if available, otherwise use gray
            cmap = 'gray'
            if hasattr(self.canvas.np_channels[channel_to_replace], 'cmap'):
                cmap = self.canvas.np_channels[channel_to_replace].cmap
            
            # Update the image
            self.canvas.update_image(cmap)
            
            # Update the channels in the UI
            self.canvas.loadChannels(self.canvas.np_channels)
            
            # Restore original active channel if different
            if current_active_channel != 1 and hasattr(self.canvas, 'swap_channel'):
                self.canvas.swap_channel(current_active_channel)
            
            # Update progress bar
            self.update_progress_bar(100, f"Replaced {channel_to_replace} with aligned image")
            return True
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Error replacing layer: {str(e)}")
            import traceback
            traceback.print_exc()
            return False

    def load_image_on_canvas(self, image):
        """Load the given image directly onto the canvas"""
        try:
            if image is None:
                return

            # Convert the image to a pixmap and display it on the canvas
            from utils import numpy_to_qimage
            from PyQt6.QtGui import QPixmap
            
            # Make sure image is in the right format for display
            if image.dtype != np.uint8:
                # Scale to 8-bit for display if needed
                if image.max() > 255:
                    image_for_display = ((image / image.max()) * 255).astype(np.uint8)
                else:
                    image_for_display = image.astype(np.uint8)
            else:
                image_for_display = image
            
            # Convert to QImage and then QPixmap
            q_image = numpy_to_qimage(image_for_display)
            pixmap = QPixmap(q_image)
            
            # Check if we already have a pixmapItem
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
            
            # Signal the canvas to update
            self.canvas.canvasUpdated.emit(pixmap)
            
            # Update progress bar
            self.update_progress_bar(100, f"Loaded aligned image onto canvas")
            
            return True
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Error loading image onto canvas: {str(e)}")
            import traceback
            traceback.print_exc()
            return False