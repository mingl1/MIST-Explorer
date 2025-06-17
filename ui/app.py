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

        self.args = (
            self._parse_arguments()
        )  # Enables passing in image & reference as cli arguments
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
            prog="MIST-Explorer",
            description="Working on it...",
            epilog="Intended for testing",
        )
        parser.add_argument("-i", "--image")  # image path
        parser.add_argument("-r", "--reference")  # reference path
        return parser.parse_args()

    def _setup_main_window(self):
        """Setup main window properties"""
        self.resize(1280, 800)
        self.setMinimumSize(1200, 800)

    def _add_shortcuts(self):
        """Add keyboard shortcuts"""

        # Can add more shortcuts by adding tuple in form: (key_press_string, function)
        shortcuts = [
            ("Ctrl+R", self.select),
            ("Ctrl+C", self.circle_select),
            ("Ctrl+P", self.poly_select),
            ("Ctrl+S", self.save),
        ]

        for key_sequence, slot in shortcuts:
            shortcut = QShortcut(QKeySequence(key_sequence), self)
            shortcut.activated.connect(slot)

    def _setup_central_widget(self):
        widget = QWidget(self)
        self.central_widget_layout = QHBoxLayout(widget)
        self.main_layout = QHBoxLayout()
        self.setCentralWidget(widget)

    def _setup_menubar_and_toolbar(self):
        self.menuBarUI = MenuBarUI(self)
        self.setMenuBar(self.menuBarUI.get_menubar())

        self.toolBarUI = ToolBarUI(self)
        self.addToolBar(Qt.ToolBarArea.TopToolBarArea, self.toolBarUI.get_toolbar())

    def _setup_side_panel(self):
        """Setup the collapsible side panel"""

        self.sidePanelContainer = QHBoxLayout()
        self.sidePanelContainer.setContentsMargins(0, 0, 0, 0)
        self.sidePanelContainer.setSpacing(0)

        self.sidePanel = QWidget(self.centralWidget())
        self.sidePanelLayout = QVBoxLayout(self.sidePanel)
        self.sidePanelLayout.setContentsMargins(10, 5, 10, 5)
        self.sidePanelLayout.setSpacing(10)
        self.sidePanel.setMinimumWidth(400)
        self.sidePanel.setMinimumWidth(500)
        self.sidePanel.setSizePolicy(
            QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Expanding
        )

        self.toggleButton = QPushButton("◀", self.sidePanel)
        self.toggleButton.setFixedSize(20, 60)
        self.toggleButton.clicked.connect(self.toggleSidePanel)
        self.toggleButton.setStyleSheet(
            """
            QPushButton:hover {
                background-color: #e0e0e0;
            }
        """
        )

        # Create stacked widget for tabs
        self.stackedWidget = QStackedWidget(self.sidePanel)
        self.sidePanelLayout.addWidget(self.stackedWidget)

        # Add to container
        self.sidePanelContainer.addWidget(self.sidePanel)
        self.sidePanelContainer.addWidget(self.toggleButton)

    def _setup_canvas(self):
        """Setup the main canvas and reference view"""
        self.canvas = ImageGraphicsViewUI(self.centralWidget(), enc=self)
        self.canvas.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
        )
        self.small_view = ReferenceGraphicsViewUI(self.centralWidget())
        self.small_view.setParent(self.canvas)
        self.small_view.hide()

    def _setup_images_tab(self):
        """Setup the images workspace tab"""
        images_scroll = self._create_scroll_area()

        self.images_tab = Manager(self.canvas)
        self.images_tab.setSizePolicy(
            QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Preferred
        )

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
        self.save_button = QPushButton("Save Canvas")
        self.save_button.clicked.connect(self.save_canvas)

        # Crop and rotate components
        self._setup_crop_rotate_components()

        # Flip components
        self._setup_flip_components()

        # Cell layer alignment
        self._setup_cell_layer_alignment()

        # Other processing components
        self.register_groupbox = RegisterUI(
            self.preprocessing_tab, self.preprocessing_dockwidget_main_vlayout
        )
        self.gaussian_blur = GaussianBlur(
            self.preprocessing_tab, self.preprocessing_dockwidget_main_vlayout
        )

        # StarDist and Cell Intensity
        self.stardist_groupbox = StarDistUI(
            self.preprocessing_tab, self.preprocessing_dockwidget_main_vlayout
        )
        self.cellIntensity_groupbox = CellIntensityUI(
            self.preprocessing_tab, self.preprocessing_dockwidget_main_vlayout
        )

        # Add save button at the end
        self.preprocessing_dockwidget_main_vlayout.addWidget(self.save_button)

    def _setup_crop_rotate_components(self):
        """Setup crop and rotate components side by side"""
        self.crop_groupbox = CropUI(self.preprocessing_tab)
        self.rotate_groupbox = RotateUI(self.preprocessing_tab)

        # Layout crop and rotate next to each other
        self.rotate_crop_hlayout = QHBoxLayout()
        self.rotate_crop_hlayout.setSpacing(3)

        self.images_tab.layout().addWidget(self.crop_groupbox.crop_groupbox)
        self.images_tab.layout().addWidget(self.rotate_groupbox.rotate_groupbox)

        self.rotate_crop_hlayout.addWidget(self.crop_groupbox.crop_groupbox)
        self.rotate_crop_hlayout.addWidget(self.rotate_groupbox.rotate_groupbox)
        self.rotate_crop_hlayout.setSpacing(3)

        self.images_tab.layout().addLayout(self.rotate_crop_hlayout)

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

        # Ensure images_tab has a layout before adding widgets
        images_tab_layout = self.images_tab.layout()

        images_tab_layout.addWidget(self.flip_groupbox)

        self.register_groupbox = RegisterUI(
            self.preprocessing_tab, self.preprocessing_dockwidget_main_vlayout
        )
        self.gaussian_blur = GaussianBlur(
            self.preprocessing_tab, self.preprocessing_dockwidget_main_vlayout
        )
        images_layout = self.images_tab.layout()
        assert isinstance(images_layout, QVBoxLayout)
        self.cell_layer_alignment = CellLayerAlignmentUI(
            images_layout,
            self.images_tab.storage,
            self.images_tab,
        )

        # Now that cell_layer_alignment exists, connect the signals
        self.images_tab.tissue_target_selected.connect(
            self.cell_layer_alignment.set_target_image
        )
        self.images_tab.tissue_unaligned_selected.connect(
            self.cell_layer_alignment.set_unaligned_image
        )

        # Connect to progress bar
        self.cell_layer_alignment.aligner.progress.connect(self.update_progress_bar)

        # stardist UI
        self.stardist_groupbox = StarDistUI(
            self.preprocessing_tab, self.preprocessing_dockwidget_main_vlayout
        )

        self.cellIntensity_groupbox = CellIntensityUI(
            self.preprocessing_tab, self.preprocessing_dockwidget_main_vlayout
        )
        self.preprocessing_dockwidget_main_vlayout.addWidget(self.save_button)

    def _setup_cell_layer_alignment(self):
        """Setup cell layer alignment component and its connections"""
        self.cell_layer_alignment = CellLayerAlignmentUI(
            self.preprocessing_dockwidget_main_vlayout,
            self.images_tab.storage,
            self.preprocessing_tab,
        )

    def _setup_view_tab(self):
        """Setup the view tab"""
        view_scroll = self._create_scroll_area()

        self.view_tab = ImageOverlay(self.canvas, enc=self)
        self.view_tab.setObjectName("view_tab")
        self.view_tab.setSizePolicy(
            QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Preferred
        )

        view_scroll.setWidget(self.view_tab)
        self.stackedWidget.addWidget(view_scroll)

    def _setup_analysis_tab(self):
        """Setup the analysis tab"""
        analysis_scroll = self._create_scroll_area()

        self.analysis_tab = AnalysisTab(self.canvas, self)
        self.analysis_tab.setSizePolicy(
            QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Preferred
        )
        self.analysis_tab.setObjectName("analysis_tab")

        analysis_scroll.setWidget(self.analysis_tab)
        self.stackedWidget.addWidget(analysis_scroll)

    def _setup_metadata_tab(self):
        """Setup the metadata tab"""
        metadata_scroll = self._create_scroll_area()

        self.metadata_tab = MetaData(self.canvas)
        self.metadata_tab.setSizePolicy(
            QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Preferred
        )

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
        self.toolBarUI.tabChanged.connect(self.stackedWidget.setCurrentIndex)
        # Start with Images tab
        self.stackedWidget.setCurrentIndex(0)

    def _retranslate_UI(self):
        """Set UI text and translations"""
        _translate = QCoreApplication.translate
        self.setWindowTitle(_translate("MainWindow", "MIST-Explorer"))

    # Event handlers and utility methods
    def updateMousePositionLabel(self, text):
        """Update mouse position in toolbar"""
        self.toolBarUI.statusLine.setText(text)

    def save_canvas(self):
        """Save canvas signal emission"""
        print("saving")
        self.saveSignal.emit()

    def update_progress_bar(self, value, text):
        """Update progress bar with value and text"""
        self.progressBar.setValue(value)
        self.progressBarLabel.setText(text + " ...")
        if value >= 100:
            self.progressBar.hide()
            self.progressBarLabel.hide()
        else:
            self.progressBar.show()
            self.progressBarLabel.show()

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
            None, "Save File", "image.png", "*.png;;*.jpg;;*.tif;; All Files(*)"
        )

        if file_name:
            # Use pixmapItem directly if available
            if hasattr(self.canvas, "pixmapItem") and self.canvas.pixmapItem:
                pixmap = self.canvas.pixmapItem.pixmap()
            else:
                pixmap = self.canvas.grab()

            # Convert to numpy array and save
            qimage = pixmap.toImage()
            buffer = qimage.bits().asstring(
                qimage.width() * qimage.height() * qimage.depth() // 8
            )
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
