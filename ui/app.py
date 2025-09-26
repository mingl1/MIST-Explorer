import argparse
import os
import sys

import numpy as np
from PyQt6.QtCore import *
from PyQt6.QtGui import *
from PyQt6.QtWidgets import *

from core import MetaData
from ui.alignment.cell_intensity_ui import CellIntensityUI
from ui.alignment.cell_layer_alignment_ui import CellLayerAlignmentUI
from ui.alignment.register_ui import RegisterUI
from ui.analysis.AnalysisTab import AnalysisTab
from ui.canvas_ui import ImageGraphicsViewUI, ReferenceGraphicsViewUI
from ui.ImageManager import Manager
from ui.processing.crop_ui import CropUI
from ui.processing.gaussian_blur import GaussianBlur
from ui.processing.rotation_ui import RotateUI
from ui.stardist.stardist_ui import StarDistUI
from ui.toolbar.menubar_ui import MenuBarUI
from ui.toolbar.toolbar_ui import ToolBarUI
from ui.view_tab import ImageOverlay


class Ui_MainWindow(QMainWindow):

    def __init__(self, parent=None):
        QImageReader.setAllocationLimit(0)
        super().__init__()

        if sys.platform == "win32":
            self.dragPos = QPoint()

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
        if sys.platform == "win32":
            self.setWindowFlags(Qt.WindowType.FramelessWindowHint)
        self.resize(1440, 1000)
        self.setMinimumSize(1200, 800)

    def toggle_maximize(self):
        if self.isMaximized():
            self.showNormal()
        else:
            self.showMaximized()

    def eventFilter(self, obj, event):
        if sys.platform == "win32":
            if obj == self.menuBarUI:
                if event.type() == QEvent.Type.MouseButtonPress:
                    self.dragPos = event.globalPosition().toPoint()
                    return False  # Allow the event to propagate for clicks
                elif event.type() == QEvent.Type.MouseMove:
                    if (
                        event.buttons() == Qt.MouseButton.LeftButton
                        and hasattr(self, "dragPos")
                        and self.dragPos is not None
                    ):
                        self.move(
                            self.pos() + event.globalPosition().toPoint() - self.dragPos
                        )
                        self.dragPos = event.globalPosition().toPoint()
                        return True  # Consume the event if dragging
                elif event.type() == QEvent.Type.MouseButtonRelease:
                    self.dragPos = QPoint()  # Reset dragPos
                    return False  # Allow the event to propagate
        return super().eventFilter(obj, event)

    def _add_shortcuts(self):
        """Add keyboard shortcuts"""

        # Can add more shortcuts by adding tuple in form: (key_press_string, function)
        shortcuts = [
            ("Ctrl+R", self.select),
            ("Ctrl+C", self.circle_select),
            ("Ctrl+P", self.poly_select),
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

        self.toolBarUI = ToolBarUI(self)
        self.addToolBar(Qt.ToolBarArea.TopToolBarArea, self.toolBarUI)
        self.setMenuBar(self.menuBarUI)
        if sys.platform == "win32":
            self.menuBarUI.installEventFilter(self)

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
        self.sidePanel.setMaximumWidth(500)
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
        self.canvas.reference_view = self.small_view

    def _setup_images_tab(self):
        """Setup the images workspace tab with an image manager and processing tabs."""
        images_scroll = self._create_scroll_area()

        # The main widget for the "Images" tab, containing the manager and the processing tabs
        self.images_tab_container = QWidget()
        images_tab_layout = QVBoxLayout(self.images_tab_container)
        images_tab_layout.setContentsMargins(0, 0, 0, 0)

        # Image Manager (the file tree)
        self.images_tab = Manager(self.canvas)
        
        # Processing Tabs
        self.processing_tabs = QTabWidget()

        # Create a splitter to allow resizing of the image manager and processing tabs
        splitter = QSplitter(Qt.Orientation.Vertical)
        splitter.addWidget(self.images_tab)
        splitter.addWidget(self.processing_tabs)

        # Add splitter to layout
        images_tab_layout.addWidget(splitter)

        # Add individual processing tabs
        self._setup_transform_tab()
        self._setup_alignment_tab()
        self._setup_segmentation_tab()
        self._setup_quantification_tab()

        images_scroll.setWidget(self.images_tab_container)
        self.stackedWidget.addWidget(images_scroll)

    def _setup_transform_tab(self):
        """Sets up the 'Transform' tab with Crop, Rotate, and Flip tools."""
        transform_tab = QWidget()
        transform_layout = QVBoxLayout(transform_tab)
        transform_layout.setAlignment(Qt.AlignmentFlag.AlignTop)

        # Crop and Rotate
        self.crop_groupbox = CropUI(transform_tab)
        self.rotate_groupbox = RotateUI(transform_tab)
        rotate_crop_hlayout = QHBoxLayout()
        rotate_crop_hlayout.addWidget(self.crop_groupbox.crop_groupbox)
        rotate_crop_hlayout.addWidget(self.rotate_groupbox.rotate_groupbox)
        transform_layout.addLayout(rotate_crop_hlayout)

        # Flip
        self.flip_groupbox = QGroupBox("Flip Image")
        flip_layout = QHBoxLayout(self.flip_groupbox)
        self.flip_horizontal_btn = QPushButton("Flip Horizontal")
        self.flip_vertical_btn = QPushButton("Flip Vertical")
        self.flip_horizontal_btn.clicked.connect(self.canvas.flip_horizontal)
        self.flip_vertical_btn.clicked.connect(self.canvas.flip_vertical)
        flip_layout.addWidget(self.flip_horizontal_btn)
        flip_layout.addWidget(self.flip_vertical_btn)
        transform_layout.addWidget(self.flip_groupbox)

        # Cell Layer Alignment
        self.cell_layer_alignment = CellLayerAlignmentUI(
            transform_layout,
            self.images_tab.storage,
            transform_tab,
        )
        self.images_tab.image_tree_view.tissue_target_selected.connect(
            self.cell_layer_alignment.set_target_image
        )
        self.images_tab.image_tree_view.tissue_unaligned_selected.connect(
            self.cell_layer_alignment.set_unaligned_image
        )
        self.cell_layer_alignment.aligner.progress.connect(self.update_progress_bar)

        self.processing_tabs.addTab(transform_tab, "Transform")

    def _setup_alignment_tab(self):
        """Sets up the 'Alignment' tab with Register tools."""
        alignment_tab = QWidget()
        alignment_layout = QVBoxLayout(alignment_tab)
        alignment_layout.setAlignment(Qt.AlignmentFlag.AlignTop)

        # Register UI
        self.register_groupbox = RegisterUI(alignment_tab, alignment_layout)

        self.processing_tabs.addTab(alignment_tab, "Alignment")

    def _setup_segmentation_tab(self):
        """Sets up the 'Segmentation' tab with Gaussian Blur and StarDist."""
        segmentation_tab = QWidget()
        segmentation_layout = QVBoxLayout(segmentation_tab)
        segmentation_layout.setAlignment(Qt.AlignmentFlag.AlignTop)

        # Gaussian Blur
        self.gaussian_blur = GaussianBlur(segmentation_tab, segmentation_layout)

        # StarDist
        self.stardist_groupbox = StarDistUI(segmentation_tab, segmentation_layout)

        self.processing_tabs.addTab(segmentation_tab, "Segmentation")

    def _setup_quantification_tab(self):
        """Sets up the 'Quantification' tab with Cell Intensity."""
        quantification_tab = QWidget()
        quantification_layout = QVBoxLayout(quantification_tab)
        quantification_layout.setAlignment(Qt.AlignmentFlag.AlignTop)

        self.cellIntensity_groupbox = CellIntensityUI(quantification_tab, quantification_layout)

        self.processing_tabs.addTab(quantification_tab, "Generation")

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
