"""
Main application window module.
"""
import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Optional

# pylint: disable=no-name-in-module
from PyQt6.QtCore import (QCoreApplication, QEvent, QMetaObject, QPoint, Qt,
                          QTimer)
from PyQt6.QtGui import QIcon, QImageReader, QKeySequence, QShortcut
from PyQt6.QtWidgets import (QFileDialog, QGroupBox, QHBoxLayout, QLabel,
                             QMainWindow, QMenu, QProgressBar, QPushButton,
                             QScrollArea, QSizePolicy, QSplitter,
                             QStackedWidget, QStatusBar, QTabWidget,
                             QVBoxLayout, QWidget)

from core import MetaData
from core.project_manager import ProjectManager
from core.project_naming import is_stardist_label_name
from ui.toolbar.menubar_ui import MenuBarUI
from ui.toolbar.toolbar_ui import ToolBarUI
from utils import resource_path

logger = logging.getLogger(__name__)


class MainWindow(QMainWindow):
    """
    Main Application Window.
    """

    # pylint: disable=too-many-instance-attributes, attribute-defined-outside-init
    def __init__(self, parent=None, project_path: Optional[Path] = None):
        QImageReader.setAllocationLimit(0)
        super().__init__(parent)

        self._init_attributes()

        self.current_project_path = project_path

        if sys.platform == "win32":
            self.drag_pos = QPoint()

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
        self._retranslate_ui()
        QMetaObject.connectSlotsByName(self)

        if self.current_project_path:
            self._load_project(self.current_project_path)

    def _init_attributes(self):
        """Initialize instance attributes."""
        self.drag_pos: Optional[QPoint] = None
        self.menu_bar: Optional[MenuBarUI] = None
        self.tool_bar: Optional[ToolBarUI] = None
        self.central_widget_layout: Optional[QHBoxLayout] = None
        self.main_layout: Optional[QHBoxLayout] = None
        self.side_panel_container: Optional[QWidget] = None
        self.side_panel: Optional[QWidget] = None
        self.side_panel_layout: Optional[QVBoxLayout] = None
        self.toggle_button: Optional[QPushButton] = None
        self.stacked_widget: Optional[QStackedWidget] = None
        self.canvas: Optional[ImageGraphicsViewUI] = None
        self.small_view: Optional[ReferenceGraphicsViewUI] = None
        self.images_tab_container: Optional[QWidget] = None
        self.images_tab: Optional[ImageManager] = None
        self.processing_tabs: Optional[QTabWidget] = None
        self.crop_groupbox: Optional[CropUI] = None
        self.rotate_groupbox: Optional[RotateUI] = None
        self.flip_groupbox: Optional[QGroupBox] = None
        self.flip_horizontal_btn: Optional[QPushButton] = None
        self.flip_vertical_btn: Optional[QPushButton] = None
        self.cell_layer_alignment: Optional[CellLayerAlignmentUI] = None
        self.register_groupbox: Optional[RegisterUI] = None
        self.gaussian_blur: Optional[GaussianBlur] = None
        self.stardist_groupbox: Optional[StarDistUI] = None
        self.cell_intensity_groupbox: Optional[CellIntensityUI] = None
        self.view_tab: Optional[ImageOverlay] = None
        self.analysis_tab: Optional[AnalysisTab] = None
        self.metadata_tab: Optional[MetaData] = None
        self.statusbar: Optional[QStatusBar] = None
        self.progress_bar_label: Optional[QLabel] = None
        self.progress_bar: Optional[QProgressBar] = None
        self.metadata: Optional[dict] = None
        self.splitter: Optional[QSplitter] = None
        self.last_sidebar_width: int = 400
        self.current_project_path: Optional[Path] = None
        self.log_dialog = None

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
        self.setWindowIcon(QIcon(resource_path("assets/final_icon.png")))
        self.resize(1440, 1000)
        self.setMinimumSize(1200, 800)

    def toggle_maximize(self):
        """Toggle between maximized and normal window state"""
        if self.isMaximized():
            self.showNormal()
        else:
            self.showMaximized()

    def _load_project(self, project_path: Path):
        """Load an existing project."""
        from core.canvas import ImageWrapper
        from models.workspace import ImageMetadata

        self.current_project_path = project_path

        metadata = ProjectManager.load_project(project_path)
        if metadata is None:
            return

        self.setWindowTitle(f"MIST-Explorer - {metadata.name}")

        if self.images_tab:
            self.images_tab.set_project_path(project_path)

        for image_meta in metadata.images:
            image_data = {}
            channel_names = ProjectManager.list_saved_channels(
                project_path, image_meta.uuid
            )
            if not channel_names:
                channel_names = [
                    f"Channel {channel_num}"
                    for channel_num in range(1, image_meta.channel_count + 1)
                ]

            for channel_name in channel_names:
                channel_array = ProjectManager.load_image(
                    project_path, image_meta.uuid, channel_name
                )
                if channel_array is not None:
                    contrast = image_meta.contrast_settings.get(channel_name, (0, 255))
                    display_name = image_meta.channel_display_names.get(
                        channel_name, channel_name
                    )
                    channel_cmap = (
                        "label_image" if is_stardist_label_name(display_name) else "gray"
                    )
                    wrapper = ImageWrapper(
                        channel_array,
                        name=display_name,
                        cmap=channel_cmap,
                    )
                    wrapper.contrast_min = contrast[0]
                    wrapper.contrast_max = contrast[1]
                    image_data[channel_name] = wrapper

            if image_data:
                image_uuid = str(image_meta.uuid)
                self.images_tab.storage.add_data(
                    image_uuid,
                    {
                        "name": image_meta.name,
                        "data": image_data,
                        "original_filename": image_meta.original_filename,
                    }
                )
                self.images_tab.add_item(image_uuid)

    def open_project_folder(self):
        """Open the current project folder in file explorer."""
        if self.current_project_path:
            ProjectManager.open_project_folder(self.current_project_path)

    def switch_project(self):
        """Cleanly restart the application to return to the project launcher."""
        logger.info("Switching project. Restarting application...")
        os.execl(sys.executable, sys.executable, *sys.argv)

    # pylint: disable=invalid-name
    def eventFilter(self, obj, event): # type: ignore
        """Event filter for handling window dragging"""
        if sys.platform == "win32":
            if obj == self.menu_bar:
                if event.type() == QEvent.Type.MouseButtonPress:
                    self.drag_pos = event.globalPosition().toPoint()
                    return False  # Allow the event to propagate for clicks
                if event.type() == QEvent.Type.MouseMove:
                    if (
                        event.buttons() == Qt.MouseButton.LeftButton
                        and hasattr(self, "drag_pos")
                        and self.drag_pos is not None
                    ):
                        self.move(
                            self.pos() + event.globalPosition().toPoint() - self.drag_pos
                        )
                        self.drag_pos = event.globalPosition().toPoint()
                        return True  # Consume the event if dragging
                if event.type() == QEvent.Type.MouseButtonRelease:
                    self.drag_pos = QPoint()  # Reset dragPos
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
        self.menu_bar = MenuBarUI(self)

        self.tool_bar = ToolBarUI(self)
        self.addToolBar(Qt.ToolBarArea.TopToolBarArea, self.tool_bar)
        self.setMenuBar(self.menu_bar)
        if sys.platform == "win32":
            self.menu_bar.installEventFilter(self)

    def _setup_side_panel(self):
        """Setup the collapsible side panel"""

        self.side_panel_container = QWidget(self.centralWidget())
        self.side_panel_container.setMinimumWidth(35)
        container_layout = QHBoxLayout(self.side_panel_container)
        container_layout.setContentsMargins(0, 0, 0, 0)
        container_layout.setSpacing(10)

        self.side_panel = QWidget(self.side_panel_container)
        self.side_panel_layout = QVBoxLayout(self.side_panel)
        # self.sidePanelLayout.setSpacing(10)
        self.side_panel.setMinimumWidth(400)
        
        self.side_panel.setSizePolicy(
            QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Expanding
        )

        self.toggle_button = QPushButton("◀", self.side_panel_container)
        self.toggle_button.setFixedSize(20, 60)
        self.toggle_button.clicked.connect(self.toggle_side_panel)
        self.toggle_button.setStyleSheet(
            """
            QPushButton:hover {
                background-color: #e0e0e0;
            }
        """
        )

        # Create stacked widget for tabs
        self.stacked_widget = QStackedWidget(self.side_panel)
        self.side_panel_layout.addWidget(self.stacked_widget)

        # Add to container
        container_layout.addWidget(self.side_panel)
        container_layout.addWidget(self.toggle_button)

    def _setup_canvas(self):
        """Setup the main canvas and reference view"""
        from ui.canvas import ImageGraphicsViewUI, ReferenceGraphicsViewUI

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
        from ui.image_manager import ImageManager

        images_scroll = self._create_scroll_area()

        # The main widget for the "Images" tab, containing the manager and the processing tabs
        self.images_tab_container = QWidget(self.side_panel)
        images_tab_layout = QVBoxLayout(self.images_tab_container)

        # Image Manager (the file tree)
        self.images_tab = ImageManager(self.canvas)

        if self.current_project_path:
            self.images_tab.set_project_path(self.current_project_path)

        # Processing Tabs
        self.processing_tabs = QTabWidget(self.side_panel)
        splitter = QSplitter(Qt.Orientation.Vertical, self.side_panel)
        splitter.addWidget(self.images_tab)
        splitter.setStretchFactor(0, 1)
        splitter.addWidget(self.processing_tabs)

        # Add splitter to layout
        images_tab_layout.addWidget(splitter)

        # Add individual processing tabs
        self._setup_transform_tab()
        self._setup_alignment_tab()
        self._setup_segmentation_tab()
        self._setup_quantification_tab()

        images_scroll.setWidget(self.images_tab_container)
        self.stacked_widget.addWidget(images_scroll)

    def _setup_transform_tab(self):
        """Sets up the 'Transform' tab with Crop, Rotate, and Flip tools."""
        from ui.alignment.cell_layer_alignment_ui import CellLayerAlignmentUI
        from ui.processing.crop_ui import CropUI
        from ui.processing.rotation_ui import RotateUI

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
        from ui.alignment.register_ui import RegisterUI

        alignment_tab = QWidget()
        alignment_layout = QVBoxLayout(alignment_tab)
        alignment_layout.setAlignment(Qt.AlignmentFlag.AlignTop)

        # Register UI
        self.register_groupbox = RegisterUI(alignment_tab, alignment_layout)

        self.processing_tabs.addTab(alignment_tab, "Alignment")

    def _setup_segmentation_tab(self):
        """Sets up the 'Segmentation' tab with Gaussian Blur and StarDist."""
        from ui.processing.gaussian_blur import GaussianBlur
        from ui.stardist.stardist_ui import StarDistUI

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
        from ui.alignment.cell_intensity_ui import CellIntensityUI

        quantification_tab = QWidget()
        quantification_layout = QVBoxLayout(quantification_tab)
        quantification_layout.setAlignment(Qt.AlignmentFlag.AlignTop)

        self.cell_intensity_groupbox = CellIntensityUI(
            quantification_tab, quantification_layout
        )

        self.processing_tabs.addTab(quantification_tab, "Generation")

    def _setup_view_tab(self):
        """Setup the view tab"""
        from ui.view_tab import ImageOverlay

        view_scroll = self._create_scroll_area()

        self.view_tab = ImageOverlay(self.canvas, enc=self)
        self.view_tab.setObjectName("view_tab")
        self.view_tab.setSizePolicy(
            QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Preferred
        )

        view_scroll.setWidget(self.view_tab)
        self.stacked_widget.addWidget(view_scroll)

    def _setup_analysis_tab(self):
        """Setup the analysis tab"""
        from ui.analysis.AnalysisTab import AnalysisTab

        analysis_scroll = self._create_scroll_area()

        self.analysis_tab = AnalysisTab(self.canvas, self)
        self.analysis_tab.setSizePolicy(
            QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Preferred
        )
        self.analysis_tab.setObjectName("analysis_tab")

        analysis_scroll.setWidget(self.analysis_tab)
        self.stacked_widget.addWidget(analysis_scroll)

    def _setup_metadata_tab(self):
        """Setup the metadata tab"""
        metadata_scroll = self._create_scroll_area()

        self.metadata_tab = MetaData(self.canvas)
        self.metadata_tab.setSizePolicy(
            QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Preferred
        )

        metadata_scroll.setWidget(self.metadata_tab)
        self.stacked_widget.addWidget(metadata_scroll)

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

        self.progress_bar_label = QLabel("")
        self.progress_bar = QProgressBar()
        self.progress_bar.setMaximum(100)

        progress_bar_layout = QHBoxLayout()
        progress_bar_layout.addWidget(self.progress_bar_label)
        progress_bar_layout.addWidget(self.progress_bar)
        container.setLayout(progress_bar_layout)

        self.statusbar.addPermanentWidget(container)

        # Style the progress bar
        progress_bar_style = """
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
        self.progress_bar.setStyleSheet(progress_bar_style)

    def _setup_layout(self):
        """Setup the final layout structure"""
        # Create a horizontal splitter
        self.splitter = QSplitter(Qt.Orientation.Horizontal)
        
        # Add sidebar container and canvas to splitter
        # Note: side_panel_container now contains both the panel and the toggle button
        self.splitter.addWidget(self.side_panel_container)
        self.splitter.addWidget(self.canvas)

        # Prevent dragging the splitter to fully collapse the sidebar
        self.splitter.setCollapsible(0, False)

        # Set stretch factors so canvas takes available space
        self.splitter.setStretchFactor(1, 1)

        # Add splitter to main layout
        self.main_layout.addWidget(self.splitter)

        # Add main layout to central widget
        self.central_widget_layout.addLayout(self.main_layout)

    def _connect_signals(self):
        """Connect remaining signals"""
        # Connect cell intensity error signal to statusbar
        if self.cell_intensity_groupbox:
            self.cell_intensity_groupbox.errorSignal.connect(
                lambda msg: self.statusbar.showMessage(msg, 5000)
            )

        # Connect toolbar tab change signal
        # Start with Images tab
        self.stacked_widget.setCurrentIndex(0)


    def _retranslate_ui(self):
        """Set UI text and translations"""
        _translate = QCoreApplication.translate
        self.setWindowTitle(_translate("MainWindow", "MIST-Explorer"))

    # Event handlers and utility methods
    def update_mouse_position_label(self, text):
        """Update mouse position in toolbar"""
        self.tool_bar.statusLine.setText(text)

    def update_progress_bar(self, value, text):
        """Update progress bar with value and text"""
        self.progress_bar.setValue(value)
        self.progress_bar_label.setText(text + " ...")
        if value >= 100:
            self.progress_bar.hide()
            self.progress_bar_label.hide()
        else:
            self.progress_bar.show()
            self.progress_bar_label.show()

    def select(self):
        """Set rectangle selection mode"""
        logger.debug("selecting")
        self.canvas.select = "rect"

    def circle_select(self):
        """Set circle selection mode"""
        logger.debug("selecting")
        self.canvas.select = "circle"

    def poly_select(self):
        """Toggle polygon selection mode"""
        if self.canvas.select == "poly":
            self.canvas.select = False
            return
        self.canvas.select = "poly"

    def toggle_side_panel(self):
        """Toggle side panel visibility"""
        if self.side_panel.isVisible():
            # Save width before hiding, if it's substantial
            current_sizes = self.splitter.sizes()
            if current_sizes[0] > 100:
                self.last_sidebar_width = current_sizes[0]
            
            self.side_panel.hide()
            self.toggle_button.setText("▶")
            
            # Collapse splitter to just the button width
            # We add a small buffer for margins/spacing
            btn_width = self.toggle_button.width() + 15 
            self.splitter.setSizes([btn_width, sum(current_sizes) - btn_width])
        else:
            self.side_panel.show()
            self.toggle_button.setText("◀")
            
            # Restore previous width
            current_sizes = self.splitter.sizes()
            total = sum(current_sizes)
            target = getattr(self, "last_sidebar_width", 400)
            self.splitter.setSizes([target, total - target])

    def show_log_dialog(self) -> None:
        """Show the application log dialog."""
        from ui.log_dialog import LogDialog

        if self.log_dialog is None:
            self.log_dialog = LogDialog(self)
        self.log_dialog.show()
        self.log_dialog.raise_()

    def get_metadata(self, metadata: dict):
        """Update metadata tab with new metadata"""
        self.metadata = metadata
        self.metadata_tab.populate_table(self.metadata)
