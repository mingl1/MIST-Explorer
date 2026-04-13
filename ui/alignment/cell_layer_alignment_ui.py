"""
Cell layer alignment UI module.
"""

# pylint: disable=no-name-in-module
from PyQt6.QtCore import QCoreApplication, Qt, pyqtSignal, pyqtSlot
from PyQt6.QtWidgets import (
    QCheckBox,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from core import CellLayerAligner, ImageStorage
from ui.alignment.alignment_preview_dialog import AlignmentPreviewDialog
from ui.processing.segmentation_image_selector import SegmentationImageSelector


# pylint: disable=too-many-instance-attributes
class CellLayerAlignmentUI(QWidget):
    """
    UI for managing cell layer alignment parameters.
    """

    errorSignal = pyqtSignal(str)
    alignmentCompleteSignal = pyqtSignal(object, str)
    loadOnCanvasSignal = pyqtSignal(dict, bool, str)
    channelChanged = pyqtSignal(int)
    progress = pyqtSignal(int, str)

    def __init__(
        self,
        containing_layout: QVBoxLayout,
        storage: ImageStorage,
        parent=None,
    ):
        super().__init__(parent)

        self.target_image = None
        self.target_uuid = ""
        self.target_name = "not loaded"

        self.unaligned_image = None
        self.unaligned_uuid = ""
        self.unaligned_name = "not loaded"

        self.aligner = CellLayerAligner()
        self.storage = storage

        # Attributes initialized in setup methods
        self.alignment_groupbox = None
        self.main_layout = None
        self._target_groupbox = None
        self.target_image_selector = None
        self._unaligned_groupbox = None
        self.unaligned_image_selector = None
        self.unaligned_spacing_row = None
        self.unaligned_spacing_label = None
        self.unaligned_spacing_x = None
        self.unaligned_spacing_y = None
        self.checkbox_layout = None
        self.need_centering = None
        self.need_gradient_descent = None
        self.button_layout = None
        self.register_button = None
        self.manually_align_button = None

        self._setup_ui(parent, containing_layout)
        self._setup_connections()

    def _setup_ui(self, parent, containing_layout: QVBoxLayout):
        """Initialize the UI components."""
        self.alignment_groupbox = QGroupBox(parent)
        self.main_layout = QVBoxLayout(self.alignment_groupbox)

        self._setup_image_layouts()
        self._setup_spacing_layouts()
        self._setup_checkboxes()
        self._setup_buttons()

        self.main_layout.addWidget(self._target_groupbox)
        self.main_layout.addWidget(self._unaligned_groupbox)
        self.main_layout.addLayout(self.unaligned_spacing_row)
        self.main_layout.addLayout(self.checkbox_layout)
        self.main_layout.addSpacing(10)
        self.main_layout.addLayout(self.button_layout)

        if containing_layout:
            containing_layout.addWidget(self.alignment_groupbox)

        self._retranslate_ui()

    def _setup_image_layouts(self):
        """Setup image selectors for target and moving images."""
        self._target_groupbox = QGroupBox("Protein Array Image", self)
        target_layout = QVBoxLayout(self._target_groupbox)
        target_layout.setContentsMargins(4, 4, 4, 4)
        self.target_image_selector = SegmentationImageSelector(self)
        target_layout.addWidget(self.target_image_selector)

        self._unaligned_groupbox = QGroupBox("Aligned by this Image", self)
        unaligned_layout = QVBoxLayout(self._unaligned_groupbox)
        unaligned_layout.setContentsMargins(4, 4, 4, 4)
        self.unaligned_image_selector = SegmentationImageSelector(self)
        unaligned_layout.addWidget(self.unaligned_image_selector)

    def _setup_spacing_layouts(self):
        """Setup layouts for scaling factor input."""
        self.unaligned_spacing_row = QHBoxLayout()
        self.unaligned_spacing_label = QLabel("Scaling Factor (x, y):")
        self.unaligned_spacing_x = QLineEdit("1.0")
        self.unaligned_spacing_y = QLineEdit("1.0")
        self.unaligned_spacing_row.setAlignment(Qt.AlignmentFlag.AlignLeft)
        self.unaligned_spacing_row.addWidget(self.unaligned_spacing_label)
        self.unaligned_spacing_row.addWidget(self.unaligned_spacing_x)
        self.unaligned_spacing_row.addWidget(QLabel(","))
        self.unaligned_spacing_row.addWidget(self.unaligned_spacing_y)

    def _setup_checkboxes(self):
        """Setup checkboxes for alignment options."""
        self.checkbox_layout = QHBoxLayout()
        self.checkbox_layout.setContentsMargins(0, 0, 0, 0)
        self.need_centering = QCheckBox("Skip Centering Images")
        self.need_centering.setChecked(True)
        self.need_centering.setToolTip(
            "Check this to center the images before alignment."
        )
        self.need_gradient_descent = QCheckBox("Skip Gradient Descent")
        self.need_gradient_descent.setChecked(False)
        self.need_gradient_descent.setToolTip(
            "Check this to skip the gradient descent step."
        )
        self.checkbox_layout.addWidget(self.need_centering)
        self.checkbox_layout.addWidget(self.need_gradient_descent)

    def _setup_buttons(self):
        """Setup registration buttons."""
        self.button_layout = QHBoxLayout()
        self.button_layout.setContentsMargins(0, 0, 0, 0)
        self.register_button = QPushButton("Register Images")
        self.register_button.setEnabled(False)
        self.register_button.clicked.connect(self.register_images)
        self.register_button.setMinimumHeight(30)
        self.register_button.setStyleSheet(self._get_button_style())

        self.manually_align_button = QPushButton("Manually Align")
        self.manually_align_button.setEnabled(False)
        self.manually_align_button.setMinimumHeight(30)
        self.manually_align_button.setStyleSheet(self._get_button_style())
        self.manually_align_button.clicked.connect(self.manually_align_images)

        self.button_layout.addWidget(self.manually_align_button)
        self.button_layout.addWidget(self.register_button)

    def _get_button_style(self):
        """Return stylesheet for registration buttons."""
        from ui.theme import ThemeManager
        c = ThemeManager.instance().get_current()
        return f"""
            QPushButton {{
                background-color: {c["success"]};
                color: {c["text_on_accent"]};
                font-weight: bold;
                border-radius: 4px;
            }}
            QPushButton:disabled {{
                background-color: {c["bg_tertiary"]};
                color: {c["text_secondary"]};
            }}
            QPushButton:hover {{
                background-color: {c["success_hover"]};
            }}
        """

    def _setup_connections(self):
        """Set up the connections to the aligner thread"""
        self.aligner.progress.connect(self._handle_progress)
        self.aligner.error.connect(self._handle_error)
        self.aligner.snapshot.connect(self._handle_snapshot)

        self.target_image_selector.image_changed.connect(self._on_target_image_changed)
        self.unaligned_image_selector.image_changed.connect(
            self._on_unaligned_image_changed
        )

        self.aligner.error.connect(self.errorSignal)

        self.need_centering.stateChanged.connect(
            lambda state: self.aligner.skip_coarse_alignment(
                state == Qt.CheckState.Checked.value
            )
        )

        self.need_gradient_descent.stateChanged.connect(
            lambda state: self.aligner.skip_gradient_descent(
                state == Qt.CheckState.Checked.value
            )
        )

    @pyqtSlot(dict)
    def _handle_snapshot(self, snapshot_data):
        """Handle snapshot data from aligner."""
        snapshot_dialog = AlignmentPreviewDialog(
            snapshot_data,
            False,
        )
        snapshot_dialog.exec()

    def _on_target_image_changed(self, uuid: str):
        """Called when user picks a different reference image in the selector."""
        item = self.storage.get_data(uuid)
        if item is None:
            return
        self.target_uuid = uuid
        self.target_image = item["data"]
        self.target_name = item["name"]
        self.target_image_selector.set_channels(item["data"])
        self._check_can_register()

    def _on_unaligned_image_changed(self, uuid: str):
        """Called when user picks a different moving image in the selector."""
        item = self.storage.get_data(uuid)
        if item is None:
            return
        self.unaligned_uuid = uuid
        self.unaligned_image = item["data"]
        self.unaligned_name = item["name"]
        self.unaligned_image_selector.set_channels(item["data"])
        self._check_can_register()

    def _retranslate_ui(self):
        """Retranslate UI components."""
        _translate = QCoreApplication.translate
        self.alignment_groupbox.setTitle(
            _translate("MainWindow", "Register and Replace Selected Reference Channel")
        )
        self.register_button.setText(_translate("MainWindow", "Register Images"))
        self.manually_align_button.setText(_translate("MainWindow", "Manually Align"))

    # pylint: disable=unused-argument
    def set_target_image(self, item_uuid, is_leaf, channel):
        """Set the Reference image for alignment (called by context menu)."""
        item = self.storage.get_data(item_uuid)
        assert item is not None, f"No data found for UUID: {item_uuid}"
        obj, name = item["data"], item["name"]
        self.target_uuid = item_uuid
        self.target_image = obj
        self.target_name = name
        # Update selector silently so context-menu picks don't re-trigger image_changed
        self.target_image_selector.set_selected_uuid_silent(item_uuid)
        self.target_image_selector.set_channels(obj)
        self.target_image_selector.set_current_channel_silent(f"Channel {channel + 1}")
        self._check_can_register()

    # pylint: disable=unused-argument
    def set_unaligned_image(self, item_uuid, is_leaf, channel):
        """Set the unaligned image that will be registered to the target (called by context menu)."""
        item = self.storage.get_data(item_uuid)
        assert item is not None, f"No data found for UUID: {item_uuid}"
        obj, name = item["data"], item["name"]
        self.unaligned_uuid = item_uuid
        self.unaligned_image = obj
        self.unaligned_name = name
        self.unaligned_image_selector.set_selected_uuid_silent(item_uuid)
        self.unaligned_image_selector.set_channels(obj)
        self.unaligned_image_selector.set_current_channel_silent(
            f"Channel {channel + 1}"
        )
        self._check_can_register()

    def _check_can_register(self):
        """Check if both images are loaded and enable/disable register button"""
        can_register = (
            self.target_image is not None and self.unaligned_image is not None
        )
        self.register_button.setEnabled(can_register)
        self.manually_align_button.setEnabled(can_register)

    def prepare_aligner(self):
        """Prepare the aligner with the current settings"""
        if self.target_image is None or self.unaligned_image is None:
            QMessageBox.warning(
                self,
                "Alignment Error",
                "Please load both target and unaligned images before alignment.",
            )
            return

        self.register_button.setEnabled(False)
        self.manually_align_button.setEnabled(False)

        try:
            scale = (
                float(self.unaligned_spacing_x.text().strip()),
                float(self.unaligned_spacing_y.text().strip()),
            )
            self.aligner.set_scale(scale)
        except ValueError:
            self._handle_error(
                "Invalid scale values. Please enter numeric values for x and y."
            )
            return

        target_ch_name = self.target_image_selector.current_channel()
        unaligned_ch_name = self.unaligned_image_selector.current_channel()

        self.aligner.set_target_image(
            self.target_image[target_ch_name].data,
            target_ch_name,
            self.target_uuid,
        )
        self.aligner.set_unaligned_image(
            self.unaligned_image[unaligned_ch_name].data,
            unaligned_ch_name,
            self.unaligned_uuid,
        )

    def register_images(self):
        """Start the image registration process"""
        if self.target_image is not None and self.unaligned_image is not None:
            self.prepare_aligner()
            self.aligner.progress.emit(0, "Starting alignment...")
            self.aligner.start()

    def manually_align_images(self):
        """Open a dialog for manual alignment of images"""
        if self.target_image is None or self.unaligned_image is None:
            QMessageBox.warning(
                self,
                "Alignment Error",
                "Please load both target and unaligned images before manual alignment.",
            )
            return
        self.prepare_aligner()
        target_ch_name = self.target_image_selector.current_channel()
        unaligned_ch_name = self.unaligned_image_selector.current_channel()

        preview_dialog = AlignmentPreviewDialog(
            {
                "target_image": self.target_image[target_ch_name].data,
                "aligned_image": self.unaligned_image[unaligned_ch_name].data,
            },
            True,
        )
        preview_dialog.transformation_matrix.connect(self.aligner.manually_align)
        preview_dialog.exec()
        self.register_button.setEnabled(True)
        self.manually_align_button.setEnabled(True)

    def _handle_progress(self, value, message):
        """Handle progress updates from the aligner thread"""
        self.progress.emit(value, message)
        if value >= 100:
            self._handle_finished()

    def _handle_error(self, error_message):
        """Handle error messages from the aligner thread"""
        QMessageBox.critical(self, "Alignment Error", error_message)
        self.progress.emit(100, "Error occurred during alignment.")

    def _handle_finished(self):
        """Handle when the alignment thread finishes"""
        self.register_button.setEnabled(True)
        self.manually_align_button.setEnabled(True)
