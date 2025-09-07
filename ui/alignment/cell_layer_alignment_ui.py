from re import L
from uuid import UUID

from PyQt6.QtCore import QCoreApplication, QMetaObject, Qt, pyqtSignal, pyqtSlot
from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
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


class CellLayerAlignmentUI(QWidget):
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
        super().__init__()
        self.image_channels = [1, 1]

        self.target_image = None
        self.target_uuid = ""
        self.target_name = "not loaded"

        self.unaligned_image = None
        self.unaligned_uuid = ""
        self.unaligned_name = "not loaded"

        self.aligner = CellLayerAligner()
        self.storage = storage
        self._setup_ui(parent, containing_layout)
        self._setup_connections()

    def _setup_ui(self, parent, containing_layout: QVBoxLayout):
        self.alignment_groupbox = QGroupBox(parent)

        # Main layout for the groupbox
        self.main_layout = QVBoxLayout(self.alignment_groupbox)

        # Title and labels

        # Image 1 layout
        self.image1_layout = QHBoxLayout()
        self.image1_label = QLabel("Target Image:")
        self.image1_status = QLabel("not loaded")
        self.image1_status.setStyleSheet("font-weight: bold; color: #555;")
        self.image1_layout.addWidget(self.image1_label)
        self.image1_layout.addWidget(self.image1_status)
        self.image1_layout.setStretch(1, 1)
        self.target_channel_selector = QComboBox(self)
        self.image1_layout.addWidget(self.target_channel_selector)
        self.target_channel_selector.setVisible(False)

        # Image 2 layout
        self.image2_layout = QHBoxLayout()
        self.image2_label = QLabel("Unaligned Image:")
        self.image2_status = QLabel("not loaded")
        self.image2_status.setStyleSheet("font-weight: bold; color: #555;")
        self.image2_layout.addWidget(self.image2_label)
        self.image2_layout.addWidget(self.image2_status)
        self.unaligned_channel_selector = QComboBox(self)
        self.image2_layout.addWidget(self.unaligned_channel_selector)
        self.unaligned_channel_selector.setVisible(False)
        self.image2_layout.setStretch(1, 1)

        # Ensure both image labels occupy same width; to align status labels
        max_label_width = max(
            self.image1_label.sizeHint().width(), self.image2_label.sizeHint().width()
        )
        self.image1_label.setMinimumWidth(max_label_width)
        self.image2_label.setMinimumWidth(max_label_width)

        self.target_spacing_row = QHBoxLayout()
        # self.target_spacing_row.addSpacing(max_label_width)  # Align with label
        self.target_spacing_label = QLabel("Spacing (x, y) um/px:")
        self.target_spacing_x = QLineEdit("1.0")
        self.target_spacing_y = QLineEdit("1.0")
        # self.target_spacing_x.setMaximumWidth(60)
        # self.target_spacing_y.setMaximumWidth(60)
        self.target_spacing_row.setAlignment(Qt.AlignmentFlag.AlignLeft)

        self.target_spacing_row.addWidget(self.target_spacing_label)
        self.target_spacing_row.addWidget(self.target_spacing_x)
        self.target_spacing_row.addWidget(QLabel(","))
        self.target_spacing_row.addWidget(self.target_spacing_y)
        # Unaligned spacing input (x, y)
        self.unaligned_spacing_row = QHBoxLayout()
        # self.unaligned_spacing_row.addSpacing(max_label_width)  # Align with label
        self.unaligned_spacing_label = QLabel("Spacing (x, y) um/px:")
        self.unaligned_spacing_x = QLineEdit("1.0")
        self.unaligned_spacing_y = QLineEdit("1.0")
        # self.unaligned_spacing_x.setMaximumWidth(60)
        # self.unaligned_spacing_y.setMaximumWidth(60)
        self.unaligned_spacing_row.setAlignment(Qt.AlignmentFlag.AlignLeft)

        self.unaligned_spacing_row.addWidget(self.unaligned_spacing_label)
        self.unaligned_spacing_row.addWidget(self.unaligned_spacing_x)
        self.unaligned_spacing_row.addWidget(QLabel(","))
        self.unaligned_spacing_row.addWidget(self.unaligned_spacing_y)
        self.unaligned_spacing_label.setAlignment(Qt.AlignmentFlag.AlignLeft)
        self.target_spacing_label.setAlignment(Qt.AlignmentFlag.AlignLeft)

        self.checkbox_layout = QHBoxLayout()
        self.checkbox_layout.setContentsMargins(0, 0, 0, 0)
        self.need_centering = QCheckBox("Skip Centering Images")
        self.need_centering.setChecked(True)
        self.need_centering.setToolTip(
            "Check this to center the images before alignment. Uncheck to skip coarse alignment."
        )
        self.need_gradient_descent = QCheckBox("Skip Gradient Descent")
        self.need_gradient_descent.setChecked(False)
        self.need_gradient_descent.setToolTip(
            "Check this to skip the gradient descent step during alignment."
        )
        self.checkbox_layout.addWidget(self.need_centering)
        self.checkbox_layout.addWidget(self.need_gradient_descent)
        # Register button
        self.button_layout = QHBoxLayout()
        self.button_layout.setContentsMargins(0, 0, 0, 0)
        self.register_button = QPushButton("Register Images")
        self.register_button.setEnabled(
            False
        )  # Initially disabled until both images are loaded
        self.register_button.clicked.connect(self.register_images)

        self.register_button.setMinimumHeight(30)
        self.register_button.setStyleSheet(
            """
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
        """
        )
        self.manually_align_button = QPushButton("Manually Align")
        self.manually_align_button.setEnabled(False)
        self.manually_align_button.setMinimumHeight(30)
        self.manually_align_button.setStyleSheet(
            """
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
        """
        )
        self.manually_align_button.clicked.connect(self.manually_align_images)

        self.button_layout.addWidget(self.manually_align_button)
        self.button_layout.addWidget(self.register_button)

        # Add all elements to the main layout
        # self.main_layout.addSpacing(5)
        self.main_layout.addLayout(self.image1_layout)
        self.main_layout.addLayout(self.target_spacing_row)
        self.main_layout.addLayout(self.image2_layout)
        self.main_layout.addLayout(self.unaligned_spacing_row)
        self.main_layout.addLayout(self.checkbox_layout)
        self.main_layout.addSpacing(10)
        self.main_layout.addLayout(self.button_layout)

        # Add the groupbox to the containing layout
        if containing_layout:
            containing_layout.addWidget(self.alignment_groupbox)

        self.__retranslate_UI()
        QMetaObject.connectSlotsByName(self)

    def _setup_connections(self):
        """Set up the connections to the aligner thread"""
        self.aligner.progress.connect(self._handle_progress)
        self.aligner.error.connect(self._handle_error)
        self.aligner.snapshot.connect(self._handle_snapshot)

        self.target_channel_selector.currentIndexChanged.connect(
            self.change_target_channel
        )
        self.unaligned_channel_selector.currentIndexChanged.connect(
            self.change_unaligned_channel
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
        # Handle the snapshot data (e.g., update the UI)
        snapshot_dialog = AlignmentPreviewDialog(
            snapshot_data,
            False,
        )
        result = snapshot_dialog.exec()
        return

    @pyqtSlot(int)
    def change_target_channel(self, index):
        self.image_channels[0] = index

    @pyqtSlot(int)
    def change_unaligned_channel(self, index):
        self.image_channels[1] = index

    def __retranslate_UI(self):
        _translate = QCoreApplication.translate
        self.alignment_groupbox.setTitle(
            _translate("MainWindow", "Cell Layer Alignment")
        )
        self.image1_label.setText(_translate("MainWindow", "Target Image:"))
        self.image2_label.setText(_translate("MainWindow", "Unaligned Image:"))
        self.register_button.setText(_translate("MainWindow", "Register Images"))
        self.manually_align_button.setText(_translate("MainWindow", "Manually Align"))

    def set_target_image(self, uuid, is_leaf, channel):
        """Set the target image for alignment"""
        self.target_uuid = uuid
        item = self.storage.get_data(uuid)
        assert item is not None, f"No data found for UUID: {uuid}"
        obj, name = item["data"], item["name"]
        self.target_image = obj

        self.target_name = name
        self.image1_status.setText(name)
        self.image1_status.setStyleSheet(
            "font-weight: bold; color: #007700;"
        )  # Green to indicate it's loaded
        self.image1_status.setWordWrap(True)
        self.target_channel_selector.setVisible(True)
        self.target_channel_selector.clear()
        self.target_channel_selector.addItems(obj.keys())
        self.target_channel_selector.setCurrentIndex(channel)
        self._check_can_register()

    def set_unaligned_image(self, uuid: UUID, is_leaf: bool, channel: int):
        """Set the unaligned image that will be registered to the target"""
        self.unaligned_uuid = uuid
        item = self.storage.get_data(uuid)
        assert item is not None, f"No data found for UUID: {uuid}"
        obj, name = item["data"], item["name"]
        self.unaligned_image = obj
        self.unaligned_name = name
        self.image2_status.setText(name)
        self.image2_status.setWordWrap(True)
        self.image2_status.setStyleSheet(
            "font-weight: bold; color: #007700;"
        )  # Green to indicate it's loaded
        self.unaligned_channel_selector.setVisible(True)
        self.unaligned_channel_selector.clear()
        self.unaligned_channel_selector.addItems(obj.keys())
        self.unaligned_channel_selector.setCurrentIndex(channel)

        self._check_can_register()

    def _check_can_register(self):
        """Check if both images are loaded and enable/disable register button"""
        if self.target_image is not None and self.unaligned_image is not None:
            self.register_button.setEnabled(True)
            self.manually_align_button.setEnabled(True)
        else:
            self.register_button.setEnabled(False)
            self.manually_align_button.setEnabled(False)

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
            target_spacing = (
                float(self.target_spacing_x.text().strip()),
                float(self.target_spacing_y.text().strip()),
            )
            self.aligner.set_target_spacing(target_spacing)

            unaligned_spacing = (
                float(self.unaligned_spacing_x.text().strip()),
                float(self.unaligned_spacing_y.text().strip()),
            )
            self.aligner.set_unaligned_spacing(unaligned_spacing)
        except ValueError:
            self._handle_error(
                "Invalid spacing values. Please enter numeric values for x and y."
            )
            return
        self.aligner.set_target_image(
            self.target_image[f"Channel {self.image_channels[0]+1}"].data,
            f"Channel {self.image_channels[0]+1}",
            self.target_uuid,
        )
        self.aligner.set_unaligned_image(
            self.unaligned_image[f"Channel {self.image_channels[1]+1}"].data,
            f"Channel {self.image_channels[1]+1}",
            self.unaligned_uuid,
        )

    def register_images(self):
        """Start the image registration process"""
        if not self.target_image is None and not self.unaligned_image is None:
            self.prepare_aligner()
            self.aligner.progress.emit(0, "Starting alignment...")
            # Start the alignment process in a separate thread
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
        tc, uc = self.image_channels
        tc = f"Channel {tc + 1}"
        uc = f"Channel {uc + 1}"
        # Create and show the alignment preview dialog
        preview_dialog = AlignmentPreviewDialog(
            {
                "target_image": self.target_image[tc].data,
                "aligned_image": self.unaligned_image[uc].data,
            },
            True,
        )
        preview_dialog.moving_image_changed.connect(self.aligner.manually_align)
        preview_dialog.exec()
        self.register_button.setEnabled(True)
        self.manually_align_button.setEnabled(True)

    def _handle_progress(self, value, message):
        """Handle progress updates from the aligner thread"""
        # You could add a progress bar to the UI if needed
        # For now, we'll just update the button text
        # self.register_button.setText(f"{message} ({value}%)")
        self.progress.emit(value, message)
        if value >= 100:
            self._handle_finished()

    def _handle_error(self, error_message):
        """Handle error messages from the aligner thread"""
        QMessageBox.critical(self, "Alignment Error", error_message)
        self.progress.emit(100, "Error occurred during alignment.")

        # Reset the image status colors to indicate failure
        if self.target_image is not None:
            self.image1_status.setStyleSheet(
                "font-weight: bold; color: #FF0000;"
            )  # Red to indicate error
        if self.unaligned_image is not None:
            self.image2_status.setStyleSheet(
                "font-weight: bold; color: #FF0000;"
            )  # Red to indicate error

    def _handle_finished(self):
        """Handle when the alignment thread finishes"""
        self.register_button.setEnabled(True)
        self.manually_align_button.setEnabled(True)
