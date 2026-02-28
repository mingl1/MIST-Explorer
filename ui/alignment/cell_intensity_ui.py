"""
Cell intensity UI module.
"""
import os
import typing

import numpy as np
import pandas as pd
# pylint: disable=no-name-in-module
from PyQt6.QtCore import QCoreApplication, pyqtSignal
from PyQt6.QtWidgets import (
    QComboBox,
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from core.project_naming import is_stardist_label_name


# pylint: disable=too-many-instance-attributes
class CellIntensityUI(QWidget):
    """
    UI for managing cell intensity analysis parameters.
    """
    errorSignal = pyqtSignal(str)
    emitBeadData = pyqtSignal(np.ndarray)
    emitColorCodes = pyqtSignal(dict)
    generate_cell_data = pyqtSignal()
    requestFilteredStats = pyqtSignal()

    def __init__(self, parent=None, containing_layout: typing.Optional[QVBoxLayout] = None):
        super().__init__(parent)
        self.channel_rows = []
        self.channel_to_color_code = {}
        self.available_channels = []
        self.bead_data_file = None

        # Attributes initialized in setup_ui
        self.cell_intensity_groupbox = None
        self.main_layout = None
        self.components_widget = None
        self.cellintensity_components_vlayout = None
        self.bead_data_layout = None
        self.bead_data = None
        self.bead_data_label = None
        self.channels_layout = None
        self.add_channel_button = None
        self.radius_fg_layout = None
        self.radius_fg_label = None
        self.radius_fg = None
        self.radius_bg_layout = None
        self.radius_bg_label = None
        self.radius_bg = None
        self.run_button = None
        self.cancel_button = None
        self.save_button = None
        self.filtered_stats_button = None

        self.setup_ui(parent, containing_layout)

    def update_channels(self, channels: dict):
        """Update available channels in dropdowns."""
        valid_channels = {}
        for channel_name, wrapper in channels.items():
            if is_stardist_label_name(getattr(wrapper, "name", "")):
                continue
            valid_channels[channel_name] = wrapper

        self.available_channels = sorted(
            valid_channels.keys(), key=lambda x: int(x.replace("Channel ", ""))
        )
        for row_widget in self.channel_rows:
            combo_box = row_widget.findChild(QComboBox)
            if combo_box:
                current_text = combo_box.currentText()
                combo_box.clear()
                combo_box.addItems(self.available_channels)
                if current_text in self.available_channels:
                    combo_box.setCurrentText(current_text)

    def setup_ui(self, parent, containing_layout: typing.Optional[QVBoxLayout]):
        """Initialize the UI components."""
        self.cell_intensity_groupbox = QGroupBox(parent)
        self.cell_intensity_groupbox.setObjectName("cell_intensity_groupbox")

        self.main_layout = QVBoxLayout(self.cell_intensity_groupbox)

        self.components_widget = QWidget()
        self.cellintensity_components_vlayout = QVBoxLayout(self.components_widget)

        # bead data
        self.bead_data_layout = QHBoxLayout()
        self.bead_data = QPushButton(self.cell_intensity_groupbox)
        self.bead_data.clicked.connect(self.load_bead_data)
        self.bead_data_label = QLabel()
        self.bead_data_layout.addWidget(self.bead_data)
        self.bead_data_layout.addWidget(self.bead_data_label)
        self.cellintensity_components_vlayout.addLayout(self.bead_data_layout)

        # Channels layout
        self.channels_layout = QVBoxLayout()
        self.cellintensity_components_vlayout.addLayout(self.channels_layout)

        self.add_channel_button = QPushButton()
        self.add_channel_button.clicked.connect(self.add_channel_row)
        self.cellintensity_components_vlayout.addWidget(self.add_channel_button)

        # radius fg
        self.radius_fg_layout = QHBoxLayout()
        self.radius_fg_label = QLabel(self.cell_intensity_groupbox)
        self.radius_fg_layout.addWidget(self.radius_fg_label)
        self.radius_fg = QSpinBox(self.cell_intensity_groupbox)
        self.radius_fg.setProperty("value", 2)
        self.radius_fg_layout.addWidget(self.radius_fg)
        self.cellintensity_components_vlayout.addLayout(self.radius_fg_layout)

        # radius bg
        self.radius_bg_layout = QHBoxLayout()
        self.radius_bg_label = QLabel(self.cell_intensity_groupbox)
        self.radius_bg_layout.addWidget(self.radius_bg_label)
        self.radius_bg = QSpinBox(self.cell_intensity_groupbox)
        self.radius_bg.setProperty("value", 6)
        self.radius_bg_layout.addWidget(self.radius_bg)
        self.cellintensity_components_vlayout.addLayout(self.radius_bg_layout)

        # run button
        self.run_button = QPushButton(self.cell_intensity_groupbox)
        self.run_button.clicked.connect(self._handle_generate_cell_data)
        self.run_button.setEnabled(False)
        self.cellintensity_components_vlayout.addWidget(self.run_button)

        # cancel button
        self.cancel_button = QPushButton(self.cell_intensity_groupbox)
        self.cellintensity_components_vlayout.addWidget(self.cancel_button)

        # save button
        self.save_button = QPushButton(self.cell_intensity_groupbox)
        self.cellintensity_components_vlayout.addWidget(self.save_button)

        # filtered bead stats button
        self.filtered_stats_button = QPushButton(self.cell_intensity_groupbox)
        self.filtered_stats_button.clicked.connect(self._show_filtered_bead_stats)
        self.cellintensity_components_vlayout.addWidget(self.filtered_stats_button)

        self.main_layout.addWidget(self.components_widget)

        if containing_layout is not None:
            containing_layout.addWidget(self.cell_intensity_groupbox)

        self._retranslate_ui()

    def _handle_generate_cell_data(self):
        if self.bead_data_file is not None:
            self.emitBeadData.emit(self.bead_data_file)
            self.emitColorCodes.emit(self.channel_to_color_code)
            self.generate_cell_data.emit()
        else:
            self.errorSignal.emit("Please load bead data first.")

    def add_channel_row(self):
        """Add a new row for channel color code mapping."""
        row_widget = QWidget(self)
        row_layout = QHBoxLayout(row_widget)

        channel_combo = QComboBox()
        channel_combo.setEditable(True)
        if self.available_channels:
            channel_combo.addItems(self.available_channels)
        else:
            channel_combo.addItems([f"Channel {i+1}" for i in range(10)])

        load_button = QPushButton("Load Color Code")
        remove_button = QPushButton("Remove")

        row_layout.addWidget(channel_combo)
        row_layout.addWidget(load_button)
        row_layout.addWidget(remove_button)

        self.channels_layout.addWidget(row_widget)
        self.channel_rows.append(row_widget)

        load_button.clicked.connect(
            lambda: self.load_color_code_for_channel(channel_combo, load_button)
        )
        remove_button.clicked.connect(lambda: self.remove_channel_row(row_widget))

    def remove_channel_row(self, row_widget):
        """Remove a channel row."""
        channel_combo = row_widget.findChild(QComboBox)
        if channel_combo:
            channel_name = channel_combo.currentText()
            if channel_name in self.channel_to_color_code:
                del self.channel_to_color_code[channel_name]

        if row_widget in self.channel_rows:
            self.channel_rows.remove(row_widget)

        row_widget.deleteLater()

    def load_color_code_for_channel(self, channel_combo, button):
        """Load color code from file for a specific channel."""
        channel_name = channel_combo.currentText()
        if not channel_name:
            self.errorSignal.emit("Please specify a channel name.")
            return

        file_name, _ = QFileDialog.getOpenFileName(
            self,
            f"Open Color Code for {channel_name}",
            "",
            "Color Code(*.csv *.xlsx);;All Files (*)",
        )
        if file_name:
            try:
                if file_name.endswith(".csv"):
                    color_code = pd.read_csv(file_name)
                else:
                    color_code = pd.read_excel(file_name)
                self.channel_to_color_code[channel_name] = color_code

                base_name = os.path.basename(file_name)
                stem, ext = os.path.splitext(base_name)
                if len(stem) > 10:
                    button_text = f"{stem[:5]}...{stem[-5:]}{ext}"
                else:
                    button_text = base_name
                button.setText(button_text)
            except (UnicodeDecodeError, pd.errors.ParserError):
                self.errorSignal.emit("Please select a valid file type")

    def load_bead_data(self):
        """Load bead data from file."""
        file_name, _ = QFileDialog.getOpenFileName(
            self, "Open Bead Data", "", "Bead Data(*.csv *.xlsx);;All Files (*)"
        )
        if file_name:
            try:
                self.bead_data_file = (
                    pd.read_csv(file_name).to_numpy().astype("uint16")
                )
                self.bead_data_label.setText(os.path.basename(file_name))
            except UnicodeDecodeError:
                self.errorSignal.emit("Please select a valid file type")

    def _show_filtered_bead_stats(self):
        if self.bead_data_file is not None:
            self.emitBeadData.emit(self.bead_data_file)
            self.requestFilteredStats.emit()
        else:
            self.errorSignal.emit("Please load bead data first.")

    def set_generation_enabled(self, enabled: bool):
        """Enable generation only when a source image is selected in the image manager."""
        self.run_button.setEnabled(bool(enabled))

    def show_filtered_stats(self, percentage: float):
        QMessageBox.information(
            self,
            "Filtered Bead Stats",
            f"Filtered beads (cy0 >= 254) inside cell boundaries: {percentage:.2f}%",
        )

    def _retranslate_ui(self):
        _translate = QCoreApplication.translate
        self.cell_intensity_groupbox.setTitle(
            _translate("MainWindow", "Generate Protein Data of Cells")
        )

        self.add_channel_button.setText(_translate("MainWindow", "+ Add Channel"))
        self.bead_data_label.setText(_translate("MainWindow", "none selected"))
        self.bead_data.setText(_translate("MainWindow", "Open Bead Data"))
        self.radius_fg_label.setText(_translate("MainWindow", "Radius fg"))
        self.radius_bg_label.setText(_translate("MainWindow", "Radius bg"))
        self.run_button.setText(_translate("MainWindow", "Run"))
        self.save_button.setText(_translate("MainWindow", "Save"))
        self.cancel_button.setText(_translate("MainWindow", "Cancel"))
        self.filtered_stats_button.setText(_translate("MainWindow", "Filtered Bead Stats"))
