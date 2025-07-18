import os

import numpy as np
import pandas as pd
from PyQt6.QtCore import QCoreApplication, QMetaObject, QSize, pyqtSignal
from PyQt6.QtWidgets import (
    QComboBox,
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)


class CellIntensityUI(QWidget):
    errorSignal = pyqtSignal(str)
    emitBeadData = pyqtSignal(np.ndarray)
    emitColorCode = pyqtSignal(pd.DataFrame)

    def __init__(self, parent=None, containing_layout: QVBoxLayout = None):
        super().__init__()
        self.setupUI(parent, containing_layout)

    def setupUI(self, parent, containing_layout: QVBoxLayout):
        self.cell_intensity_groupbox = QGroupBox(parent)
        self.cell_intensity_groupbox.setObjectName("cell_intensity_groupbox")

        # Main layout for the groupbox - this is essential for proper layout
        self.main_layout = QVBoxLayout(self.cell_intensity_groupbox)

        # Container widget for all components
        self.components_widget = QWidget()
        self.cellintensity_components_vlayout = QVBoxLayout(self.components_widget)
        self.cellintensity_components_vlayout.setSpacing(6)  # Better spacing
        self.cellintensity_components_vlayout.setContentsMargins(5, 5, 5, 5)

        # bead data
        self.bead_data_layout = QHBoxLayout()
        self.bead_data = QPushButton(self.cell_intensity_groupbox)
        self.bead_data_label = QLabel()
        self.bead_data_layout.addWidget(self.bead_data)
        self.bead_data_layout.addWidget(self.bead_data_label)
        self.cellintensity_components_vlayout.addLayout(self.bead_data_layout)

        # color code
        self.color_code_layout = QHBoxLayout()
        self.color_code = QPushButton(self.cell_intensity_groupbox)
        self.color_code_label = QLabel()
        self.color_code_layout.addWidget(self.color_code)
        self.color_code_layout.addWidget(self.color_code_label)
        self.cellintensity_components_vlayout.addLayout(self.color_code_layout)

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

        # Spacer to push buttons to bottom
        self.cellintensity_components_vlayout.addStretch()

        # run button
        self.run_button = QPushButton(self.cell_intensity_groupbox)
        self.cellintensity_components_vlayout.addWidget(self.run_button)

        # cancel button
        self.cancel_button = QPushButton(self.cell_intensity_groupbox)
        self.cellintensity_components_vlayout.addWidget(self.cancel_button)

        # save button
        self.save_button = QPushButton(self.cell_intensity_groupbox)
        self.cellintensity_components_vlayout.addWidget(self.save_button)

        # Add the components widget to the main layout
        self.main_layout.addWidget(self.components_widget)

        # Add the groupbox to the containing layout
        containing_layout.addWidget(self.cell_intensity_groupbox)

        # Setup UI text and connections
        self.__retranslate_UI()

    def loadBeadData(self):
        file_name, _ = QFileDialog.getOpenFileName(
            None, "Open Bead Data", "", "Bead Data(*.csv *.xlsx);;All Files (*)"
        )
        if file_name:
            try:
                bead_data = (
                    pd.read_csv(file_name).to_numpy().astype("uint16")
                )  # this is the output from the registration->decoding program
                self.emitBeadData.emit(bead_data)
                self.bead_data_label.setText(os.path.basename(file_name))
            except UnicodeDecodeError:
                self.errorSignal.emit("Please select a valid file type")

    def loadColorCode(self):
        file_name, _ = QFileDialog.getOpenFileName(
            None, "Open Color Code", "", "Color Code(*.csv *.xlsx);;All Files (*)"
        )
        if file_name:
            try:
                color_code = pd.read_csv(file_name)
                self.emitColorCode.emit(color_code)
                self.color_code_label.setText(os.path.basename(file_name))
            except UnicodeDecodeError:
                self.errorSignal.emit("Please select a valid file type")

    def __retranslate_UI(self):
        _translate = QCoreApplication.translate
        self.cell_intensity_groupbox.setTitle(
            _translate("MainWindow", "Generate Protein Data of Cells")
        )

        self.color_code_label.setText(_translate("MainWindow", "none selected"))
        self.color_code.setText(_translate("MainWindow", "Open Color Code"))
        self.bead_data_label.setText(_translate("MainWindow", "none selected"))
        self.bead_data.setText(_translate("MainWindow", "Open Bead Data"))
        self.radius_fg_label.setText(_translate("MainWindow", "Radius fg"))
        self.radius_bg_label.setText(_translate("MainWindow", "Radius bg"))
        self.run_button.setText(_translate("MainWindow", "Run"))
        self.save_button.setText(_translate("MainWindow", "Save"))
        self.cancel_button.setText(_translate("MainWindow", "Cancel"))
