import numpy as np
import pandas as pd
from PyQt6.QtCore import QCoreApplication, QMetaObject, Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSlider,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from ui.processing.segmentation_image_selector import SegmentationImageSelector


class RegisterUI(QWidget):
    errorSignal = pyqtSignal(str)
    emitBeadData = pyqtSignal(np.ndarray)
    emitColorCode = pyqtSignal(pd.DataFrame)

    def __init__(self, parent, containing_layout: QVBoxLayout):
        super().__init__()
        self.setupUI(parent, containing_layout)

    def setupUI(self, parent, containing_layout: QVBoxLayout):
        self.register_groupbox = QGroupBox(parent)

        self.horizontalLayout_4 = QHBoxLayout(self.register_groupbox)
        self.register_components_vlayout = QVBoxLayout()
        self.register_components_vlayout.setSpacing(0)
        self.register_components_vlayout.setContentsMargins(0, 0, 0, 0)

        # Reference image selector
        self.reference_label = QLabel("Reference Image:")
        self.register_components_vlayout.addWidget(self.reference_label)
        self.reference_selector = SegmentationImageSelector(self.register_groupbox)
        self.register_components_vlayout.addWidget(self.reference_selector)

        # Moving image selector
        self.moving_label = QLabel("Moving Image:")
        self.register_components_vlayout.addWidget(self.moving_label)
        self.image_selector = SegmentationImageSelector(self.register_groupbox)
        self.register_components_vlayout.addWidget(self.image_selector)

        # max size
        self.max_size_layout = QHBoxLayout()
        self.max_size_label = QLabel(self.register_groupbox)
        self.max_size_layout.addWidget(self.max_size_label)
        self.max_size = QSpinBox(self.register_groupbox)
        self.max_size.setMaximum(100000)
        self.max_size.setSingleStep(100)
        self.max_size.setProperty("value", 10000)
        self.max_size_layout.addWidget(self.max_size)
        self.register_components_vlayout.addLayout(self.max_size_layout)

        # num tiles
        self.num_tiles_layout = QHBoxLayout()
        self.num_tiles_label = QLabel(self.register_groupbox)
        self.num_tiles_layout.addWidget(self.num_tiles_label)
        self.num_tiles = QSpinBox(self.register_groupbox)
        self.num_tiles.setMaximum(30)
        self.num_tiles.setSingleStep(1)
        self.num_tiles.setProperty("value", 10)
        self.num_tiles_layout.addWidget(self.num_tiles)
        self.register_components_vlayout.addLayout(self.num_tiles_layout)

        # overlap
        self.overlap_layout = QHBoxLayout()
        self.overlap_label = QLabel(self.register_groupbox)
        self.overlap_layout.addWidget(self.overlap_label)
        self.overlap = QSpinBox(self.register_groupbox)
        self.overlap.setMaximum(1000)
        self.overlap.setSingleStep(10)
        self.overlap.setProperty("value", 250)
        self.overlap_layout.addWidget(self.overlap)
        self.register_components_vlayout.addLayout(self.overlap_layout)

        # NCC threshold
        self.ncc_threshold_layout = QHBoxLayout()
        self.ncc_threshold_label = QLabel(self.register_groupbox)
        self.ncc_threshold_layout.addWidget(self.ncc_threshold_label)
        self.ncc_threshold = QSlider(Qt.Orientation.Horizontal, self.register_groupbox)
        self.ncc_threshold.setMinimum(85)
        self.ncc_threshold.setMaximum(100)
        self.ncc_threshold.setValue(85)
        self.ncc_threshold.setTickPosition(QSlider.TickPosition.TicksBelow)
        self.ncc_threshold.setTickInterval(10)
        self.ncc_threshold_value_label = QLabel("0.85", self.register_groupbox)
        self.ncc_threshold_value_label.setFixedWidth(30)
        self.ncc_threshold.valueChanged.connect(
            lambda v: self.ncc_threshold_value_label.setText(f"{v / 100:.2f}")
        )
        self.ncc_threshold_layout.addWidget(self.ncc_threshold)
        self.ncc_threshold_layout.addWidget(self.ncc_threshold_value_label)
        self.register_components_vlayout.addLayout(self.ncc_threshold_layout)

        # run button
        self.run_button = QPushButton(self.register_groupbox)
        self.register_components_vlayout.addWidget(self.run_button)

        # cancel button
        self.cancel_button = QPushButton(self.register_groupbox)
        self.register_components_vlayout.addWidget(self.cancel_button)
        self.horizontalLayout_4.addLayout(self.register_components_vlayout)
        containing_layout.addWidget(self.register_groupbox)

        self.__retranslate_UI()
        QMetaObject.connectSlotsByName(self)

    def __retranslate_UI(self):
        _translate = QCoreApplication.translate
        self.register_groupbox.setTitle(_translate("MainWindow", "Align Arrays"))
        self.max_size_label.setText(_translate("MainWindow", "Max Size (px)"))
        self.num_tiles_label.setText(_translate("MainWindow", "Number of Tiles"))
        self.overlap_label.setText(_translate("MainWindow", "Overlap (px)"))
        self.ncc_threshold_label.setText(_translate("MainWindow", "NCC Threshold"))
        self.run_button.setText(_translate("MainWindow", "Run Full Image"))
        self.cancel_button.setText(_translate("MainWindow", "Cancel"))
