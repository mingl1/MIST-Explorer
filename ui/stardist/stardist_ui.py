from PyQt6.QtCore import QCoreApplication, QMetaObject
from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSpinBox,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from core.project_naming import is_stardist_label_name
from ui.stardist.cellprofiler_ui import CellProfilerAdvancedSettings


class StarDistUI(QWidget):
    def __init__(self, parent=None, containing_layout: QVBoxLayout = None):
        super().__init__()
        self.setupUI(parent, containing_layout)

    def setupUI(self, parent, containing_layout: QVBoxLayout):

        self.stardist_groupbox = QGroupBox(parent)
        self.horizontalLayout_4 = QVBoxLayout(self.stardist_groupbox)
        self.stardist_components_vlayout = QVBoxLayout()
        self.stardist_components_vlayout.setSpacing(0)
        self.stardist_components_vlayout.setContentsMargins(0, 0, 0, 0)

        self.segmentation_tabs = QTabWidget(self.stardist_groupbox)
        self.basic_tab = QWidget(self.segmentation_tabs)
        self.advanced_tab = QWidget(self.segmentation_tabs)
        self.segmentation_tabs.addTab(self.basic_tab, "")
        self.segmentation_tabs.addTab(self.advanced_tab, "")

        self.basic_tab_layout = QVBoxLayout(self.basic_tab)
        self.basic_tab_layout.setSpacing(0)
        self.basic_tab_layout.setContentsMargins(0, 0, 0, 0)

        self.advanced_tab_layout = QVBoxLayout(self.advanced_tab)
        self.advanced_tab_layout.setSpacing(0)
        self.advanced_tab_layout.setContentsMargins(0, 0, 0, 0)

        # channel selector
        self.stardist_channel_selector_layout = QHBoxLayout()
        self.stardist_channel_selector = QComboBox(self.stardist_groupbox)
        self.stardist_channel_selector_label = QLabel(self.stardist_groupbox)
        self.stardist_channel_selector_layout.addWidget(
            self.stardist_channel_selector_label
        )
        self.stardist_channel_selector_layout.addWidget(self.stardist_channel_selector)
        self.basic_tab_layout.addLayout(self.stardist_channel_selector_layout)

        # segmentation method selector
        self.segmentation_method_layout = QHBoxLayout()
        self.segmentation_method_label = QLabel(self.stardist_groupbox)
        self.segmentation_method_selector = QComboBox(self.stardist_groupbox)
        self.segmentation_method_selector.addItems(["StarDist", "CellProfiler-like"])
        self.segmentation_method_layout.addWidget(self.segmentation_method_label)
        self.segmentation_method_layout.addWidget(self.segmentation_method_selector)
        self.horizontalLayout_4.addLayout(self.segmentation_method_layout)
        self.horizontalLayout_4.addWidget(self.segmentation_tabs)

        # dilation
        self.stardist_hlayout8 = QHBoxLayout()
        self.stardist_label8 = QLabel(self.stardist_groupbox)
        self.enable_dilation = QCheckBox(self.stardist_groupbox)
        self.enable_dilation.setChecked(True)
        self.stardist_hlayout8.addWidget(self.enable_dilation)
        self.stardist_hlayout8.addWidget(self.stardist_label8)
        self.radius = QSpinBox(self.stardist_groupbox)
        self.radius.setProperty("value", 5)
        self.stardist_hlayout8.addWidget(self.radius)
        self.basic_tab_layout.addLayout(self.stardist_hlayout8)

        # min size
        self.min_size_layout = QHBoxLayout()
        self.min_size_label = QLabel(self.stardist_groupbox)
        self.min_size_layout.addWidget(self.min_size_label)
        self.min_size = QSpinBox(self.stardist_groupbox)
        self.min_size.setRange(1, 10000)
        self.min_size.setProperty("value", 60)
        self.min_size_layout.addWidget(self.min_size)
        self.basic_tab_layout.addLayout(self.min_size_layout)

        # max size
        self.max_size_layout = QHBoxLayout()
        self.max_size_label = QLabel(self.stardist_groupbox)
        self.max_size_layout.addWidget(self.max_size_label)
        self.max_size = QSpinBox(self.stardist_groupbox)
        self.max_size.setRange(1, 10000)
        self.max_size.setProperty("value", 180)
        self.max_size_layout.addWidget(self.max_size)
        self.basic_tab_layout.addLayout(self.max_size_layout)

        # Load Preset button (visible only for CellProfiler-like)
        self.load_preset_button = QPushButton(self.stardist_groupbox)
        self.load_preset_button.setText("Load Preset")
        self.basic_tab_layout.addWidget(self.load_preset_button)

        # pretrained 2D Model
        self.stardist_hlayout1 = QHBoxLayout()
        self.stardist_label1 = QLabel(self.stardist_groupbox)
        self.stardist_hlayout1.addWidget(self.stardist_label1)
        self.stardist_pretrained_models = QComboBox(self.stardist_groupbox)
        self.stardist_pretrained_models.addItems(
            ["2D_versatile_fluo", "2D_paper_dsb2018", "2D_versatile_he"]
        )
        self.stardist_hlayout1.addWidget(self.stardist_pretrained_models)
        self.advanced_tab_layout.addLayout(self.stardist_hlayout1)

        # percentile low
        self.stardist_hlayout2 = QHBoxLayout()
        self.stardist_label2 = QLabel(self.stardist_groupbox)
        self.stardist_hlayout2.addWidget(self.stardist_label2)
        self.percentile_low = QDoubleSpinBox(self.stardist_groupbox)
        self.percentile_low.setProperty("value", 3.0)
        self.stardist_hlayout2.addWidget(self.percentile_low)
        self.advanced_tab_layout.addLayout(self.stardist_hlayout2)

        # percentile high
        self.stardist_hlayout3 = QHBoxLayout()
        self.stardist_label3 = QLabel(self.stardist_groupbox)
        self.stardist_hlayout3.addWidget(self.stardist_label3)
        self.percentile_high = QDoubleSpinBox(self.stardist_groupbox)
        self.percentile_high.setProperty("value", 99.80)
        self.stardist_hlayout3.addWidget(self.percentile_high)
        self.advanced_tab_layout.addLayout(self.stardist_hlayout3)

        # prob threshold
        self.stardist_hlayout4 = QHBoxLayout()
        self.stardist_label4 = QLabel(self.stardist_groupbox)
        self.stardist_hlayout4.addWidget(self.stardist_label4)
        self.prob_threshold = QDoubleSpinBox(self.stardist_groupbox)
        self.prob_threshold.setProperty("value", 0.48)
        self.stardist_hlayout4.addWidget(self.prob_threshold)
        self.advanced_tab_layout.addLayout(self.stardist_hlayout4)

        # nms threshold
        self.stardist_hlayout5 = QHBoxLayout()
        self.stardist_label5 = QLabel(self.stardist_groupbox)
        self.stardist_hlayout5.addWidget(self.stardist_label5)
        self.nms_threshold = QDoubleSpinBox(self.stardist_groupbox)
        self.nms_threshold.setProperty("value", 0.3)
        self.stardist_hlayout5.addWidget(self.nms_threshold)
        self.advanced_tab_layout.addLayout(self.stardist_hlayout5)

        # scale
        self.stardist_hlayout6 = QHBoxLayout()
        self.stardist_label6 = QLabel(self.stardist_groupbox)
        self.stardist_hlayout6.addWidget(self.stardist_label6)
        self.scale_factor = QDoubleSpinBox(self.stardist_groupbox)
        self.scale_factor.setDecimals(2)
        self.scale_factor.setRange(0.1, 10.0)
        self.scale_factor.setSingleStep(0.1)
        self.scale_factor.setProperty("value", 1.0)
        self.stardist_hlayout6.addWidget(self.scale_factor)
        self.advanced_tab_layout.addLayout(self.stardist_hlayout6)

        # number of tiles
        self.stardist_hlayout7 = QHBoxLayout()
        self.stardist_label7 = QLabel(self.stardist_groupbox)
        self.stardist_hlayout7.addWidget(self.stardist_label7)
        self.n_tiles = QSpinBox(self.stardist_groupbox)
        self.n_tiles.setProperty("value", 0)
        self.stardist_hlayout7.addWidget(self.n_tiles)
        self.advanced_tab_layout.addLayout(self.stardist_hlayout7)

        # CellProfiler-like advanced settings widget
        self.cp_advanced = CellProfilerAdvancedSettings()
        self.advanced_tab_layout.addWidget(self.cp_advanced)

        self.use_contrasted_checkbox = QCheckBox(self.basic_tab)
        self.use_contrasted_checkbox.setChecked(False)
        self.basic_tab_layout.addWidget(self.use_contrasted_checkbox)

        self.enable_dilation.toggled.connect(self.radius.setEnabled)

        # run button
        self.stardist_run_button = QPushButton(self.stardist_groupbox)
        self.stardist_components_vlayout.addWidget(self.stardist_run_button)

        # overlay toggle
        self.overlay_toggle_button = QPushButton(self.stardist_groupbox)
        self.overlay_toggle_button.setCheckable(True)
        self.overlay_toggle_button.setChecked(False)
        self.stardist_components_vlayout.addWidget(self.overlay_toggle_button)

        # cancel button
        self.cancel_button = QPushButton(self.stardist_groupbox)
        self.stardist_components_vlayout.addWidget(self.cancel_button)
        self.horizontalLayout_4.addLayout(self.stardist_components_vlayout)
        containing_layout.addWidget(self.stardist_groupbox)

        self.__retranslate_UI()
        self._stardist_specific_rows = [
            self.stardist_hlayout1,
            self.stardist_hlayout2,
            self.stardist_hlayout3,
            self.stardist_hlayout4,
            self.stardist_hlayout5,
            self.stardist_hlayout6,
            self.stardist_hlayout7,
        ]
        self._basic_stardist_rows = []
        self._basic_primary_rows = [self.min_size_layout, self.max_size_layout, self.load_preset_button]
        self.segmentation_method_selector.currentTextChanged.connect(
            self._update_method_controls
        )
        self._update_method_controls()
        QMetaObject.connectSlotsByName(self)

    def set_groupbox_title(self, name, channel):
        self.stardist_groupbox.setTitle(f"Cell Segmentation - {name}")
        self.stardist_channel_selector.setCurrentText(channel)

    def updateChannelSelector(self, channels: dict):
        previous_channel = self.stardist_channel_selector.currentText()
        self.clearChannelSelector()
        channel_keys = sorted(
            [
                key
                for key, wrapper in channels.items()
                if not is_stardist_label_name(getattr(wrapper, "name", ""))
            ],
            key=lambda x: int(x.replace("Channel ", "")),
        )
        self.stardist_channel_selector.addItems(channel_keys)
        if previous_channel in channel_keys:
            self.stardist_channel_selector.setCurrentText(previous_channel)
        self.overlay_toggle_button.setChecked(False)
        self.__retranslate_UI()

    def clearChannelSelector(self):
        self.stardist_channel_selector.clear()

    def _update_method_controls(self):
        use_stardist = self.segmentation_method_selector.currentText() == "StarDist"
        self._set_row_visibility(self._stardist_specific_rows, use_stardist)
        self._set_row_visibility(self._basic_stardist_rows, use_stardist)
        self._set_row_visibility(self._basic_primary_rows, not use_stardist)
        self.cp_advanced.setVisible(not use_stardist)
        self.segmentation_tabs.setTabEnabled(1, True)
        self.segmentation_tabs.setTabToolTip(1, "")

    def _set_row_visibility(self, rows, visible):
        for row in rows:
            if isinstance(row, QWidget):
                row.setVisible(visible)
                continue
            self._set_layout_visibility(row, visible)

    def _set_layout_visibility(self, layout, visible):
        for idx in range(layout.count()):
            item = layout.itemAt(idx)
            child_widget = item.widget()
            if child_widget is not None:
                child_widget.setVisible(visible)
                continue
            child_layout = item.layout()
            if child_layout is not None:
                self._set_layout_visibility(child_layout, visible)

    def __retranslate_UI(self):
        _translate = QCoreApplication.translate
        self.stardist_groupbox.setTitle(
            _translate(
                "MainWindow", "Cell Segmentation - Current Canvas Image"
            )
        )
        self.segmentation_tabs.setTabText(0, _translate("MainWindow", "Basic"))
        self.segmentation_tabs.setTabText(1, _translate("MainWindow", "Advanced"))
        self.stardist_channel_selector_label.setText(
            _translate("MainWindow", "Cell Channel")
        )
        self.segmentation_method_label.setText(
            _translate("MainWindow", "Method")
        )
        self.stardist_label1.setText(_translate("MainWindow", "Pre-trained 2D Model"))
        self.stardist_label2.setText(_translate("MainWindow", "Percentile Low"))
        self.stardist_label3.setText(_translate("MainWindow", "Percentile High"))
        self.stardist_label4.setText(
            _translate("MainWindow", "Probability/ Score Threshold")
        )
        self.stardist_label5.setText(_translate("MainWindow", "Overlap Threshold"))
        self.stardist_label6.setText(_translate("MainWindow", "Scale"))
        self.stardist_label7.setText(_translate("MainWindow", "Number of Tiles"))
        self.enable_dilation.setText(_translate("MainWindow", "Enable Dilation"))
        self.stardist_label8.setText(_translate("MainWindow", "Dilation Radius"))
        self.min_size_label.setText(_translate("MainWindow", "Min Size (diameter px)"))
        self.max_size_label.setText(_translate("MainWindow", "Max Size (diameter px)"))
        self.use_contrasted_checkbox.setText(
            _translate("MainWindow", "Use contrasted image (toolbar sliders)")
        )
        self.stardist_run_button.setText(_translate("MainWindow", "Run"))
        self.overlay_toggle_button.setText(_translate("MainWindow", "Show Overlay"))
        self.cancel_button.setText(_translate("MainWindow", "Cancel"))
