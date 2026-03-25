from PyQt6.QtCore import QCoreApplication
from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)


class CellProfilerAdvancedSettings(QWidget):
    """Advanced settings panel for CellProfiler-like segmentation."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._setup_ui()
        self._setup_connections()
        self._retranslate_ui()
        self._update_threshold_method_controls()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(0)
        layout.setContentsMargins(0, 0, 0, 0)

        # -- Threshold method --
        self.threshold_method_layout = QHBoxLayout()
        self.threshold_method_label = QLabel()
        self.threshold_method = QComboBox()
        self.threshold_method.addItems(
            [
                "MCT",
                "Otsu",
                "MoG",
                "Background",
                "RobustBackground",
                "RidlerCalvard",
                "Kapur",
                "Manual",
            ]
        )
        self.threshold_method_layout.addWidget(self.threshold_method_label)
        self.threshold_method_layout.addWidget(self.threshold_method)
        layout.addLayout(self.threshold_method_layout)

        # -- Threshold scope --
        self.threshold_scope_layout = QHBoxLayout()
        self.threshold_scope_label = QLabel()
        self.threshold_scope = QComboBox()
        self.threshold_scope.addItems(["Global", "Adaptive"])
        self.threshold_scope_layout.addWidget(self.threshold_scope_label)
        self.threshold_scope_layout.addWidget(self.threshold_scope)
        layout.addLayout(self.threshold_scope_layout)

        # -- Threshold correction factor --
        self.tcf_layout = QHBoxLayout()
        self.tcf_label = QLabel()
        self.threshold_correction_factor = QDoubleSpinBox()
        self.threshold_correction_factor.setDecimals(2)
        self.threshold_correction_factor.setRange(0.1, 3.0)
        self.threshold_correction_factor.setSingleStep(0.1)
        self.threshold_correction_factor.setValue(1.0)
        self.tcf_layout.addWidget(self.tcf_label)
        self.tcf_layout.addWidget(self.threshold_correction_factor)
        layout.addLayout(self.tcf_layout)

        # -- Threshold smoothing scale --
        self.tss_layout = QHBoxLayout()
        self.tss_label = QLabel()
        self.threshold_smoothing_scale = QDoubleSpinBox()
        self.threshold_smoothing_scale.setDecimals(4)
        self.threshold_smoothing_scale.setRange(0.0, 10.0)
        self.threshold_smoothing_scale.setSingleStep(0.1)
        self.threshold_smoothing_scale.setValue(1.3488)
        self.tss_layout.addWidget(self.tss_label)
        self.tss_layout.addWidget(self.threshold_smoothing_scale)
        layout.addLayout(self.tss_layout)

        # -- Lower bound on threshold --
        self.tlb_layout = QHBoxLayout()
        self.tlb_label = QLabel()
        self.threshold_lower_bound = QDoubleSpinBox()
        self.threshold_lower_bound.setDecimals(2)
        self.threshold_lower_bound.setRange(0.0, 1.0)
        self.threshold_lower_bound.setSingleStep(0.01)
        self.threshold_lower_bound.setValue(0.0)
        self.tlb_layout.addWidget(self.tlb_label)
        self.tlb_layout.addWidget(self.threshold_lower_bound)
        layout.addLayout(self.tlb_layout)

        # -- Upper bound on threshold --
        self.tub_layout = QHBoxLayout()
        self.tub_label = QLabel()
        self.threshold_upper_bound = QDoubleSpinBox()
        self.threshold_upper_bound.setDecimals(2)
        self.threshold_upper_bound.setRange(0.0, 1.0)
        self.threshold_upper_bound.setSingleStep(0.01)
        self.threshold_upper_bound.setValue(1.0)
        self.tub_layout.addWidget(self.tub_label)
        self.tub_layout.addWidget(self.threshold_upper_bound)
        layout.addLayout(self.tub_layout)

        # -- Manual threshold (visible when method=Manual) --
        self.manual_layout = QHBoxLayout()
        self.manual_label = QLabel()
        self.manual_threshold = QDoubleSpinBox()
        self.manual_threshold.setDecimals(3)
        self.manual_threshold.setRange(0.0, 1.0)
        self.manual_threshold.setSingleStep(0.01)
        self.manual_threshold.setValue(0.5)
        self.manual_layout.addWidget(self.manual_label)
        self.manual_layout.addWidget(self.manual_threshold)
        layout.addLayout(self.manual_layout)

        # -- Otsu: two/three-class --
        self.otsu_class_layout = QHBoxLayout()
        self.otsu_class_label = QLabel()
        self.otsu_class = QComboBox()
        self.otsu_class.addItems(["Two classes", "Three classes"])
        self.otsu_class_layout.addWidget(self.otsu_class_label)
        self.otsu_class_layout.addWidget(self.otsu_class)
        layout.addLayout(self.otsu_class_layout)

        # -- Otsu: assign middle to --
        self.otsu_middle_layout = QHBoxLayout()
        self.otsu_middle_label = QLabel()
        self.otsu_middle = QComboBox()
        self.otsu_middle.addItems(["Foreground", "Background"])
        self.otsu_middle_layout.addWidget(self.otsu_middle_label)
        self.otsu_middle_layout.addWidget(self.otsu_middle)
        layout.addLayout(self.otsu_middle_layout)

        # -- MoG: object fraction --
        self.mog_layout = QHBoxLayout()
        self.mog_label = QLabel()
        self.object_fraction = QDoubleSpinBox()
        self.object_fraction.setDecimals(2)
        self.object_fraction.setRange(0.01, 0.99)
        self.object_fraction.setSingleStep(0.05)
        self.object_fraction.setValue(0.2)
        self.mog_layout.addWidget(self.mog_label)
        self.mog_layout.addWidget(self.object_fraction)
        layout.addLayout(self.mog_layout)

        # -- RobustBackground: lower outlier fraction --
        self.rb_lower_layout = QHBoxLayout()
        self.rb_lower_label = QLabel()
        self.lower_outlier_fraction = QDoubleSpinBox()
        self.lower_outlier_fraction.setDecimals(2)
        self.lower_outlier_fraction.setRange(0.0, 0.5)
        self.lower_outlier_fraction.setSingleStep(0.01)
        self.lower_outlier_fraction.setValue(0.05)
        self.rb_lower_layout.addWidget(self.rb_lower_label)
        self.rb_lower_layout.addWidget(self.lower_outlier_fraction)
        layout.addLayout(self.rb_lower_layout)

        # -- RobustBackground: upper outlier fraction --
        self.rb_upper_layout = QHBoxLayout()
        self.rb_upper_label = QLabel()
        self.upper_outlier_fraction = QDoubleSpinBox()
        self.upper_outlier_fraction.setDecimals(2)
        self.upper_outlier_fraction.setRange(0.0, 0.5)
        self.upper_outlier_fraction.setSingleStep(0.01)
        self.upper_outlier_fraction.setValue(0.05)
        self.rb_upper_layout.addWidget(self.rb_upper_label)
        self.rb_upper_layout.addWidget(self.upper_outlier_fraction)
        layout.addLayout(self.rb_upper_layout)

        # -- RobustBackground: averaging method --
        self.rb_avg_layout = QHBoxLayout()
        self.rb_avg_label = QLabel()
        self.averaging_method = QComboBox()
        self.averaging_method.addItems(["Mean", "Median", "Mode"])
        self.rb_avg_layout.addWidget(self.rb_avg_label)
        self.rb_avg_layout.addWidget(self.averaging_method)
        layout.addLayout(self.rb_avg_layout)

        # -- RobustBackground: variance method --
        self.rb_var_layout = QHBoxLayout()
        self.rb_var_label = QLabel()
        self.variance_method = QComboBox()
        self.variance_method.addItems(["Standard deviation", "MAD"])
        self.rb_var_layout.addWidget(self.rb_var_label)
        self.rb_var_layout.addWidget(self.variance_method)
        layout.addLayout(self.rb_var_layout)

        # -- RobustBackground: number of deviations --
        self.rb_dev_layout = QHBoxLayout()
        self.rb_dev_label = QLabel()
        self.number_of_deviations = QDoubleSpinBox()
        self.number_of_deviations.setDecimals(1)
        self.number_of_deviations.setRange(0.0, 10.0)
        self.number_of_deviations.setSingleStep(0.5)
        self.number_of_deviations.setValue(2.0)
        self.rb_dev_layout.addWidget(self.rb_dev_label)
        self.rb_dev_layout.addWidget(self.number_of_deviations)
        layout.addLayout(self.rb_dev_layout)

        # -- Adaptive: window size --
        self.adaptive_layout = QHBoxLayout()
        self.adaptive_label = QLabel()
        self.adaptive_window_size = QSpinBox()
        self.adaptive_window_size.setRange(10, 500)
        self.adaptive_window_size.setValue(50)
        self.adaptive_layout.addWidget(self.adaptive_label)
        self.adaptive_layout.addWidget(self.adaptive_window_size)
        layout.addLayout(self.adaptive_layout)

        # -- Declumping settings --

        self.fill_holes_thresholding = QCheckBox()
        self.fill_holes_thresholding.setChecked(True)
        layout.addWidget(self.fill_holes_thresholding)

        self.fill_holes_declumping = QCheckBox()
        self.fill_holes_declumping.setChecked(True)
        layout.addWidget(self.fill_holes_declumping)

        self.automatic_smoothing = QCheckBox()
        self.automatic_smoothing.setChecked(True)
        layout.addWidget(self.automatic_smoothing)

        # -- Smoothing filter size (visible when auto smoothing unchecked) --
        self.sfs_layout = QHBoxLayout()
        self.sfs_label = QLabel()
        self.smoothing_filter_size = QDoubleSpinBox()
        self.smoothing_filter_size.setDecimals(1)
        self.smoothing_filter_size.setRange(0.0, 100.0)
        self.smoothing_filter_size.setSingleStep(0.5)
        self.smoothing_filter_size.setValue(10.0)
        self.sfs_layout.addWidget(self.sfs_label)
        self.sfs_layout.addWidget(self.smoothing_filter_size)
        layout.addLayout(self.sfs_layout)

        self.automatic_maxima_suppression = QCheckBox()
        self.automatic_maxima_suppression.setChecked(True)
        layout.addWidget(self.automatic_maxima_suppression)

        # -- Maxima suppression size (visible when auto maxima unchecked) --
        self.mss_layout = QHBoxLayout()
        self.mss_label = QLabel()
        self.maxima_suppression_size = QDoubleSpinBox()
        self.maxima_suppression_size.setDecimals(1)
        self.maxima_suppression_size.setRange(0.0, 100.0)
        self.maxima_suppression_size.setSingleStep(0.5)
        self.maxima_suppression_size.setValue(7.0)
        self.mss_layout.addWidget(self.mss_label)
        self.mss_layout.addWidget(self.maxima_suppression_size)
        layout.addLayout(self.mss_layout)

        self.low_res_maxima = QCheckBox()
        self.low_res_maxima.setChecked(True)
        layout.addWidget(self.low_res_maxima)

        self.exclude_border_objects = QCheckBox()
        self.exclude_border_objects.setChecked(True)
        layout.addWidget(self.exclude_border_objects)

        self.crop_experiment_button = QPushButton()
        layout.addWidget(self.crop_experiment_button)

        # Preset buttons
        self.save_preset_button = QPushButton()
        layout.addWidget(self.save_preset_button)
        self.load_preset_button_adv = QPushButton()
        layout.addWidget(self.load_preset_button_adv)

        # Row groups for conditional visibility
        self._manual_rows = [self.manual_layout]
        self._otsu_rows = [self.otsu_class_layout]
        self._otsu_middle_rows = [self.otsu_middle_layout]
        self._mog_rows = [self.mog_layout]
        self._robust_bg_rows = [
            self.rb_lower_layout,
            self.rb_upper_layout,
            self.rb_avg_layout,
            self.rb_var_layout,
            self.rb_dev_layout,
        ]
        self._adaptive_rows = [self.adaptive_layout]
        self._scope_rows = [self.threshold_scope_layout]
        self._correction_rows = [
            self.tcf_layout,
            self.tss_layout,
            self.tlb_layout,
            self.tub_layout,
        ]

    def _setup_connections(self):
        self.threshold_method.currentTextChanged.connect(
            self._update_threshold_method_controls
        )
        self.threshold_scope.currentTextChanged.connect(self._update_scope_controls)
        self.otsu_class.currentTextChanged.connect(self._update_otsu_controls)
        self.automatic_smoothing.toggled.connect(
            lambda checked: self._set_layout_visible(self.sfs_layout, not checked)
        )
        self.automatic_maxima_suppression.toggled.connect(
            lambda checked: self._set_layout_visible(self.mss_layout, not checked)
        )

    def _update_threshold_method_controls(self):
        method = self.threshold_method.currentText()

        # Hide all method-specific rows
        for rows in (
            self._manual_rows,
            self._otsu_rows,
            self._otsu_middle_rows,
            self._mog_rows,
            self._robust_bg_rows,
        ):
            for row in rows:
                self._set_layout_visible(row, False)

        # Show relevant rows
        if method == "Manual":
            for row in self._manual_rows:
                self._set_layout_visible(row, True)
            # Manual: hide scope, correction, bounds
            for row in self._scope_rows + self._correction_rows:
                self._set_layout_visible(row, False)
        else:
            # Show scope and correction for non-Manual
            for row in self._scope_rows + self._correction_rows:
                self._set_layout_visible(row, True)

            if method == "Otsu":
                for row in self._otsu_rows:
                    self._set_layout_visible(row, True)
                self._update_otsu_controls()
            elif method == "MoG":
                for row in self._mog_rows:
                    self._set_layout_visible(row, True)
            elif method == "RobustBackground":
                for row in self._robust_bg_rows:
                    self._set_layout_visible(row, True)

        self._update_scope_controls()

    def _update_otsu_controls(self):
        three_class = self.otsu_class.currentText() == "Three classes"
        for row in self._otsu_middle_rows:
            self._set_layout_visible(row, three_class)

    def _update_scope_controls(self):
        method = self.threshold_method.currentText()
        is_adaptive = self.threshold_scope.currentText() == "Adaptive"
        show_adaptive = is_adaptive and method != "Manual"
        for row in self._adaptive_rows:
            self._set_layout_visible(row, show_adaptive)

    def _set_layout_visible(self, layout, visible):
        """Show/hide all widgets in a layout."""
        if isinstance(layout, QWidget):
            layout.setVisible(visible)
            return
        for idx in range(layout.count()):
            item = layout.itemAt(idx)
            w = item.widget()
            if w is not None:
                w.setVisible(visible)
            child = item.layout()
            if child is not None:
                self._set_layout_visible(child, visible)

    def _retranslate_ui(self):
        _t = QCoreApplication.translate
        self.threshold_method_label.setText(_t("MainWindow", "Threshold Method"))
        self.threshold_scope_label.setText(_t("MainWindow", "Threshold Scope"))
        self.tcf_label.setText(_t("MainWindow", "Threshold Correction Factor"))
        self.tss_label.setText(_t("MainWindow", "Threshold Smoothing Scale"))
        self.tlb_label.setText(_t("MainWindow", "Lower Bound on Threshold"))
        self.tub_label.setText(_t("MainWindow", "Upper Bound on Threshold"))
        self.manual_label.setText(_t("MainWindow", "Manual Threshold"))
        self.otsu_class_label.setText(_t("MainWindow", "Otsu Classes"))
        self.otsu_middle_label.setText(_t("MainWindow", "Assign Middle To"))
        self.mog_label.setText(_t("MainWindow", "Object Fraction"))
        self.rb_lower_label.setText(_t("MainWindow", "Lower Outlier Fraction"))
        self.rb_upper_label.setText(_t("MainWindow", "Upper Outlier Fraction"))
        self.rb_avg_label.setText(_t("MainWindow", "Averaging Method"))
        self.rb_var_label.setText(_t("MainWindow", "Variance Method"))
        self.rb_dev_label.setText(_t("MainWindow", "Number of Deviations"))
        self.adaptive_label.setText(_t("MainWindow", "Adaptive Window Size"))
        self.fill_holes_thresholding.setText(
            _t("MainWindow", "Fill Holes After Thresholding")
        )
        self.fill_holes_declumping.setText(
            _t("MainWindow", "Fill Holes After Declumping")
        )
        self.automatic_smoothing.setText(
            _t("MainWindow", "Auto Smoothing for Declumping")
        )
        self.sfs_label.setText(_t("MainWindow", "Smoothing Filter Size"))
        self.automatic_maxima_suppression.setText(
            _t("MainWindow", "Auto Maxima Suppression")
        )
        self.mss_label.setText(_t("MainWindow", "Maxima Suppression Size"))
        self.low_res_maxima.setText(_t("MainWindow", "Low-Res Image for Maxima"))
        self.exclude_border_objects.setText(_t("MainWindow", "Exclude Border Objects"))
        self.crop_experiment_button.setText(_t("MainWindow", "Sample && Test"))
        self.save_preset_button.setText(_t("MainWindow", "Save Preset"))
        self.load_preset_button_adv.setText(_t("MainWindow", "Load Preset"))
