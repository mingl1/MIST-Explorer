import logging
import os
import sys
import traceback

os.environ["QT_API"] = "PyQt6"

import matplotlib
import matplotlib.patheffects as PathEffects
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyqtgraph as pg
import scanpy as sc
import tifffile
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from PyQt6.QtCore import Qt, QTimer, pyqtSignal
from PyQt6.QtGui import QColor, QIcon, QPixmap
from PyQt6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSlider,
    QSpinBox,
    QSplitter,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)
from ui.analysis.graphing.UMAPDataModel import DataModel
from ui.analysis.graphing.UMAPLocale import Locale
from ui.analysis.graphing.UMAPWorkers import AnalysisWorker, ClusteringWorker

matplotlib.use("QtAgg")

# --- 1. Debugging Setup ---

log_format = "%(asctime)s - %(levelname)s - %(message)s"
logging.basicConfig(
    level=logging.DEBUG,
    format=log_format,
    handlers=[logging.FileHandler("app_debug.log"), logging.StreamHandler(sys.stdout)],
)

logger = logging.getLogger("UMAP_App")

logging.getLogger("matplotlib").setLevel(logging.WARNING)


def excepthook(exc_type, exc_value, exc_tb):
    logger.critical("Uncaught exception:", exc_info=(exc_type, exc_value, exc_tb))
    app = QApplication.instance()
    if app:
        tb_str = "".join(traceback.format_exception(exc_type, exc_value, exc_tb))
        msg = QMessageBox()
        msg.setIcon(QMessageBox.Icon.Critical)
        msg.setText("An unexpected error occurred.")
        msg.setInformativeText(str(exc_value))
        msg.setDetailedText(tb_str)
        msg.exec()


# ==========================================
# 0. AESTHETICS ENGINE
# ==========================================


class ThemeManager:
    """
    Manages the 'Abyssal Void' (Dark) and 'Sterile Lab' (Light) themes.
    """

    THEMES = {
        "DARK": {
            "bg_dark": "#0b0c10",  # Deep Void
            "bg_panel": "#1f2833",  # Dark Slate
            "text_main": "#c5c6c7",  # Soft Grey
            "accent": "#66fcf1",  # Electric Cyan
            "accent_dim": "#45a29e",  # Dull Cyan
            "alert": "#ffaa00",  # Signal Amber
            "grid": "#2d3436",  # Subtle Grid
            "plot_cmap": "plasma",  # Matplotlib colormap
            "plot_palette": "tab20",  # Categorical palette
            "pg_bg": "#0b0c10",  # PyQtGraph Background
        },
        "LIGHT": {
            "bg_dark": "#ffffff",  # Clinical White
            "bg_panel": "#f0f2f5",  # Lab Coat Grey
            "text_main": "#2d3436",  # Ink Black
            "accent": "#0984e3",  # Blueprint Blue (Sharp, Technical)
            "accent_dim": "#74b9ff",  # Soft Blue
            "alert": "#d63031",  # Safety Red
            "grid": "#dfe6e9",  # Faint Grey Lines
            "plot_cmap": "viridis",  # Viridis reads better on white
            "plot_palette": "tab20",  # Categorical palette
            "pg_bg": "#ffffff",  # PyQtGraph Background
        },
    }

    current_mode = "DARK"

    @classmethod
    def get_current(cls):
        return cls.THEMES[cls.current_mode]

    @classmethod
    def toggle(cls):
        cls.current_mode = "LIGHT" if cls.current_mode == "DARK" else "DARK"
        return cls.get_current()

    @classmethod
    def get_stylesheet(cls):
        c = cls.get_current()
        
        # Resolve icon path
        base_dir = os.path.dirname(os.path.abspath(__file__))
        icons_dir = os.path.join(base_dir, "icons").replace("\\", "/")
        
        # Select icon based on theme
        tick_icon = "checkbox_tick_dark.svg" if cls.current_mode == "DARK" else "checkbox_tick_light.svg"
        tick_path = f"{icons_dir}/{tick_icon}"

        return f"""
        /* GLOBAL RESET */
        QWidget {{
            background-color: {c["bg_dark"]};
            color: {c["text_main"]};
            font-family: "Consolas", "Menlo", "Monaco", monospace;
            font-size: 10pt;
            selection-background-color: {c["accent"]};
            selection-color: {c["bg_dark"]};
        }}

        /* TABS */
        QTabWidget::pane {{
            border: 2px solid {c["grid"]};
            background: {c["bg_dark"]};
            margin-top: -1px; 
        }}

        QTabBar::tab {{
            background: {c["bg_panel"]};
            color: {c["text_main"]};
            padding: 8px 20px;
            border: 1px solid {c["grid"]};
            border-bottom: none;
            margin-right: 2px;
            font-weight: bold;
            text-transform: uppercase;
            letter-spacing: 1px;
        }}

        QTabBar::tab:selected {{
            background: {c["bg_dark"]};
            color: {c["accent"]};
            border: 2px solid {c["accent"]};
            border-bottom: 2px solid {c["bg_dark"]};
        }}

        /* INPUTS */
        QLineEdit, QComboBox, QSpinBox {{
            background-color: {c["bg_panel"]};
            border: 1px solid {c["grid"]};
            padding: 4px;
            color: {c["accent"]}; /* Distinctive colored text input */
            font-weight: bold;
        }}

        QLineEdit:focus, QComboBox:focus {{
            border: 2px solid {c["accent"]};
            background-color: {c["bg_dark"]};
        }}

        /* LISTS */
        QListWidget {{
            background-color: {c["bg_panel"]};
            border: 2px solid {c["grid"]};
            outline: none;
        }}
        QListWidget::item {{
            padding: 5px;
            border-bottom: 1px solid {c["grid"]};
            color: {c["text_main"]};
        }}
        QListWidget::item:selected {{
            background-color: {c["accent"]};
            color: {c["bg_dark"]};
        }}

        /* BUTTONS */
        QPushButton {{
            background-color: {c["bg_panel"]};
            border: 1px solid {c["accent_dim"]};
            color: {c["accent"]};
            padding: 8px;
            font-weight: bold;
            text-transform: uppercase;
            letter-spacing: 1px;
        }}
        QPushButton:hover {{
            background-color: {c["accent"]};
            color: {c["bg_dark"] if cls.current_mode == "DARK" else "#ffffff"};
            border: 1px solid {c["accent"]};
        }}
        QPushButton:pressed {{
            background-color: {c["alert"]};
            color: #ffffff;
            border: 1px solid {c["alert"]};
        }}

        /* GROUPBOX */
        QGroupBox {{
            border: 1px solid {c["grid"]};
            margin-top: 1.5em;
            padding-top: 10px;
            font-weight: bold;
            color: {c["accent_dim"]};
        }}
        QGroupBox::title {{
            subcontrol-origin: margin;
            subcontrol-position: top left;
            padding: 0 5px;
            left: 10px;
            background-color: {c["bg_dark"]};
            color: {c["text_main"]};
        }}

        /* SLIDERS */
        QSlider::groove:horizontal {{
            border: 1px solid {c["grid"]};
            height: 6px;
            background: {c["bg_panel"]};
            margin: 2px 0;
        }}
        QSlider::handle:horizontal {{
            background: {c["accent"]};
            border: 1px solid {c["accent"]};
            width: 14px;
            height: 14px;
            margin: -6px 0;
            border-radius: 0px; /* Sharp edges */
        }}

        /* SCROLLBARS */
        QScrollBar:vertical {{
            border: none;
            background: {c["bg_panel"]};
            width: 12px;
        }}
        QScrollBar::handle:vertical {{
            background: {c["accent_dim"]};
            min-height: 20px;
        }}

        /* CHECKBOX INDICATORS */
        QCheckBox::indicator, QListWidget::indicator {{
            width: 14px;
            height: 14px;
            background-color: {c["bg_panel"]};
            border: 1px solid {c["text_main"]};
            border-radius: 3px;
        }}
        QCheckBox::indicator:checked, QListWidget::indicator:checked {{
            background-color: {c["accent"]};
            border: 1px solid {c["accent"]};
            image: url({tick_path});
        }}
        QCheckBox::indicator:hover, QListWidget::indicator:hover {{
            border: 1px solid {c["accent"]};
        }}
        
        /* SPECIFIC OVERRIDES */
        QLabel#h1 {{
            font-size: 14pt;
            color: {c["accent"]};
            font-weight: 900;
        }}
        
        /* MAIN ACTION BUTTON OVERRIDE */
        QPushButton#MainRunBtn {{
            background-color: {c["bg_panel"]};
            border: 3px solid {c["accent"]};
            color: {c["accent"]};
            font-size: 11pt;
            letter-spacing: 2px;
        }}
        QPushButton#MainRunBtn:hover {{
            background-color: {c["accent"]};
            color: {c["bg_dark"] if cls.current_mode == "DARK" else "#ffffff"};
        }}
        """


def apply_matplotlib_theme(figure, ax=None):
    """Force Matplotlib to respect the current theme."""
    c = ThemeManager.get_current()

    figure.patch.set_facecolor(c["bg_dark"])

    if ax:
        ax.set_facecolor(c["bg_dark"])
        ax.tick_params(colors=c["text_main"], which="both")

        # Spines
        for spine in ax.spines.values():
            spine.set_edgecolor(c["accent_dim"])
            spine.set_linewidth(1.5)

        # Labels
        ax.xaxis.label.set_color(c["accent"])
        ax.yaxis.label.set_color(c["accent"])
        ax.title.set_color(c["accent"])

        # Legend
        legend = ax.get_legend()
        if legend:
            frame = legend.get_frame()
            frame.set_facecolor(c["bg_panel"])
            frame.set_edgecolor(c["grid"])
            for text in legend.get_texts():
                text.set_color(c["text_main"])


def to_rgb(hex_color):
    h = hex_color.lstrip("#")
    return tuple(int(h[i : i + 2], 16) / 255.0 for i in (0, 2, 4))


sys.excepthook = excepthook


# ==========================================
# 1. UI COMPONENTS
# ==========================================


class UMAPControls(QWidget):
    apply_data_signal = pyqtSignal(str, list)
    run_signal = pyqtSignal(dict)
    resolution_change_signal = pyqtSignal(float)

    def __init__(self, model, max_pca_components=50, all_features=[]):
        super().__init__()
        self.model = model
        self.res_debounce_timer = QTimer()
        self.res_debounce_timer.setSingleShot(True)
        self.res_debounce_timer.setInterval(400)
        self.res_debounce_timer.timeout.connect(self.emit_resolution_change)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(15)

        # === 0. Data Setup ===
        data_group = QGroupBox(Locale.get("GRP_DATA"))
        data_layout = QVBoxLayout()
        data_layout.addWidget(QLabel(Locale.get("LBL_NORM")))
        self.combo_norm = QComboBox()
        self.combo_norm.addItems(
            self.model.get_normalization_methods()
            if hasattr(self.model, "get_normalization_methods")
            else ["Log1p", "Z-Score"]
        )
        # on change emit
        self.combo_norm.currentIndexChanged.connect(self.emit_apply_data)
        data_layout.addWidget(self.combo_norm)

        input_row = QHBoxLayout()
        self.search_bar = QLineEdit()
        self.search_bar.setPlaceholderText(Locale.get("PLACEHOLDER_SEARCH"))
        self.search_bar.textChanged.connect(self.filter_feature_list)
        input_row.addWidget(self.search_bar)

        self.btn_select_input = QPushButton(Locale.get("BTN_SCAN"))
        self.btn_select_input.setFixedWidth(60)
        self.btn_select_input.clicked.connect(self.select_features_from_input)
        input_row.addWidget(self.btn_select_input)
        data_layout.addLayout(input_row)

        self.feature_list = QListWidget()
        self.feature_list.setMinimumHeight(200)
        self.populate_features(all_features)
        data_layout.addWidget(self.feature_list)
        self.feature_list.itemChanged.connect(self.emit_apply_data)

        btn_row = QHBoxLayout()
        btn_all = QPushButton(Locale.get("BTN_ALL"))
        btn_all.clicked.connect(self.select_all)
        btn_none = QPushButton(Locale.get("BTN_CLR"))
        btn_none.clicked.connect(self.select_none)
        btn_row.addWidget(btn_all)
        btn_row.addWidget(btn_none)
        data_layout.addLayout(btn_row)

        data_group.setLayout(data_layout)
        layout.addWidget(data_group)

        # === 1. PCA ===
        pca_group = QGroupBox(Locale.get("GRP_PCA"))
        pca_layout = QVBoxLayout()
        self.lbl_pca = QLabel(Locale.get("LBL_COMPONENTS", 10))
        self.slider_pca = QSlider(Qt.Orientation.Horizontal)
        self.slider_pca.setRange(2, max_pca_components)
        self.slider_pca.setValue(10)
        self.slider_pca.valueChanged.connect(
            lambda v: self.lbl_pca.setText(Locale.get("LBL_COMPONENTS", v))
        )
        pca_layout.addWidget(self.lbl_pca)
        pca_layout.addWidget(self.slider_pca)
        pca_group.setLayout(pca_layout)
        layout.addWidget(pca_group)

        # === 2. UMAP ===
        umap_group = QGroupBox(Locale.get("GRP_UMAP"))
        umap_layout = QVBoxLayout()
        self.lbl_neighbors = QLabel(Locale.get("LBL_NEIGHBORS", 15))
        self.slider_neighbors = QSlider(Qt.Orientation.Horizontal)
        self.slider_neighbors.setRange(2, 200)
        self.slider_neighbors.setValue(15)
        self.slider_neighbors.valueChanged.connect(
            lambda v: self.lbl_neighbors.setText(Locale.get("LBL_NEIGHBORS", v))
        )
        umap_layout.addWidget(self.lbl_neighbors)
        umap_layout.addWidget(self.slider_neighbors)

        self.lbl_dist = QLabel(Locale.get("LBL_MINDIST", 0.10))
        self.slider_dist = QSlider(Qt.Orientation.Horizontal)
        self.slider_dist.setRange(1, 99)
        self.slider_dist.setValue(10)
        self.slider_dist.valueChanged.connect(
            lambda v: self.lbl_dist.setText(Locale.get("LBL_MINDIST", v / 100.0))
        )
        umap_layout.addWidget(self.lbl_dist)
        umap_layout.addWidget(self.slider_dist)
        umap_group.setLayout(umap_layout)
        layout.addWidget(umap_group)

        # === 3. Clustering ===
        cluster_group = QGroupBox(Locale.get("GRP_CLUSTER"))
        cluster_layout = QVBoxLayout()
        self.lbl_res = QLabel(Locale.get("LBL_RES", 0.30))
        self.slider_res = QSlider(Qt.Orientation.Horizontal)
        self.slider_res.setRange(1, 200)
        self.slider_res.setValue(30)
        self.slider_res.valueChanged.connect(self.on_res_slider_change)
        cluster_layout.addWidget(self.lbl_res)
        cluster_layout.addWidget(self.slider_res)
        cluster_group.setLayout(cluster_layout)

        self.btn_run = QPushButton(Locale.get("BTN_EXECUTE"))
        self.btn_run.setObjectName("MainRunBtn")  # ID for specific styling
        self.btn_run.setMinimumHeight(50)
        self.btn_run.clicked.connect(self.emit_run)
        layout.addWidget(self.btn_run)
        layout.addStretch()
        layout.addWidget(cluster_group)

    def on_res_slider_change(self, value):
        self.lbl_res.setText(Locale.get("LBL_RES", value / 100.0))
        self.res_debounce_timer.start()

    def emit_resolution_change(self):
        val = self.slider_res.value() / 100.0
        self.resolution_change_signal.emit(val)

    def select_features_from_input(self):
        raw_text = self.search_bar.text()
        if not raw_text:
            return
        tokens = raw_text.split(",")
        target_set = set()
        for t in tokens:
            clean = t.strip().strip("'").strip("'")
            if clean:
                target_set.add(clean)
        if not target_set:
            return
        self.feature_list.blockSignals(True)
        self.select_none(emit=False)
        count = 0
        for i in range(self.feature_list.count()):
            item = self.feature_list.item(i)
            if item.text() in target_set:
                item.setCheckState(Qt.CheckState.Checked)
                count += 1
        self.feature_list.blockSignals(False)
        self.search_bar.clear()
        self.emit_apply_data()

    def populate_features(self, features):
        self.feature_list.clear()
        for f in features:
            item = QListWidgetItem(f)
            item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            item.setCheckState(Qt.CheckState.Checked)
            self.feature_list.addItem(item)

    def filter_feature_list(self, text):
        for i in range(self.feature_list.count()):
            item = self.feature_list.item(i)
            item.setHidden(text.lower() not in item.text().lower())

    def select_all(self, emit=True):
        self.feature_list.blockSignals(True)
        for i in range(self.feature_list.count()):
            item = self.feature_list.item(i)
            if not item.isHidden():
                item.setCheckState(Qt.CheckState.Checked)
        self.feature_list.blockSignals(False)
        if emit:
            self.emit_apply_data()

    def select_none(self, emit=True):
        if emit:
            self.feature_list.blockSignals(True)
        for i in range(self.feature_list.count()):
            self.feature_list.item(i).setCheckState(Qt.CheckState.Unchecked)
        if emit:
            self.feature_list.blockSignals(False)
            self.emit_apply_data()

    def get_selected_features(self):
        selected = []
        for i in range(self.feature_list.count()):
            item = self.feature_list.item(i)
            if item.checkState() == Qt.CheckState.Checked:
                selected.append(item.text())
        return selected

    def emit_apply_data(self):
        self.apply_data_signal.emit(
            self.combo_norm.currentText(), self.get_selected_features()
        )

    def emit_run(self):
        self.run_signal.emit(
            {
                "n_components": self.slider_pca.value(),
                "n_neighbors": self.slider_neighbors.value(),
                "min_dist": self.slider_dist.value() / 100.0,
                "random_state": 42,
                "resolution": self.slider_res.value() / 100.0,
            }
        )

    def set_max_pca(self, max_val):
        self.slider_pca.setMaximum(max_val)


class PlotView(QWidget):
    def __init__(self):
        super().__init__()
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self.figure = Figure(figsize=(5, 5), dpi=100)
        apply_matplotlib_theme(self.figure)

        self.canvas = FigureCanvas(self.figure)
        layout.addWidget(self.canvas)

        self.ax = self.figure.add_subplot(111)
        self.ax.text(0.5, 0.5, "NO_SIGNAL", ha="center", va="center")
        self.ax.axis("off")
        self.current_adata = None
        self.current_key = None

    def refresh_theme(self):
        """Called when theme toggles"""
        apply_matplotlib_theme(self.figure, self.ax)
        if self.current_adata and self.current_key:
            self.update_plot(self.current_adata, self.current_key)
        else:
            self.canvas.draw()

    def update_plot(self, adata, color_key):
        self.current_adata = adata
        self.current_key = color_key

        self.figure.clear()
        apply_matplotlib_theme(self.figure)
        self.ax = self.figure.add_subplot(111)
        apply_matplotlib_theme(self.figure, self.ax)

        c = ThemeManager.get_current()
        try:
            sc.pl.umap(
                adata,
                color=color_key,
                ax=self.ax,
                show=False,
                title=f"PROJECTION :: {color_key.upper()}",
                legend_loc="on data",
                palette=c["plot_palette"],
                frameon=False,
            )
            for text_obj in self.ax.texts:
                text_obj.set_fontfamily("monospace")
                text_obj.set_fontsize(10)
                text_obj.set_weight("bold")

                fg_color = (
                    "#ffffff" if ThemeManager.current_mode == "DARK" else "#000000"
                )
                text_obj.set_color(fg_color)

                text_obj.set_path_effects(
                    [PathEffects.withStroke(linewidth=4, foreground=c["bg_dark"])]
                )
            self.figure.tight_layout()
            self.canvas.draw()
        except Exception:
            self.ax.text(
                0.5, 0.5, Locale.get("PLOT_NO_SIGNAL"), ha="center", color=c["alert"]
            )
            self.canvas.draw()


class HeatmapView(QWidget):
    def __init__(self):
        super().__init__()
        self.layout_container = QVBoxLayout(self)
        self.layout_container.setContentsMargins(0, 0, 0, 0)

        # Controls
        self.controls_layout = QHBoxLayout()

        self.chk_scale = QCheckBox("Scale Data (Max=3)")
        self.chk_scale.stateChanged.connect(self.on_ui_change)
        self.controls_layout.addWidget(self.chk_scale)

        self.chk_sort_y = QCheckBox("Sort Y (Features)")
        self.chk_sort_y.stateChanged.connect(self.on_ui_change)
        self.controls_layout.addWidget(self.chk_sort_y)

        self.chk_equiwidth = QCheckBox("Equiwidth")
        self.chk_equiwidth.stateChanged.connect(self.on_ui_change)
        self.controls_layout.addWidget(self.chk_equiwidth)

        self.controls_layout.addStretch()
        self.layout_container.addLayout(self.controls_layout)

        self.figure = Figure(figsize=(8, 6), dpi=100)
        apply_matplotlib_theme(self.figure)
        self.canvas = FigureCanvas(self.figure)
        self.layout_container.addWidget(self.canvas)
        # Store last params to allow re-drawing on theme switch
        self.last_params = None

    def on_ui_change(self):
        if self.last_params:
            self.update_heatmap(*self.last_params)

    def refresh_theme(self):
        if self.last_params:
            self.update_heatmap(*self.last_params)
        else:
            self.figure.clear()
            apply_matplotlib_theme(self.figure)
            self.canvas.draw()

    def update_heatmap(
        self,
        adata,
        cluster_key,
        visible_indices,
        rename_map,
        selected_features,
    ):
        self.last_params = (
            adata,
            cluster_key,
            visible_indices,
            rename_map,
            selected_features,
        )

        # Cleanup old canvas
        try:
            self.layout_container.removeWidget(self.canvas)
            self.canvas.setParent(None)
            self.canvas.deleteLater()
        except:
            pass

        c = ThemeManager.get_current()

        def show_msg(text):
            self.figure = Figure()
            apply_matplotlib_theme(self.figure)
            ax = self.figure.add_subplot(111)
            ax.text(0.5, 0.5, text, ha="center", color=c["alert"])
            ax.axis("off")
            self.canvas = FigureCanvas(self.figure)
            self.layout_container.addWidget(self.canvas)

        if adata is None:
            show_msg(Locale.get("HM_WAITING"))
            return

        try:
            # Data subsetting logic
            all_codes = adata.obs[cluster_key].cat.codes
            mask_cells = np.isin(all_codes, visible_indices)
            if np.sum(mask_cells) == 0:
                show_msg(Locale.get("HM_VOID"))
                return

            heatmap_data = adata[mask_cells, :][:, selected_features].copy()

            # Apply scaling if checkbox is checked
            if self.chk_scale.isChecked():
                try:
                    sc.pp.scale(heatmap_data, max_value=3)
                except Exception as e:
                    logger.warning(f"Scaling failed: {e}")

            # Sort Features (Y)
            final_features = list(selected_features)
            if self.chk_sort_y.isChecked():
                final_features.sort()

            # Handle Group Sorting (X)
            current_codes = heatmap_data.obs[cluster_key].cat.codes
            new_labels = [rename_map.get(c, str(c)) for c in current_codes]

            # Determine unique labels in natural order (based on code)
            present_codes = np.unique(current_codes)
            natural_order = [rename_map.get(c, str(c)) for c in sorted(present_codes)]

            final_categories = natural_order

            heatmap_data.obs["display_label"] = pd.Categorical(
                new_labels, categories=final_categories, ordered=True
            )

            # Apply Equiwidth Resampling
            if self.chk_equiwidth.isChecked():
                try:
                    target_n = 200  # Fixed width per cluster
                    resampled_indices = []

                    # We use the categories we just set
                    for cat in final_categories:
                        # Find integer indices in the current heatmap_data
                        cat_mask = heatmap_data.obs["display_label"] == cat
                        cat_indices = np.where(cat_mask)[0]

                        if len(cat_indices) > 0:
                            # Interpolate indices to stretch/shrink to target_n
                            # linspace gives us N evenly spaced numbers between 0 and len-1
                            sample_pos = np.linspace(0, len(cat_indices) - 1, target_n)
                            # Round to nearest integer index
                            sample_idx = np.round(sample_pos).astype(int)
                            # Map back to original indices
                            resampled_indices.append(cat_indices[sample_idx])

                    if resampled_indices:
                        all_indices = np.concatenate(resampled_indices)
                        heatmap_data = heatmap_data[all_indices].copy()
                except Exception as e:
                    logger.warning(f"Equiwidth resampling failed: {e}")

            # Generate Plot
            plt.close("all")
            # Force Matplotlib RC for this plot
            with plt.rc_context(
                {
                    "axes.facecolor": c["bg_dark"],
                    "figure.facecolor": c["bg_dark"],
                    "text.color": c["text_main"],
                    "axes.labelcolor": c["accent"],
                    "xtick.color": c["text_main"],
                    "ytick.color": c["text_main"],
                }
            ):
                ax_dict = sc.pl.heatmap(
                    heatmap_data,
                    var_names=final_features,
                    groupby="display_label",
                    swap_axes=True,
                    show=False,
                    cmap="magma" if ThemeManager.current_mode == "DARK" else "viridis",
                    dendrogram=False,
                )

            # Extract Figure
            new_fig = None
            if isinstance(ax_dict, dict):
                new_fig = (
                    ax_dict["heatmap"].figure
                    if "heatmap" in ax_dict
                    else list(ax_dict.values())[0].figure
                )
            elif isinstance(ax_dict, list):
                new_fig = ax_dict[0].figure
            else:
                new_fig = ax_dict.figure

            apply_matplotlib_theme(new_fig)
            self.figure = new_fig
            self.canvas = FigureCanvas(self.figure)
            self.layout_container.addWidget(self.canvas)

        except Exception as e:
            logger.exception("HEATMAP_FAIL")
            show_msg(f"ERR: Need to rerunning UMAP.\n{str(e)}")


class RankedGenesView(QWidget):
    def __init__(self):
        super().__init__()
        layout = QVBoxLayout(self)

        # Controls
        controls_layout = QHBoxLayout()

        self.cluster_combo = QComboBox()
        self.cluster_combo.setPlaceholderText("Select Cluster")
        self.cluster_combo.currentIndexChanged.connect(self.on_update)

        self.genes_spin = QSpinBox()
        self.genes_spin.setRange(1, 20)
        self.genes_spin.setValue(5)
        self.genes_spin.valueChanged.connect(self.on_update)

        controls_layout.addWidget(QLabel("Cluster:"))
        controls_layout.addWidget(self.cluster_combo)
        controls_layout.addWidget(QLabel("Top Genes:"))
        controls_layout.addWidget(self.genes_spin)
        controls_layout.addStretch()

        layout.addLayout(controls_layout)

        # Scroll Area
        self.scrollArea = QScrollArea()
        self.scrollArea.setWidgetResizable(True)
        layout.addWidget(self.scrollArea)

        # Container for plots (Grid Layout)
        self.plot_container = QWidget()
        self.plot_layout = QGridLayout(self.plot_container)
        self.plot_layout.setSpacing(10)
        self.scrollArea.setWidget(self.plot_container)

        self.adata = None
        self.key = None

    def refresh_theme(self):
        # Simply regenerating plots picks up the current theme
        self.on_update()

    def set_data(self, adata, key):
        self.adata = adata
        self.key = key

        if self.adata and self.key and self.key in self.adata.obs:
            cats = self.adata.obs[self.key].cat.categories
            self.cluster_combo.blockSignals(True)
            self.cluster_combo.clear()
            for c in cats:
                name = str(c)
                self.cluster_combo.addItem(name, name)
            self.cluster_combo.blockSignals(False)

            # Check validity of existing rank_genes_groups
            recompute = True
            if "rank_genes_groups" in self.adata.uns:
                rgg = self.adata.uns["rank_genes_groups"]
                if rgg.get("params", {}).get("groupby") == self.key:
                    # Check if all current categories are present in the ranked results
                    stored_names = (
                        set(rgg["names"].dtype.names)
                        if hasattr(rgg["names"], "dtype")
                        else set()
                    )
                    current_cats = set(str(c) for c in cats)
                    if current_cats.issubset(stored_names):
                        recompute = False

            if recompute:
                try:
                    sc.tl.rank_genes_groups(
                        self.adata, groupby=self.key, method="t-test"
                    )
                except Exception as e:
                    logger.error(f"Failed to rank genes: {e}")

    def update_cluster_names(self, renames):
        """
        Updates the displayed names in the combo box based on the dictionary
        renames: { int_code : str_new_name }
        """
        self.cluster_combo.blockSignals(True)
        for i in range(self.cluster_combo.count()):
            # Assuming the combo box items were added in order of codes (which they are in set_data)
            if i in renames:
                self.cluster_combo.setItemText(i, renames[i])
        self.cluster_combo.blockSignals(False)

    def on_update(self):
        if self.adata is None or self.key is None:
            return

        # Use the UserData (original name) for data lookup
        group = self.cluster_combo.currentData()
        if not group:
            # Fallback for safety
            group = self.cluster_combo.currentText()
        n_genes = self.genes_spin.value()

        if not group:
            return

        try:
            dc_cluster_genes = sc.get.rank_genes_groups_df(
                self.adata, group=group
            ).head(n_genes)["names"]
        except KeyError:
            # Fallback: try recomputing if group is missing
            logger.warning(f"Group {group} not found in ranked genes. Recomputing...")
            try:
                sc.tl.rank_genes_groups(self.adata, groupby=self.key, method="t-test")
                dc_cluster_genes = sc.get.rank_genes_groups_df(
                    self.adata, group=group
                ).head(n_genes)["names"]
            except Exception as e:
                logger.error(f"Retry failed: {e}")
                return

        # Clear existing plots
        while self.plot_layout.count():
            item = self.plot_layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()

        try:
            colors = [self.key, *dc_cluster_genes]
            ncols = 3
            c = ThemeManager.get_current()

            for i, color_key in enumerate(colors):
                # Create individual Figure and Canvas
                fig = Figure(figsize=(4, 4), dpi=100)
                apply_matplotlib_theme(fig)
                canvas = FigureCanvas(fig)
                canvas.setMinimumSize(350, 350)  # Ensure minimum size per plot

                ax = fig.add_subplot(111)

                is_categorical = color_key == self.key

                sc.pl.umap(
                    self.adata,
                    color=color_key,
                    ax=ax,
                    show=False,
                    legend_loc="on data" if is_categorical else None,
                    frameon=False,
                    title=color_key,
                    cmap=c["plot_cmap"],
                    palette=c["plot_palette"],
                    colorbar_loc=None,
                )

                # Enforce square aspect ratio
                ax.set_aspect("equal", "box")

                # --- COLOR SETTINGS ---
                fg_color = (
                    "#ffffff" if ThemeManager.current_mode == "DARK" else "#000000"
                )
                ax.title.set_color(fg_color)
                ax.title.set_fontfamily("monospace")
                ax.title.set_weight("bold")

                for text_obj in ax.texts:
                    text_obj.set_fontfamily("monospace")
                    text_obj.set_fontsize(8)
                    text_obj.set_weight("bold")
                    text_obj.set_color(fg_color)
                    text_obj.set_path_effects(
                        [PathEffects.withStroke(linewidth=3, foreground=c["bg_dark"])]
                    )

                # Legend styling
                legend = ax.get_legend()
                if legend:
                    for text in legend.get_texts():
                        text.set_color(fg_color)

                # Manually add colorbar for continuous data
                if not is_categorical and ax.collections:
                    mappable = ax.collections[-1]
                    fig.colorbar(mappable, ax=ax, fraction=0.046, pad=0.04)
                else:
                    ax.axis("off")

                fig.tight_layout()

                # Add to Grid Layout
                row = i // ncols
                col = i % ncols
                self.plot_layout.addWidget(canvas, row, col)

        except Exception as e:
            logger.error(f"Error updating ranked genes view: {e}")
            # Show error in a label if plotting fails
            lbl = QLabel(f"Error: {str(e)}")
            lbl.setStyleSheet(f"color: {c['alert']}")
            self.plot_layout.addWidget(lbl, 0, 0)


class SegmentationView(QWidget):
    cluster_state_changed = pyqtSignal(list, dict)

    def __init__(self, segmentation_data=None):
        super().__init__()
        main_layout = QHBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        self.seg_data = segmentation_data
        if self.seg_data is not None:
            self.seg_data = self.seg_data.astype(int)

        self.splitter = QSplitter(Qt.Orientation.Horizontal)

        # Left: Image
        left_w = QWidget()
        l_layout = QVBoxLayout(left_w)
        l_layout.setContentsMargins(0, 0, 0, 0)

        self.info_label = QLabel(Locale.get("SEG_STATUS_OFF"))
        self.info_label.setStyleSheet("font-weight: bold;")
        l_layout.addWidget(self.info_label)

        self.pg_view = pg.ImageView()
        self.pg_view.ui.roiBtn.hide()
        self.pg_view.ui.menuBtn.hide()
        self.pg_view.ui.histogram.hide()
        pg.setConfigOption("background", ThemeManager.get_current()["pg_bg"])
        l_layout.addWidget(self.pg_view)
        self.splitter.addWidget(left_w)

        # Right: Controls
        right_w = QWidget()
        r_layout = QVBoxLayout(right_w)

        r_layout.addWidget(QLabel(Locale.get("SEG_CLUSTER_FILTER")))

        btn_row = QHBoxLayout()
        self.btn_all = QPushButton(Locale.get("BTN_ALL"))
        self.btn_all.clicked.connect(self.select_all)
        self.btn_none = QPushButton(Locale.get("BTN_CLR"))
        self.btn_none.clicked.connect(self.select_none)
        btn_row.addWidget(self.btn_all)
        btn_row.addWidget(self.btn_none)
        r_layout.addLayout(btn_row)

        self.cluster_list = QListWidget()
        self.cluster_list.itemChanged.connect(self.on_item_changed)
        r_layout.addWidget(self.cluster_list)
        self.splitter.addWidget(right_w)
        self.splitter.setStretchFactor(0, 4)
        main_layout.addWidget(self.splitter)

        if self.seg_data is not None:
            self.pg_view.setImage(self.seg_data)

        self.npy_id_to_cat_idx = None
        self.npy_cat_idx_to_rgb = None
        self._last_visible_set = set()

    def refresh_theme(self):
        c = ThemeManager.get_current()
        pg.setConfigOption("background", c["pg_bg"])
        self.info_label.setStyleSheet(f"color: {c['alert']}; font-weight: bold;")

    def get_state(self):
        visible = []
        renames = {}
        for i in range(self.cluster_list.count()):
            item = self.cluster_list.item(i)
            original_idx = item.data(Qt.ItemDataRole.UserRole)
            renames[original_idx] = item.text()
            if item.checkState() == Qt.CheckState.Checked:
                visible.append(original_idx)
        return visible, renames

    def render_clusters(self, adata, cluster_key):
        if self.seg_data is None:
            return

        self.refresh_theme()
        if f"{cluster_key}_colors" not in adata.uns:
            sc.pl.umap(adata, color=cluster_key, show=False)
        cluster_colors = adata.uns[f"{cluster_key}_colors"]
        categories = adata.obs[cluster_key].cat.categories
        self.npy_cat_idx_to_rgb = np.array([to_rgb(c) for c in cluster_colors])
        valid_ids = adata.obs["cell_id"].values.astype(int)
        max_id = max(self.seg_data.max(), valid_ids.max())
        self.npy_id_to_cat_idx = np.full(max_id + 1, -1, dtype=np.int32)
        self.npy_id_to_cat_idx[valid_ids] = adata.obs[cluster_key].cat.codes.values

        self.cluster_list.blockSignals(True)
        self.cluster_list.clear()
        for cat_name, cat_code in zip(categories, range(len(categories))):
            item = QListWidgetItem(str(cat_name))
            item.setFlags(
                item.flags()
                | Qt.ItemFlag.ItemIsUserCheckable
                | Qt.ItemFlag.ItemIsEditable
            )
            item.setCheckState(Qt.CheckState.Checked)
            item.setData(Qt.ItemDataRole.UserRole, int(cat_code))
            self.cluster_list.addItem(item)
        self.regenerate_image(force=True)
        self.cluster_list.blockSignals(False)
        self.cluster_state_changed.emit(*self.get_state())

    def on_item_changed(self, item):
        self.regenerate_image()
        self.cluster_state_changed.emit(*self.get_state())

    def select_all(self):
        self.cluster_list.blockSignals(True)
        for i in range(self.cluster_list.count()):
            self.cluster_list.item(i).setCheckState(Qt.CheckState.Checked)
        self.cluster_list.blockSignals(False)
        self.regenerate_image()
        self.cluster_state_changed.emit(*self.get_state())

    def select_none(self):
        self.cluster_list.blockSignals(True)
        for i in range(self.cluster_list.count()):
            self.cluster_list.item(i).setCheckState(Qt.CheckState.Unchecked)
        self.cluster_list.blockSignals(False)
        self.regenerate_image()
        self.cluster_state_changed.emit(*self.get_state())

    def regenerate_image(self, force=False):
        if self.seg_data is None or self.npy_id_to_cat_idx is None:
            return
        active_palette = np.zeros_like(self.npy_cat_idx_to_rgb)
        name_to_target = {}
        count = 0
        self.cluster_list.blockSignals(True)
        for i in range(self.cluster_list.count()):
            item = self.cluster_list.item(i)
            orig = item.data(Qt.ItemDataRole.UserRole)
            txt = item.text()
            target = name_to_target.get(txt, orig)
            name_to_target[txt] = target

            if 0 <= target < len(self.npy_cat_idx_to_rgb):
                rgb = self.npy_cat_idx_to_rgb[target]
                pix = QPixmap(12, 12)
                pix.fill(QColor.fromRgbF(*rgb))
                item.setIcon(QIcon(pix))

            if item.checkState() == Qt.CheckState.Checked:
                count += 1
                if 0 <= orig < len(active_palette):
                    active_palette[orig] = self.npy_cat_idx_to_rgb[target]
        self.cluster_list.blockSignals(False)
        safe_seg = self.seg_data.copy()
        mask_overflow = safe_seg >= len(self.npy_id_to_cat_idx)
        safe_seg[mask_overflow] = 0
        pixel_cat = self.npy_id_to_cat_idx[safe_seg]
        mask_valid = pixel_cat >= 0
        final = np.zeros(self.seg_data.shape + (3,), dtype=np.float32)
        final[mask_valid] = active_palette[pixel_cat[mask_valid]]
        self.pg_view.setImage(
            final.transpose(1, 0, 2),
            autoRange=False,
            autoLevels=False,
            levels=(0, 1),
            axes={"x": 0, "y": 1, "c": 2},
        )
        self.info_label.setText(Locale.get("SEG_ACTIVE_COUNT", count))


class UMAPVisualizer(QMainWindow):
    def __init__(self, df, segmentation=None):
        super().__init__()
        self.setWindowTitle(Locale.get("APP_TITLE"))
        self.resize(1300, 900)
        self.setStyleSheet(ThemeManager.get_stylesheet())
        self.is_running_full_pipeline = False

        try:
            self.model = DataModel(df, normalization="RC")
            self.segmentation = segmentation
        except Exception as e:
            QMessageBox.critical(self, "SYSTEM_FAILURE", f"INIT_ERR: {e}")
            return

        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)

        # === TOOLBAR & THEME TOGGLE ===
        top_bar = QHBoxLayout()
        header = QLabel(Locale.get("HEADER"))
        header.setObjectName("h1")
        top_bar.addWidget(header)
        top_bar.addStretch()

        self.btn_theme = QPushButton(Locale.get("THEME_BTN"))
        self.btn_theme.setFixedWidth(200)
        self.btn_theme.clicked.connect(self.toggle_theme)
        top_bar.addWidget(self.btn_theme)

        main_layout.addLayout(top_bar)

        # === TABS ===
        self.tabs = QTabWidget()
        main_layout.addWidget(self.tabs)

        self.umap_tab = QWidget()
        self.setup_umap_tab()
        self.tabs.addTab(self.umap_tab, Locale.get("TAB_UMAP"))

        self.seg_view = SegmentationView(segmentation)
        self.seg_view.cluster_state_changed.connect(self.trigger_heatmap_update)
        self.tabs.addTab(self.seg_view, Locale.get("TAB_SEG"))

        self.heatmap_view = HeatmapView()
        self.tabs.addTab(self.heatmap_view, Locale.get("TAB_HEATMAP"))

        self.ranked_genes_view = RankedGenesView()
        self.tabs.addTab(self.ranked_genes_view, "RANKED GENES")

        # === STATUS BAR ===
        self.status_bar = self.statusBar()
        self.progress = QProgressBar()
        self.progress.setVisible(False)
        self.progress.setMaximumWidth(200)
        self.status_bar.addPermanentWidget(self.progress)
        self.status_label = QLabel(Locale.get("STATUS_READY"))
        self.status_bar.addWidget(self.status_label)

        self.update_status_style()

    def setup_umap_tab(self):
        layout = QHBoxLayout(self.umap_tab)
        layout.setContentsMargins(10, 10, 10, 10)
        splitter = QSplitter(Qt.Orientation.Horizontal)

        all_feats = self.model.get_all_features()
        max_pcs = self.model.get_max_pcs()
        self.controls = UMAPControls(
            max_pca_components=max_pcs, all_features=all_feats, model=self.model
        )

        self.controls.run_signal.connect(self.start_full_analysis)
        self.controls.apply_data_signal.connect(self.apply_data_changes)
        self.controls.resolution_change_signal.connect(self.start_clustering_only)

        self.plot_view = PlotView()
        splitter.addWidget(self.controls)
        splitter.addWidget(self.plot_view)
        splitter.setStretchFactor(1, 4)
        layout.addWidget(splitter)

    def trigger_heatmap_update(self, visible=None, renames=None):
        if self.model.adata is None or "leiden" not in self.model.adata.obs:
            return
        if visible is None or renames is None:
            visible, renames = self.seg_view.get_state()
        
        self.ranked_genes_view.update_cluster_names(renames)
        
        selected_features = self.controls.get_selected_features()
        self.heatmap_view.update_heatmap(
            self.model.adata, "leiden", visible, renames, selected_features
        )

    def start_full_analysis(self, params):
        self.set_busy_state(True)
        self.is_running_full_pipeline = True

        self.analysis_worker = AnalysisWorker(
            model=self.model,
            n_neighbors=params["n_neighbors"],
            min_dist=params["min_dist"],
            n_components=params["n_components"],
            random_state=params["random_state"],
            resolution=params["resolution"],
        )
        self.analysis_worker.finished.connect(self.on_analysis_finished)
        self.analysis_worker.error.connect(self.on_analysis_error)
        self.analysis_worker.start()

    def start_clustering_only(self, resolution):
        if self.is_running_full_pipeline:
            return
        if self.model.adata is None or "neighbors" not in self.model.adata.uns:
            self.status_label.setText("ERR: PIPELINE_REQUIRED")
            return

        self.status_label.setText(f"REPARTITIONING [RES: {resolution}]")
        self.progress.setVisible(True)
        self.progress.setRange(0, 0)

        self.clustering_worker = ClusteringWorker(self.model, resolution)
        self.clustering_worker.finished.connect(self.on_analysis_finished)
        self.clustering_worker.error.connect(self.on_analysis_error)
        self.clustering_worker.start()

    def on_analysis_finished(self, adata, key):
        self.set_busy_state(False)
        self.is_running_full_pipeline = False
        self.plot_view.update_plot(adata, key)
        n_clusters = len(adata.obs[key].unique())
        self.status_label.setText(
            f"COMPLETED: {n_clusters} CLUSTERS IDENTIFIED ({key.upper()})"
        )
        if self.segmentation is not None:
            self.seg_view.render_clusters(adata, key)
        self.trigger_heatmap_update()
        self.ranked_genes_view.set_data(adata, key)

    def on_analysis_error(self, error_msg):
        self.set_busy_state(False)
        self.is_running_full_pipeline = False
        self.status_label.setText("CRITICAL_FAILURE")
        QMessageBox.critical(self, "PROCESS_TERMINATED", error_msg)

    def apply_data_changes(self, normalization, features):
        if len(features) < 1:
            QMessageBox.warning(self, "INPUT_ERR", "MIN_FEATURES_REQUIRED: 1")
            return

        self.set_busy_state(True)
        self.status_label.setText("REPROCESSING_STREAM...")
        QApplication.processEvents()

        try:
            self.model.reprocess_data(normalization, features)
            new_max_pcs = self.model.get_max_pcs()
            self.controls.set_max_pca(new_max_pcs)
            self.plot_view.figure.clear()
            apply_matplotlib_theme(self.plot_view.figure)  # Re-apply theme on clear
            self.plot_view.canvas.draw()

            self.heatmap_view.figure.clear()
            apply_matplotlib_theme(self.heatmap_view.figure)
            self.heatmap_view.canvas.draw()

            msg = f"UPDATED: {len(features)} FEATS | {normalization}"
            self.status_label.setText(msg)
        except Exception as e:
            QMessageBox.critical(self, "ERR", str(e))
        finally:
            self.set_busy_state(False)

    def set_busy_state(self, is_busy):
        self.controls.btn_run.setEnabled(not is_busy)
        if is_busy:
            self.controls.btn_run.setText(Locale.get("BTN_PROCESSING"))
            self.progress.setVisible(True)
            self.progress.setRange(0, 0)
        else:
            self.controls.btn_run.setText(Locale.get("BTN_EXECUTE"))
            self.progress.setVisible(False)

    def toggle_theme(self):
        """Switches between Abyssal (Dark) and Sterile (Light)"""
        ThemeManager.toggle()

        # 1. Update Global Stylesheet
        self.setStyleSheet(ThemeManager.get_stylesheet())

        # 2. Update Matplotlib Canvases
        self.plot_view.refresh_theme()
        self.heatmap_view.refresh_theme()
        self.ranked_genes_view.refresh_theme()

        # 3. Update PyQtGraph
        self.seg_view.refresh_theme()

        # 4. Update Status Bar
        self.update_status_style()

        logger.info(f"THEME SWITCHED TO: {ThemeManager.current_mode}")

    def update_status_style(self):
        c = ThemeManager.get_current()
        self.status_bar.setStyleSheet(
            f"background: {c['bg_panel']}; color: {c['accent']}; border-top: 2px solid {c['grid']}"
        )
        self.progress.setStyleSheet(
            f"""
            QProgressBar {{ border: 1px solid {c["accent_dim"]}; background-color: {c["bg_dark"]}; text-align: center; }}
            QProgressBar::chunk {{ background-color: {c["accent"]}; width: 5px; }}
        """
        )


if __name__ == "__main__":
    app = QApplication(sys.argv)

    logging.basicConfig(
        level=logging.INFO,
    )
    try:
        logger.info("Starting Application...")
        df = pd.read_csv("sp13_2_6.csv")
        segmentation = tifffile.imread("SP13 Stardist.tif")

        window = UMAPVisualizer(df=df, segmentation=segmentation)
        window.show()
        sys.exit(app.exec())
    except FileNotFoundError:
        logger.error("Demo files not found. Exiting.")
        print("Demo files (csv/tif) not found. Please check paths.")
