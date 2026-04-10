import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtWidgets import (
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QScrollArea,
    QSpinBox,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from core.dataframe_utils import get_marker_columns

sns.set_style("whitegrid")


def calculate_weighted_centroids(data, proteins, signal_threshold, custom_thresholds):
    """Calculate weighted centroid for each protein."""
    centroids = {}
    for protein in proteins:
        threshold = custom_thresholds.get(protein, signal_threshold)
        protein_cells = data[data[protein] >= threshold]
        if len(protein_cells) > 0:
            weights = protein_cells[protein]
            avg_x = np.average(protein_cells["Global X"], weights=weights)
            avg_y = np.average(protein_cells["Global Y"], weights=weights)
            centroids[protein] = (avg_x, avg_y)
    return centroids


def compute_distance_matrix(centroids, proteins):
    """Compute pairwise Euclidean distance matrix."""
    n = len(proteins)
    distances = np.full((n, n), np.nan)
    for i, p1 in enumerate(proteins):
        for j, p2 in enumerate(proteins):
            if p1 in centroids and p2 in centroids:
                x1, y1 = centroids[p1]
                x2, y2 = centroids[p2]
                distances[i, j] = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
    return distances


class HeatmapWindow(QMainWindow):
    def __init__(
        self,
        data,
        signal_threshold=1000,
        custom_thresholds=None,
        x_min=None,
        y_min=None,
        x_max=None,
        y_max=None,
        parent=None,
    ):
        super().__init__(parent)

        if custom_thresholds is None:
            custom_thresholds = {}

        font = {"size": 8}
        matplotlib.rc("font", **font)

        # Optional ROI filtering
        self._data = data
        if all(v is not None for v in (x_min, y_min, x_max, y_max)):
            self._data = data[
                (data["Global X"] >= x_min)
                & (data["Global X"] <= x_max)
                & (data["Global Y"] >= y_min)
                & (data["Global Y"] <= y_max)
            ]

        self._proteins = list(get_marker_columns(self._data))

        # Track which per-protein spinboxes the user has manually edited
        self._manually_edited = set()

        # Debounce timer for live updates
        self._debounce = QTimer(self)
        self._debounce.setSingleShot(True)
        self._debounce.setInterval(300)
        self._debounce.timeout.connect(self._update_heatmap)

        # --- Build UI ---
        central_widget = QWidget(self)
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)

        # Default threshold row
        default_row = QHBoxLayout()
        default_row.addWidget(QLabel("Default Threshold:"))
        self._default_spin = QSpinBox()
        self._default_spin.setRange(0, 50000)
        self._default_spin.setSingleStep(100)
        self._default_spin.setValue(signal_threshold)
        self._default_spin.valueChanged.connect(self._on_default_threshold_changed)
        default_row.addWidget(self._default_spin)
        default_row.addStretch()
        main_layout.addLayout(default_row)

        # Collapsible per-protein section
        self._toggle_btn = QToolButton()
        self._toggle_btn.setStyleSheet("QToolButton { border: none; }")
        self._toggle_btn.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        self._toggle_btn.setArrowType(Qt.ArrowType.RightArrow)
        self._toggle_btn.setText("Per-Protein Thresholds")
        self._toggle_btn.setCheckable(True)
        self._toggle_btn.toggled.connect(self._toggle_protein_section)
        main_layout.addWidget(self._toggle_btn)

        # Scrollable grid of per-protein spinboxes
        self._protein_group = QGroupBox()
        self._protein_group.setFlat(True)
        self._protein_group.setVisible(False)
        grid = QGridLayout(self._protein_group)
        grid.setContentsMargins(4, 4, 4, 4)

        self._protein_spins = {}
        cols = 2  # two protein-threshold pairs per row
        for idx, protein in enumerate(self._proteins):
            row, col = divmod(idx, cols)
            label = QLabel(protein)
            spin = QSpinBox()
            spin.setRange(0, 50000)
            spin.setSingleStep(100)
            initial = custom_thresholds.get(protein, signal_threshold)
            spin.setValue(initial)
            if protein in custom_thresholds:
                self._manually_edited.add(protein)
            spin.valueChanged.connect(
                lambda _v, p=protein: self._on_protein_threshold_changed(p)
            )
            self._protein_spins[protein] = spin
            grid.addWidget(label, row, col * 2)
            grid.addWidget(spin, row, col * 2 + 1)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(self._protein_group)
        scroll.setMaximumHeight(160)
        self._scroll_area = scroll
        scroll.setVisible(False)
        main_layout.addWidget(scroll)

        # Matplotlib figure + canvas
        self.figure, self.ax = plt.subplots(figsize=(12, 10))
        self._canvas = FigureCanvas(self.figure)
        main_layout.addWidget(self._canvas, stretch=1)

        # Initial draw
        self._update_heatmap()

    # --- Signal handlers ---

    def _on_default_threshold_changed(self, value):
        # Propagate to per-protein spinboxes that haven't been manually customized
        for protein, spin in self._protein_spins.items():
            if protein not in self._manually_edited:
                spin.blockSignals(True)
                spin.setValue(value)
                spin.blockSignals(False)
        self._debounce.start()

    def _on_protein_threshold_changed(self, protein):
        self._manually_edited.add(protein)
        self._debounce.start()

    def _toggle_protein_section(self, checked):
        self._protein_group.setVisible(checked)
        self._scroll_area.setVisible(checked)
        arrow = Qt.ArrowType.DownArrow if checked else Qt.ArrowType.RightArrow
        self._toggle_btn.setArrowType(arrow)

    # --- Heatmap computation + render ---

    def _update_heatmap(self):
        signal_threshold = self._default_spin.value()
        custom_thresholds = {}
        for protein, spin in self._protein_spins.items():
            val = spin.value()
            if val != signal_threshold:
                custom_thresholds[protein] = val

        centroids = calculate_weighted_centroids(
            self._data, self._proteins, signal_threshold, custom_thresholds
        )
        distances = compute_distance_matrix(centroids, self._proteins)

        # Normalize
        if np.all(np.isnan(distances)):
            distances_normalized = np.full_like(distances, np.nan)
        elif np.nanmax(distances) == np.nanmin(distances):
            distances_normalized = np.zeros_like(distances)
        else:
            distances_normalized = (distances - np.nanmin(distances)) / (
                np.nanmax(distances) - np.nanmin(distances)
            )

        median_distance = np.nanmedian(distances_normalized)
        vmax_value = max(2 * median_distance, 0.01)

        # Redraw on existing axes
        self.figure.clear()
        self.ax = self.figure.add_subplot(111)
        sns.heatmap(
            distances_normalized,
            cmap="coolwarm_r",
            xticklabels=self._proteins,
            yticklabels=self._proteins,
            vmin=0,
            vmax=vmax_value,
            cbar_kws={"label": "Normalized Distance"},
            ax=self.ax,
        )
        self.ax.set_xticklabels(
            self.ax.get_xticklabels(),
            rotation=90,
            fontsize=12,
            fontweight="bold",
            fontname="Arial",
        )
        self.ax.set_yticklabels(
            self.ax.get_yticklabels(),
            fontsize=12,
            fontweight="bold",
            fontname="Arial",
        )
        self.ax.set_title(
            "Normalized Protein Spatial Distance Heatmap",
            fontsize=14,
            weight="bold",
        )
        self.figure.tight_layout()
        self._canvas.draw_idle()
