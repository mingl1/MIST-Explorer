import math

import cv2
import numpy as np
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import (
    QDoubleValidator,
    QImage,
    QIntValidator,
    QKeyEvent,
    QPainter,
    QPixmap,
)
from PyQt6.QtWidgets import (
    QCheckBox,
    QDialog,
    QFormLayout,
    QGraphicsPixmapItem,
    QGraphicsScene,
    QGraphicsView,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QSlider,
    QVBoxLayout,
)

from utils import adjust_contrast, to_uint8


class ZoomableImageView(QGraphicsView):
    def __init__(self, parent=None):
        super().__init__(parent)

        self._scene = QGraphicsScene(self)
        self._pixmap_item = QGraphicsPixmapItem()
        self._scene.addItem(self._pixmap_item)
        self.setScene(self._scene)

        # Configure view properties for better interaction
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.ViewportAnchor.AnchorViewCenter)
        self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
        self.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)

    def set_image(self, pixmap: QPixmap, reset_view: bool = False):
        self._pixmap_item.setPixmap(pixmap)
        if reset_view:
            self.reset_zoom()

    def reset_zoom(self):
        self.fitInView(self._pixmap_item, Qt.AspectRatioMode.KeepAspectRatio)
        self.centerOn(self._pixmap_item)

    def wheelEvent(self, event):
        """Handle mouse wheel events for zooming."""
        angle = event.angleDelta().y()
        if angle > 0:
            zoom_factor = 1.15  # Zoom in
        else:
            zoom_factor = 1 / 1.15  # Zoom out

        self.scale(zoom_factor, zoom_factor)

    # doesnt work...
    def update_pixmap_preserve_center(self, new_pixmap: QPixmap):
        center = self.mapToScene(self.viewport().rect().center())

        self._pixmap_item.setPixmap(new_pixmap)

        self.centerOn(center)


class AlignmentPreviewDialog(QDialog):
    moving_image_changed = pyqtSignal(np.ndarray)

    SLIDER_SCALE_MULTIPLIER = 100.0
    MIN_DOWNSCALE_FACTOR = 1.0
    MAX_DOWNSCALE_FACTOR = 32.0

    def __init__(self, snapshot_data: dict, can_edit: bool = False):
        super().__init__(None)

        self.target_image = snapshot_data["target_image"].copy()
        self.aligned_image = snapshot_data["aligned_image"].copy()
        self.metadata = snapshot_data.get("metadata", {})
        self.can_edit = can_edit
        self.original_aligned_image = self.aligned_image.copy()
        self.result_accepted = False
        self.transformations = [[0.0, []]]
        self.offset_x, self.offset_y, self.move_step = 0, 0, 1
        self.display_max_size = 1024
        self.initial_scale_factor = self._calculate_initial_scale_factor()
        self.scale_factor = self.initial_scale_factor
        self.target_display = self.scale_image_for_display(self.target_image)
        self.aligned_display = self.scale_image_for_display(self.aligned_image)
        self.original_aligned_display = self.aligned_display.copy()
        self.adjust_contrast = False
        self._setup_ui()
        self.create_direct_overlay()
        self.image_view.mouseDoubleClickEvent = self.reset_zoom

    def _setup_ui(self):
        stage_name = self.metadata.get("stage", "Preview").replace("_", " ").title()
        self.setWindowTitle(f"Alignment Preview: {stage_name}")
        self.resize(1000, 800)
        main_layout = QVBoxLayout(self)
        self.enhance_contrast_checkbox = QCheckBox("Enhance Contrast")
        self.enhance_contrast_checkbox.stateChanged.connect(
            self._on_contrast_checkbox_changed
        )
        instruction_text = (
            "Arrow keys/Inputs: move, Mouse wheel: zoom, Drag: pan, Double-click: reset view"
            if self.can_edit
            else "Mouse wheel: zoom, Drag: pan, Double-click: reset view"
        )
        self.preview_label = QLabel(
            f"Red = Target, Green = Aligned | {instruction_text}"
        )
        self.preview_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.offset_label = QLabel()
        self.offset_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.offset_label.setVisible(self.can_edit)
        self.update_offset_label()

        self.metadata_groupbox = QGroupBox("Stage Information")
        metadata_layout = QFormLayout()
        for key, value in self.metadata.items():
            key_str = key.replace("_", " ").title()
            if (
                isinstance(value, (list, tuple, np.ndarray))
                and np.array(value).ndim == 2
            ):
                val_str = readable_matrix_string(np.array(value))
            elif isinstance(value, float):
                val_str = f"{value:.4f}"
            else:
                val_str = str(value)
            metadata_layout.addRow(QLabel(f"{key_str}:"), QLabel(val_str))
        self.metadata_groupbox.setLayout(metadata_layout)

        self.image_view = ZoomableImageView(self)
        self.image_view.setMinimumSize(800, 500)

        self.control_layout = QHBoxLayout()
        self.button_layout = QHBoxLayout()
        if self.can_edit:
            self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
            self._setup_editable_controls()
        else:
            self.setFocusPolicy(Qt.FocusPolicy.NoFocus)
            self._setup_view_only_controls()

        main_layout.addWidget(self.preview_label)
        main_layout.addWidget(self.offset_label)
        main_layout.addWidget(self.metadata_groupbox)
        main_layout.addWidget(self._setup_scale_controls())
        main_layout.addWidget(self.enhance_contrast_checkbox)
        main_layout.addWidget(self.image_view)
        main_layout.addLayout(self.control_layout)
        main_layout.addLayout(self.button_layout)
        self.setLayout(main_layout)

    def _on_contrast_checkbox_changed(self, state):
        self.adjust_contrast = self.enhance_contrast_checkbox.isChecked()
        self.create_direct_overlay()

    def _setup_editable_controls(self):
        """Create UI controls for when manual editing is enabled."""

        trans_group = QGroupBox("Translate (Display Pixels)")
        trans_layout = QHBoxLayout()
        int_validator = QIntValidator(-99999, 99999)

        self.dx_input = QLineEdit("0")
        self.dx_input.setValidator(int_validator)
        self.dx_input.setFixedWidth(50)

        self.dy_input = QLineEdit("0")
        self.dy_input.setValidator(int_validator)
        self.dy_input.setFixedWidth(50)

        self.apply_trans_button = QPushButton("Apply")
        trans_layout.addWidget(QLabel("dx:"))
        trans_layout.addWidget(self.dx_input)
        trans_layout.addWidget(QLabel("dy:"))
        trans_layout.addWidget(self.dy_input)
        trans_layout.addWidget(self.apply_trans_button)
        trans_group.setLayout(trans_layout)

        rot_group = QGroupBox("Rotate (°)")
        rot_layout = QHBoxLayout()
        self.rotation_input = QLineEdit()
        self.rotation_input.setPlaceholderText("Angle")
        self.rotation_input.setValidator(QDoubleValidator(-360.0, 360.0, 2))
        self.rotate_button = QPushButton("Apply")
        rot_layout.addWidget(self.rotation_input)
        rot_layout.addWidget(self.rotate_button)
        rot_group.setLayout(rot_layout)

        self.apply_trans_button.clicked.connect(self.apply_manual_translation)
        self.dx_input.returnPressed.connect(self.apply_manual_translation)
        self.dy_input.returnPressed.connect(self.apply_manual_translation)
        self.rotate_button.clicked.connect(self.apply_rotation)
        self.rotation_input.returnPressed.connect(self.apply_rotation)

        self.reset_button = QPushButton("Reset Position & Rotation")
        self.reset_button.clicked.connect(self.reset_position)

        self.control_layout.addWidget(trans_group)
        self.control_layout.addWidget(rot_group)
        self.control_layout.addStretch()
        self.control_layout.addWidget(self.reset_button)

        self.confirm_button = QPushButton("Confirm Alignment")
        self.confirm_button.clicked.connect(self.accept_alignment)
        self.cancel_button = QPushButton("Cancel")
        self.cancel_button.clicked.connect(self.reject)
        self.button_layout.addStretch()
        self.button_layout.addWidget(self.confirm_button)
        self.button_layout.addWidget(self.cancel_button)
        self.button_layout.addStretch()

    def apply_manual_translation(self):
        """Applies translation based on the dx/dy input fields."""
        try:
            dx = int(self.dx_input.text())
            dy = int(self.dy_input.text())
        except ValueError:
            QMessageBox.warning(
                self,
                "Invalid Input",
                "Please enter valid integer values for dx and dy.",
            )
            return

        if dx == 0 and dy == 0:
            return  # Nothing to do

        self.move_aligned_image(dx, dy)

        self.dx_input.setText("0")
        self.dy_input.setText("0")

    def apply_rotation(self):
        """Rotate the display image and start a new transformation group."""
        if not self.rotation_input.text():
            return
        try:
            angle = float(self.rotation_input.text())
            self.aligned_display = self.rotate_image(self.aligned_display, angle)
            self.transformations.append([angle, []])
            self.update_offset_label()
            self.create_direct_overlay()
            self.rotation_input.clear()
        except ValueError:
            QMessageBox.warning(
                self, "Invalid Input", "Please enter a valid rotation angle."
            )

    def move_aligned_image(self, dx, dy):
        """Move the display image and record the transformation. Central method for all translations."""
        self.offset_x += dx
        self.offset_y += dy
        self.transformations[-1][1].append((dx, dy))
        trans_mat = np.float32([[1, 0, dx], [0, 1, dy]])
        self.aligned_display = cv2.warpAffine(
            self.aligned_display,
            trans_mat,
            (self.aligned_display.shape[1], self.aligned_display.shape[0]),
        )
        self.update_offset_label()
        self.create_direct_overlay()

    def reset_zoom(self, event=None):
        self.image_view.reset_zoom()
        if event:
            event.accept()

    def _setup_view_only_controls(self):
        self.close_button = QPushButton("Close")
        self.close_button.clicked.connect(self.accept)
        self.button_layout.addStretch()
        self.button_layout.addWidget(self.close_button)
        self.button_layout.addStretch()

    def _setup_scale_controls(self):
        group_box = QGroupBox("Display Downscale Factor (1.0 = Full Resolution)")
        layout = QHBoxLayout()
        self.scale_slider = QSlider(Qt.Orientation.Horizontal)
        self.scale_slider.setMinimum(
            int(self.MIN_DOWNSCALE_FACTOR * self.SLIDER_SCALE_MULTIPLIER)
        )
        self.scale_slider.setMaximum(
            int(self.MAX_DOWNSCALE_FACTOR * self.SLIDER_SCALE_MULTIPLIER)
        )
        self.scale_slider.setValue(
            int(self.initial_scale_factor * self.SLIDER_SCALE_MULTIPLIER)
        )
        self.scale_slider.valueChanged.connect(self._on_slider_value_changed)
        self.scale_input = QLineEdit(f"{self.scale_factor:.2f}")
        self.scale_input.setValidator(
            QDoubleValidator(self.MIN_DOWNSCALE_FACTOR, self.MAX_DOWNSCALE_FACTOR, 2)
        )
        self.scale_input.setMaximumWidth(70)
        self.scale_input.editingFinished.connect(self._on_scale_text_changed)
        self.reset_scale_button = QPushButton("Reset Scale")
        self.reset_scale_button.setToolTip(
            f"Reset scale to initial fit-to-screen value ({self.initial_scale_factor:.2f})"
        )
        self.reset_scale_button.clicked.connect(self._reset_scale_to_initial)
        layout.addWidget(QLabel("Factor:"))
        layout.addWidget(self.scale_input)
        layout.addWidget(self.scale_slider)
        layout.addWidget(self.reset_scale_button)
        group_box.setLayout(layout)
        return group_box

    def _reset_scale_to_initial(self):
        self.scale_slider.setValue(
            int(self.initial_scale_factor * self.SLIDER_SCALE_MULTIPLIER)
        )

    def _calculate_initial_scale_factor(self) -> float:
        max_dim = max(self.target_image.shape)
        if max_dim == 0 or max_dim <= self.display_max_size:
            return 1.0
        return max_dim / self.display_max_size

    def scale_image_for_display(self, image: np.ndarray) -> np.ndarray:
        if math.isclose(self.scale_factor, 1.0):
            return image.copy()
        fx_fy = 1.0 / self.scale_factor
        return cv2.resize(
            image, (0, 0), fx=fx_fy, fy=fx_fy, interpolation=cv2.INTER_AREA
        )

    def display_to_actual_coordinates(self, dx: int, dy: int) -> tuple[float, float]:
        return dx * self.scale_factor, dy * self.scale_factor

    def _on_slider_value_changed(self, value: int):
        new_scale = value / self.SLIDER_SCALE_MULTIPLIER
        self.scale_input.setText(f"{new_scale:.2f}")
        self._update_for_new_scale(new_scale)

    def _on_scale_text_changed(self):
        try:
            new_scale = float(self.scale_input.text())
            if not (
                self.MIN_DOWNSCALE_FACTOR <= new_scale <= self.MAX_DOWNSCALE_FACTOR
            ):
                raise ValueError("Scale out of bounds")
            self.scale_slider.blockSignals(True)
            self.scale_slider.setValue(int(new_scale * self.SLIDER_SCALE_MULTIPLIER))
            self.scale_slider.blockSignals(False)
            self._update_for_new_scale(new_scale)
        except ValueError:
            self.scale_input.setText(f"{self.scale_factor:.2f}")

    def _update_for_new_scale(self, new_scale: float):
        if math.isclose(new_scale, self.scale_factor):
            return
        self.scale_factor = new_scale
        self.target_display = self.scale_image_for_display(self.target_image)
        self.original_aligned_display = self.scale_image_for_display(
            self.original_aligned_image
        )
        self._reapply_display_transformations()
        self.update_offset_label()
        self.create_direct_overlay()

    def _reapply_display_transformations(self):
        self.aligned_display = self.original_aligned_display.copy()
        for angle, translations in self.transformations:
            if angle != 0.0:
                self.aligned_display = self.rotate_image(self.aligned_display, angle)
            if translations:
                total_dx, total_dy = sum(t[0] for t in translations), sum(
                    t[1] for t in translations
                )
                if total_dx != 0 or total_dy != 0:
                    trans_mat = np.float32([[1, 0, total_dx], [0, 1, total_dy]])
                    h, w = self.aligned_display.shape[:2]
                    self.aligned_display = cv2.warpAffine(
                        self.aligned_display, trans_mat, (w, h)
                    )

    def update_offset_label(self):
        actual_dx, actual_dy = self.display_to_actual_coordinates(
            self.offset_x, self.offset_y
        )
        self.offset_label.setText(
            f"Manual Offset: ({self.offset_x}, {self.offset_y}) display | ({actual_dx:.1f}, {actual_dy:.1f}) actual"
        )

    def accept_alignment(self):
        self.result_accepted = True
        final_image = self.original_aligned_image.copy()
        for angle, translations in self.transformations:
            if angle != 0.0:
                final_image = self.rotate_image(final_image, angle)
            if translations:
                total_dx_display, total_dy_display = sum(
                    t[0] for t in translations
                ), sum(t[1] for t in translations)
                actual_dx, actual_dy = self.display_to_actual_coordinates(
                    total_dx_display, total_dy_display
                )
                if not math.isclose(actual_dx, 0) or not math.isclose(actual_dy, 0):
                    trans_mat = np.float32([[1, 0, actual_dx], [0, 1, actual_dy]])
                    final_image = cv2.warpAffine(
                        final_image,
                        trans_mat,
                        (final_image.shape[1], final_image.shape[0]),
                    )
        self.aligned_image = final_image
        self.moving_image_changed.emit(self.aligned_image)
        self.accept()

    def keyPressEvent(self, event: QKeyEvent):
        if not self.can_edit:
            super().keyPressEvent(event)
            return
        # Prevent arrow keys from being processed if a text input has focus
        if self.focusWidget() in [self.dx_input, self.dy_input, self.rotation_input]:
            super().keyPressEvent(event)
            return
        key_map = {
            Qt.Key.Key_Left: (-self.move_step, 0),
            Qt.Key.Key_Right: (self.move_step, 0),
            Qt.Key.Key_Up: (0, -self.move_step),
            Qt.Key.Key_Down: (0, self.move_step),
        }
        if event.key() in key_map:
            self.move_aligned_image(*key_map[event.key()])
        else:
            super().keyPressEvent(event)

    def reset_position(self):
        self.offset_x, self.offset_y = 0, 0
        self.transformations = [[0.0, []]]
        self.aligned_display = self.original_aligned_display.copy()
        self.update_offset_label()
        self.create_direct_overlay()

    def create_direct_overlay(self):
        target_gray = self.to_uint8(self.target_display)
        aligned_gray = self.to_uint8(self.aligned_display)
        if self.adjust_contrast:
            target_gray = to_uint8(
                adjust_contrast(target_gray.astype(np.float64), 30, 99)
            )
            aligned_gray = to_uint8(
                adjust_contrast(aligned_gray.astype(np.float64), 30, 99)
            )
        h, w = target_gray.shape
        overlay = np.zeros((h, w, 3), dtype=np.uint8)
        overlay[:, :, 0] = target_gray
        overlay[:, :, 1] = aligned_gray
        q_image = QImage(overlay.data, w, h, w * 3, QImage.Format.Format_RGB888)
        self.image_view.update_pixmap_preserve_center(QPixmap.fromImage(q_image))

    def rotate_image(self, image, angle):
        h, w = image.shape[:2]
        center = (w / 2, h / 2)
        rot_mat = cv2.getRotationMatrix2D(center, angle, 1.0)
        return cv2.warpAffine(
            image,
            rot_mat,
            (w, h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )

    def to_uint8(self, image):
        if image.dtype == np.uint8:
            return image
        img_min, img_max = image.min(), image.max()
        if img_max > img_min:
            return cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        return np.zeros_like(image, dtype=np.uint8)

    def get_current_aligned_image(self):
        return self.aligned_image


def readable_matrix_string(matrix: np.ndarray) -> str:
    if matrix.shape != (2, 3):
        return str(matrix)
    a, b, tx = matrix[0]
    c, d, ty = matrix[1]
    angle_rad = math.atan2(c, a)
    angle_deg = math.degrees(angle_rad)
    scale_x = math.sqrt(a**2 + c**2)
    scale_y = math.sqrt(b**2 + d**2)
    return f"Translation: ({tx:.2f}, {ty:.2f}), Rotation: {angle_deg:.2f}°, Scale: (x: {scale_x:.2f}, y: {scale_y:.2f})"
