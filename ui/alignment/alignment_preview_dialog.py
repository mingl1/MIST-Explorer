import math

import cv2
import numpy as np
from PyQt6.QtCore import QPointF, Qt, QTimer, pyqtSignal
from PyQt6.QtGui import (
    QBrush,
    QColor,
    QDoubleValidator,
    QFont,
    QImage,
    QIntValidator,
    QKeyEvent,
    QPainter,
    QPen,
    QPixmap,
    QTransform,
)
from PyQt6.QtWidgets import (
    QCheckBox,
    QDialog,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGraphicsEllipseItem,
    QGraphicsPixmapItem,
    QGraphicsScene,
    QGraphicsSimpleTextItem,
    QGraphicsView,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
)

from utils import adjust_contrast, apply_ransac_affine_warp, to_uint8


class NullableIntValidator(QIntValidator):
    def validate(self, input_str, pos):
        if input_str == "":
            return (self.State.Acceptable, input_str, pos)
        return super().validate(input_str, pos)


class ZoomableImageView(QGraphicsView):
    def __init__(self, parent=None):
        super().__init__(parent)

        self._scene = QGraphicsScene(self)
        self.target_item = QGraphicsPixmapItem()
        self.moving_item = QGraphicsPixmapItem()
        self.moving_item.setZValue(0.5)
        self.moving_item.setOpacity(0.5)
        self._scene.addItem(self.target_item)
        self._scene.addItem(self.moving_item)
        self.setScene(self._scene)

        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.ViewportAnchor.AnchorViewCenter)
        self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
        self.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)

        # For shift+drag to move moving image
        self._is_dragging_layer = False
        self._drag_start_pos = None
        self._parent_dialog = None

        # Max zoom level (relative to original size)
        self._max_zoom = 150.0
        self._min_zoom = 0.1
        self._current_zoom = 1.0

    def set_images(self, target_pixmap: QPixmap, moving_pixmap: QPixmap):
        self.target_item.setPixmap(target_pixmap)
        self.moving_item.setPixmap(moving_pixmap)
        QTimer.singleShot(0, self.reset_zoom)  # center after render updates

    def reset_zoom(self):
        self.get_scene().setSceneRect(self.get_scene().itemsBoundingRect())
        self.fitInView(self.target_item, Qt.AspectRatioMode.KeepAspectRatio)
        self.centerOn(self.target_item)
        self._current_zoom = 1.0

    def get_scene(self):
        s = self.scene()
        assert s is not None
        return s

    def update_moving_image(self, new_pixmap: QPixmap):
        self.moving_item.setPixmap(new_pixmap)

    def wheelEvent(self, event):
        """Handle mouse wheel events for zooming."""
        if event is None:
            return
        angle = event.angleDelta().y()
        if angle > 0:
            zoom_factor = 1.15  # Zoom in
        else:
            zoom_factor = 1 / 1.15  # Zoom out

        # Calculate new zoom level
        new_zoom = self._current_zoom * zoom_factor

        # Enforce max zoom limit
        if new_zoom > self._max_zoom:
            zoom_factor = self._max_zoom / self._current_zoom
            new_zoom = self._max_zoom

        # Enforce min zoom limit (0.1x)
        if new_zoom < 0.1:
            zoom_factor = 0.1 / self._current_zoom
            new_zoom = 0.1

        self._current_zoom = new_zoom
        self.scale(zoom_factor, zoom_factor)

    def mousePressEvent(self, event):
        """Handle mouse press events for shift+drag layer movement and landmark placement."""
        if (
            event.button() == Qt.MouseButton.LeftButton
            and event.modifiers() == Qt.KeyboardModifier.ShiftModifier
        ):
            if self._parent_dialog and self._parent_dialog.can_edit:
                self._is_dragging_layer = True
                self._drag_start_pos = self.mapToScene(event.pos())
                event.accept()
                return
        # Landmark mode: intercept plain left-click to place landmark
        if (
            self._parent_dialog
            and getattr(self._parent_dialog, "_lm_mode", False)
            and event.button() == Qt.MouseButton.LeftButton
            and event.modifiers() == Qt.KeyboardModifier.NoModifier
        ):
            scene_pt = self.mapToScene(event.pos())
            self._parent_dialog._on_landmark_click(scene_pt.x(), scene_pt.y())
            event.accept()
            return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        """Handle mouse move events for shift+drag layer movement."""
        if self._is_dragging_layer and self._drag_start_pos is not None:
            current_pos = self.mapToScene(event.pos())
            delta = current_pos - self._drag_start_pos

            transform = self.moving_item.transform()

            # Extract current translation and add delta
            current_dx = transform.dx()
            current_dy = transform.dy()

            # Create new transform with updated translation, preserving rotation/scale
            new_transform = QTransform(
                transform.m11(),
                transform.m12(),
                transform.m21(),
                transform.m22(),
                current_dx + delta.x(),
                current_dy + delta.y(),
            )
            self.moving_item.setTransform(new_transform)
            if self._parent_dialog:
                self._parent_dialog.update_offset_label()

            self._drag_start_pos = current_pos
            event.accept()
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        """Handle mouse release events for shift+drag layer movement."""
        if event.button() == Qt.MouseButton.LeftButton and self._is_dragging_layer:
            self._is_dragging_layer = False
            self._drag_start_pos = None
            event.accept()
            return
        super().mouseReleaseEvent(event)


class AlignmentPreviewDialog(QDialog):
    moving_image_changed = pyqtSignal(np.ndarray)
    transformation_matrix = pyqtSignal(np.ndarray)
    combined_transform_ready = pyqtSignal(object)  # dict: ransac_affine_then_affine

    def __init__(self, snapshot_data: dict, can_edit: bool = False, can_emit=False):
        super().__init__(None)

        self.target_image = snapshot_data["target_image"]
        self.aligned_image = snapshot_data["aligned_image"].copy()
        self.metadata = snapshot_data.get("metadata", {})
        self.can_edit = can_edit
        self.original_aligned_image = snapshot_data["aligned_image"]
        self.result_accepted = False
        self.transformations = [[0.0, []]]
        self.offset_x, self.offset_y, self.move_step = 0, 0, 1
        self.adjust_contrast = True
        self.can_emit = can_emit
        self.downscaled = False

        self._ransac_M: np.ndarray | None = None

        # Landmark state
        self._lm_mode = False
        self._lm_waiting_for = "reference"  # "reference" | "moving"
        self._lm_src_pts: list[tuple[float, float]] = []   # (col, row) in target image
        self._lm_dst_pts: list[tuple[float, float]] = []   # (col, row) in original moving image
        self._lm_pending_ref: tuple[float, float] | None = None
        self._lm_ref_markers: list[tuple] = []  # (ellipse, text) graphics items
        self._lm_mov_markers: list[tuple] = []

        self._setup_ui()
        self.create_direct_overlay()
        self.image_view.mouseDoubleClickEvent = self.reset_zoom

    def _setup_ui(self):
        stage_name = self.metadata.get("stage", "Preview").replace("_", " ").title()
        self.setWindowTitle(f"Alignment Preview: {stage_name}")
        self.resize(1000, 800)
        main_layout = QVBoxLayout(self)
        self.enhance_contrast_checkbox = QCheckBox("Enhance Contrast")
        self.enhance_contrast_checkbox.setChecked(self.adjust_contrast)
        self.enhance_contrast_checkbox.stateChanged.connect(
            self._on_contrast_checkbox_changed
        )
        self._default_instruction = (
            "Arrow keys/Inputs: move, Shift+Drag: move layer, Mouse wheel: zoom, Drag: pan, Double-click: reset view"
            if self.can_edit
            else "Mouse wheel: zoom, Drag: pan, Double-click: reset view"
        )
        self.preview_label = QLabel()
        self.preview_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._update_preview_label()
        self.offset_label = QLabel()
        self.offset_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.offset_label.setVisible(self.can_edit)

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
        self.image_view._parent_dialog = self

        self.control_layout = QHBoxLayout()
        self.button_layout = QHBoxLayout()
        if self.can_edit:
            self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
            self._setup_editable_controls()
            self._setup_landmark_controls()
        else:
            self.setFocusPolicy(Qt.FocusPolicy.NoFocus)
            if self.can_emit:
                self._setup_confirm_cancel_buttons()
            else:
                self._setup_view_only_controls()

        main_layout.addWidget(self.preview_label)
        main_layout.addWidget(self.offset_label)
        if self.metadata:
            main_layout.addWidget(self.metadata_groupbox)
        main_layout.addWidget(self.enhance_contrast_checkbox)
        main_layout.addWidget(self.image_view)
        main_layout.addLayout(self.control_layout)
        if self.can_edit:
            main_layout.addLayout(self.landmark_layout)
        main_layout.addLayout(self.button_layout)
        self.setLayout(main_layout)
        self.update_offset_label()

    def _on_contrast_checkbox_changed(self, state):
        self.adjust_contrast = self.enhance_contrast_checkbox.isChecked()
        self.create_direct_overlay()

    def _setup_editable_controls(self):
        """Create UI controls for when manual editing is enabled."""

        self.trans_group = QGroupBox("Translate (Display Pixels)")
        trans_layout = QHBoxLayout()
        int_validator = NullableIntValidator(-99999, 99999)

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
        self.trans_group.setLayout(trans_layout)

        self.rot_group = QGroupBox("Rotate (°)")
        rot_layout = QHBoxLayout()
        self.rotation_input = QLineEdit()
        self.rotation_input.setPlaceholderText("Angle")
        self.rotation_input.setValidator(QDoubleValidator(-360.0, 360.0, 6))
        self.rotate_button = QPushButton("Apply")
        rot_layout.addWidget(self.rotation_input)
        rot_layout.addWidget(self.rotate_button)
        self.rot_group.setLayout(rot_layout)

        self.scale_group = QGroupBox("Scale")
        scale_layout = QHBoxLayout()
        self.scale_input = QLineEdit()
        self.scale_input.setPlaceholderText("1.0")
        self.scale_input.setValidator(QDoubleValidator(0.000001, 10000, 6))
        self.scale_button = QPushButton("Apply")
        scale_layout.addWidget(self.scale_input)
        scale_layout.addWidget(self.scale_button)
        self.scale_group.setLayout(scale_layout)

        self.flip_group = QGroupBox("Flip")
        flip_layout = QHBoxLayout()
        self.flip_horizontal_btn = QPushButton("Flip Horizontal")
        self.flip_vertical_btn = QPushButton("Flip Vertical")
        flip_layout.addWidget(self.flip_horizontal_btn)
        flip_layout.addWidget(self.flip_vertical_btn)
        self.flip_group.setLayout(flip_layout)

        self.apply_trans_button.clicked.connect(self.apply_manual_translation)
        # self.dx_input.returnPressed.connect(self.apply_manual_translation)
        # self.dy_input.returnPressed.connect(self.apply_manual_translation)
        self.rotate_button.clicked.connect(self.apply_rotation)
        self.scale_button.clicked.connect(self.apply_scale)
        # self.rotation_input.returnPressed.connect(self.apply_rotation)
        self.flip_horizontal_btn.clicked.connect(self.apply_flip_horizontal)
        self.flip_vertical_btn.clicked.connect(self.apply_flip_vertical)

        self.reset_button = QPushButton("Reset Transformations")
        self.reset_button.clicked.connect(self.reset_zoom)

        self.control_layout.addWidget(self.trans_group)
        self.control_layout.addWidget(self.rot_group)
        self.control_layout.addWidget(self.scale_group)
        self.control_layout.addWidget(self.flip_group)
        self.control_layout.addStretch()
        self.control_layout.addWidget(self.reset_button)
        self._setup_confirm_cancel_buttons()

    def _setup_confirm_cancel_buttons(self):
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
            xtext = self.dx_input.text()
            ytext = self.dy_input.text()
            if xtext == "":
                xtext = "0"
            if ytext == "":
                ytext = "0"
            dx = int(xtext)
            dy = int(ytext)
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

    def apply_rotation(self):
        if not self.rotation_input.text():
            return
        try:
            angle = float(self.rotation_input.text())
            self.transformations.append([angle, []])

            transform = self.image_view.moving_item.transform()
            center = self.image_view.moving_item.boundingRect().center()
            t = QTransform()
            t.translate(center.x(), center.y())
            t.rotate(angle)
            t.translate(-center.x(), -center.y())

            self.image_view.moving_item.setTransform(transform * t)
            self.update_offset_label()
            # self.rotation_input.clear()
        except ValueError:
            QMessageBox.warning(
                self, "Invalid Input", "Please enter a valid rotation angle."
            )

    def apply_scale(self):
        if not self.scale_input.text():
            return
        try:
            scale = float(self.scale_input.text())
            if scale < 1.0:
                self.downscaled = True
            self.transformations[-1].append("x" + str(scale))

            transform = self.image_view.moving_item.transform()
            center = self.image_view.moving_item.boundingRect().center()
            t = QTransform()
            t.translate(center.x(), center.y())
            t.scale(scale, scale)
            t.translate(-center.x(), -center.y())

            self.image_view.moving_item.setTransform(transform * t)
            self.update_offset_label()
        except ValueError:
            QMessageBox.warning(
                self, "Invalid Input", "Please enter a valid scale factor."
            )

    def apply_flip_horizontal(self):
        """Apply horizontal flip to the moving image."""
        self.transformations[-1].append("flip_h")

        transform = self.image_view.moving_item.transform()
        center = self.image_view.moving_item.boundingRect().center()
        t = QTransform()
        t.translate(center.x(), center.y())
        t.scale(-1, 1)  # Flip horizontal
        t.translate(-center.x(), -center.y())

        self.image_view.moving_item.setTransform(transform * t)
        self.update_offset_label()

    def apply_flip_vertical(self):
        """Apply vertical flip to the moving image."""
        self.transformations[-1].append("flip_v")

        transform = self.image_view.moving_item.transform()
        center = self.image_view.moving_item.boundingRect().center()
        t = QTransform()
        t.translate(center.x(), center.y())
        t.scale(1, -1)  # Flip vertical
        t.translate(-center.x(), -center.y())

        self.image_view.moving_item.setTransform(transform * t)
        self.update_offset_label()

    def move_aligned_image(self, dx, dy):
        """Move the moving image by dx, dy pixels in screen coordinates."""
        self.offset_x += dx
        self.offset_y += dy
        self.transformations[-1][1].append((dx, dy))

        transform = self.image_view.moving_item.transform()

        # Extract current translation and add delta
        current_dx = transform.dx()
        current_dy = transform.dy()

        # Create new transform with updated translation, preserving rotation/scale
        new_transform = QTransform(
            transform.m11(),
            transform.m12(),
            transform.m21(),
            transform.m22(),
            current_dx + dx,
            current_dy + dy,
        )
        self.image_view.moving_item.setTransform(new_transform)

        self.update_offset_label()

    def reset_zoom(self, event=None):
        self.image_view.moving_item.resetTransform()
        self.image_view.reset_zoom()
        if event:
            event.accept()

    def _setup_view_only_controls(self):
        self.close_button = QPushButton("Close")
        self.close_button.clicked.connect(self.accept)
        self.button_layout.addStretch()
        self.button_layout.addWidget(self.close_button)
        self.button_layout.addStretch()

    def _setup_landmark_controls(self) -> None:
        """Create the inline landmark RANSAC Affine row shown below transform controls."""
        self.landmark_layout = QHBoxLayout()
        self.landmark_button = QPushButton("Start Landmark")
        self.landmark_undo_button = QPushButton("Undo Last")
        self.landmark_undo_button.setEnabled(False)
        self.landmark_cancel_button = QPushButton("Cancel Landmark")
        self.landmark_cancel_button.setEnabled(False)
        self.landmark_cancel_button.setToolTip("Cancel landmark mode (Esc)")
        self.landmark_status_label = QLabel("")

        self.import_landmarks_btn = QPushButton("Import Landmarks JSON")
        self.import_landmarks_btn.clicked.connect(self._import_landmarks_from_json)

        self.export_landmarks_btn = QPushButton("Export Landmarks JSON")
        self.export_landmarks_btn.setEnabled(False)
        self.export_landmarks_btn.clicked.connect(self._export_landmarks_to_json)

        self.landmark_button.clicked.connect(self._on_landmark_button_clicked)
        self.landmark_undo_button.clicked.connect(self._undo_landmark)
        self.landmark_cancel_button.clicked.connect(self._cancel_landmark_mode)

        ransac_threshold_label = QLabel("RANSAC Threshold (px):")
        self.landmark_ransac_threshold_spinbox = QDoubleSpinBox()
        self.landmark_ransac_threshold_spinbox.setRange(1.0, 50.0)
        self.landmark_ransac_threshold_spinbox.setSingleStep(0.5)
        self.landmark_ransac_threshold_spinbox.setValue(5.0)
        self.landmark_ransac_threshold_spinbox.setDecimals(1)
        self.landmark_ransac_threshold_spinbox.setToolTip(
            "RANSAC inlier threshold in pixels (lower = stricter outlier rejection)"
        )

        self.landmark_layout.addWidget(self.landmark_button)
        self.landmark_layout.addWidget(self.landmark_undo_button)
        self.landmark_layout.addWidget(self.landmark_cancel_button)
        self.landmark_layout.addWidget(self.landmark_status_label)
        self.landmark_layout.addStretch()
        self.landmark_layout.addWidget(ransac_threshold_label)
        self.landmark_layout.addWidget(self.landmark_ransac_threshold_spinbox)
        self.landmark_layout.addWidget(self.import_landmarks_btn)
        self.landmark_layout.addWidget(self.export_landmarks_btn)

    def _set_transform_controls_enabled(self, enabled: bool) -> None:
        """Enable/disable affine controls + confirm so they can't be triggered
        while the user is in landmark mode."""
        widgets = [
            getattr(self, "trans_group", None),
            getattr(self, "rot_group", None),
            getattr(self, "scale_group", None),
            getattr(self, "flip_group", None),
            getattr(self, "reset_button", None),
            getattr(self, "confirm_button", None),
            getattr(self, "enhance_contrast_checkbox", None),
        ]
        for w in widgets:
            if w is not None:
                w.setEnabled(enabled)

    # ------------------------------------------------------------------
    # Landmark TPS helpers
    # ------------------------------------------------------------------

    def _on_landmark_button_clicked(self) -> None:
        if not self._lm_mode:
            self._start_landmark_mode()
        elif len(self._lm_src_pts) >= 3:
            self._confirm_ransac_affine()

    def _start_landmark_mode(self) -> None:
        self._lm_mode = True
        self._lm_waiting_for = "reference"
        self.image_view.setDragMode(QGraphicsView.DragMode.NoDrag)
        self.image_view.setCursor(Qt.CursorShape.CrossCursor)
        # Highlight the canvas with an orange border so the mode is unmissable.
        self.image_view.setStyleSheet(
            "QGraphicsView { border: 2px solid #FFA500; }"
        )
        # Disable the rest of the editor so the user can't accidentally mutate
        # the affine while placing landmark pairs.
        self._set_transform_controls_enabled(False)
        self._update_preview_label()
        self._update_landmark_ui()

    def _exit_landmark_mode(self) -> None:
        """Shared cleanup for both confirm and cancel paths."""
        self._lm_mode = False
        self.image_view.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
        self.image_view.unsetCursor()
        self.image_view.setStyleSheet("")
        self._set_transform_controls_enabled(True)
        self._update_preview_label()

    def _cancel_landmark_mode(self) -> None:
        """Bail out of landmark mode without applying anything.
        Clears any pending pair and all unconfirmed markers."""
        if not self._lm_mode:
            return
        # Wipe markers / pending pair from current batch
        for e, t in self._lm_ref_markers + self._lm_mov_markers:
            self.image_view.get_scene().removeItem(e)
            self.image_view.get_scene().removeItem(t)
        self._lm_src_pts = []
        self._lm_dst_pts = []
        self._lm_pending_ref = None
        self._lm_ref_markers = []
        self._lm_mov_markers = []
        self._exit_landmark_mode()
        self._update_landmark_ui()

    def _on_landmark_click(self, scene_x: float, scene_y: float) -> None:
        """Called by ZoomableImageView when user clicks in landmark mode."""
        if self._lm_waiting_for == "reference":
            self._lm_pending_ref = (scene_x, scene_y)
            n = len(self._lm_src_pts) + 1
            marker = self._add_scene_marker(scene_x, scene_y, n, QColor(255, 80, 80))
            self._lm_ref_markers.append(marker)
            self._lm_waiting_for = "moving"
        elif self._lm_waiting_for == "moving" and self._lm_pending_ref is not None:
            # Map scene position → original moving-image pixel coordinates
            local_pt = self.image_view.moving_item.mapFromScene(
                QPointF(scene_x, scene_y)
            )
            self._lm_src_pts.append(self._lm_pending_ref)
            self._lm_dst_pts.append((local_pt.x(), local_pt.y()))
            self._lm_pending_ref = None
            n = len(self._lm_src_pts)
            marker = self._add_scene_marker(scene_x, scene_y, n, QColor(80, 210, 80))
            self._lm_mov_markers.append(marker)
            self._lm_waiting_for = "reference"
        self._update_landmark_ui()

    def _add_scene_marker(
        self, x: float, y: float, number: int, color: QColor
    ) -> tuple:
        r = 8
        ellipse = QGraphicsEllipseItem(x - r, y - r, r * 2, r * 2)
        fill = QColor(color.red(), color.green(), color.blue(), 130)
        ellipse.setBrush(QBrush(fill))
        ellipse.setPen(QPen(color, 2))
        ellipse.setZValue(20)
        self.image_view.get_scene().addItem(ellipse)

        text = QGraphicsSimpleTextItem(str(number))
        font = QFont("Arial", 8, QFont.Weight.Bold)
        text.setFont(font)
        text.setBrush(QBrush(Qt.GlobalColor.white))
        text.setPos(x + r + 1, y - r - 1)
        text.setZValue(21)
        self.image_view.get_scene().addItem(text)

        return (ellipse, text)

    def _undo_landmark(self) -> None:
        if self._lm_waiting_for == "moving" and self._lm_pending_ref is not None:
            self._lm_pending_ref = None
            if self._lm_ref_markers:
                e, t = self._lm_ref_markers.pop()
                self.image_view.get_scene().removeItem(e)
                self.image_view.get_scene().removeItem(t)
            self._lm_waiting_for = "reference"
        elif self._lm_src_pts:
            self._lm_src_pts.pop()
            self._lm_dst_pts.pop()
            for marker_list in (self._lm_ref_markers, self._lm_mov_markers):
                if marker_list:
                    e, t = marker_list.pop()
                    self.image_view.get_scene().removeItem(e)
                    self.image_view.get_scene().removeItem(t)
        self._update_landmark_ui()

    def _update_preview_label(self) -> None:
        """Swap the top instruction strip based on whether landmark mode is active."""
        if getattr(self, "_lm_mode", False):
            self.preview_label.setText(
                "LANDMARK MODE — click reference (red), then matching point on moving layer (green). "
                "Esc to cancel."
            )
            self.preview_label.setStyleSheet(
                "color: #FFA500; font-weight: bold; padding: 4px;"
            )
        else:
            self.preview_label.setText(
                f"Red = Target, Green = Aligned | {self._default_instruction}"
            )
            self.preview_label.setStyleSheet("")

    def _update_landmark_ui(self) -> None:
        n = len(self._lm_src_pts)
        has_pending = self._lm_pending_ref is not None
        self.landmark_undo_button.setEnabled(self._lm_mode and (bool(n) or has_pending))
        self.landmark_cancel_button.setEnabled(self._lm_mode)
        has_data = bool(n) or (self._ransac_M is not None)
        self.export_landmarks_btn.setEnabled(has_data)

        if not self._lm_mode:
            self.landmark_button.setText("Start Landmark")
            self.landmark_button.setEnabled(True)
            self.landmark_status_label.setText("")
            self.landmark_status_label.setStyleSheet("")
            return

        # Color-code the status by which marker color comes next
        if self._lm_waiting_for == "reference":
            self.landmark_status_label.setText(
                f"Pair {n + 1} — click REFERENCE point (red)"
            )
            self.landmark_status_label.setStyleSheet(
                "color: #d33; font-weight: bold;"
            )
        else:
            self.landmark_status_label.setText(
                f"Pair {n + 1} — click MATCHING point on moving layer (green)"
            )
            self.landmark_status_label.setStyleSheet(
                "color: #2a9d2a; font-weight: bold;"
            )

        if n >= 3:
            self.landmark_button.setText(f"Compute RANSAC Affine ({n} pairs)")
            self.landmark_button.setEnabled(True)
        else:
            self.landmark_button.setText(f"Need {3 - n} more pair(s)")
            self.landmark_button.setEnabled(False)

    def _bake_ui_transform(self, qt_matrix: np.ndarray, w: int, h: int) -> None:
        """Apply the current Qt overlay transform into aligned_image and compose into _ransac_M."""
        identity = np.array([[1, 0, 0], [0, 1, 0]], dtype=np.float32)
        if np.allclose(qt_matrix, identity):
            return
        original_dtype = self.aligned_image.dtype
        _cv2_ok = {np.uint8, np.uint16, np.int16, np.float32, np.float64}
        bake_src = self.aligned_image if self.aligned_image.dtype in _cv2_ok else self.aligned_image.astype(np.float32)
        baked = cv2.warpAffine(bake_src, qt_matrix, (w, h), flags=cv2.INTER_LINEAR, borderValue=0)
        self.aligned_image = baked.astype(original_dtype) if baked.dtype != original_dtype else baked
        qt_3x3 = np.vstack([qt_matrix, [0, 0, 1]])
        if self._ransac_M is None:
            self._ransac_M = qt_matrix
        else:
            prev_3x3 = np.vstack([self._ransac_M, [0, 0, 1]])
            self._ransac_M = (qt_3x3 @ prev_3x3)[:2, :]

    def _confirm_ransac_affine(self) -> None:
        if len(self._lm_src_pts) < 3:
            return
        src = np.array(self._lm_src_pts, dtype=np.float64)
        dst = np.array(self._lm_dst_pts, dtype=np.float64)
        threshold = self.landmark_ransac_threshold_spinbox.value()

        # Bake any pending UI affine into aligned_image before RANSAC estimation
        h, w = self.target_image.shape[:2]
        qt_matrix = transform_to_matrix(self.image_view.moving_item.transform())
        self._bake_ui_transform(qt_matrix, w, h)

        try:
            warped, meta = apply_ransac_affine_warp(
                self.aligned_image, src, dst,
                ransac_threshold=threshold,
                out_shape=(h, w),
            )
        except Exception as exc:
            QMessageBox.warning(self, "RANSAC Affine Error", str(exc))
            return
        new_M = meta["M"]
        if self._ransac_M is None:
            self._ransac_M = new_M
        else:
            prev_3x3 = np.vstack([self._ransac_M, [0, 0, 1]])
            new_3x3 = np.vstack([new_M, [0, 0, 1]])
            self._ransac_M = (new_3x3 @ prev_3x3)[:2, :]
        self.aligned_image = warped
        self.create_direct_overlay()
        self.image_view.moving_item.resetTransform()
        self.update_offset_label()
        for e, t in self._lm_ref_markers + self._lm_mov_markers:
            self.image_view.get_scene().removeItem(e)
            self.image_view.get_scene().removeItem(t)
        self._lm_src_pts = []
        self._lm_dst_pts = []
        self._lm_pending_ref = None
        self._lm_ref_markers = []
        self._lm_mov_markers = []
        self._exit_landmark_mode()
        self._update_landmark_ui()
        self.landmark_status_label.setText(
            f"Inliers: {meta['inliers']}/{len(src)}  reproj: {meta['reprojection_px']:.1f}px"
        )

    def _import_landmarks_from_json(self) -> None:
        import json

        path, _ = QFileDialog.getOpenFileName(
            self, "Import Landmarks", "", "JSON Files (*.json)"
        )
        if not path:
            return
        try:
            with open(path) as f:
                data = json.load(f)
            batch = data["current_batch"]
            landmarks = batch["landmarks"]
            threshold = float(batch.get("ransac_threshold_px", 5.0))
        except Exception as exc:
            QMessageBox.warning(self, "Import Error", f"Could not parse landmarks file:\n{exc}")
            return

        if not landmarks:
            QMessageBox.warning(self, "Import Error", "No landmarks found in file.")
            return

        if not self._lm_mode:
            self._start_landmark_mode()

        self.landmark_ransac_threshold_spinbox.setValue(threshold)

        for lm in landmarks:
            src = lm["src"]   # [col, row] in target/reference image (scene coords)
            dst = lm["dst"]   # [col, row] in moving image (item-local coords)
            n = len(self._lm_src_pts) + 1

            ref_marker = self._add_scene_marker(src[0], src[1], n, QColor(255, 80, 80))
            self._lm_ref_markers.append(ref_marker)
            self._lm_src_pts.append((src[0], src[1]))

            mov_scene = self.image_view.moving_item.mapToScene(QPointF(dst[0], dst[1]))
            mov_marker = self._add_scene_marker(mov_scene.x(), mov_scene.y(), n, QColor(80, 210, 80))
            self._lm_mov_markers.append(mov_marker)
            self._lm_dst_pts.append((dst[0], dst[1]))

        self._lm_pending_ref = None
        self._lm_waiting_for = "reference"
        self._update_landmark_ui()

    def _export_landmarks_to_json(self) -> None:
        import datetime
        import json

        path, _ = QFileDialog.getSaveFileName(
            self, "Export Landmarks", "landmarks.json", "JSON Files (*.json)"
        )
        if not path:
            return

        data = {
            "metadata": {
                "exported_at": datetime.datetime.now().isoformat(),
                "coordinate_format": "col_row",
                "note": "src = reference image, dst = moving image",
            },
            "current_batch": {
                "ransac_threshold_px": self.landmark_ransac_threshold_spinbox.value(),
                "landmarks": [
                    {"index": i + 1, "src": list(map(float, s)), "dst": list(map(float, d))}
                    for i, (s, d) in enumerate(zip(self._lm_src_pts, self._lm_dst_pts))
                ],
            },
        }

        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    def update_offset_label(self):
        transform_matrix = self.image_view.moving_item.transform()
        transform_text = readable_matrix_string(transform_to_matrix(transform_matrix))
        self.offset_label.setText(transform_text)

    def accept_alignment(self):
        self.result_accepted = True
        final_transformation = self.image_view.moving_item.transform()
        transf_matrix = transform_to_matrix(final_transformation)
        if self._ransac_M is not None:
            self.combined_transform_ready.emit({
                "type": "ransac_affine_then_affine",
                "M": self._ransac_M,
                "affine": transf_matrix,
            })
        else:
            # Pure-affine path — unchanged from original behavior
            h, w = self.target_image.shape[:2]
            img = self.original_aligned_image
            original_dtype = img.dtype
            _supported = {np.uint8, np.uint16, np.int16, np.float32, np.float64}
            if img.dtype not in _supported:
                img = img.astype(np.float32)
            final_image = cv2.warpAffine(img, transf_matrix, (w, h))
            if final_image.dtype != original_dtype:
                final_image = final_image.astype(original_dtype)
            self.transformation_matrix.emit(transf_matrix)
            self.moving_image_changed.emit(final_image)
        self.accept()

    def keyPressEvent(self, event: QKeyEvent):
        # Esc cancels landmark mode regardless of can_edit (only relevant when active)
        if event.key() == Qt.Key.Key_Escape and getattr(self, "_lm_mode", False):
            self._cancel_landmark_mode()
            event.accept()
            return
        if not self.can_edit:
            super().keyPressEvent(event)
            return
        # Block arrow-key nudging while landmarking — the user is placing pairs,
        # not adjusting the affine.
        if self._lm_mode:
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

    def create_direct_overlay(self):
        target_img = self.target_image
        if self.target_image.ndim == 3:
            target_img = cv2.cvtColor(self.target_image, cv2.COLOR_RGB2GRAY)

        aligned_img = self.aligned_image
        if self.aligned_image.ndim == 3:
            aligned_img = cv2.cvtColor(self.aligned_image, cv2.COLOR_RGB2GRAY)

        target_gray = self.to_uint8(target_img)
        aligned_gray = self.to_uint8(aligned_img)
        h, w = target_gray.shape
        ah, aw = aligned_gray.shape

        # start_y = (ah - h) // 2
        # start_x = (aw - w) // 2
        # aligned_gray = aligned_gray[start_y : start_y + h, start_x : start_x + w]

        if self.adjust_contrast:
            target_gray = to_uint8(
                adjust_contrast(target_gray.astype(np.float32), 30, 99)
            )
            aligned_gray = to_uint8(
                adjust_contrast(aligned_gray.astype(np.float32), 30, 99)
            )

        # Create separate QPixmaps for both layers
        aligned_pixmap = colorize_grayscale(aligned_gray, "green")
        target_pixmap = colorize_grayscale(target_gray, "red")

        self.image_view.set_images(target_pixmap, aligned_pixmap)

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


def colorize_grayscale(gray_img: np.ndarray, color: str) -> QPixmap:
    """Colorize grayscale image and make black pixels fully transparent."""
    h, w = gray_img.shape
    rgba = np.zeros((h, w, 4), dtype=np.uint8)

    if color == "red":
        rgba[:, :, 0] = gray_img  # R
    elif color == "green":
        rgba[:, :, 1] = gray_img  # G
    elif color == "blue":
        rgba[:, :, 2] = gray_img  # B

    # Make black (value 0) transparent
    mask = gray_img > 0
    rgba[:, :, 3] = mask.astype(np.uint8) * 255  # Alpha

    qimage = QImage(rgba.data, w, h, 4 * w, QImage.Format.Format_RGBA8888)
    return QPixmap.fromImage(qimage)


def transform_to_matrix(t: QTransform):
    matrix = np.array(
        [
            [t.m11(), t.m21(), t.dx()],
            [t.m12(), t.m22(), t.dy()],
        ],
        dtype=np.float32,
    )
    return matrix
