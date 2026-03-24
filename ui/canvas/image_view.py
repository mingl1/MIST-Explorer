"Image graphics view module."
import logging
import typing

import numpy as np
import pyqtgraph as pg
import tifffile
from PyQt6.QtCore import QPoint, QRect, QRectF, QSize, Qt, pyqtSignal
from PyQt6.QtGui import (QBrush, QColor, QCursor, QDragEnterEvent,
                         QDragMoveEvent, QIcon, QImage, QMouseEvent, QPainter,
                         QPen, QPixmap)
from PyQt6.QtWidgets import (QFileDialog, QGraphicsPixmapItem,
                             QGraphicsRectItem, QGraphicsView, QHBoxLayout,
                             QLabel, QPushButton, QToolTip, QWidget)

from ui.canvas.items import CropRectItem, ResizableRect
from ui.lassos.CircleLasso import CircleLasso
from ui.lassos.PolyLasso import PolyLasso
from ui.lassos.RectLasso import RectLasso
from utils import resource_path

logger = logging.getLogger(__name__)

if typing.TYPE_CHECKING:
    from ui.app import MainWindow

TOOLTIP_PERSIST_MS = 120000

pg.setConfigOption("imageAxisOrder", "row-major")
pg.setConfigOption("useOpenGL", True)
pg.setConfigOption("useCupy", True)
pg.setConfigOption("useNumba", False)


# pylint: disable=too-many-instance-attributes, too-many-public-methods
class ImageGraphicsViewUI(QGraphicsView):
    """Main image view with support for selection, cropping and other operations"""

    image_dropped = pyqtSignal(str)
    show_crop = pyqtSignal(QRect)
    horizontal_flip = pyqtSignal()  # Signal to request horizontal flip
    vertical_flip = pyqtSignal()  # Signal to request vertical flip
    clear_canvas = pyqtSignal()
    sigRangeChanged = pyqtSignal(object)  # placeholder for pyqtgraph compatibility
    sigTransformChanged = pyqtSignal()  # placeholder for pyqtgraph compatibility

    # pylint: disable=too-many-instance-attributes
    def __init__(self, parent, enc: "MainWindow", show_buttons=True):
        super().__init__(parent)
        self.enc = enc
        self.show_buttons = show_buttons

        self.pixmap_item = QGraphicsPixmapItem(QPixmap())
        self.rubber_band = None
        self.rubber_bands = []
        self.rubber_band_colors = []
        self.begin_crop = False
        self.origin = None
        self.crop_cursor = QCursor(Qt.CursorShape.CrossCursor)
        self.select = False
        self.zoom = 1
        self.polygons = []
        self.current_polygon = None
        self.rubber_band_positions = []
        self.select_start_pos = None

        # Improved crop system
        self.crop_mode = False
        self.crop_start_pos = None
        self.active_crop_rect = None
        self.is_resizing = False
        
        # Move ROI state
        self.moving_lasso = None
        self.last_mouse_pos = None

        self.reference_view = None

        # Setup interaction
        # self.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.setMouseTracking(True)
        self.view_pixmaps = []
        # self.view_pixmap_item = QGraphicsPixmapItem()

        # self.get_scene().addItem(self.view_pixmap_item)
        self.view_mode = False
        
        
        # Attributes initialized
        self.floating_container = None
        self.rect_button = None
        self.circle_button = None
        self.poly_button = None
        self.image_pos = None
        self.starting_x = 0
        self.starting_y = 0
        self.center = None
        self.initial_crop = None
        self.initial_crop_rect = None
        self.np_channels = None
        self.pixel_highlight = None
        self.setup_ui()
        self.get_scene().addItem(self.pixmap_item)
        self.pixmap_item.show()

    def show_view_tab_image(self):
        """Show the view tab image."""
        self.pixmap_item.hide()
        for pixmap in self.view_pixmaps:
            pixmap.show()

        # Update the view transform to properly fit the view_pixmaps
        if self.view_pixmaps:
            self._center_image()

        # Show rubberbands when in view/analyze tabs (after centering to ensure correct z-order)
        for rubber_band in self.rubber_bands:
            rubber_band.show()
            rubber_band.raise_()  # Bring to front

        # Force viewport update to ensure rubberbands are painted
        if self.viewport():
            self.viewport().update()

    def show_images_tab_image(self):
        """Show the images tab image."""
        self.pixmap_item.show()
        for pixmap in self.view_pixmaps:
            pixmap.hide()

        # Also hide rubberbands when switching to Extract tab
        for rubber_band in self.rubber_bands:
            rubber_band.hide()

    def get_scene(self):
        """Get the scene."""
        scene_obj = self.scene()
        assert scene_obj is not None, "Scene should be initialized"
        return scene_obj

    def reset_view_tab(self):
        """Reset the view tab by clearing all additional layers"""
        for pixmap in self.view_pixmaps:
            self.get_scene().removeItem(pixmap)
        self.view_pixmaps = []
        self.show_images_tab_image()
        self._center_image()

    def flip_horizontal(self):
        """Flip the image horizontally"""
        if self.pixmap_item:
            # img = self.pixmap_item.pixmap().toImage()
            # flipped_img = img.mirrored(horizontal=True, vertical=False)
            # flipped_pixmap = QPixmap.fromImage(flipped_img)
            # self.pixmap_item.setPixmap(flipped_pixmap)
            # self.get_scene().update()
            # Emit signal to update the underlying data model
            self.horizontal_flip.emit()

    def flip_vertical(self):
        """Flip the image vertically"""
        if self.pixmap_item:
            # img = self.pixmap_item.pixmap().toImage()
            # flipped_img = img.mirrored(horizontal=False, vertical=True)
            # flipped_pixmap = QPixmap.fromImage(flipped_img)
            # self.pixmap_item.setPixmap(flipped_pixmap)
            # self.get_scene().update()
            # Emit signal to update the underlying data model
            self.vertical_flip.emit()

    def setup_ui(self):
        """Setup the UI."""
        self.setMinimumSize(QSize(600, 600))
        self.setObjectName("canvas")
        self.setAcceptDrops(True)
        self.setScene(pg.GraphicsScene())
        self.get_scene().setBackgroundBrush(QBrush(QColor("black")))  # black background
        self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
        self.setRenderHint(self.renderHints() | self.renderHints().Antialiasing)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.setSceneRect(0, 0, 800, 600)
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        # self.setContentsMargins(1000, 1000, 1000, 1000)

        # Create floating selection buttons
        if self.show_buttons:
            self.create_floating_buttons()
            self.set_buttons_visible(False)

    def create_floating_buttons(self):
        """Create floating selection buttons that appear over the canvas"""
        # Create a container widget for the buttons
        self.floating_container = QWidget(self)

        # Create horizontal layout for the buttons
        button_layout = QHBoxLayout(self.floating_container)
        button_layout.setContentsMargins(10, 5, 10, 5)
        button_layout.setSpacing(10)
        button_layout.setAlignment(Qt.AlignmentFlag.AlignRight)

        # Add label
        label = QLabel("Select region of interest:", self.floating_container)
        label.setStyleSheet(
            "QLabel { color: white; padding: 5px; border-radius: 3px; }"
        )
        button_layout.addWidget(label)

        # Create the selection buttons
        self.rect_button = QPushButton()
        self.circle_button = QPushButton()
        self.poly_button = QPushButton()

        # Set button sizes and styles
        for button in [self.rect_button, self.circle_button, self.poly_button]:
            button.setFixedSize(40, 40)
            button.setStyleSheet(
                """
                QPushButton {
                    background-color: rgba(255, 255, 255, 0.1);
                    border: 1px solid #ccc;
                    border-radius: 5px;
                }
                QPushButton:hover {
                    background-color: rgba(255, 255, 255, 0.2);
                }
                QPushButton:pressed {
                    background-color: rgba(200, 200, 200, 0.2);
                }
            """
            )

        # Set icons for the buttons
        self.rect_button.setIcon(QIcon(resource_path("assets/icons/square.png")))
        self.circle_button.setIcon(QIcon(resource_path("assets/icons/circle.png")))
        self.poly_button.setIcon(QIcon(resource_path("assets/icons/poly.png")))

        self.rect_button.clicked.connect(lambda: self.set_selection_mode("rect"))
        self.circle_button.clicked.connect(lambda: self.set_selection_mode("circle"))
        self.poly_button.clicked.connect(self.enc.poly_select)

        # Add buttons to layout
        button_layout.addWidget(self.rect_button)
        button_layout.addWidget(self.circle_button)
        button_layout.addWidget(self.poly_button)

        # Position the container at the top-right of the view
        self.update_buttons_position()

    def set_buttons_visible(self, visible: bool):
        """Set the visibility of the floating buttons"""
        if self.floating_container:
            self.floating_container.setVisible(visible)
            self.update_buttons_position()

    def update_buttons_position(self):
        """Update the position of the floating buttons"""
        if self.floating_container and self.floating_container.isVisible():
            # Position at the top-right of the view with some padding
            self.floating_container.move(
                self.width() - self.floating_container.width() - 20,
                10,
            )

    # pylint: disable=invalid-name, useless-parent-delegation
    def resizeEvent(self, event):
        """Handle resize events to update floating buttons position"""
        super().resizeEvent(event)
        self.update_buttons_position()

    def set_selection_mode(self, mode):
        """Set the current selection mode"""
        # Reset all modes
        self.select = False
        self.current_polygon = None
        self.crop_mode = False
        self.begin_crop = False

        # Set the new mode
        if mode == "rect":
            self.select = "rect"
            self.enc.select()
        elif mode == "circle":
            self.select = "circle"
            self.enc.circle_select()
        elif mode == "poly":
            self.select = "poly"
            self.enc.poly_select()

    def start_crop_mode(self):
        """Start crop mode - called from external crop button"""
        self.crop_mode = True
        self.begin_crop = True
        self.select = False
        self.current_polygon = None
        self.setFocus()

        # Clear any existing crop rectangle
        if self.active_crop_rect:
            self.get_scene().removeItem(self.active_crop_rect)
            self.active_crop_rect = None

        self.setCursor(Qt.CursorShape.CrossCursor)

    def cancel_crop_mode(self):
        """Cancel crop mode and clean up"""
        self.crop_mode = False
        self.begin_crop = False
        self.crop_start_pos = None

        if self.active_crop_rect:
            self.get_scene().removeItem(self.active_crop_rect)
            self.active_crop_rect = None

        self.unsetCursor()

    def is_empty(self) -> bool:
        """Check if canvas is empty."""
        has_pixmap = self.pixmap_item is not None and not self.pixmap_item.pixmap().isNull()
        has_view_layers = len(self.view_pixmaps) > 0
        return not (has_pixmap or has_view_layers)

    # pylint: disable=invalid-name, unused-argument
    def mouseDoubleClickEvent(self, event):
        """Handle mouse double click."""
        if not self.is_empty():
            self._center_image()

    def update_canvas(self, pixmap: QPixmap):
        """Updates canvas when current image is operated on"""
        if self.pixmap_item:
            prev_pixmap_shape = self.pixmap_item.boundingRect()
            self.pixmap_item.setPixmap(pixmap)
            if self.zoom == 1 or self.pixmap_item.boundingRect() != prev_pixmap_shape:
                self._center_image()

    def update_layer_levels(self, layer_idx, value):
        """Update layer levels."""
        self.view_pixmaps[layer_idx].setLevels(value, True)

    def update_layer_cmap(self, layer_idx, cmap):
        """Update layer colormap."""
        layer = self.view_pixmaps[layer_idx]
        assert isinstance(layer, pg.ImageItem)
        layer.setLookupTable(cmap)

    def update_layer_opacity(self, layer_idx, opacity):
        """Update layer opacity."""
        layer = self.view_pixmaps[layer_idx]
        assert isinstance(layer, pg.ImageItem)
        layer.setOpacity(opacity)

    def update_layer_visibility(self, layer_idx, visible):
        """Update layer visibility."""
        layer = self.view_pixmaps[layer_idx]
        assert isinstance(layer, pg.ImageItem)
        layer.setVisible(visible)

    def update_view_tab_canvas(self, pixmap: np.ndarray, layer_idx):
        """Update view tab canvas."""
        # self._show_view_tab_image()
        if layer_idx > (len(self.view_pixmaps) - 1):
            new_layer = pg.ImageItem(pixmap, levels=None)
            new_layer.setCompositionMode(QPainter.CompositionMode.CompositionMode_Plus)
            new_layer.setZValue(2)
            self.view_pixmaps.append(new_layer)
            self.get_scene().addItem(new_layer)
            self._center_view_tab_image(layer_idx)
            # Button visibility is now controlled by tab changes
        else:
            logger.debug(f"layer {layer_idx}")
            logger.debug(f"pixmap shape {pixmap.shape}")
            # Button visibility is now controlled by tab changes
            if pixmap.shape == (0,):
                removed_item = self.view_pixmaps.pop(layer_idx)
                self.get_scene().removeItem(removed_item)
            else:
                self.view_pixmaps[layer_idx].setImage(pixmap)

    def _center_view_tab_image(self, layer_idx):
        rubber_band_positions = self._capture_rubber_band_scene_positions()
        pixmap_item = self.view_pixmaps[layer_idx]
        item_rect = pixmap_item.boundingRect()
        item_rect = QRectF(
            item_rect.x() - item_rect.width() // 2,
            item_rect.y() - item_rect.height() // 2,
            item_rect.width() * 2,
            item_rect.height() * 2,
        )
        self.setSceneRect(item_rect)
        self.fitInView(pixmap_item, Qt.AspectRatioMode.KeepAspectRatio)
        self.centerOn(pixmap_item)
        self._restore_rubber_band_scene_positions(rubber_band_positions)

    def _center_image(self):
        rubber_band_positions = self._capture_rubber_band_scene_positions()
        pixmap_item = \
            self.pixmap_item if self.pixmap_item.isVisible() else self.view_pixmaps[0]
        item_rect = pixmap_item.boundingRect()
        item_rect = QRectF(
            item_rect.x() - item_rect.width() // 2,
            item_rect.y() - item_rect.height() // 2,
            item_rect.width() * 2,
            item_rect.height() * 2,
        )
        self.setSceneRect(item_rect)
        self.fitInView(pixmap_item, Qt.AspectRatioMode.KeepAspectRatio)
        self.centerOn(pixmap_item)
        self._restore_rubber_band_scene_positions(rubber_band_positions)
        # if self.reference_view:
        #     self.reference_view.__centerImage()

    def _capture_rubber_band_scene_positions(self):
        """Capture ROI widget corners in scene coordinates before view transform changes."""
        positions = []
        for rubber_band in self.rubber_bands:
            if rubber_band is None:
                continue
            rect = rubber_band.geometry().normalized()
            top_left_scene = self.mapToScene(rect.topLeft())
            bottom_right_scene = self.mapToScene(rect.bottomRight())
            positions.append((rubber_band, top_left_scene, bottom_right_scene))
        return positions

    def _restore_rubber_band_scene_positions(self, positions):
        """Restore ROI widget geometry from scene coordinates after view transform changes."""
        for rubber_band, top_left_scene, bottom_right_scene in positions:
            if rubber_band is None:
                continue
            new_top_left_view = self.mapFromScene(top_left_scene)
            new_bottom_right_view = self.mapFromScene(bottom_right_scene)
            rubber_band.setGeometry(
                QRect(new_top_left_view, new_bottom_right_view).normalized()
            )

    # pylint: disable=invalid-name
    def dragEnterEvent(self, event: QDragEnterEvent):  # type: ignore
        """Handle drag enter."""
        mime = event.mimeData()
        if mime and mime.hasUrls():
            event.acceptProposedAction()

    # pylint: disable=invalid-name
    def dragMoveEvent(self, event: QDragMoveEvent):  # type: ignore
        """Handle drag move."""
        event.acceptProposedAction()

    # pylint: disable=invalid-name
    def dropEvent(self, event):
        """Handle drop."""
        if event is None:
            return
        mime = event.mimeData()
        if mime and mime.hasUrls():
            for url in mime.urls():
                file_path = url.toLocalFile()
                if file_path is not None:
                    self.image_dropped.emit(file_path)
            event.acceptProposedAction()

    # pylint: disable=invalid-name
    def wheelEvent(self, event):
        """Handle wheel event for zooming."""
        if event is None:
            return
        if self.pixmap_item is None:
            return
        zooming_out = event.angleDelta().y() < 0

        # Prevent excessive zooming in either direction
        # if self.zoom > 1.1**2 and zooming_out:  # Max zoom out
        #     return

        # if self.zoom < 1 / (1.1**20) and not zooming_out:
        #     return
        # if self.reference_view:
        #     self.reference_view.wheelEvent(event)
        if self.zoom <= 0.1 and zooming_out:
            return
        zoom_factor = 0.9 if zooming_out else 1.1
        self.zoom *= zoom_factor

        # Store rubber band positions before zooming
        if not self.rubber_band_positions:
            self.rubber_band_positions = []
            for rubber_band in self.rubber_bands:
                rubber_band_geometry = rubber_band.geometry()
                top_left_scene = self.mapToScene(rubber_band_geometry.topLeft())
                bottom_right_scene = self.mapToScene(rubber_band_geometry.bottomRight())
                self.rubber_band_positions.append(
                    (rubber_band, top_left_scene, bottom_right_scene)
                )

        # Perform the zoom
        self.scale(zoom_factor, zoom_factor)

        # Update rubber band positions after zooming
        for (
            rubber_band,
            top_left_scene,
            bottom_right_scene,
        ) in self.rubber_band_positions:
            new_top_left_view = self.mapFromScene(top_left_scene)
            new_bottom_right_view = self.mapFromScene(bottom_right_scene)
            new_rect = QRect(new_top_left_view, new_bottom_right_view)
            rubber_band.setGeometry(new_rect)

        # Force redraw to update polygon positions
        vp = self.viewport()
        assert vp is not None, "Viewport should be initialized"
        vp.update()

    # pylint: disable=too-many-arguments, too-many-positional-arguments
    def create_rubber_band(self, rubber_band_class, shape, x, y, origin):
        """Create a rubber band of the specified class"""
        rubber_band = rubber_band_class(shape, x, y, self)
        rubber_band.setGeometry(QRect(origin, QSize()))
        rubber_band.show()
        return rubber_band

    def update_starting_position(self, event):
        """Update the starting position for rubber band operations"""
        scene_pos = self.mapToScene(event.pos())
        self.image_pos = self.pixmap_item.mapFromScene(scene_pos)
        self.starting_x = int(self.image_pos.x())
        self.starting_y = int(self.image_pos.y())

    def _find_top_rubber_band(self, view_pos):
        """Return the top-most visible ROI under the given view position."""
        for rubber_band in reversed(self.rubber_bands):
            if (
                rubber_band
                and rubber_band.isVisible()
                and rubber_band.geometry().contains(view_pos)
            ):
                return rubber_band
        return None

    def _cancel_rubber_band_drag(self):
        """Reset any in-progress ROI drag state."""
        self.moving_lasso = None
        self.last_mouse_pos = None
        for rubber_band in self.rubber_bands:
            if hasattr(rubber_band, "is_dragging"):
                rubber_band.is_dragging = False
            if hasattr(rubber_band, "mouse_press_pos"):
                rubber_band.mouse_press_pos = None
            if hasattr(rubber_band, "mouse_move_pos"):
                rubber_band.mouse_move_pos = None

    # pylint: disable=invalid-name, too-many-branches, too-many-statements
    def mousePressEvent(self, event: QMouseEvent | None):
        """Handle mouse press."""
        if self.is_empty() or event is None:
            return
        if event.button() == Qt.MouseButton.LeftButton:
            # Handle crop mode
            if self.crop_mode and not self.active_crop_rect:
                scene_pos = self.mapToScene(event.pos())
                self.crop_start_pos = scene_pos

                # Create a new crop rectangle starting from this position
                if self.active_crop_rect:
                    self.get_scene().removeItem(self.active_crop_rect)

                # Create a resizable crop rectangle

                self.active_crop_rect = CropRectItem()
                self.active_crop_rect.setRect(scene_pos.x(), scene_pos.y(), 0, 0)

                # Style the crop rectangle
                pen = QPen(QColor(255, 255, 255), 2, Qt.PenStyle.DashLine)
                self.active_crop_rect.setPen(pen)
                self.get_scene().addItem(self.active_crop_rect)
                self.is_resizing = True

            # Handle Shift+Click to move ROI
            if event.modifiers() & Qt.KeyboardModifier.ShiftModifier:
                # Find the top-most lasso under the mouse
                top_lasso = self._find_top_rubber_band(event.pos())
                if top_lasso is not None:
                    self.moving_lasso = top_lasso
                    self.last_mouse_pos = event.pos()
                    event.accept()
                    return

            # Handle regular selection modes
            if self.begin_crop or self.select:
                logger.debug("beginning selection")
                self.origin = event.pos()
                self.update_starting_position(event)
                scene_pos = self.mapToScene(event.pos())
                image_pos = self.pixmap_item.mapFromScene(scene_pos)
                self.select_start_pos = image_pos
                if self.begin_crop:
                    if not self.rubber_band:
                        self.rubber_band = RectLasso(self)
                elif self.select == "rect":
                    self.rubber_band = RectLasso(self)
                    self.rubber_band.roi_moved.connect(lambda: self.update_roi_from_lasso(self.rubber_band))
                    self.rubber_bands.append(self.rubber_band)
                    self.rubber_band_colors.append(self.rubber_band.color)
                    self.rubber_band.setGeometry(QRect(self.origin, QSize()))
                    self.rubber_band.show()
                elif self.select == "circle":
                    self.center = QPoint(self.starting_x, self.starting_y)
                    self.rubber_band = CircleLasso(self)
                    self.rubber_band.roi_moved.connect(lambda: self.update_roi_from_lasso(self.rubber_band))
                    self.rubber_bands.append(self.rubber_band)
                    self.rubber_band_colors.append(self.rubber_band.color)
                    self.rubber_band.setGeometry(QRect(self.origin, QSize()))
                    self.rubber_band.show()
                elif self.select == "poly":
                    if not self.current_polygon:
                        self.current_polygon = PolyLasso(
                            self.pixmap_item
                        )  # Set pixmapItem as parent
                        self.get_scene().addItem(self.current_polygon)
                        # Enable mouse tracking for live preview
                        self.setMouseTracking(True)

                    # Add point in scene coordinates, but relative to the image

                    # Convert image_pos to scene coordinates relative to the image
                    polygon_pos = self.pixmap_item.mapToScene(image_pos)
                    self.current_polygon.add_point(polygon_pos, image_pos)

                if self.begin_crop:
                    self.rubber_bands.append(self.rubber_band)
                    assert (
                        self.rubber_band is not None
                    ), "Rubber band should be initialized"
                    self.rubber_band_colors.append(self.rubber_band.color)
                    self.rubber_band.setGeometry(QRect(self.origin, QSize()))
                    self.rubber_band.show()

        if not self.is_resizing and not self.select:
            super().mousePressEvent(event)

    # pylint: disable=invalid-name
    def keyPressEvent(self, event):
        """Handle key press."""
        # Handle crop confirmation with Enter key
        if event is None:
            return
        if (
            (event.key() == Qt.Key.Key_Enter or event.key() == Qt.Key.Key_Return)
            and self.crop_mode
            and self.active_crop_rect
        ):
            logger.debug("confirming")
            self.confirm_crop()
            return

        # Handle crop cancellation with Escape key
        if event.key() == Qt.Key.Key_Escape and self.crop_mode:
            self.cancel_crop_mode()
            return

        super().keyPressEvent(event)

    def confirm_crop(self):
        """Confirm the current crop selection"""
        if not self.active_crop_rect or not self.pixmap_item:
            return

        # Get the crop rectangle in scene coordinates
        crop_rect = self.active_crop_rect.rect()

        # Convert to image coordinates
        image_top_left = self.pixmap_item.mapFromScene(crop_rect.topLeft())
        image_bottom_right = self.pixmap_item.mapFromScene(crop_rect.bottomRight())

        # Create image rectangle and clamp to image bounds
        image_rect = QRect(
            max(0, int(image_top_left.x())),
            max(0, int(image_top_left.y())),
            int(image_bottom_right.x() - image_top_left.x()),
            int(image_bottom_right.y() - image_top_left.y()),
        ).normalized()

        # Make sure rect is within image bounds
        image = self.pixmap_item.pixmap().toImage()
        image_width, image_height = image.width(), image.height()
        image_rect = image_rect.intersected(QRect(0, 0, image_width, image_height))

        if image_rect.isEmpty():
            return

        # Emit the crop signal
        self.show_crop.emit(image_rect)

        # Clean up crop mode
        self.cancel_crop_mode()

    def save_as_png(self, num_layers):
        """Save as PNG."""
        if num_layers == 0:
            return
        file_name, _ = QFileDialog.getSaveFileName(
            None, "Save PNG File", "protein_layers.png", "*.png;;All Files (*)"
        )
        if not file_name:
            return

        scene = self.get_scene()
        scene_rect = scene.sceneRect()

        # Create a QImage to render the scene onto
        # Use ARGB32_Premultiplied for transparency support
        img = QImage(
            scene_rect.size().toSize(), QImage.Format.Format_ARGB32_Premultiplied
        )
        img.fill(Qt.GlobalColor.black)  # Fill with transparent background

        # Create a QPainter to draw on the QImage
        painter = QPainter(img)

        # Render the scene onto the QImage
        scene.render(painter, QRectF(img.rect()), scene_rect)

        # End the painter
        painter.end()
        img.save(file_name)

    def save_as_tif(self, num_layers):
        """Save as TIF."""
        if num_layers == 0:
            return  # the second QRectF is the source rect from the scene.
        file_name, _ = QFileDialog.getSaveFileName(
            None, "Save PNG File", "protein_layers.png", "*.png;;All Files (*)"
        )
        if not file_name:
            return

        assert num_layers == len(self.view_pixmaps)
        img_stack = []
        for i in self.view_pixmaps:
            assert isinstance(i, pg.ImageItem)
            qimage = (
                i.getPixmap().toImage().convertToFormat(QImage.Format.Format_RGB888)
            )
            ptr = qimage.bits()
            ptr.setsize(qimage.width() * qimage.height() * 4)
            # pylint: disable=too-many-function-args
            arr = np.array(ptr, dtype=np.uint8).reshape(
                qimage.height(), qimage.width(), 3
            )
            img_stack.append(arr)

        img_stack = np.array(img_stack)
        # pylint: disable=too-many-function-args
        tifffile.imwrite(file_name, img_stack, imagej=True)

    # pylint: disable=invalid-name, too-many-locals, too-many-statements, too-many-branches
    def mouseMoveEvent(self, event: QMouseEvent | None):
        """Handle mouse move."""
        super().mouseMoveEvent(event)

        # Handle crop rectangle re  sizing
        if event is None or self.is_empty():
            return
        event.accept()
        if (
            self.crop_mode
            and self.crop_start_pos is not None
            and self.active_crop_rect is not None
            and self.is_resizing
        ):
            current_pos = self.mapToScene(event.pos())
            # Update the crop rectangle
            x1, y1 = self.crop_start_pos.x(), self.crop_start_pos.y()
            x2, y2 = current_pos.x(), current_pos.y()

            # Ensure proper rectangle (top-left to bottom-right)
            left = min(x1, x2)
            top = min(y1, y2)
            width = abs(x2 - x1)
            height = abs(y2 - y1)
            r = QRectF(left, top, width, height)
            r = QRectF(left, top, width, height)
            self.active_crop_rect.setRect(r)
            
        # Handle moving lasso
        if self.moving_lasso and self.last_mouse_pos:
            delta = event.pos() - self.last_mouse_pos
            self.moving_lasso.move(self.moving_lasso.pos() + delta)
            self.last_mouse_pos = event.pos()
            event.accept()

        # Store current mouse position for polygon preview
        if self.current_polygon and len(self.current_polygon.points) > 0:
            # Update temp point in scene coordinates relative to the image
            scene_pos = self.mapToScene(event.pos())
            image_pos = self.pixmap_item.mapFromScene(scene_pos)
            polygon_pos = self.pixmap_item.mapToScene(image_pos)
            self.current_polygon.set_temp_point(polygon_pos)

        # # Handle pixel info display
        # Determine reference item for coordinates and bounds
        reference_item = self.pixmap_item
        has_main_pixmap = self.pixmap_item is not None and not self.pixmap_item.pixmap().isNull()
        
        # If main pixmap is invalid but we have view layers, use the first view layer as reference
        if not has_main_pixmap and self.view_pixmaps:
            reference_item = self.view_pixmaps[0]

        if reference_item:
            scene_pos = self.mapToScene(event.pos())
            
            # Map scene position to image coordinates relative to the reference item
            # QGraphicsItem.mapFromScene works for both QGraphicsPixmapItem and pg.ImageItem
            image_pos = reference_item.mapFromScene(scene_pos)

            x = int(image_pos.x())
            y = int(image_pos.y())
            
            # Determine dimensions of the reference item
            width = 0
            height = 0
            
            if isinstance(reference_item, QGraphicsPixmapItem) and not reference_item.pixmap().isNull():
                pixmap = reference_item.pixmap()
                width = pixmap.width()
                height = pixmap.height()
            elif isinstance(reference_item, pg.ImageItem):
                # pg.ImageItem stores image data in .image
                if reference_item.image is not None:
                    # Shape is usually (h, w) or (w, h) depending on axis order, 
                    # but ImageItem exposes width() and height() methods usually
                    width = reference_item.width()
                    height = reference_item.height()

            # Check bounds
            if width > 0 and height > 0 and 0 <= x < width and 0 <= y < height:
                
                # Create a 1x1 QImage to render the pixel onto
                img = QImage(1, 1, QImage.Format.Format_ARGB32_Premultiplied)
                img.fill(Qt.GlobalColor.black)  # Fill with transparent background

                # Create a QPainter to draw on the QImage
                painter = QPainter(img)

                # Render just the 1x1 pixel area from the scene
                # The target is the full 1x1 image (0,0,1,1)
                # The source is the 1x1 rect in scene coordinates
                scene = self.get_scene()
                scene.render(painter, QRectF(0, 0, 1, 1), QRectF(scene_pos.x(), scene_pos.y(), 1, 1))

                # End the painter
                painter.end()

                color = QColor(img.pixel(0, 0))
                r, g, b = color.red(), color.green(), color.blue()

                global_pos = self.mapToGlobal(event.pos())

                # Get layer values if available
                if self.enc and self.enc.tool_bar.tabButtonGroup.checkedId() != 0:
                    layers = self.enc.view_tab.get_layer_values_at(x, y)
                else:
                    layers = None

                combined_layers = None  # added this so we don't get reference error

                if layers:
                    layers = [f"{layer}: {value}\n" for layer, value in layers]
                    combined_layers = "".join(layers)[:-1]
                    QToolTip.showText(
                        global_pos, combined_layers, self, self.rect(), TOOLTIP_PERSIST_MS
                    )
                else:
                    raw_intensity_str = f"R: {r}, G: {g}, B: {b}"
                    try:
                        from controller import Controller
                        ctrl = Controller.get()
                        if self == self.enc.canvas and ctrl.model_canvas.image_wrapper is not None and ctrl.model_canvas.image_wrapper.data.size > 0:
                            val = ctrl.model_canvas.image_wrapper.data[y, x]
                            raw_intensity_str = f"Intensity: {val}"
                        elif hasattr(self.enc, "small_view") and self == self.enc.small_view and ctrl.model_reference_canvas.image_wrapper is not None and ctrl.model_reference_canvas.image_wrapper.data.size > 0:
                            val = ctrl.model_reference_canvas.image_wrapper.data[y, x]
                            raw_intensity_str = f"Intensity: {val}"
                    except Exception:
                        pass
                        
                    QToolTip.showText(
                        global_pos,
                        raw_intensity_str,
                        self,
                        self.rect(),
                        TOOLTIP_PERSIST_MS,
                    )

                # Update position display in main window
                if combined_layers:
                    combined_layers = combined_layers.replace("\n", ", ")
                    combined_layers += ";"
                    self.enc.update_mouse_position_label(
                        f"{combined_layers} X: {x}, Y: {y}"
                    )
                else:
                    self.enc.update_mouse_position_label(
                        f"{raw_intensity_str} X: {x}, Y: {y}"
                    )

                # Highlight the pixel under cursor
                self.highlight_pixel(x, y)
                # if self.reference_view:
                #     self.reference_view.highlight_pixel(x, y)
            else:
                self.enc.update_mouse_position_label("")
                QToolTip.hideText()
                # Hide pixel highlight when outside image bounds
                self.hide_pixel_highlight()
                # if self.reference_view:
                # self.reference_view.hide_pixel_highlight()
        # Handle rubber band updates for old crop system
        if (
            not self.is_empty()
            and self.begin_crop
            and self.rubber_band
            and not self.crop_mode
        ):
            if self.origin is None:
                self.origin = event.pos()
                self.update_starting_position(event)
            self.rubber_band.setGeometry(QRect(self.origin, event.pos()))

        if (
            self.select in ("rect", "circle")
            and self.rubber_bands
            and self.origin is not None
        ):
            if self.select == "circle":
                center = self.origin
                corner = event.pos()
                size = (
                    max(abs(center.x() - corner.x()), abs(center.y() - corner.y())) * 2
                )
                self.rubber_bands[-1].setGeometry(
                    QRect(center.x() - size // 2, center.y() - size // 2, size, size)
                )
            else:
                self.rubber_bands[-1].setGeometry(
                    QRect(self.origin, event.pos()).normalized()
                )

    # pylint: disable=invalid-name
    def mouseReleaseEvent(self, event: QMouseEvent | None):
        """Handle mouse release."""
        if event is None or self.is_empty():
            return

        super().mouseReleaseEvent(event)

        # Handle moving lasso release
        if self.moving_lasso:
            # Update analysis with new position
            scene_pos = self.mapToScene(self.moving_lasso.geometry().topLeft())
            # For CircleLasso, topLeft might not be start_pos in the same sense, but region calc depends on type
            # We need to reconstruct the region defined by the lasso geometry
            
            # Determine region type
            region_type = "rect"
            if isinstance(self.moving_lasso, CircleLasso):
                region_type = "circle"
            # PolyLasso is not in rubber_bands, so not handled here
            
            # Helper to get image pos from view pos
            def get_img_pos(view_pos):
                scene_p = self.mapToScene(view_pos)
                return self.pixmap_item.mapFromScene(scene_p)
            
            lasso_rect = self.moving_lasso.geometry()
            
            if region_type == "rect":
                p1 = get_img_pos(lasso_rect.topLeft())
                p2 = get_img_pos(lasso_rect.bottomRight())
                x1, y1 = p1.x(), p1.y()
                x2, y2 = p2.x(), p2.y()
                image_rect = (
                    region_type,
                    (min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2))
                )
            elif region_type == "circle":
                # For circle, we stored (center_x, center_y, edge_x, edge_y) or similar?
                # Actually existing code uses (start_x, start_y, end_x, end_y) where start is top-left of rect?
                # No, look at create_rubber_band: center = starting_x, starting_y.
                # In mouseReleaseEvent (original):
                # image_pos dist to initial left click pos is radius.
                # initial pos is self.select_start_pos.
                
                # We need to emulate this. The CircleLasso widget is a square/rect bounding the circle.
                # Center of circle = Center of widget.
                # Radius = width / 2.
                
                center_view = lasso_rect.center()
                center_img = get_img_pos(center_view)
                
                # We need a 2nd point to define radius. (center_x + r, center_y)
                radius_view = lasso_rect.width() / 2
                # Assuming isotropic scaling (zoom uniform), we can map a point at radius distance.
                # But safer to map edge point.
                edge_view = QPoint(lasso_rect.right(), lasso_rect.center().y())
                edge_img = get_img_pos(edge_view)
                
                image_rect = (
                    region_type,
                    (center_img.x(), center_img.y(), edge_img.x(), edge_img.y())
                )

            # Update the analysis
            self.enc.analysis_tab.update_roi_region(self.moving_lasso, image_rect)

            self.moving_lasso = None
            self.last_mouse_pos = None
            return

        self.rubber_band_positions = []
        # Handle crop mode mouse release
        if self.crop_mode and self.active_crop_rect and self.is_resizing:
            # Don't auto-confirm, let user press Enter or Escape
            self.initial_crop = True
            self.initial_crop_rect = self.active_crop_rect.rect()
            self.get_scene().removeItem(self.active_crop_rect)

            # pylint: disable=too-many-function-args
            self.active_crop_rect = ResizableRect(
                self.initial_crop_rect.x(), self.initial_crop_rect.y(),
                self.initial_crop_rect.width(), self.initial_crop_rect.height(),
            )
            self.active_crop_rect.setZValue(10)
            
            self.is_resizing = False

            self.get_scene().addItem(self.active_crop_rect)
            return
        if not self.rubber_bands:
            return

        if event.button() == Qt.MouseButton.LeftButton:
            rubberband = self.rubber_band if self.begin_crop else self.rubber_bands[-1]

            if self.select:
                self.origin = None
                assert self.select_start_pos is not None
                if self.select in ("rect", "circle"):
                    assert rubberband is not None
                    scene_pos = self.mapToScene(event.pos())
                    image_pos = self.pixmap_item.mapFromScene(scene_pos)
                    # if select is circle, then image_pos dist to initial left click pos is radius
                    # if select is rectangle, then initial is top let and image_pos is bottom right
                    # initial pos is self.select_start_pos
                    logger.debug(image_pos)
                    image_rect = (
                        self.select,
                        (
                            self.select_start_pos.x(),
                            self.select_start_pos.y(),
                            image_pos.x(),
                            image_pos.y(),
                        ),
                    )
                    # if failed to analyze region, then remove rubber band
                    if not self.enc.analysis_tab.analyze_region(rubberband, image_rect):
                        self.rubber_bands.remove(self.rubber_band)
                        self.rubber_band_colors.pop()
                        if rubberband:
                            rubberband.deleteLater()
                    self.select = False
                    return

    def leaveEvent(self, event):
        """Cancel ROI dragging when pointer leaves the view."""
        self._cancel_rubber_band_drag()
        QToolTip.hideText()
        super().leaveEvent(event)

    # pylint: disable=invalid-name
    def scrollContentsBy(self, dx, dy):
        """Keep ROI overlays anchored to image coordinates while panning."""
        rubber_band_positions = self._capture_rubber_band_scene_positions()
        super().scrollContentsBy(dx, dy)
        self._restore_rubber_band_scene_positions(rubber_band_positions)

    def load_channels(self, np_channels):
        """Load channel data"""
        self.np_channels = np_channels
        if self.pixmap_item is not None:
            self._center_image()

    def highlight_pixel(self, x, y):
        """Highlight the pixel at the given coordinates"""
        if self.pixmap_item is None:
            return

        # Create a small rectangle to highlight the pixel
        # Convert pixel coordinates to scene coordinates
        pixel_rect = QRectF(x, y, 1, 1)  # 1x1 pixel
        scene_rect = self.pixmap_item.mapRectToScene(pixel_rect)

        # Reuse existing item if possible
        if hasattr(self, "pixel_highlight") and self.pixel_highlight:
            # Ensure it's in the scene
            if self.pixel_highlight.scene() != self.get_scene():
                self.get_scene().addItem(self.pixel_highlight)
            
            self.pixel_highlight.setRect(scene_rect)
            self.pixel_highlight.show()
        else:
            # Create highlight rectangle
            self.pixel_highlight = QGraphicsRectItem(scene_rect)

            # Style the highlight (you can customize this)
            pen = QPen(QColor(255, 255, 0, 180))  # Yellow with transparency
            pen.setWidth(0)  # Cosmetic pen (always 1 pixel wide regardless of zoom)
            pen.setCosmetic(True)
            self.pixel_highlight.setPen(pen)

            # Optional: Add a semi-transparent fill
            brush = QBrush(QColor(255, 255, 0, 50))  # Light yellow fill
            self.pixel_highlight.setBrush(brush)

            # Add to scene
            self.get_scene().addItem(self.pixel_highlight)

    def hide_pixel_highlight(self):
        """Hide the pixel highlight"""
        if hasattr(self, "pixel_highlight") and self.pixel_highlight:
            self.pixel_highlight.hide()

    def update_roi_from_lasso(self, rubberband):
        """Update ROI region based on lasso position"""
        if not rubberband or not self.pixmap_item:
            return

        # Get geometry in view coordinates
        rect = rubberband.geometry()
        top_left = rect.topLeft()
        bottom_right = rect.bottomRight()

        # Convert to scene coords
        top_left_scene = self.mapToScene(top_left)
        bottom_right_scene = self.mapToScene(bottom_right)

        # Convert to image coords
        image_top_left = self.pixmap_item.mapFromScene(top_left_scene)
        image_bottom_right = self.pixmap_item.mapFromScene(bottom_right_scene)

        # Determine shape type
        shape_type = "rect"
        if isinstance(rubberband, CircleLasso):
            shape_type = "circle"
            center_x = (image_top_left.x() + image_bottom_right.x()) / 2
            center_y = (image_top_left.y() + image_bottom_right.y()) / 2
            radius = (image_bottom_right.x() - image_top_left.x()) / 2
            x2 = center_x + radius
            y2 = center_y
            region_coords = (center_x, center_y, x2, y2)
        else:
            shape_type = "rect"
            region_coords = (
                image_top_left.x(),
                image_top_left.y(),
                image_bottom_right.x(),
                image_bottom_right.y(),
            )

        region = (shape_type, region_coords)
        
        # Call into analysis tab to update
        self.enc.analysis_tab.update_roi_region(rubberband, region)
