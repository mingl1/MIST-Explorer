'''Main Class to handle display of images'''

from PyQt6.QtWidgets import (
    QToolTip, QGraphicsView, QRubberBand, QGraphicsScene, QGraphicsPixmapItem,
    QGraphicsItem, QGraphicsRectItem, QGraphicsOpacityEffect, QGraphicsItemGroup,
    QGraphicsSimpleTextItem, QApplication, QMainWindow, QWidget, QHBoxLayout, QPushButton, QLabel,
    QMenu, QMessageBox

)
from PyQt6.QtGui import (
    QDragEnterEvent, QDropEvent, QPixmap, QDragMoveEvent, QMouseEvent, QCursor,
    QImage, QPalette, QPainter, QBrush, QColor, QPen, QIcon, QAction

)
from PyQt6.QtCore import (
    Qt, QRect, QSize, QPoint, pyqtSignal, pyqtSlot, QPointF,
    QPropertyAnimation, QEasingCurve, QRectF, QSizeF,
)

import numpy as np
import cv2
import pandas as pd
import random
import traceback
import utils

from core.Worker import Worker
import ui.Dialogs as Dialogs

from ui.lassos.CircleLasso import CircleLasso
from ui.lassos.RectLasso import RectLasso
from ui.lassos.PolyLasso import PolyLasso


class ArrowItem(QGraphicsPixmapItem):
    """Arrow with hover effect"""
    def __init__(self, pixmap, position):
        super().__init__(pixmap)
        rect = QRectF(self.boundingRect())
        self.bg_rect = QGraphicsRectItem(rect, parent=self)
        self.bg_rect.setBrush(QBrush(QColor(255, 255, 255, 100)))  
        self.bg_rect.setZValue(1)  # Make sure it goes behind the arrows
        self.base_opacity = 0.4
        self.hover_opacity = 1  # Darker when hovered
        self.setPos(position)
        self.applyHoverEffect()

    def applyHoverEffect(self):
        self.effect = QGraphicsOpacityEffect()
        self.effect.setOpacity(self.base_opacity)
        self.setGraphicsEffect(self.effect)
        self.bg_rect.setGraphicsEffect(self.effect)

        self.fade_in = QPropertyAnimation(self.effect, b"opacity")
        self.fade_in.setDuration(300)
        self.fade_in.setEndValue(self.hover_opacity)
        self.fade_in.setEasingCurve(QEasingCurve.Type.InOutQuad)

        self.fade_out = QPropertyAnimation(self.effect, b"opacity")
        self.fade_out.setDuration(300)
        self.fade_out.setEndValue(self.base_opacity)
        self.fade_out.setEasingCurve(QEasingCurve.Type.InOutQuad)

        self.setAcceptHoverEvents(True)

    def hoverEnterEvent(self, event):
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.fade_out.stop() 
        self.fade_in.start()  

    def hoverLeaveEvent(self, event):
        self.fade_in.stop()   
        self.fade_out.start() 


class ReferenceGraphicsViewUI(QGraphicsView):
    """Reference view for displaying images with navigation arrows"""
    imageDropped = pyqtSignal(str)  
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.parent = parent
        self.zoom = 1
        self.left_arrow = None
        self.right_arrow = None
        self.pixmapItem = None
        self.current_index = 1
        self.np_channels = {}
        self.pixmap = None
        
        self.init_ui()
        
    def init_ui(self):
        self.setMinimumSize(QSize(300, 300))
        self.setScene(QGraphicsScene(self))
        self.setStyleSheet("QGraphicsView { border: 1px solid black; }")
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)

    def is_empty(self) -> bool:
        return self.pixmapItem is None
        
    def load_channels(self, np_channels):
        self.np_channels = np_channels

    def slideshow(self):

        # Create navigation arrows
        self.right_arrow = ArrowItem(
            QPixmap("assets/icons/right-arrow.png").scaled(
                25, 25, 
                aspectRatioMode=Qt.AspectRatioMode.KeepAspectRatio, 
                transformMode=Qt.TransformationMode.SmoothTransformation
            ),
            QPointF(250, 275)
        )
        
        self.left_arrow = ArrowItem(
            QPixmap("assets/icons/left-arrow.png").scaled(
                25, 25, 
                aspectRatioMode=Qt.AspectRatioMode.KeepAspectRatio, 
                transformMode=Qt.TransformationMode.SmoothTransformation
            ), 
            QPointF(10, 275)
        )

        self.scene().addItem(self.right_arrow)
        self.scene().addItem(self.left_arrow)

        self.left_arrow.setAcceptedMouseButtons(Qt.MouseButton.LeftButton)
        self.right_arrow.setAcceptedMouseButtons(Qt.MouseButton.LeftButton)

        self.fitInView(self.sceneRect(), Qt.AspectRatioMode.KeepAspectRatio)

    def mouseDoubleClickEvent(self, event):
        if not self.is_empty():
            self.fitInView(self.sceneRect(), Qt.AspectRatioMode.KeepAspectRatio)

    def wheelEvent(self, event):
        if self.pixmapItem is not None:
            zooming_out = event.angleDelta().y() > 0

            # Prevent excessive zooming in either direction
            if self.zoom > 1.1**90 and zooming_out:  # Max zoom out
                return
            
            if self.zoom < 1/(1.1**2) and not zooming_out:  # Max zoom in
                return

            zoom_factor = 1.1 if zooming_out else 0.9
            self.zoom *= zoom_factor
            self.scale(zoom_factor, zoom_factor)
        else:
            super().wheelEvent(event)

    def dragEnterEvent(self, event: QDragEnterEvent):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dragMoveEvent(self, event: QDragMoveEvent):
        event.acceptProposedAction()

    def dropEvent(self, event: QDropEvent):
        if event.mimeData().hasUrls():
            for url in event.mimeData().urls():
                file_path = url.toLocalFile()
                if file_path is not None:
                    self.imageDropped.emit(file_path)
            event.acceptProposedAction()

    def mousePressEvent(self, event):
        if not self.is_empty() and len(self.np_channels) > 1:
            print("mouse press")
            scene_pos = self.mapToScene(event.pos())
            if self.left_arrow and self.left_arrow.contains(self.left_arrow.mapFromScene(scene_pos)):
                self.prev_slide()
            elif self.right_arrow and self.right_arrow.contains(self.right_arrow.mapFromScene(scene_pos)):
                self.next_slide()
        super().mousePressEvent(event)

    def prev_slide(self):
        """Show previous slide"""
        if self.current_index > 1:
            self.current_index -= 1
            self.update_slide()

    def next_slide(self):
        """Show next slide"""
        if self.current_index < len(self.np_channels.keys()):
            self.current_index += 1
            self.update_slide()

    def update_slide(self):
        """Update displayed image"""
        self.pixmap = QPixmap(utils.numpy_to_qimage(self.np_channels[f"Channel {self.current_index}"].data))
        self.pixmapItem.setPixmap(self.pixmap)
        item_rect = self.pixmapItem.boundingRect()
        self.setSceneRect(item_rect)
        self.fitInView(self.sceneRect(), Qt.AspectRatioMode.KeepAspectRatio)

    def display(self, pixmap: QPixmap, is_layer: bool):
        # self.scene().clear()
        # reset
        self.current_index = 1

        # if not hasattr(self, "right_arrow"):
        self.slideshow()  # Initialize arrows

        self.pixmap = pixmap
        if not hasattr(self, 'pixmapItem') or self.pixmapItem is None:
            self.pixmapItem = QGraphicsPixmapItem(self.pixmap)
            self.scene().addItem(self.pixmapItem)

        else:
            print("setting pixmap")
            self.pixmapItem.setPixmap(self.pixmap)
        
        print("is layer: ", is_layer)
        
        # if is_layer:

        print("has np channels")
        # Scale arrows appropriately
        rw = int(self.scene().width() / 10.6)
        rh = int(self.scene().height() / 10.6)
        
        self.right_arrow.bg_rect.setRect(0, 0, rw, rh)
        self.left_arrow.bg_rect.setRect(0, 0, rw, rh)

        self.right_arrow.setPixmap(QPixmap("assets/icons/right-arrow.png").scaled(rw, rh))
        self.left_arrow.setPixmap(QPixmap("assets/icons/left-arrow.png").scaled(rw, rh))

        scene_height = self.scene().height()
        scene_width = self.scene().width()
        self.right_arrow.setToolTip("Next")
        self.left_arrow.setToolTip("Previous")

        # Position the arrows
        right_arrow_pos_x = int(scene_width)
        left_arrow_pos_x = 0
        arrow_pos_y = int(scene_height / 2)

        self.right_arrow.setPos(self.mapToScene(right_arrow_pos_x, arrow_pos_y))
        self.left_arrow.setPos(self.mapToScene(left_arrow_pos_x, arrow_pos_y))
    
        # Set Z-order
        self.pixmapItem.setZValue(0)
        self.right_arrow.setZValue(1)
        self.left_arrow.setZValue(1)

        # Setup scene
        item_rect = self.pixmapItem.boundingRect()
        self.setSceneRect(item_rect)
        self.fitInView(self.sceneRect(), Qt.AspectRatioMode.KeepAspectRatio)


    def resizeEvent(self, event):
        super().resizeEvent(event)

        # self.move(int(self.parent.width() - 2*self.parent.width()), 10)
          
class ImageGraphicsViewUI(QGraphicsView):

    """Main image view with support for selection, cropping and other operations"""
    imageDropped = pyqtSignal(str)
    showCrop = pyqtSignal(QRect)
    requestFlipHorizontal = pyqtSignal()  # Signal to request horizontal flip
    requestFlipVertical = pyqtSignal()  # Signal to request vertical flip
    
    def __init__(self, parent=None, enc=None):
        super().__init__(parent)
        self.enc = enc
        self.setupUI()
        
        # Initialize variables
        self.crop_drawing_mode = False  # True when user is drawing initial crop
        self.crop_draw_origin = None    # Starting point of the drawn crop
        self.temp_crop_rect = None      # Temporary visual set draw
        self.pixmapItem = None
        self.rubberBand = None
        self.rubberBands = []
        self.rubberBandColors = []
        self.begin_crop = False
        self.origin = None
        self.crop_cursor = QCursor(Qt.CursorShape.CrossCursor)
        self.select = False
        self.zoom = 1
        self.polygons = []
        self.current_polygon = None
        self.rubber_band_positions = []

        # NEW: for resizable crop
        self.resizable_crop_rect = None
        
        # Setup interaction
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.setMouseTracking(True)

    def flip_horizontal(self):
        """Flip the image horizontally"""
        if self.pixmapItem:
            img = self.pixmapItem.pixmap().toImage()
            flipped_img = img.mirrored(horizontal=True, vertical=False)
            flipped_pixmap = QPixmap.fromImage(flipped_img)
            self.pixmapItem.setPixmap(flipped_pixmap)
            self.scene().update()
            # Emit signal to update the underlying data model
            self.requestFlipHorizontal.emit()

    def flip_vertical(self):
        """Flip the image vertically"""
        if self.pixmapItem:
            img = self.pixmapItem.pixmap().toImage()
            flipped_img = img.mirrored(horizontal=False, vertical=True)
            flipped_pixmap = QPixmap.fromImage(flipped_img)
            self.pixmapItem.setPixmap(flipped_pixmap)
            self.scene().update()
            # Emit signal to update the underlying data model
            self.requestFlipVertical.emit()

    def setupUI(self):
        self.setMinimumSize(QSize(600, 600))
        self.setObjectName("canvas")
        self.setAcceptDrops(True)
        self.setScene(QGraphicsScene(self))
        self.setSceneRect(0, 0, 800, 600)
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        
        # Create floating selection buttons
        self.create_floating_buttons()

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
        label.setStyleSheet("QLabel { color: white; padding: 5px; border-radius: 3px; }")
        button_layout.addWidget(label)
        
        # Create the selection buttons
        self.rect_button = QPushButton()
        self.circle_button = QPushButton()
        self.poly_button = QPushButton()
        
        # Set button sizes and styles
        for button in [self.rect_button, self.circle_button, self.poly_button]:
            button.setFixedSize(40, 40)
            button.setStyleSheet("""
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
            """)
        
        # Set icons for the buttons
        self.rect_button.setIcon(QIcon("assets/icons/square.png"))
        self.circle_button.setIcon(QIcon("assets/icons/circle.png"))
        self.poly_button.setIcon(QIcon("assets/icons/poly.png"))
        
        # Connect button signals to selection modes
        self.rect_button.clicked.connect(lambda: self.set_selection_mode("rect"))
        self.circle_button.clicked.connect(lambda: self.set_selection_mode("circle"))
        self.poly_button.clicked.connect(lambda: self.enc.poly_select())
        
        # Add buttons to layout
        button_layout.addWidget(self.rect_button)
        button_layout.addWidget(self.circle_button)
        button_layout.addWidget(self.poly_button)
        
        # Position the container at the top-right of the view
        self.update_floating_buttons_position()

    def update_floating_buttons_position(self):
        """Update the position of the floating buttons"""
        if hasattr(self, 'floating_container'):
            # Position at the top-right of the view with some padding
            self.floating_container.move(
                self.width() - self.floating_container.width() - 20,
                10
            )
    
    def resizeEvent(self, event):
        """Handle resize events to update floating buttons position"""
        super().resizeEvent(event)
        self.update_floating_buttons_position()

    def set_selection_mode(self, mode):

        print("set_selection_mode", mode)
        """Set the current selection mode"""
        # Reset all modes
        self.select = False
        self.current_polygon = None
        
        # Set the new mode
        if mode == "rect":
            self.select = "rect"
            # self.setCursor(Qt.CursorShape.CrossCursor)
            self.enc.select();
        elif mode == "circle":
            # self.setCursor(Qt.CursorShape.CrossCursor)
            self.enc.circle_select();
        elif mode == "poly":
            self.select = "poly"
            # self.setCursor(Qt.CursorShape.CrossCursor)
            self.enc.poly_select();
        

    def isEmpty(self) -> bool:
        return self.pixmapItem is None
    
    def mouseDoubleClickEvent(self, event):
        if self.isEmpty():
            return
        self.__centerImage(self.pixmapItem)

    
    def updateCanvas(self, pixmap: QPixmap, reset=False, crop=False):
        """Updates canvas when current image is operated on"""
        if self.pixmapItem:
            print("updating canvas and setting pixmap")
            self.pixmapItem.setPixmap(pixmap)
            # self.__centerImage(self.pixmapItem)
            self.pixmapItem.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsMovable)

        
    def addNewImage(self, pixmapItem: QGraphicsPixmapItem):
        """Update the pixmap of the existing image or add a new one"""
        if not hasattr(self, 'pixmapItem') or self.pixmapItem is None:
            # If no pixmapItem exists, add it to the scene
            self.pixmapItem = pixmapItem
            self.pixmapItem.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsMovable)
            self.scene().addItem(self.pixmapItem)
            self.__centerImage(self.pixmapItem)
        else:
            # Update the pixmap of the existing item
            self.pixmapItem.setPixmap(pixmapItem.pixmap())

    def __centerImage(self, pixmapItem):
        item_rect = self.pixmapItem.boundingRect()
        self.setSceneRect(item_rect)
        self.fitInView(pixmapItem, Qt.AspectRatioMode.KeepAspectRatio)
        self.centerOn(pixmapItem)
        
    def dragEnterEvent(self, event: QDragEnterEvent):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dragMoveEvent(self, event: QDragMoveEvent):
        event.acceptProposedAction()

    def dropEvent(self, event: QDropEvent):
        if event.mimeData().hasUrls():
            for url in event.mimeData().urls():
                file_path = url.toLocalFile()
                if file_path is not None:
                    self.imageDropped.emit(file_path)
            event.acceptProposedAction()

    def wheelEvent(self, event):
        zooming_out = event.angleDelta().y() > 0

        # Prevent excessive zooming in either direction
        if self.zoom > 1.1**90 and zooming_out:  # Max zoom out
            return
        
        if self.zoom < 1/(1.1**2) and not zooming_out:  # Max zoom in
            return

        zoom_factor = 1.1 if zooming_out else 0.9
        self.zoom *= zoom_factor

        # Store rubber band positions before zooming
        if not self.rubber_band_positions:
            self.rubber_band_positions = []
            for rubber_band in self.rubberBands:
                rubber_band_geometry = rubber_band.geometry()
                top_left_scene = self.mapToScene(rubber_band_geometry.topLeft())
                bottom_right_scene = self.mapToScene(rubber_band_geometry.bottomRight())
                self.rubber_band_positions.append((rubber_band, top_left_scene, bottom_right_scene))

        # Perform the zoom
        self.scale(zoom_factor, zoom_factor)

        # Update rubber band positions after zooming
        for rubber_band, top_left_scene, bottom_right_scene in self.rubber_band_positions:
            new_top_left_view = self.mapFromScene(top_left_scene)
            new_bottom_right_view = self.mapFromScene(bottom_right_scene)
            new_rect = QRect(new_top_left_view, new_bottom_right_view)
            rubber_band.setGeometry(new_rect)

        # Force redraw to update polygon positions
        self.viewport().update()

    def create_rubber_band(self, rubber_band_class, shape, x, y, parent, origin):
        """Create a rubber band of the specified class"""
        rubber_band = rubber_band_class(shape, x, y, self)
        rubber_band.setGeometry(QRect(origin, QSize()))
        rubber_band.show()
        return rubber_band

    def update_starting_position(self, event):
        """Update the starting position for rubber band operations"""
        scene_pos = self.mapToScene(event.pos())
        self.image_pos = self.pixmapItem.mapFromScene(scene_pos)
        self.starting_x = int(self.image_pos.x())
        self.starting_y = int(self.image_pos.y())

    def mousePressEvent(self, event: QMouseEvent):
        if self.isEmpty():
            return

        if self.crop_drawing_mode and event.button() == Qt.MouseButton.LeftButton:
            self.crop_draw_origin = self.mapToScene(event.pos())

            if self.temp_crop_rect:
                self.scene().removeItem(self.temp_crop_rect)

            pen = QPen(QColor(0, 255, 0), 2, Qt.PenStyle.SolidLine)  # Brighter green solid
            pen.setCosmetic(True)  # Constant width regardless of zoom
            self.temp_crop_rect = QGraphicsRectItem()
            self.temp_crop_rect.setPen(pen)
            self.temp_crop_rect.setBrush(QBrush(QColor(0, 255, 0, 50)))  # translucent green fill
            self.scene().addItem(self.temp_crop_rect)
            return  # Don't pass to super or anything else if in drawing mode

        # Your existing selection/crop logic continues below
        if event.button() == Qt.MouseButton.LeftButton:
            if self.begin_crop or self.select:
                self.origin = event.pos()
                self.update_starting_position(event)

                if self.begin_crop:
                    if not self.rubberBand:
                        self.rubberBand = RectLasso(self)

                elif self.select == "rect":
                    self.rubberBand = RectLasso(self)
                    self.rubberBands.append(self.rubberBand)
                    self.rubberBandColors.append(self.rubberBand.color)
                    self.rubberBand.setGeometry(QRect(self.origin, QSize()))
                    self.rubberBand.show()
                    return
                elif self.select == "circle":
                    self.center = QPoint(self.starting_x, self.starting_y)
                    self.rubberBand = CircleLasso(self)
                    self.rubberBands.append(self.rubberBand)
                    self.rubberBandColors.append(self.rubberBand.color)
                    self.rubberBand.setGeometry(QRect(self.origin, QSize()))
                    self.rubberBand.show()
                    return
                elif self.select == "poly":
                    if not self.current_polygon:
                        self.current_polygon = PolyLasso(self.pixmapItem)
                        self.scene().addItem(self.current_polygon)
                        self.setMouseTracking(True)

                    scene_pos = self.mapToScene(event.pos())
                    image_pos = self.pixmapItem.mapFromScene(scene_pos)
                    polygon_pos = self.pixmapItem.mapToScene(image_pos)
                    self.current_polygon.add_point(polygon_pos, image_pos)
                    return

                if self.begin_crop:
                    self.rubberBands.append(self.rubberBand)
                    self.rubberBandColors.append(self.rubberBand.color)
                    self.rubberBand.setGeometry(QRect(self.origin, QSize()))
                    self.rubberBand.show()
                    return

        super().mousePressEvent(event)

        for r in self.rubberBands:
            r.mousePressEvent(event)


    def keyPressEvent(self, event):
        if event.key() == Qt.Key.Key_Return and self.resizable_crop_rect:
            rect = self.resizable_crop_rect.rect()
            top_left = self.resizable_crop_rect.mapToScene(rect.topLeft())
            bottom_right = self.resizable_crop_rect.mapToScene(rect.bottomRight())

            # Map to image coordinates
            image_top_left = self.pixmapItem.mapFromScene(top_left)
            image_bottom_right = self.pixmapItem.mapFromScene(bottom_right)

            # Clamp to image bounds
            image_rect = QRect(
                max(0, int(image_top_left.x())),
                max(0, int(image_top_left.y())),
                int(image_bottom_right.x() - image_top_left.x()),
                int(image_bottom_right.y() - image_top_left.y())
            ).normalized()

            # Make sure rect is within image bounds
            image = self.pixmapItem.pixmap().toImage()
            image_width, image_height = image.width(), image.height()
            image_rect = image_rect.intersected(QRect(0, 0, image_width, image_height))

            if image_rect.isEmpty():
                print("[✘] Invalid crop area — outside image bounds.")
                return

            # Emit the crop
            self.showCrop.emit(image_rect)

            # Clean up
            self.scene().removeItem(self.resizable_crop_rect)
            self.resizable_crop_rect = None
            self.unsetCursor()
            self.begin_crop = False
            return

        super().keyPressEvent(event)


        # Allow other keys to propagate
        super().keyPressEvent(event)


        # Allow other keys to propagate
        super().keyPressEvent(event)

    def paintEvent(self, event):
        super().paintEvent(event)
        
        # Create a painter for the viewport
        painter = QPainter(self.viewport())
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        
        # Draw rubber bands if needed
        for rubber_band in self.rubberBands:
            if rubber_band.isVisible():
                # Trigger the rubber band's own paintEvent
                rubber_band.paintEvent(event)
        
        painter.end()

    def mouseMoveEvent(self, event: QMouseEvent):
        if self.crop_drawing_mode and self.crop_draw_origin and self.temp_crop_rect:
            current_pos = self.mapToScene(event.pos())
            rect = QRectF(self.crop_draw_origin, current_pos).normalized()
            self.temp_crop_rect.setRect(rect)
            return

        super().mouseMoveEvent(event)

        if self.current_polygon and len(self.current_polygon.points) > 0:
            scene_pos = self.mapToScene(event.pos())
            image_pos = self.pixmapItem.mapFromScene(scene_pos)
            polygon_pos = self.pixmapItem.mapToScene(image_pos)
            self.current_polygon.set_temp_point(polygon_pos)

        if self.pixmapItem:
            scene_pos = self.mapToScene(event.pos())
            image_pos = self.pixmapItem.mapFromScene(scene_pos)
            x = int(image_pos.x())
            y = int(image_pos.y())
            img = self.pixmapItem.pixmap().toImage()

            if 0 <= x < img.width() and 0 <= y < img.height():
                color = QColor(img.pixel(x, y))
                r, g, b = color.red(), color.green(), color.blue()
                global_pos = self.mapToGlobal(event.pos())
                QToolTip.showText(global_pos, f"", self)

                layers = self.enc.view_tab.get_layer_values_at(x, y)
                combined_layers = None

                if layers:
                    layers = [f"{layer}: {value[0]}\n" for layer, value in layers]
                    combined_layers = ''.join(layers)[:-1]
                    QToolTip.showText(global_pos, combined_layers, self)
                else:
                    QToolTip.showText(global_pos, f"R: {r}, G: {g}, B: {b}", self)

                if combined_layers:
                    combined_layers = combined_layers.replace("\n", ", ")
                    combined_layers += ";"
                    self.enc.updateMousePositionLabel(f"{combined_layers} X: {x}, Y: {y}")
                else:
                    self.enc.updateMousePositionLabel(f"R: {r}, G: {g}, B: {b} X: {x}, Y: {y}")
            else:
                self.enc.updateMousePositionLabel(f"")

        if not self.isEmpty() and self.begin_crop and self.rubberBand:
            self.rubberBand.setGeometry(QRect(self.origin, event.pos()).normalized())

        if (self.select == "rect" or self.select == "circle") and self.rubberBands and self.origin is not None:
            if self.select == "circle":
                center = self.origin
                corner = event.pos()
                size = max(abs(center.x() - corner.x()), abs(center.y() - corner.y())) * 2
                self.rubberBands[-1].setGeometry(QRect(center.x() - size // 2, center.y() - size // 2, size, size))
            else:
                self.rubberBands[-1].setGeometry(QRect(self.origin, event.pos()).normalized())

        if not self.select:
            for r in self.rubberBands:
                r.mouseMoveEvent(event)


    def mouseReleaseEvent(self, event: QMouseEvent):
        if self.isEmpty():
            return

        if self.crop_drawing_mode and event.button() == Qt.MouseButton.LeftButton and self.crop_draw_origin:
            end_pos = self.mapToScene(event.pos())
            rect = QRectF(self.crop_draw_origin, end_pos).normalized()

            if rect.width() > 10 and rect.height() > 10:
                self.resizable_crop_rect = ResizableRect(rect.x(), rect.y(), rect.width(), rect.height())
                self.scene().addItem(self.resizable_crop_rect)
                self.resizable_crop_rect.setZValue(10)

            if self.temp_crop_rect:
                self.scene().removeItem(self.temp_crop_rect)
                self.temp_crop_rect = None

            self.crop_draw_origin = None
            self.crop_drawing_mode = False
            self.unsetCursor()
            return

        super().mouseReleaseEvent(event)
        self.rubber_band_positions = []

        if not self.rubberBands:
            return

        for r in self.rubberBands:
            r.mouseReleaseEvent(event)

        if event.button() == Qt.MouseButton.LeftButton:
            rubberband = self.rubberBand if self.begin_crop else self.rubberBands[-1]

            if self.begin_crop:
                rubberband.hide()
                selectedRect = rubberband.geometry()
                if selectedRect.isEmpty():
                    return

                scene_pos = self.mapToScene(event.pos())
                image_pos = self.pixmapItem.mapFromScene(scene_pos)

                self.image_rect = QRect(
                    int(self.starting_x),
                    int(self.starting_y),
                    int(image_pos.x() - self.starting_x),
                    int(image_pos.y() - self.starting_y)
                ).normalized()

                self.showCrop.emit(self.image_rect)

            if self.select:
                self.origin = None

                if self.select == "rect" or self.select == "circle":
                    scene_pos = self.mapToScene(event.pos())
                    image_pos = self.pixmapItem.mapFromScene(scene_pos)
                    image_rect = (self.select, (self.starting_x, self.starting_y, int(image_pos.x()), int(image_pos.y())))
                    self.enc.analysis_tab.analyze_region(rubberband, image_rect)
                    self.select = False
                    
                
    # def contextMenuEvent(self, event):
    #     # Create the menu
    #     menu = QMenu(self)

    #     # Add actions
    #     action1 = QAction("Option 1", self)
    #     action1.triggered.connect(lambda: self.show_message("Option 1 selected"))

    #     action2 = QAction("Option 2", self)
    #     action2.triggered.connect(lambda: self.show_message("Option 2 selected"))

    #     menu.addAction(action1)
    #     menu.addAction(action2)

    #     # Show the menu at the cursor position
    #     menu.exec(event.globalPos())

    # def show_message(self, message):
    #     QMessageBox.information(self, "Selection", message)


    def set_crop_status(self, status):
        """Enter and exit crop mode that allows drawing the initial crop box"""
        self.crop_drawing_mode = status
        self.select = False
        self.unsetCursor()

        if status:
            self.begin_crop = False  # disable legacy
            self.setCursor(Qt.CursorShape.CrossCursor)
            self.setFocus()
        else:
            self.crop_drawing_mode = False
            self.begin_crop = False
            self.unsetCursor()
            if self.resizable_crop_rect:
                self.scene().removeItem(self.resizable_crop_rect)
                self.resizable_crop_rect = None

            
        
    def loadChannels(self, np_channels):
        """Load channel data"""
        self.np_channels = np_channels
    
    def setCurrentChannel(self, channel_num: int) -> None:
        """Set the current channel to display"""
        self.currentChannelNum = channel_num
        
    def update_image(self, cmap=None):
        """Update the displayed image with current channel data"""
        if hasattr(self, 'np_channels') and self.np_channels and hasattr(self, 'currentChannelNum'):
            channel_key = f"Channel {self.currentChannelNum + 1}"
            if channel_key in self.np_channels:
                # Convert numpy data to QImage
                channel_data = self.np_channels[channel_key].data
                from utils import numpy_to_qimage
                from PyQt6.QtGui import QPixmap
                
                # Make sure channel data is in the right format
                if channel_data.dtype != np.uint8:
                    # Scale to 8-bit for display if needed
                    if channel_data.max() > 255:
                        channel_data = ((channel_data / channel_data.max()) * 255).astype(np.uint8)
                    else:
                        channel_data = channel_data.astype(np.uint8)
                
                q_image = numpy_to_qimage(channel_data)
                pixmap = QPixmap(q_image)
                
                # Update the pixmap
                if self.pixmapItem:
                    self.pixmapItem.setPixmap(pixmap)
                    
                return True
        return False
        
    def swap_channel(self, channel_num):
        """Switch to a different channel"""
        if hasattr(self, 'np_channels') and self.np_channels:
            self.setCurrentChannel(channel_num)
            return self.update_image()
        return False


class ResizeHandle(QGraphicsRectItem):
    def __init__(self, cursor_shape: Qt.CursorShape, parent=None):
        super().__init__(-4, -4, 8, 8, parent)  # Centered, big, clickable
        self.setBrush(QBrush(Qt.GlobalColor.white))
        self.setPen(QPen(Qt.GlobalColor.black, 2))
        self.setZValue(11)
        self.setAcceptHoverEvents(True)
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIgnoresTransformations)  # <-- Fixes size at all zoom levels
        self.cursor_shape = cursor_shape

    def hoverEnterEvent(self, event):
        QApplication.setOverrideCursor(QCursor(self.cursor_shape))

    def hoverLeaveEvent(self, event):
        QApplication.restoreOverrideCursor()


class ResizableRect(QGraphicsRectItem):
    def __init__(self, x, y, width, height, onCenter=False):
        if onCenter:
            super().__init__(-width / 2, -height / 2, width, height)
        else:
            super().__init__(0, 0, width, height)

        self.setPos(x, y)
        self.setFlags(
            QGraphicsRectItem.GraphicsItemFlag.ItemIsMovable |
            QGraphicsRectItem.GraphicsItemFlag.ItemIsSelectable |
            QGraphicsRectItem.GraphicsItemFlag.ItemIsFocusable
        )
        self.setAcceptHoverEvents(True)
        self.setPen(QPen(QBrush(Qt.GlobalColor.blue), 3, Qt.PenStyle.DotLine))
        self.selected_edge = None

        self.posItem = QGraphicsSimpleTextItem(f'{self.x()}, {self.y()}', parent=self)
        self.posItem.setPos(
            self.boundingRect().x(),
            self.boundingRect().y() - self.posItem.boundingRect().height()
        )

        # Create 8 resize handles with cursor + role
        self.handles = []
        self.handle_roles = [
            ('top_left', Qt.Edge.TopEdge | Qt.Edge.LeftEdge),
            ('top_center', Qt.Edge.TopEdge),
            ('top_right', Qt.Edge.TopEdge | Qt.Edge.RightEdge),
            ('mid_right', Qt.Edge.RightEdge),
            ('bottom_right', Qt.Edge.BottomEdge | Qt.Edge.RightEdge),
            ('bottom_center', Qt.Edge.BottomEdge),
            ('bottom_left', Qt.Edge.BottomEdge | Qt.Edge.LeftEdge),
            ('mid_left', Qt.Edge.LeftEdge),
        ]

        cursor_shapes = [
            Qt.CursorShape.SizeFDiagCursor,    # top-left
            Qt.CursorShape.SizeVerCursor,      # top-center
            Qt.CursorShape.SizeBDiagCursor,    # top-right
            Qt.CursorShape.SizeHorCursor,      # mid-right
            Qt.CursorShape.SizeFDiagCursor,    # bottom-right
            Qt.CursorShape.SizeVerCursor,      # bottom-center
            Qt.CursorShape.SizeBDiagCursor,    # bottom-left
            Qt.CursorShape.SizeHorCursor       # mid-left
        ]

        for (cursor, (role_name, edge_flags)) in zip(cursor_shapes, self.handle_roles):
            handle = ResizeHandle(cursor, self)
            handle.role = role_name
            handle.edges = edge_flags
            self.handles.append(handle)

        self.updateHandles()

    def updateHandles(self):
        rect = self.rect()
        x, y, w, h = rect.x(), rect.y(), rect.width(), rect.height()

        positions = [
            QPointF(x, y),                      # top-left
            QPointF(x + w / 2, y),              # top-center
            QPointF(x + w, y),                  # top-right
            QPointF(x + w, y + h / 2),          # mid-right
            QPointF(x + w, y + h),              # bottom-right
            QPointF(x + w / 2, y + h),          # bottom-center
            QPointF(x, y + h),                  # bottom-left
            QPointF(x, y + h / 2),              # mid-left
        ]

        for handle, pos in zip(self.handles, positions):
            handle.setPos(pos)

    def setRect(self, rect):
        super().setRect(rect)
        self.updateHandles()

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            scene_pos = self.mapToScene(event.pos())
            for handle in self.handles:
                if handle.contains(handle.mapFromScene(scene_pos)):
                    self.selected_edge = handle.edges
                    self.offset = QPointF()
                    break
            else:
                self.selected_edge = Qt.Edge(0)
        else:
            self.selected_edge = Qt.Edge(0)

        super().mousePressEvent(event)



    def mouseMoveEvent(self, event):
        if self.selected_edge:
            rect = self.rect()
            pos = event.pos()
            new_rect = QRectF(rect)

            if self.selected_edge & Qt.Edge.LeftEdge:
                diff = pos.x()
                new_width = rect.right() - diff
                if new_width > 10:
                    new_rect.setLeft(diff)

            if self.selected_edge & Qt.Edge.RightEdge:
                diff = pos.x()
                new_width = diff - rect.left()
                if new_width > 10:
                    new_rect.setRight(diff)

            if self.selected_edge & Qt.Edge.TopEdge:
                diff = pos.y()
                new_height = rect.bottom() - diff
                if new_height > 10:
                    new_rect.setTop(diff)

            if self.selected_edge & Qt.Edge.BottomEdge:
                diff = pos.y()
                new_height = diff - rect.top()
                if new_height > 10:
                    new_rect.setBottom(diff)

            self.setRect(new_rect)
        else:
            super().mouseMoveEvent(event)

        self.posItem.setText(f'{self.x()}, {self.y()} ({self.rect().getRect()})')
        self.posItem.setPos(
            self.boundingRect().x(),
            self.boundingRect().y() - self.posItem.boundingRect().height()
        )

    def mouseReleaseEvent(self, event):
        self.selected_edge = Qt.Edge(0)
        super().mouseReleaseEvent(event)
        self.updateHandles()

    def getEdges(self, pos):
        """Detect which edge or corner the user is near — prioritize corners."""
        edges = Qt.Edge(0)
        rect = self.rect()
        buffer = 30  # How far outside/inside the user can be and still grab

        # Corners first (priority!)
        if (pos - rect.topLeft()).manhattanLength() < buffer:
            return Qt.Edge.TopEdge | Qt.Edge.LeftEdge
        if (pos - rect.topRight()).manhattanLength() < buffer:
            return Qt.Edge.TopEdge | Qt.Edge.RightEdge
        if (pos - rect.bottomLeft()).manhattanLength() < buffer:
            return Qt.Edge.BottomEdge | Qt.Edge.LeftEdge
        if (pos - rect.bottomRight()).manhattanLength() < buffer:
            return Qt.Edge.BottomEdge | Qt.Edge.RightEdge

        # Edges (only if not corner)
        if abs(pos.x() - rect.left()) < buffer:
            edges |= Qt.Edge.LeftEdge
        elif abs(pos.x() - rect.right()) < buffer:
            edges |= Qt.Edge.RightEdge

        if abs(pos.y() - rect.top()) < buffer:
            edges |= Qt.Edge.TopEdge
        elif abs(pos.y() - rect.bottom()) < buffer:
            edges |= Qt.Edge.BottomEdge

        return edges
