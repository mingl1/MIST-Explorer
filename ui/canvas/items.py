"""
Graphics items for the canvas.
"""
# pylint: disable=no-name-in-module
from PyQt6.QtCore import (QEasingCurve, QPointF, QPropertyAnimation,
                          QRectF, QSize, Qt)
from PyQt6.QtGui import (QBrush, QColor, QCursor, QFont, QPen)
from PyQt6.QtWidgets import (QApplication, QGraphicsItem, QGraphicsItemGroup,
                             QGraphicsOpacityEffect, QGraphicsPixmapItem,
                             QGraphicsRectItem, QGraphicsSimpleTextItem)


class ArrowItem(QGraphicsItemGroup):
    """Arrow with hover effect"""

    def __init__(self, pixmap, position, action):
        fixed_size = QSize(30, 30)  # Fixed size for the background rectangle
        if pixmap is None:
            raise ValueError("Pixmap cannot be None")
        scaled = QGraphicsPixmapItem(
            pixmap.scaled(
                fixed_size,
                aspectRatioMode=Qt.AspectRatioMode.KeepAspectRatio,
                transformMode=Qt.TransformationMode.SmoothTransformation,
            )
        )
        super().__init__()
        self.bg_rect = QGraphicsRectItem(0, 0, fixed_size.width(), fixed_size.height())
        self.bg_rect.setBrush(QBrush(QColor(255, 255, 255, 100)))
        self.bg_rect.setZValue(999)
        self.base_opacity = 0.4
        self.hover_opacity = 1
        self.action = action
        self.addToGroup(self.bg_rect)
        self.addToGroup(scaled)
        self.setZValue(100)
        self.setPos(position)
        self.apply_hover_effect()
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIgnoresTransformations, True)
        self.effect = None
        self.fade_in = None
        self.fade_out = None

    def apply_hover_effect(self):
        """Apply hover effect to the arrow."""
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

    # pylint: disable=invalid-name, unused-argument
    def hoverEnterEvent(self, event):
        """Handle hover enter event."""
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        if self.fade_out:
            self.fade_out.stop()
        if self.fade_in:
            self.fade_in.start()

    # pylint: disable=invalid-name, unused-argument
    def hoverLeaveEvent(self, event):
        """Handle hover leave event."""
        if self.fade_in:
            self.fade_in.stop()
        if self.fade_out:
            self.fade_out.start()

    # pylint: disable=invalid-name
    def mousePressEvent(self, event):
        """Handle mouse press events to prevent propagation"""
        assert event is not None, "Event should not be None"
        if event.button() == Qt.MouseButton.LeftButton:
            self.setCursor(Qt.CursorShape.PointingHandCursor)
            event.accept()
            # Slideshow action
            if self.action:
                self.action()
        else:
            super().mousePressEvent(event)


class ResizeHandle(QGraphicsRectItem):
    """Resize handle for ResizableRect"""
    def __init__(self, cursor_shape: Qt.CursorShape, parent):
        """Resize handle for ResizableRect"""
        w_rect, h_rect = parent.boundingRect().width(), parent.boundingRect().height()
        w_handle, h_handle = w_rect // 16, h_rect // 16
        w_handle, h_handle = max(16, w_handle), max(16, h_handle)  # Ensure minimum size

        # Center the handle
        super().__init__(-w_handle // 2, -h_handle // 2, w_handle, h_handle, parent)
        self.setBrush(QBrush(Qt.GlobalColor.white))
        self.setPen(QPen(Qt.GlobalColor.black))
        self.setZValue(11)
        self.setAcceptHoverEvents(True)
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsSelectable)
        self.cursor_shape = cursor_shape
        self.edge = Qt.Edge(0)
        self.parent_item = parent

    def set_edge(self, edge):
        """Set the edge this handle is associated with"""
        self.edge = edge

    # pylint: disable=invalid-name, unused-argument
    def hoverEnterEvent(self, event):
        """Handle hover enter."""
        QApplication.setOverrideCursor(QCursor(self.cursor_shape))

    # pylint: disable=invalid-name, unused-argument
    def hoverLeaveEvent(self, event):
        """Handle hover leave."""
        QApplication.restoreOverrideCursor()

    # pylint: disable=invalid-name, unused-argument
    def mousePressEvent(self, event):
        """Handle mouse press events on the resize handle"""
        self.parent_item.selected_edge = self.edge
        # super().mousePressEvent(event)

    # pylint: disable=invalid-name, unused-argument
    def mouseReleaseEvent(self, event):
        """Handle mouse release events on the resize handle"""
        self.parent_item.selected_edge = Qt.Edge(0)
        # super().mouseReleaseEvent(event)

    # pylint: disable=invalid-name
    def mouseMoveEvent(self, event):
        """Handle mouse move events on the resize handle"""
        super().mouseMoveEvent(event)
        # Update the parent rectangle's position if needed
        self.parent_item.mouseMoveEvent(event)


class ResizableRect(QGraphicsRectItem):
    """A rectangle that can be resized."""
    # pylint: disable=too-many-instance-attributes, too-many-arguments, too-many-positional-arguments
    def __init__(self, x=0.0, y=0.0, width=0.0, height=0.0, on_center=False):
        if on_center:
            super().__init__(-width / 2, -height / 2, width, height)
        else:
            super().__init__(x, y, width, height)

        self.setFlags(
            QGraphicsRectItem.GraphicsItemFlag.ItemIsMovable
            | QGraphicsRectItem.GraphicsItemFlag.ItemIsSelectable
            | QGraphicsRectItem.GraphicsItemFlag.ItemIsFocusable
        )
        self.selected_edge = None
        self.setAcceptHoverEvents(True)
        self.setPen(QPen(QBrush(Qt.GlobalColor.blue), 3, Qt.PenStyle.DotLine))
        self.setBrush(QBrush(QColor(0, 255, 0, 30)))
        # Create 8 resize handles with correct cursor
        self.handles = []
        cursor_shapes = [
            Qt.CursorShape.SizeFDiagCursor,  # top-left
            Qt.CursorShape.SizeVerCursor,  # top-center
            Qt.CursorShape.SizeBDiagCursor,  # top-right
            Qt.CursorShape.SizeHorCursor,  # mid-right
            Qt.CursorShape.SizeFDiagCursor,  # bottom-right
            Qt.CursorShape.SizeVerCursor,  # bottom-center
            Qt.CursorShape.SizeBDiagCursor,  # bottom-left
            Qt.CursorShape.SizeHorCursor,  # mid-left
        ]
        edges = [
            Qt.Edge.TopEdge | Qt.Edge.LeftEdge,  # top-left
            Qt.Edge.TopEdge,  # top-center
            Qt.Edge.TopEdge | Qt.Edge.RightEdge,  # top-right
            Qt.Edge.RightEdge,  # mid-right
            Qt.Edge.BottomEdge | Qt.Edge.RightEdge,  # bottom-right
            Qt.Edge.BottomEdge,  # bottom-center
            Qt.Edge.BottomEdge | Qt.Edge.LeftEdge,  # bottom-left
            Qt.Edge.LeftEdge,  # mid-left
        ]

        for cursor, edge in zip(cursor_shapes, edges):
            handle = ResizeHandle(cursor, self)
            handle.set_edge(edge)
            self.handles.append(handle)

        self.update_handles()
        self.text_item = QGraphicsSimpleTextItem(self)
        self.text_item.setBrush(QColor("yellow"))
        self.text_item.setFont(QFont("Arial", 10))
        self.text_item.setFlag(
            QGraphicsItem.GraphicsItemFlag.ItemIgnoresTransformations, True
        )
        self.update_text()
        # ⬆ ensures text doesn’t scale with zoom

    def update_handles(self):
        """Update positions of resize handles."""
        rect = self.rect()
        x_pos, y_pos = rect.x(), rect.y()
        width, height = rect.width(), rect.height()

        positions = [
            QPointF(x_pos, y_pos),  # top-left
            QPointF(x_pos + width / 2, y_pos),  # top-center
            QPointF(x_pos + width, y_pos),  # top-right
            QPointF(x_pos + width, y_pos + height / 2),  # mid-right
            QPointF(x_pos + width, y_pos + height),  # bottom-right
            QPointF(x_pos + width / 2, y_pos + height),  # bottom-center
            QPointF(x_pos, y_pos + height),  # bottom-left
            QPointF(x_pos, y_pos + height / 2),  # mid-left
        ]
        handle_size = 16
        for handle, pos in zip(self.handles, positions):
            handle.setPos(pos)

            # fixed-size rect centered at (0,0)
            handle.setRect(-handle_size / 2, -handle_size / 2, handle_size, handle_size)

            # ensure handle doesn't scale with zoom
            handle.setFlag(
                QGraphicsItem.GraphicsItemFlag.ItemIgnoresTransformations, True
            )

    # pylint: disable=invalid-name
    def setRect(self, *args, **kwargs):
        """Override setRect to update handles."""
        if len(args) == 1 and hasattr(args[0], "x"):
            # Called with QRectF object
            rect = args[0]
            super().setRect(rect.x(), rect.y(), rect.width(), rect.height())
        elif len(args) == 4:
            # Called with separate x, y, width, height parameters
            super().setRect(*args)
        elif "rect" in kwargs:
            # Called with rect keyword parameter
            super().setRect(**kwargs)
        else:
            # Fallback to base implementation
            super().setRect(*args, **kwargs)
        self.update_handles()
        self.update_text()

    # pylint: disable=invalid-name
    def mouseMoveEvent(self, event):
        """Handle mouse move for resizing."""
        if event is None:
            return
        if self.selected_edge:
            rect = self.rect()
            pos = self.mapFromScene(event.scenePos())
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

    def update_text(self):
        """Update the size text."""
        margin = 360
        rect = self.rect()
        w_size = int(rect.width())
        h_size = int(rect.height())
        x_pos = rect.x()
        y_pos = rect.y()
        # self.text_item.setFlag()

        self.text_item.setText(f"{w_size} × {h_size}")

        # Fixed margin (5 px in scene coordinates, stays constant due to IgnoresTransformations)
        self.text_item.setPos(x_pos - margin, y_pos - margin)

    # pylint: disable=invalid-name
    def mouseReleaseEvent(self, event):
        """Handle mouse release."""
        self.selected_edge = Qt.Edge(0)
        super().mouseReleaseEvent(event)
        self.update_handles()

    # pylint: disable=invalid-name
    def get_edges(self, pos):
        """Fallback edge hit detection for resize dragging (not hover)"""
        edges = Qt.Edge(0)
        rect = self.rect()
        buffer = 30

        if pos.x() < rect.x() + buffer:
            edges |= Qt.Edge.LeftEdge
        elif pos.x() > rect.right() - buffer:
            edges |= Qt.Edge.RightEdge
        if pos.y() < rect.y() + buffer:
            edges |= Qt.Edge.TopEdge
        elif pos.y() > rect.bottom() - buffer:
            edges |= Qt.Edge.BottomEdge

        return edges


class CropRectItem(QGraphicsRectItem):
    """Crop rectangle item."""
    def __init__(self, parent=None):
        super().__init__(parent)

        # Rectangle style
        self.setPen(QPen(QColor("red"), 2, Qt.PenStyle.DashLine))
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsSelectable, True)
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsMovable, True)
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemSendsGeometryChanges, True)

        # Size text overlay
        self.text_item = QGraphicsSimpleTextItem(self)
        self.text_item.setBrush(QColor("yellow"))
        self.text_item.setFont(QFont("Arial", 10))
        self.text_item.setFlag(
            QGraphicsItem.GraphicsItemFlag.ItemIgnoresTransformations, True
        )
        # ⬆ ensures text doesn’t scale with zoom

    def update_text(self):
        """Update size text."""
        rect = self.rect()
        w_size = int(rect.width())
        h_size = int(rect.height())
        self.text_item.setText(f"{w_size} × {h_size}")

        # Place text at top-left of rect with a small margin
        self.text_item.setPos(rect.x() + 5, rect.y() + 5)

    # pylint: disable=invalid-name
    def setRect(self, *args):
        """Override setRect to keep text updated"""
        super().setRect(*args)
        self.update_text()
