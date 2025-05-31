from PyQt6.QtWidgets import (
    QApplication, QGraphicsView, QGraphicsScene, QGraphicsRectItem, QGraphicsSimpleTextItem
)
from PyQt6.QtGui import QPen, QBrush, QColor, QMouseEvent, QPainter  # <-- include QPainter here
from PyQt6.QtCore import Qt, QPointF
from PyQt6.QtCore import QRectF

import sys


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

    def getEdges(self, pos):
        edges = Qt.Edge(0)
        rect = self.rect()
        border = self.pen().width() / 2

        if pos.x() < rect.x() + border:
            edges |= Qt.Edge.LeftEdge
        elif pos.x() > rect.right() - border:
            edges |= Qt.Edge.RightEdge
        if pos.y() < rect.y() + border:
            edges |= Qt.Edge.TopEdge
        elif pos.y() > rect.bottom() - border:
            edges |= Qt.Edge.BottomEdge

        return edges

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self.selected_edge = self.getEdges(event.pos())
            self.offset = QPointF()
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
                if new_width > 10:  # minimum width
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

        # Update the position label
        self.posItem.setText(f'{self.x()}, {self.y()} ({self.rect().getRect()})')
        self.posItem.setPos(
            self.boundingRect().x(), 
            self.boundingRect().y() - self.posItem.boundingRect().height()
        )



    def mouseReleaseEvent(self, event):
        self.selected_edge = Qt.Edge(0)
        super().mouseReleaseEvent(event)

    def hoverMoveEvent(self, event):
        edges = self.getEdges(event.pos())
        if not edges:
            self.unsetCursor()
        elif edges in (Qt.Edge.TopEdge | Qt.Edge.LeftEdge, Qt.Edge.BottomEdge | Qt.Edge.RightEdge):
            self.setCursor(Qt.CursorShape.SizeFDiagCursor)
        elif edges in (Qt.Edge.BottomEdge | Qt.Edge.LeftEdge, Qt.Edge.TopEdge | Qt.Edge.RightEdge):
            self.setCursor(Qt.CursorShape.SizeBDiagCursor)
        elif edges in (Qt.Edge.LeftEdge, Qt.Edge.RightEdge):
            self.setCursor(Qt.CursorShape.SizeHorCursor)
        else:
            self.setCursor(Qt.CursorShape.SizeVerCursor)

class ResizableRectTestView(QGraphicsView):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("ResizableRect Test")
        self.setScene(QGraphicsScene())
        self.setSceneRect(0, 0, 800, 600)
        self.setRenderHint(QPainter.RenderHint.Antialiasing)
        self.setInteractive(True)

        rect = ResizableRect(100, 100, 150, 150)
        self.scene().addItem(rect)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = ResizableRectTestView()
    window.resize(800, 600)
    window.show()
    sys.exit(app.exec())
