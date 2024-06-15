from PyQt6 import QtCore, QtGui, QtWidgets


class ReferenceGraphicsView(QtWidgets.QGraphicsView):
    def __init__(self, parent=None):
        super().__init__(parent)

        self.reset_referencepixmap=None
        self.reset_reference_pixmapItem=None
        self.reference_pixmap=None
        self.reference_pixmapItem=None

        self.setMinimumSize(QtCore.QSize(300, 300))
        self.setObjectName("reference_canvas")
        self.setScene(QtWidgets.QGraphicsScene(self))
        self.setSceneRect(0, 0, 200, 150)

    def dragEnterEvent(self, event: QtGui.QDragEnterEvent):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dragMoveEvent(self, event: QtGui.QDragEnterEvent):
        event.acceptProposedAction()

    def dropEvent(self, event: QtGui.QDropEvent):
        if event.mimeData().hasUrls():
            for url in event.mimeData().urls():
                file_path = url.toLocalFile()
                pixmap = QtGui.QPixmap(file_path) 
                if not pixmap.isNull():
                    self.addImage(pixmap)
            event.acceptProposedAction()

    def addImage(self, pixmap: QtGui.QPixmap):
        # check if canvas already has an image
        self.resetTransform()
        if self.reference_pixmapItem:
            self.deleteImage() 
        # else 
        self.reset_reference_pixmap=pixmap
        self.reference_pixmap = pixmap
        self.reset_reference_pixmapItem = QtWidgets.QGraphicsPixmapItem(pixmap)
        self.reference_pixmapItem = QtWidgets.QGraphicsPixmapItem(pixmap)

        # center the image

        self.scene().addItem(self.reference_pixmapItem)

        item_rect = self.reference_pixmapItem.boundingRect()
        self.setSceneRect(item_rect)
        self.fitInView(self.reference_pixmapItem, QtCore.Qt.AspectRatioMode.KeepAspectRatio)
        self.centerOn(self.reference_pixmapItem)

        

    def deleteImage(self):
        self.scene().removeItem(self.reference_pixmapItem)
        self.reset_reference_pixmapItem = None
        self.reference_pixmapItem=None

    def resetImage(self):
        if self.reference_pixmapItem:
            self.reference_pixmapItem.setPixmap(self.reset_reference_pixmap)
            self.scene().update()



class ImageGraphicsView(QtWidgets.QGraphicsView):
    def __init__(self, parent=None):
        super().__init__(parent)

        self.reset_pixmap=None
        self.reset_pixmapItem=None
        self.pixmap=None
        self.pixmapItem=None
        self.scale_factor = 1.25

        self.setMinimumSize(QtCore.QSize(600, 600))
        self.setObjectName("canvas")
        self.setAcceptDrops(True) # allow drag and drop
        self.setScene(QtWidgets.QGraphicsScene(self))
        self.setSceneRect(0, 0, 800, 600)
        self.setDragMode(QtWidgets.QGraphicsView.DragMode.ScrollHandDrag)
        self.setTransformationAnchor(QtWidgets.QGraphicsView.ViewportAnchor.AnchorUnderMouse)

    def dragEnterEvent(self, event: QtGui.QDragEnterEvent):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dragMoveEvent(self, event: QtGui.QDragEnterEvent):
        event.acceptProposedAction()

    def dropEvent(self, event: QtGui.QDropEvent):
        if event.mimeData().hasUrls():
            for url in event.mimeData().urls():
                file_path = url.toLocalFile()
                pixmap = QtGui.QPixmap(file_path) 
                if not pixmap.isNull():
                    self.addImage(pixmap)
            event.acceptProposedAction()

    def addImage(self, pixmap: QtGui.QPixmap):
        # check if canvas already has an image
        self.resetTransform()
        if self.pixmapItem:
            self.deleteImage() 
        # else 
        self.reset_pixmap=pixmap
        self.pixmap = pixmap
        self.reset_pixmapItem = QtWidgets.QGraphicsPixmapItem(pixmap)
        self.pixmapItem = QtWidgets.QGraphicsPixmapItem(pixmap)

        # center image
        self.scene().addItem(self.pixmapItem)
        item_rect = self.pixmapItem.boundingRect()
        self.setSceneRect(item_rect)
        self.fitInView(self.pixmapItem, QtCore.Qt.AspectRatioMode.KeepAspectRatio)
        self.centerOn(self.pixmapItem)

        self.pixmapItem.setFlag(QtWidgets.QGraphicsItem.GraphicsItemFlag.ItemIsMovable)
        
    def wheelEvent(self, event):
        if event.angleDelta().y() > 0:
            self.scale(self.scale_factor, self.scale_factor)
        else:
            self.scale(1/self.scale_factor, 1/self.scale_factor)

    def deleteImage(self):
        self.scene().removeItem(self.pixmapItem)
        self.reset_pixmapItem = None
        self.pixmapItem=None

    def resetImage(self):
        if self.pixmapItem:
            self.pixmapItem.setPixmap(self.reset_pixmap)
            self.scene().update()