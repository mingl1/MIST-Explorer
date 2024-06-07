from PyQt5 import QtCore, QtGui, QtWidgets

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

        # check if the item has multiple channels
        self.scene().addItem(self.pixmapItem)
        self.fitInView(self.pixmapItem, QtCore.Qt.AspectRatioMode.KeepAspectRatio)


        self.pixmapItem.setPos(0, 0)  # You can set position as needed
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