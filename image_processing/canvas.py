from PyQt6.QtWidgets import QGraphicsView, QRubberBand, QGraphicsScene, QGraphicsPixmapItem, QGraphicsItem
from PyQt6.QtGui import QDragEnterEvent, QDropEvent, QPixmap, QDragMoveEvent, QMouseEvent, QCursor, QImage, QTransform
from PyQt6.QtCore import Qt, QRect, QSize, QPoint, pyqtSignal
import Dialogs, tifffile as tiff, numpy as np
from PIL import Image, ImageSequence



class ReferenceGraphicsView(QGraphicsView):
    def __init__(self, parent=None):
        super().__init__(parent)

        self.reset_referencepixmap=None
        self.reset_reference_pixmapItem=None
        self.reference_pixmap=None
        self.reference_pixmapItem=None

        self.setMinimumSize(QSize(300, 300))
        self.setObjectName("reference_canvas")
        self.setScene(QGraphicsScene(self))
        self.setSceneRect(0, 0, 200, 150)
        self.setStyleSheet("QGraphicsView { border: 1px solid black; }")


# drag and drog has issue with some tiff images, need to fix
    def dragEnterEvent(self, event: QDragEnterEvent):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dragMoveEvent(self, event: QDragEnterEvent):
        event.acceptProposedAction()

    def dropEvent(self, event: QDropEvent):
        if event.mimeData().hasUrls():
            for url in event.mimeData().urls():
                file_path = url.toLocalFile()
                pixmap = QPixmap(file_path) 
                if not pixmap.isNull():
                    self.addImage(pixmap)
            event.acceptProposedAction()

    def addImage(self, pixmap: QPixmap):
        # check if canvas already has an image
        self.resetTransform()
        if self.reference_pixmapItem:
            self.deleteImage() 
        # else 
        self.reset_reference_pixmap=pixmap
        self.reference_pixmap = pixmap
        self.reset_reference_pixmapItem = QGraphicsPixmapItem(pixmap)
        self.reference_pixmapItem = QGraphicsPixmapItem(pixmap)

        # center the image
        self.scene().addItem(self.reference_pixmapItem)

        item_rect = self.reference_pixmapItem.boundingRect()
        self.setSceneRect(item_rect)
        self.fitInView(self.reference_pixmapItem, Qt.AspectRatioMode.KeepAspectRatio)
        self.centerOn(self.reference_pixmapItem)

        
    def deleteImage(self):
        self.scene().clear()

    def resetImage(self):
        if self.reference_pixmapItem:
            self.reference_pixmapItem.setPixmap(self.reset_reference_pixmap)
            self.scene().update()


##########################################################
class ImageGraphicsView(QGraphicsView):
    
    canvasUpdated = pyqtSignal(QGraphicsPixmapItem)
    newImageAdded = pyqtSignal(QGraphicsPixmapItem)
    channelLoaded = pyqtSignal(dict, dict)
    channelNotLoaded = pyqtSignal(np.ndarray)

    def __init__(self):

        super().__init__()
        self.channels= None
        # self.channel_selector_combobox = self.view.toolBar.channelSelector
        self.reset_pixmap=None
        self.reset_pixmapItem=None
        self.pixmap=None
        self.pixmapItem=None
        self.begin_crop = False
        self.crop_cursor =  QCursor(QPixmap("icons/clicks.png").scaled(30,30, Qt.AspectRatioMode.KeepAspectRatio), 0,0)


    # def connectSignals(self):
    #     self.channel_selector_combobox.textActivated.connect(self.on_channelSelector_currentIndexChanged)
        

    def toPixmapItem(self, pixmap):
        #convert pixmap to pixmapItem
        self.pixmap = pixmap
        self.pixmapItem = QGraphicsPixmapItem(pixmap)
        self.canvasUpdated.emit(self.pixmapItem)


    def addImage(self, file:str):
        '''add a new image'''

        self.channels = None
        self.resetTransform()

        if isinstance(file, str):
            pixmap = self.__filename_to_pixmap(file)
        
        else:
            raise ValueError
        
        self.reset_pixmap=pixmap
        self.reset_pixmapItem = QGraphicsPixmapItem(pixmap)
        self.pixmap = pixmap
        self.pixmapItem = QGraphicsPixmapItem(pixmap)
        self.newImageAdded.emit(self.pixmapItem)


    def __filename_to_pixmap(self, file_name:str):
        if file_name.endswith((".tiff", ".tif")):

            # image_data = tiff.imread(file_name) 
            Image.MAX_IMAGE_PIXELS = 9999999999999
            image_data = Image.open(file_name)
            num_channels = sum(1 for _ in ImageSequence.Iterator(image_data))

            bytesPerPixel = 1 # only supports uint8 and uint16
            format = QImage.Format.Format_Grayscale8

            if (num_channels > 1):
                # num_channels, height, width = image_data.shape
                resultSize = QSize(image_data.width, image_data.height)
                self.resultImage = QImage(resultSize, QImage.Format.Format_ARGB32_Premultiplied)

                self.channels = {}
                self.np_channels = {}
                for channel_num in range(num_channels):
                    image_data.seek(channel_num)
                    image_data_np = np.array(image_data)
                    channel_name = f'Channel {channel_num + 1}'
                    
                    # self.channel_selector_combobox.addItem(channel_name)
                    __scaled = self.scale_adjust(image_data_np) # uint8
                    image_adjusted = self.adjustContrast(__scaled) # uint8
                    self.np_channels[channel_name] = image_adjusted # for stardist and other image processing, maybe consider keeping it as uint16
                    qimage_channel = QImage(image_adjusted, image_data.width, image_data.height, image_data.width*bytesPerPixel, format)
                    # qpixmap_channel = QtGui.QPixmap(qimage_channel)
                    self.channels[channel_name] = qimage_channel # for displaying on canvas

                channel_one_qimage = next(iter(self.channels.values()))
                self.channelLoaded.emit(self.channels, self.np_channels)
                
            else:
                image_data_np = np.array(image_data)
                __scaled = self.scale_adjust(image_data_np) # uint8
                image_adjusted = self.adjustContrast(__scaled) # uint8
                channel_one_qimage = QImage(image_adjusted, image_data.width, image_data.height, image_data.width*bytesPerPixel, format)
                self.channelNotLoaded.emit(image_adjusted)

            pixmap = QPixmap(channel_one_qimage)

        else:
            pixmap = QPixmap(file_name)

        return pixmap
    

    def adjustContrast(self,img, min=2, max = 98):  
        minval = np.percentile(img, min) 
        maxval = np.percentile(img, max) 
        img = np.clip(img, minval, maxval)
        img = ((img - minval) / (maxval - minval)) * 255
        return (img.astype(np.uint8))

    def scale_adjust(self, arr:np.ndarray):
        return ((arr - arr.min()) * (1/(arr.max() - arr.min()) * 255)).astype('uint8')
        
    def deleteImage(self):
        self.scene().clear()

    def resetImage(self):
        if self.pixmapItem: 
            self.toPixmapItem(self.reset_pixmap)

    
    # rotate
    def rotateImage(self, angle_text:str):
        angle = None
        try:
            angle = float(angle_text)
        except ValueError:
            print("Error: Please enter a valid number.")

        if self.pixmap and angle != None:
            image = self.pixmap.toImage()
            transform = QTransform()
            
            transform.rotate(angle)
            rotated_image = image.transformed(transform, Qt.TransformationMode.SmoothTransformation)
            rotated_pixmapItem = QGraphicsPixmapItem(QPixmap.fromImage(rotated_image))
            self.canvasUpdated.emit(rotated_pixmapItem)
