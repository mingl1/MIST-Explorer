
from PyQt6.QtGui import QImage, QPixmap
from PyQt6.QtCore import pyqtSignal, QObject
from PyQt6.QtWidgets import QMessageBox
import numpy as np, cv2 as cv
from image_processing.canvas import ImageGraphicsView
import ui.app
# STARDIST
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
from stardist.models import StarDist2D
from csbdeep.utils import normalize

class StarDist(QObject):
    stardistDone = pyqtSignal(QPixmap)
    def __init__(self):
        super().__init__()
        self.np_channels = None
        self.np_image = None
        self.params = {
        'channel': 'Channel 1',
        'model': '2D_versatile_fluo',
        'percentile_low' : 1.00,
        'percentile_high': 99.80,
        'prob_threshold': 0.48,
        'nms_threshold': 0.3,
        'n_tiles': 0,
        'kernel_size': 3,
        'iterations': 1
        }

    def runStarDist(self):
        

        model = StarDist2D.from_pretrained(str(self.params['model']))

        try:
            # case: image has only one channel
            if not self.np_channels:
                arr = self.np_image
            # case: image has multiple channels
            else:
                arr = self.np_channels[self.params['channel']] 

            # scale down image if it's a large image
            scaleDown = arr.shape[0] > 10000
            if scaleDown:
                scale_factor = 20
                cell_image = cv.resize(arr, (0, 0), fx = 1 / scale_factor , fy = 1 / scale_factor)
            else:
                cell_image = arr
                            
            if self.params['n_tiles'] == 0:
                stardist_labels, _ = model.predict_instances(normalize(cell_image, self.params['percentile_low'], self.params['percentile_high']), prob_thresh=self.params['prob_threshold'], nms_thresh=self.params['nms_threshold'])
            else:
                stardist_labels, _ = model.predict_instances(normalize(cell_image, self.params['percentile_low'], self.params['percentile_high']), prob_thresh=.48, nms_thresh=.3, n_tiles =self.params['n_tiles'])

            kernel = np.ones((self.params['kernel_size'], self.params['kernel_size']), np.uint8)

            # size it back to original
            if scaleDown:
                stardist_labels = cv.resize(np.array(stardist_labels).astype('uint16'), (0, 0), fx = scale_factor , fy = scale_factor, interpolation=cv.INTER_NEAREST)

            # dilate
            stardist_labels = cv.dilate(self.normalize_to_uint8(stardist_labels), kernel = kernel, iterations=self.params['iterations'])

            # convert to pixmap
            height, width = stardist_labels.shape
            stardist_qimage = QImage(stardist_labels.data, width, height, width, QImage.Format.Format_Grayscale8)
            stardist_pixmap = QPixmap(stardist_qimage)
            self.stardistDone.emit(stardist_pixmap)

        except TypeError: # should probably start defining custom exceptions
            QMessageBox.critical(ui.app.Ui_MainWindow(), "Error", "Empty canvas, please an load image first")
        
    
    def normalize_to_uint8(self, data: np.ndarray) -> QImage:
        normalized_data = 255 * (data - np.min(data)) / (np.max(data) - np.min(data))
        normalized_data = normalized_data.astype(np.uint8)
        return normalized_data

    def updateChannels(self,_, channels):
        self.np_image = None
        self.np_channels = channels

    # def numpy_to_qimage(self, array:np.ndarray) -> QImage:
    #     if len(array.shape) == 2:
    #         # Grayscale image
    #         height, width = array.shape
    #         qimage =  QImage(array.data, width, height, width, QImage.Format.Format_Grayscale8)
    #     elif len(array.shape) == 3:
    #         height, width, channels = array.shape
    #         if channels == 3:
    #             # RGB image
    #             qimage = QImage(array.data, width, height, width * channels, QImage.Format.Format_RGB888)
    #         elif channels == 4:
    #             # RGBA image
    #             qimage = QImage(array.data, width, height, width * channels, QImage.Format.Format_RGBA8888)
    #     else:
    #         raise ValueError("Unsupported array shape: {}".format(array.shape))
    #     return qimage

    # def qimage_to_numpy(self, qimage:QImage):
    #     # Ensure the QImage format is suitable for conversion
    #     if qimage.format() == QImage.Format.Format_Grayscale8:
    #         width = qimage.width()
    #         height = qimage.height()
    #         ptr = qimage.constBits()
    #         ptr.setsize(qimage.sizeInBytes())  # Ensure the pointer size matches the image byte count
            
    #         # Convert QImage to a 2D numpy array
    #         arr = np.ndarray(shape=(height, width), buffer=ptr, dtype=np.uint8)
    #         return arr
    #     else:
    #         raise ValueError("Unsupported QImage format for conversion to NumPy array")
        
    def setImageToProcess(self, np_image):
        self.np_channels = None
        self.np_image = np_image

    def setChannel(self, channel):
        self.params['channel'] = channel

    def setModel(self, model):
        self.params['model'] = model

    def setPercentileLow(self, value):
        self.params['percentile_low'] = value

    def setPercentileHigh(self, value):
        self.params['percentile_high'] = value

    def setProbThresh(self, value):
        self.params['prob_threshold'] = value

    def setNMSThresh(self, value):
        self.params['nms_threshold'] = value

    def setNumberTiles(self, value):
        self.params['n_tiles'] = value

    def setDilationKernelSize(self, value):
        self.params['kernel_size'] = value

    def setDilationIterations(self, value):
        self.params['iterations'] = value

