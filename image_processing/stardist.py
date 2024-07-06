
from PyQt6.QtGui import QImage
from PyQt6.QtCore import Qt, QRect, QSize, QPoint, pyqtSignal
import tifffile as tiff, numpy as np
# STARDIST
import image_processing.canvas
from stardist.models import StarDist2D
from csbdeep.utils import normalize

class StarDist():
    def __init__(self):
        
        self.channels = None
        self.params = {
        'channel': 'Channel 1',
        'model': '2D_versatile_fluo',
        'percentile_low' : 1.00,
        'percentile_high': 99.80,
        'prob_theshold': 0.48,
        'nms_threshold': 0.3,
        'n_tiles': 0
        }

    def runStarDist(self):
        model = StarDist2D.from_pretrained(str(self.params['model']))

        qimage = self.channels[self.params['channel']]
        arr = self.qimage_to_numpy(qimage)
        stardist_labels, _ = model.predict_instances(normalize(arr), prob_thresh=.48, nms_thresh=.3)

    # def set_param(self, key, value):
    #     if key in self.params:
    #         self.params[key] = value

    # def get_param(self, key):
    #     return self.params.get(key, None)

    # def get_all_params(self):
    #     return self.params

    def updateChannels(self, channels):
        self.channels = channels

    def qimage_to_numpy(self, qimage:QImage):
        qimage = qimage.convertToFormat(QImage.Format.Format_Grayscale8)

        width = qimage.width()
        height = qimage.height()

        ptr = qimage.bits()
        ptr.setsize(height * width * 4)
        arr = np.frombuffer(ptr, np.uint8).reshape((height, width, 4))
        return arr

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
        print(self.params['n_tiles'])