from PyQt6.QtCore import pyqtSignal, QThread
import numpy as np
from pyclesperanto_prototype import dilate_labels
from core.Worker import Worker
import os

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
from stardist.models import StarDist2D
from csbdeep.utils import normalize
import tensorflow as tf


class StarDist(QThread):
    stardistDone = pyqtSignal(np.ndarray)
    progress = pyqtSignal(int, str)
    errorSignal = pyqtSignal(str)

    def __init__(self):
        super().__init__()
        self.protein_channels = None
        self.np_image = None
        self.params = {
            "channel": "Channel 2",
            "model": "2D_versatile_fluo",
            "percentile_low": 3,
            "percentile_high": 99.80,
            "prob_threshold": 0.48,
            "nms_threshold": 0.3,
            "n_tiles": 0,
            "radius": 5,
        }
        self.aligned = False

    def loadCellImage(self, arr):
        self.cell_image = arr
        self.aligned = True

    def runStarDist(self):
        if self.protein_channels is None and self.np_image is None:
            self._fatal_error_message("please load image first")  # emit error message
            return
        elif self.protein_channels and self.np_image:
            self._fatal_error_message(
                "unknown error, canvas has both single channel image and multi-channel image initiated"
            )  # emit error message
            return

        import platform

        system = platform.system()
        print("system: ", system)
        print("tensorflow version: ", tf.__version__)
        gpu = len(tf.config.list_physical_devices("GPU")) > 0
        if gpu:
            device_name = tf.test.gpu_device_name()
            print("gpu name: ", device_name)
        else:
            device_name = "/CPU:0"

        # if system == "Windows":
        with tf.device(device_name):
            self.star_dist_worker = Worker(self.run)
            self.star_dist_worker.finished.connect(self.quit)
            self.star_dist_worker.finished.connect(self.deleteLater)
            self.star_dist_worker.error.connect(self._fatal_error_message)
            self.star_dist_worker.start()

        # else:
        #     print("on MacOS ")
        #     self.stardist_worker = Worker(self.stardistTask)
        #     self.stardist_worker.start()

        print("here")

    def __get_cell_image(self):
        if self.aligned:
            return self.cell_image
        elif self.protein_channels is None and self.np_image:
            return self.np_image
        elif self.protein_channels and self.np_image is None:
            return self.protein_channels[self.params["channel"]].data

    def run(self):
        cell_image = self.__get_cell_image()
        if cell_image is None:
            self._fatal_error_message("No image to process")
            return
        assert isinstance(cell_image, np.ndarray), "cell_image must be a numpy array"

        self.progress.emit(0, "Loading StarDist model")
        model = StarDist2D.from_pretrained(str(self.params["model"]))
        if model is None:
            self._fatal_error_message("Failed to load StarDist model")
            return
        self.progress.emit(25, "Training model")

        print("here2")
        guess_tiles = self.params["n_tiles"]
        if guess_tiles == 0:
            guess_tiles = model._guess_n_tiles(cell_image)
        stardist_labels, _ = model.predict_instances(
            normalize(
                cell_image,
                self.params["percentile_low"],
                self.params["percentile_high"],
            ),
            prob_thresh=self.params["prob_threshold"],
            nms_thresh=self.params["nms_threshold"],
            n_tiles=guess_tiles,
        )

        # dilate
        print("here3")
        radius = self.params["radius"]
        self.progress.emit(95, "Dilating")
        # If error is platform not found, ask user to install run "sudo apt install pocl-opencl-icd"
        try:
            self.stardist_labels_grayscale = np.array(
                dilate_labels(stardist_labels, radius=radius), dtype=np.uint16
            )
        except Exception as e:
            self._fatal_error_message(
                f"Error during dilation: {e}. You may need to install pocl-opencl-icd(wsl2 users)."
            )
            return
        print("here 4")
        self.progress.emit(100, "Stardist Done")
        self.stardistDone.emit(self.stardist_labels_grayscale)

    def cancel(self):
        self.terminate()

    def saveImage(self):
        from PIL import Image
        from PyQt6.QtWidgets import QFileDialog

        file_name, _ = QFileDialog.getSaveFileName(
            None, "Save File", "image.png", "*.png;;*.jpg;;*.tif;; All Files(*)"
        )
        if not self.stardist_labels_grayscale is None:
            Image.fromarray(self.stardist_labels_grayscale).save(file_name)
        else:
            self._fatal_error_message("Cannot save. No stardist labels available")

    def updateChannels(self, protein_channels, _):
        self.np_image = None
        self.protein_channels = protein_channels

    def setProteinImage(self, protein_channels):
        self.protein_channels = protein_channels
        self.np_image = None

    def setImageToProcess(self, np_image):
        self.protein_channels = None
        self.np_image = np_image

    def setChannel(self, channel):
        self.params["channel"] = channel

    def setModel(self, model):
        self.params["model"] = model

    def setPercentileLow(self, value):
        self.params["percentile_low"] = value

    def setPercentileHigh(self, value):
        self.params["percentile_high"] = value

    def setProbThresh(self, value):
        self.params["prob_threshold"] = value

    def setNMSThresh(self, value):
        self.params["nms_threshold"] = value

    def setNumberTiles(self, value):
        self.params["n_tiles"] = value

    def setDilationRadius(self, value):
        self.params["radius"] = value

    def _fatal_error_message(self, msg):
        self.errorSignal.emit(msg)
        self.progress.emit(100, "")
