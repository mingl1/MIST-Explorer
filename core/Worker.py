import logging

from PyQt6.QtCore import QThread, pyqtSignal

logger = logging.getLogger(__name__)


class Worker(QThread):
    signal = pyqtSignal(object)
    error = pyqtSignal(str)

    def __init__(self, func, *args, **kwargs):
        super().__init__()
        self.func = func
        self.args = args
        self.kwargs = kwargs

    def run(self):
        try:
            logger.debug("Worker executing: %s", self.func.__name__)
            self.result = self.func(*self.args, **self.kwargs)
            self.signal.emit(self.result)
        except Exception as e:
            logger.error("Worker thread error in %s: %s", self.func.__name__, e, exc_info=True)
            self.error.emit(str(e))
