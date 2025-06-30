"""
Main script for starting the MIST-Explorer application.
"""

import numpy as np

if not hasattr(np, "bool"):
    np.bool = np.bool_  # type: ignore
from PyQt6.QtWidgets import QApplication
import ui.app as app
from controller import Controller


if __name__ == "__main__":

    import sys

    __app = QApplication(sys.argv)
    window = app.Ui_MainWindow()

    Controller.init(window)
    window.show()

    __app.exec()
