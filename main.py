"""
Main script for starting the MIST-Explorer application.
"""

import numpy as np
if not hasattr(np, "bool"):
    np.bool = np.bool_  # type: ignore
import os
script_dir = os.path.dirname(os.path.abspath(__file__))
bundled_cache_path = os.path.join(script_dir, "matplotlib")
font_cache_file = os.path.join(bundled_cache_path, "fontlist-v330.json")
if os.path.exists(font_cache_file):
    os.environ["MPLCONFIGDIR"] = bundled_cache_path

from PyQt6.QtWidgets import QApplication
import ui.app as app
from controller import Controller

# Prevent matplotlib cache building after compiling data


if __name__ == "__main__":

    import sys

    __app = QApplication(sys.argv)
    window = app.Ui_MainWindow()

    Controller.init(window)
    window.show()

    __app.exec()
