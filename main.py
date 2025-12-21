"""
Main script for starting the MIST-Explorer application.
"""
import io
import os
import sys

import numpy as np
from PyQt6.QtWidgets import QApplication  # pylint: disable=no-name-in-module

from ui import app
from controller import Controller

if sys.stdout is None:
    sys.stdout = io.StringIO()
if sys.stderr is None:
    sys.stderr = io.StringIO()


if not hasattr(np, "bool"):
    np.bool = np.bool_  # type: ignore

# Prevent matplotlib cache building after compiling data
script_dir = os.path.dirname(os.path.abspath(__file__))
bundled_cache_path = os.path.join(script_dir, "matplotlib")
font_cache_file = os.path.join(bundled_cache_path, "fontlist-v330.json")
if os.path.exists(font_cache_file):
    os.environ["MPLCONFIGDIR"] = bundled_cache_path


if __name__ == "__main__":
    __app = QApplication(sys.argv)
    window = app.MainWindow()

    Controller.init(window)
    window.show()

    __app.exec()
