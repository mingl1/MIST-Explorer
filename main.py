"""
Main script for starting the MIST-Explorer application.
"""
import io
import logging
import os
import sys
import traceback
from logging.handlers import RotatingFileHandler
from pathlib import Path

# --- Log directory ---
_log_dir = Path.home() / ".mist-explorer" / "logs"
_log_dir.mkdir(parents=True, exist_ok=True)
_log_file = _log_dir / "mist-explorer.log"
_crash_file = _log_dir / "crash.log"

# --- Root logger with file + console output ---
_fmt = logging.Formatter(
    "%(asctime)s [%(levelname)s] %(name)s: %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
)

# Rotating file handler: 5 MB per file, keep 3 backups
_file_handler = RotatingFileHandler(
    _log_file, maxBytes=5 * 1024 * 1024, backupCount=3, encoding="utf-8"
)
_file_handler.setLevel(logging.DEBUG)
_file_handler.setFormatter(_fmt)

_console_handler = logging.StreamHandler()
_console_handler.setLevel(logging.WARNING)
_console_handler.setFormatter(_fmt)

logging.basicConfig(level=logging.WARNING, handlers=[_file_handler, _console_handler])
for _name in ("core", "ui", "controller", "UMAP_App", "MIST_UMAP", "MIST_MAIN"):
    logging.getLogger(_name).setLevel(logging.DEBUG)

_logger = logging.getLogger("MIST_MAIN")
_logger.info("=== MIST-Explorer started ===")
_logger.info(f"Log file: {_log_file}")


# --- Global crash handler ---
def _crash_handler(exc_type, exc_value, exc_tb):
    """Write unhandled exceptions to crash.log and the main log, then exit."""
    tb_text = "".join(traceback.format_exception(exc_type, exc_value, exc_tb))
    _logger.critical(f"Unhandled exception:\n{tb_text}")
    try:
        with open(_crash_file, "w", encoding="utf-8") as f:
            f.write(f"MIST-Explorer Crash Report\n{'=' * 40}\n\n{tb_text}")
    except Exception:
        pass
    sys.__excepthook__(exc_type, exc_value, exc_tb)


sys.excepthook = _crash_handler

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
from PyQt6.QtCore import QCoreApplication, Qt
from PyQt6.QtWidgets import (QApplication, QDialog, QLabel, QProgressBar,
                             QVBoxLayout)

if sys.stdout is None:
    sys.stdout = io.StringIO()
if sys.stderr is None:
    sys.stderr = io.StringIO()
try:
    import pyi_splash
except ImportError:
    class PyiSplashMock:
        @staticmethod
        def close():
            pass
    pyi_splash = PyiSplashMock()


class LoadingDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Loading")
        self.setMinimumSize(300, 100)
        self.setWindowFlags(Qt.WindowType.Dialog | Qt.WindowType.WindowTitleHint)
        self.setModal(True)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 20, 20, 20)

        self.label = QLabel("Loading MIST-Explorer...")
        self.label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.label)

        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        layout.addWidget(self.progress_bar)

    def update_progress(self, value, text):
        self.progress_bar.setValue(value)
        self.label.setText(text)
        QCoreApplication.processEvents()


if __name__ == "__main__":
    __app = QApplication(sys.argv)

    loading_dialog = LoadingDialog()
    loading_dialog.show()
    try:
        pyi_splash.close()
    except Exception:
        pass
    __app.processEvents()

    loading_dialog.update_progress(5, "Initializing...")
    __app.processEvents()

    loading_dialog.update_progress(10, "Loading project manager...")
    __app.processEvents()

    from core.project_manager import ProjectManager

    loading_dialog.update_progress(15, "Loading recent projects...")
    __app.processEvents()

    projects = ProjectManager.get_recent_projects()

    loading_dialog.accept()
    __app.processEvents()

    from ui.project_launcher import ProjectLauncher

    launcher = ProjectLauncher()
    if launcher.exec() == ProjectLauncher.DialogCode.Accepted:
        project_path = launcher.get_selected_project()
        if project_path:
            loading_dialog = LoadingDialog()
            loading_dialog.show()
            __app.processEvents()

            steps = [
                ("Loading numpy...", 20),
                ("Initializing core modules...", 40),
                ("Loading image processing...", 50),
                ("Loading analysis tools...", 60),
                ("Loading UI components...", 80),
                ("Starting application...", 90),
            ]

            for message, value in steps:
                loading_dialog.update_progress(value, message)
                __app.processEvents()

            import numpy as np

            if not hasattr(np, "bool"):
                np.bool = np.bool_

            from controller import Controller
            from ui import app

            loading_dialog.update_progress(100, "Done")
            __app.processEvents()

            window = app.MainWindow(project_path=project_path)
            Controller.init(window)
            window.show()
            loading_dialog.hide()

            __app.exec()
    else:
        sys.exit(0)
