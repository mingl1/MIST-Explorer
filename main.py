"""
Main script for starting the MIST-Explorer application.
"""
import io
import sys

from PyQt6.QtWidgets import (
    QApplication,
    QDialog,
    QLabel,
    QProgressBar,
    QVBoxLayout,
)
from PyQt6.QtCore import Qt, QCoreApplication

if sys.stdout is None:
    sys.stdout = io.StringIO()
if sys.stderr is None:
    sys.stderr = io.StringIO()


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
