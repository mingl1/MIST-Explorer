"""
Log dialog module — surfaces Python logging messages in a PyQt6 dialog.
"""
import logging
import os
import subprocess
import sys
from pathlib import Path

from PyQt6.QtCore import pyqtSignal, QObject
from PyQt6.QtWidgets import (
    QDialog,
    QHBoxLayout,
    QPlainTextEdit,
    QPushButton,
    QVBoxLayout,
)


class _SignalEmitter(QObject):
    """Helper QObject that owns the signal (signals must live on a QObject)."""
    log_message = pyqtSignal(str)


class QtLogHandler(logging.Handler):
    """Logging handler that emits each record as a Qt signal."""

    def __init__(self) -> None:
        super().__init__()
        self._emitter = _SignalEmitter()
        self.signal = self._emitter.log_message
        self.setFormatter(
            logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s",
                              datefmt="%H:%M:%S")
        )

    def emit(self, record: logging.LogRecord) -> None:
        try:
            msg = self.format(record)
            self._emitter.log_message.emit(msg)
        except Exception:
            try:
                print(self.format(record), file=sys.stderr)
            except Exception:
                pass
            self.handleError(record)


class LogDialog(QDialog):
    """Non-modal dialog that displays application log messages."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Application Log")
        self.resize(700, 400)

        layout = QVBoxLayout(self)

        self._log_view = QPlainTextEdit()
        self._log_view.setReadOnly(True)
        layout.addWidget(self._log_view)

        btn_layout = QHBoxLayout()
        open_folder_btn = QPushButton("Open Log Folder")
        open_folder_btn.clicked.connect(self._open_log_folder)
        btn_layout.addWidget(open_folder_btn)
        btn_layout.addStretch()
        clear_btn = QPushButton("Clear")
        clear_btn.clicked.connect(self._log_view.clear)
        btn_layout.addWidget(clear_btn)
        layout.addLayout(btn_layout)

        # Attach handler to root logger
        self._handler = QtLogHandler()
        self._handler.signal.connect(self._append_log)
        logging.getLogger().addHandler(self._handler)

    @staticmethod
    def _open_log_folder() -> None:
        log_dir = Path.home() / ".mist-explorer" / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        if sys.platform == "win32":
            os.startfile(str(log_dir))
        elif sys.platform == "darwin":
            subprocess.Popen(["open", str(log_dir)])
        else:
            subprocess.Popen(["xdg-open", str(log_dir)])

    def _append_log(self, message: str) -> None:
        self._log_view.appendPlainText(message)
