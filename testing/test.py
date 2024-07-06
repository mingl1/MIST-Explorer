import sys
import time
from PyQt6.QtWidgets import QApplication, QMainWindow, QPushButton, QVBoxLayout, QWidget, QLabel
from PyQt6.QtCore import QRunnable, QThreadPool, pyqtSignal, QObject

class WorkerSignals(QObject):
    progress = pyqtSignal(int)

class Worker(QRunnable):
    def __init__(self):
        super().__init__()
        self.signals = WorkerSignals()

    def run(self):
        for i in range(1, 11):
            time.sleep(1)  # Simulate a long computation
            self.signals.progress.emit(i * 10)

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        
        self.label = QLabel("Progress: 0%")
        self.start_button = QPushButton("Start Computation")
        self.start_button.clicked.connect(self.start_computation)
        
        layout = QVBoxLayout()
        layout.addWidget(self.label)
        layout.addWidget(self.start_button)
        
        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)
        
        self.threadpool = QThreadPool()

    def start_computation(self):
        worker = Worker()
        worker.signals.progress.connect(self.update_progress)
        self.threadpool.start(worker)

    def update_progress(self, value):
        self.label.setText(f"Progress: {value}%")

app = QApplication(sys.argv)
window = MainWindow()
window.show()
app.exec()
