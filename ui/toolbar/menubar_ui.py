import sys
from PyQt6.QtGui import QAction
from PyQt6.QtWidgets import QMainWindow, QMenuBar, QPushButton, QWidget, QHBoxLayout


class MenuBarUI(QMenuBar):
    def __init__(self, parent: QMainWindow):
        super().__init__(parent)

        menufile = self.addMenu("&File")
        assert menufile is not None, "Menu 'File' could not be created"
        self.menuFile = menufile
        menu_open = self.menuFile.addMenu("&Open")
        assert menu_open is not None, "Menu 'Open' could not be created"
        self.menuOpen = menu_open

        open_image = QAction("&Open Image", self.menuOpen)
        assert open_image is not None, "Action 'Open Image' could not be created"
        self.open_image = open_image
        self.menuOpen.addAction(open_image)
        open_reference = QAction("&Open Reference", self.menuOpen)
        assert (
            open_reference is not None
        ), "Action 'Open Reference' could not be created"
        self.open_reference = open_reference
        self.menuOpen.addAction(open_reference)

        self.menuFile.addSeparator()

        self.save_action = QAction("Save", self.menuFile)
        self.save_action.triggered.connect(lambda: parent._on_save_project(False))
        self.menuFile.addAction(self.save_action)

        self.save_with_data_action = QAction("Save with Data", self.menuFile)
        self.save_with_data_action.triggered.connect(lambda: parent._on_save_project(True))
        self.menuFile.addAction(self.save_with_data_action)

        self.menuFile.addSeparator()

        self.open_project_folder = QAction("Open Project Folder", self.menuFile)
        self.open_project_folder.triggered.connect(parent.open_project_folder)
        self.menuFile.addAction(self.open_project_folder)

        self.switch_project_action = QAction("Switch Project...", self.menuFile)
        self.switch_project_action.triggered.connect(parent.switch_project)
        self.menuFile.addAction(self.switch_project_action)

        # View menu
        menu_view = self.addMenu("&View")
        assert menu_view is not None, "Menu 'View' could not be created"
        self.menuView = menu_view

        view_log_action = QAction("View Log", self.menuView)
        view_log_action.triggered.connect(parent.show_log_dialog)
        self.menuView.addAction(view_log_action)

        if sys.platform == "win32" and getattr(
            parent, "custom_windows_chrome_enabled", False
        ):
            # Window controls
            self.controls_widget = QWidget()
            self.controls_layout = QHBoxLayout(self.controls_widget)
            self.controls_layout.setContentsMargins(0, 0, 0, 0)
            self.controls_layout.setSpacing(0)

            self.minimize_button = QPushButton("—")
            self.maximize_button = QPushButton("[]")
            self.close_button = QPushButton("X")

            self.minimize_button.setFixedSize(30, 30)
            self.maximize_button.setFixedSize(30, 30)
            self.close_button.setFixedSize(30, 30)

            self.minimize_button.clicked.connect(parent.showMinimized)
            self.maximize_button.clicked.connect(parent.toggle_maximize)
            self.close_button.clicked.connect(parent.close)

            self.controls_layout.addWidget(self.minimize_button)
            self.controls_layout.addWidget(self.maximize_button)
            self.controls_layout.addWidget(self.close_button)

            self.setCornerWidget(self.controls_widget)
