from PyQt6.QtGui import QAction, QKeySequence
from PyQt6.QtWidgets import QMainWindow, QMenuBar


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

        self.save_all_to_action = QAction("Save all images to...", self.menuFile)
        self.save_all_to_action.triggered.connect(parent.save_all_images_to_folder)
        self.menuFile.addAction(self.save_all_to_action)

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

        self.menuView.addSeparator()

        toggle_theme_action = QAction("Toggle Theme", self.menuView)
        toggle_theme_action.setShortcut(QKeySequence("Ctrl+Shift+T"))
        toggle_theme_action.triggered.connect(self._toggle_theme)
        self.menuView.addAction(toggle_theme_action)

    def _toggle_theme(self):
        from ui.theme import ThemeManager
        ThemeManager.instance().toggle()
