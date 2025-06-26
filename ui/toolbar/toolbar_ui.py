from PyQt6.QtWidgets import (
    QToolBar,
    QWidget,
    QComboBox,
    QLabel,
    QSizePolicy,
    QPushButton,
    QToolButton,
    QButtonGroup,
)
from PyQt6.QtCore import Qt, QCoreApplication, pyqtSignal, QSize, pyqtSlot
from PyQt6.QtGui import QIcon, QImage, QPixmap
from ui.toolbar.Action import Action
import matplotlib.pyplot as plt
import numpy as np
from qtrangeslider import QRangeSlider


class ToolBarUI(QWidget):
    # Public signals
    tabChanged = pyqtSignal(int)
    channelChanged = pyqtSignal(int)
    cmapChanged = pyqtSignal(str)

    def __init__(self, parent):
        super().__init__()
        self.toolbar = QToolBar()

        self._init_tab_buttons()
        self._init_actions(parent)
        self._init_channel_selector(parent)
        self._init_status_line()
        self._init_cmap_selector(parent)
        self._init_contrast_slider()

        self._populate_toolbar()

        self._retranslateUI()

    def get_toolbar(self):
        return self.toolbar

    def _init_tab_buttons(self):
        self.tab_buttons = []
        tab_names = ["Images", "Data Processing", "View", "Analysis", "Details"]
        for name in tab_names:
            button = QToolButton()
            button.setText(name)
            button.setStyleSheet(self._tab_button_style())
            button.setCheckable(True)
            self.tab_buttons.append(button)

        # Group buttons for exclusive behavior
        self.tabButtonGroup = QButtonGroup(self)
        self.tabButtonGroup.setExclusive(True)
        for idx, button in enumerate(self.tab_buttons):
            self.tabButtonGroup.addButton(button, idx)

        self.tabButtonGroup.idClicked.connect(self.onTabButtonClicked)
        self.tab_buttons[0].setChecked(True)

    def _tab_button_style(self):
        return """
            QToolButton {
                padding: 8px 16px;
                border: none;
                background: transparent;
                font-size: 12px;
            }
            QToolButton:hover {
                background: rgba(0, 0, 0, 0.1);
            }
            QToolButton:checked {
                border-bottom: 2px solid #007AFF;
                font-weight: Bold;
            }
        """

    @pyqtSlot(int)
    def onTabButtonClicked(self, index):
        self.tabChanged.emit(index)

    def _init_actions(self, parent):
        self.actionRotate = Action(
            parent, "actionRotate", "assets/icons/rotate-right.png"
        )
        self.actionReset = Action(parent, "actionReset", "assets/icons/reset.png")
        self.actionOpenBrightnessContrast = Action(
            parent, "actionBC", "assets/icons/brightness.png"
        )

    def _init_channel_selector(self, parent):
        self.channelSelector = QComboBox(parent)
        self.channelSelector.setMinimumWidth(100)
        self.channelSelector.currentIndexChanged.connect(
            self.on_channelSelector_currentIndexChanged
        )

    def updateChannelSelector(self, channels: dict, clear=False):
        if clear:
            self.clearChannelSelector()
        self.channelSelector.addItems(list(channels.keys()))

    def clearChannelSelector(self):
        self.channelSelector.clear()

    @pyqtSlot(int)
    def on_channelSelector_currentIndexChanged(self, index):
        if self.channelSelector.count() > 0:
            self.channelChanged.emit(index)

    def _init_status_line(self):
        self.statusLine = QLabel("Welcome! Please load an image to get started.")
        self.statusLine.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.statusLine.setAlignment(
            Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter
        )
        self.statusLine.setStyleSheet("margin: 10px;")

    def _init_cmap_selector(self, parent):
        self.cmapSelector = QComboBox(parent)
        self.cmapSelector.setMinimumSize(QSize(100, 20))
        self.cmapSelector.currentTextChanged.connect(self.on_cmapTextChanged)

        thumbnails = self._generate_cmap_thumbnails()
        for index, thumbnail in enumerate(thumbnails):
            icon = self._numpy_to_QIcon(thumbnail)
            self.cmapSelector.setIconSize(QSize(100, 20))
            self.cmapSelector.addItem(icon, self.cmap_names[index])

    @pyqtSlot(str)
    def on_cmapTextChanged(self, cmap_str: str):
        self.cmapChanged.emit(cmap_str)

    def update_cmap_selector(self, cmap_value):
        self.cmapSelector.setCurrentText(cmap_value)

    def _generate_cmap_thumbnails(self):
        self.cmap_names = ["gray", "viridis", "plasma", "inferno", "magma", "cividis"]
        thumbnails = []
        for cmap_name in self.cmap_names:
            gradient = np.linspace(0, 1, 256)
            gradient = np.vstack((gradient, gradient))
            fig, ax = plt.subplots(figsize=(4, 1))
            ax.imshow(gradient, aspect="auto", cmap=cmap_name)
            ax.set_xticks([])
            ax.set_yticks([])
            fig.tight_layout(pad=0)
            fig.canvas.draw()
            thumbnails.append(np.array(fig.canvas.renderer.buffer_rgba()))
            plt.close()
        return thumbnails

    def _numpy_to_QIcon(self, array: np.ndarray):
        height, width, channels = array.shape
        image = QImage(array.tobytes(), width, height, QImage.Format.Format_RGBA8888)
        return QIcon(QPixmap.fromImage(image))

    def _init_contrast_slider(self):
        self.auto_contrast_button = QPushButton("Auto Contrast")
        self.contrastSlider = QRangeSlider()
        self.contrastSlider.setOrientation(Qt.Orientation.Horizontal)
        self.contrastSlider.setRange(0, 255)
        self.contrastSlider.setMaximumWidth(200)

    def update_contrast_slider(self, values):
        self.contrastSlider.blockSignals(True)
        self.contrastSlider.setValue(values)
        self.contrastSlider.blockSignals(False)

    def _populate_toolbar(self):
        # Tabs first
        for button in self.tab_buttons:
            self.toolbar.addWidget(button)

        self.toolbar.addSeparator()

        # Actions & widgets
        self.toolbar.addAction(self.actionReset)
        self.toolbar.addWidget(self.channelSelector)
        self.toolbar.addWidget(self.cmapSelector)
        self.toolbar.addWidget(self.statusLine)
        self.toolbar.addWidget(self.auto_contrast_button)
        self.toolbar.addWidget(self.contrastSlider)

    def _retranslateUI(self):
        _translate = QCoreApplication.translate
        self.toolbar.setWindowTitle(_translate("MainWindow", "toolBar"))
        self.actionReset.setText(_translate("MainWindow", "Reset"))
        self.actionReset.setToolTip(_translate("MainWindow", "Reset Image"))
        self.channelSelector.setToolTip(_translate("MainWindow", "Select a channel"))
