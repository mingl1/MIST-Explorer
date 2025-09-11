import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
from PyQt6.QtCore import QCoreApplication, QSize, Qt, pyqtSignal, pyqtSlot
from PyQt6.QtGui import QAction, QIcon, QImage, QPixmap
from PyQt6.QtWidgets import (
    QApplication,
    QButtonGroup,
    QComboBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSizePolicy,
    QToolBar,
    QToolButton,
    QWidget,
)
from qtrangeslider import QRangeSlider

from ui.toolbar.Action import Action
from utils import resource_path


class ToolBarUI(QToolBar):
    # Public signals
    tabChanged = pyqtSignal(int)
    channelChanged = pyqtSignal(int)
    cmapChanged = pyqtSignal(str)

    def __init__(self, parent):
        super().__init__(parent=parent)
        self._init_tab_buttons()
        self._init_actions(parent)
        # self._init_channel_selector(parent)
        self._init_status_line()
        self._init_cmap_selector(parent)
        self._init_contrast_slider()
        self._init_channel_selector(parent)

        self._populate_toolbar()

        self._retranslateUI()

    def _init_tab_buttons(self):
        self.tab_buttons = []
        tab_names = ["Images", "Data Processing", "View", "Analysis"]
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
                background: rgba(0, 0, 0, 0.3);
            }
            QToolButton:checked {
                border-bottom: 2px solid #007AFF;
                font-weight: Bold;
            }
        """

    @pyqtSlot(int)
    def onTabButtonClicked(self, index):
        self.tabChanged.emit(index)
        if index==2 or index==3:  # View or Analysis tab
            # hide all actions
            self.actionReset.setVisible(False)
            self.channel_selector_action.setVisible(True)
            self.cmap_action.setVisible(False)
            self.auto_contrast_button_action.setVisible(False)
            self.contrast_slider_action.setVisible(False)

        else:
            self.actionReset.setVisible(True)
            self.channel_selector_action.setVisible(True)
            self.cmap_action.setVisible(True)
            self.auto_contrast_button_action.setVisible(True)
            self.contrast_slider_action.setVisible(True)

    def _init_actions(self, parent):
        self.actionRotate = Action(
            parent, "actionRotate", resource_path("assets/icons/rotate-right.png")
        )
        self.actionReset = Action(
            parent, "actionReset", resource_path("assets/icons/reset.png")
        )
        self.actionOpenBrightnessContrast = Action(
            parent, "actionBC", resource_path("assets/icons/brightness.png")
        )

    def _init_channel_selector(self, parent):
        self.channelSelector = QComboBox(parent)
        self.channelSelector.setMinimumWidth(100)
        self.channelSelector.currentIndexChanged.connect(
            self.on_channelSelector_currentIndexChanged
        )

    def updateChannelSelector(self, channels: dict, clear=False):
        self.initialized = False
        if clear:
            self.clearChannelSelector()
        channel_keys = sorted(
            channels.keys(), key=lambda x: int(x.replace("Channel ", ""))
        )
        self.channelSelector.addItems(channel_keys)
        return
    def setChannelSelector(self, channel:int):
        self.channelSelector.blockSignals(True)
        self.channelSelector.setCurrentIndex(channel)
        self.channelSelector.blockSignals(False)

    def clearChannelSelector(self):
        self.channelSelector.clear()
        return

    @pyqtSlot(int)
    def on_channelSelector_currentIndexChanged(self, index):
        if self.initialized is False:
            self.initialized = True
        if self.channelSelector.count() > 0:
            self.channelChanged.emit(index)

    def _init_status_line(self):
        self.statusLine = QLabel("Welcome! Please load an image to get started.")
        self.statusLine.setSizePolicy(
            QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed
        )
        self.statusLine.setAlignment(
            Qt.AlignmentFlag.AlignLeft
        )
        self.statusLine.setStyleSheet("margin: 10px;")

    def _init_cmap_selector(self, parent):
        self.cmapSelector = QComboBox(parent)
        self.cmapSelector.setMinimumSize(QSize(200, 20))
        self.cmapSelector.setMaximumWidth(200)
        self.cmapSelector.currentTextChanged.connect(self.on_cmapTextChanged)

        thumbnails = self._generate_cmap_thumbnails()
        for index, thumbnail in enumerate(thumbnails):
            icon = self._numpy_to_QIcon(thumbnail)
            self.cmapSelector.setIconSize(QSize(80, 20))
            self.cmapSelector.addItem(icon, self.cmap_names[index])

    @pyqtSlot(str)
    def on_cmapTextChanged(self, cmap_str: str):
        self.cmapChanged.emit(cmap_str)

    def update_cmap_selector(self, cmap_value):
        self.cmapSelector.setCurrentText(cmap_value)

    def _generate_cmap_thumbnails(self):
        self.cmap_names = [
            "gray",
            "viridis",
            "plasma",
            "inferno",
            "magma",
            "cividis",
            "label_image",
        ]
        thumbnails = []
        for cmap_name in self.cmap_names:
            if cmap_name == "label_image":
                # Make a thumbnail with discrete colors (like skimage.label2rgb default)
                from skimage.color import color_dict

                # Get a list of default categorical colors
                colors = [
                    "red",  # (1, 0, 0)
                    "blue",  # (0, 0, 1)
                    "green",  # (0, 0.502, 0)
                    "gold",  # (1, 0.843, 0)
                    "purple",  # (0.502, 0, 0.502)
                    "orange",  # (1, 0.647, 0)
                    "cyan",  # (0, 1, 1)
                    "magenta",  # (1, 0, 1)
                    "brown",  # (0.647, 0.165, 0.165)
                    "lime",  # (0, 1, 0)
                ]

                colors = [
                    color_dict[i] for i in colors
                ]  # take first N colors (e.g. 10)
                n = len(colors)

                # Make an array of shape (1, n) with categorical colors
                gradient = np.arange(n).reshape(1, -1)

                fig, ax = plt.subplots(figsize=(4, 1))
                cmap = ListedColormap(colors)
                ax.imshow(gradient, aspect="auto", cmap=cmap)
                ax.set_xticks([])
                ax.set_yticks([])
                fig.tight_layout(pad=0)
                fig.canvas.draw()
                thumbnails.append(np.array(fig.canvas.renderer.buffer_rgba()))
                plt.close()
            else:
                # Continuous cmap preview
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
        self.auto_contrast_button = QPushButton("Auto Contrast", self)
        self.contrast_slider = QRangeSlider(parent=self)
        self.contrast_slider.setOrientation(Qt.Orientation.Horizontal)
        self.contrast_slider.setRange(0, 255)
        self.contrast_slider.setMaximumWidth(200)
        self.contrast_slider.setMinimumWidth(100)

    def update_contrast_slider(self, values):
        self.contrast_slider.blockSignals(True)
        self.contrast_slider.setValue(values)
        self.contrast_slider.blockSignals(False)

    def _create_tab_button_bar(self, fixed_width=400):
        container = QWidget()
        layout = QHBoxLayout(container)
        layout.setContentsMargins(8, 0, 0, 0)
        layout.setSpacing(0)

        for btn in self.tab_buttons:
            btn.setSizePolicy(
                QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred
            )
            layout.addWidget(btn)

        container.setFixedWidth(fixed_width)

        return container

    def _populate_toolbar(self):
        tab_bar = self._create_tab_button_bar(fixed_width=488)
        self.addWidget(tab_bar)
        self.addSeparator()
        self.addAction(self.actionReset)
        self.cmap_action = self.addWidget(self.cmapSelector)
        self.channel_selector_action = self.addWidget(self.channelSelector)
        self.auto_contrast_button_action = self.addWidget(self.auto_contrast_button)
        self.contrast_slider_action = self.addWidget(self.contrast_slider)
        self.addWidget(self.statusLine)

        self.disable_actions()

    def enable_actions(self):
        assert self.cmap_action is not None
        assert self.channel_selector_action is not None
        assert self.auto_contrast_button_action is not None
        assert self.contrast_slider_action is not None
        self.actionReset.setEnabled(True)
        self.cmap_action.setEnabled(True)
        self.auto_contrast_button_action.setEnabled(True)
        self.contrast_slider_action.setEnabled(True)

    def disable_actions(self):
        assert self.cmap_action is not None
        assert self.channel_selector_action is not None
        assert self.auto_contrast_button_action is not None
        assert self.contrast_slider_action is not None
        self.actionReset.setEnabled(False)
        self.cmap_action.setEnabled(False)
        self.auto_contrast_button_action.setEnabled(False)
        self.contrast_slider_action.setEnabled(False)

    def _retranslateUI(self):
        _translate = QCoreApplication.translate
        self.setWindowTitle(_translate("MainWindow", "toolBar"))
        self.actionReset.setText(_translate("MainWindow", "Reset"))
        self.actionReset.setToolTip(_translate("MainWindow", "Reset Image"))
