from PyQt6.QtWidgets import QHBoxLayout, QGroupBox, QToolButton, QWidget
from PyQt6.QtCore import QSize, QCoreApplication
from ui.toolbar.Action import Action

class TissueUI(QWidget):
    def __init__(self, parent=None, main_layout=None):
        super().__init__()
        self.tissue_groupbox = QGroupBox(parent)
        self.tissue_groupbox.setTitle("Tissue Image")
        self.tissue_groupbox.setMinimumWidth(100)
        self.tissue_groupbox.setMaximumHeight(80)
        
        # Create button to replace channel 2
        self.replace_channel_button = QToolButton(self.tissue_groupbox)
        replaceAction = Action(self.tissue_groupbox, "replaceChannelAction", "assets/icons/replace.png")
        self.replace_channel_button.setDefaultAction(replaceAction)
        self.replace_channel_button.setIconSize(QSize(40,40))
        
        # Create button to align tissue image
        self.align_button = QToolButton(self.tissue_groupbox)
        alignAction = Action(self.tissue_groupbox, "alignTissueAction", "assets/icons/align.png")
        self.align_button.setDefaultAction(alignAction)
        self.align_button.setIconSize(QSize(40,40))

        # Layout
        tissue_layout = QHBoxLayout(self.tissue_groupbox)
        tissue_layout.addWidget(self.align_button)
        tissue_layout.addWidget(self.replace_channel_button)

        self.__retranslateUI()

    def __retranslateUI(self):
        _translate = QCoreApplication.translate
        self.align_button.setToolTip(_translate("MainWindow", "Align Tissue Image"))
        self.replace_channel_button.setToolTip(_translate("MainWindow", "Replace Channel 2 with Tissue")) 