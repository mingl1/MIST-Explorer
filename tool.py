from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPixmap
import sys, os


class Tool(QtWidgets.QAction):
    def __init__(self, window, tool_name, icon_file_path="", is_checkable=False):
        super().__init__(window)
        self.setCheckable(is_checkable)
        self.setObjectName(tool_name)
        tool_icon = QtGui.QIcon()
        tool_icon.addPixmap(QtGui.QPixmap(icon_file_path), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.setIcon(tool_icon)
        self.setToolTip(tool_name)



class Rotate_Dialog(QtWidgets.QDialog):
    def __init__(self, canvas:QtWidgets.QGraphicsView, pixmap:QtGui.QPixmap = None):
        super().__init__()
        self.pixmap = pixmap
        self.canvas = canvas
        self.setObjectName("Dialog")
        self.setWindowTitle("Rotate")
        self.resize(274, 72)
        self.horizontalLayout_2 = QtWidgets.QHBoxLayout(self)
        self.horizontalLayout_2.setObjectName("horizontalLayout_2")
        self.horizontalLayout = QtWidgets.QHBoxLayout()
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.verticalLayout = QtWidgets.QVBoxLayout()
        self.verticalLayout.setObjectName("verticalLayout")
        self.label = QtWidgets.QLabel(self)
        self.label.setObjectName("label")
        self.verticalLayout.addWidget(self.label)
        self.lineEdit = QtWidgets.QLineEdit(self)
        self.lineEdit.setObjectName("lineEdit")
        self.verticalLayout.addWidget(self.lineEdit)
        self.horizontalLayout.addLayout(self.verticalLayout)
        self.buttonBox = QtWidgets.QDialogButtonBox(self)
        self.buttonBox.setOrientation(QtCore.Qt.Orientation.Vertical)
        self.buttonBox.setStandardButtons(QtWidgets.QDialogButtonBox.StandardButton.Cancel|QtWidgets.QDialogButtonBox.StandardButton.Ok)
        self.buttonBox.setObjectName("buttonBox")
        self.horizontalLayout.addWidget(self.buttonBox)
        self.horizontalLayout_2.addLayout(self.horizontalLayout)
        self.buttonBox.accepted.connect(self.accept) 
        self.buttonBox.rejected.connect(self.reject) 
        QtCore.QMetaObject.connectSlotsByName(self)
        self.label.setText("Enter degrees of rotation")

    def accept(self):
        transform = QtGui.QTransform()
        transform.rotate(int(self.lineEdit.text()))
        self.canvas.setTransform(transform)
        self.destroy()

