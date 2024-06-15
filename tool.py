from PyQt6 import QtCore, QtGui, QtWidgets
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QPixmap
import sys, os, canvas


class Tool(QtGui.QAction):
    def __init__(self, window, tool_name, icon_file_path="", is_checkable=False):
        super().__init__(window)
        self.setCheckable(is_checkable)
        self.setObjectName(tool_name)
        tool_icon = QtGui.QIcon()
        tool_icon.addPixmap(QtGui.QPixmap(icon_file_path), QtGui.QIcon.Mode.Normal, QtGui.QIcon.State.Off)
        self.setIcon(tool_icon)
        self.setToolTip(tool_name)
