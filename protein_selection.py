from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPixmap
import sys, os


class Group(QtWidgets.QWidget):
    pass


class Protein_Selector(QtWidgets.QGroupBox):
    def __init__(self, container, size=(100,100), objectName=""):
        super().__init__(container)
        self.setMinimumSize(QtCore.QSize(size[0], size[1]))
        self.setTitle("")
        self.setObjectName(objectName)
        self.horizontalLayout = QtWidgets.QHBoxLayout(self)
        self.protein_vlayout = QtWidgets.QVBoxLayout()
        self.protein_combobox = QtWidgets.QComboBox(self)
        self.protein_combobox.addItem("")
        self.protein_vlayout.addWidget(self.protein_combobox)
        self.protein_contrast_slider = QtWidgets.QSlider(self)
        self.protein_contrast_slider.setOrientation(QtCore.Qt.Orientation.Horizontal)
        self.protein_vlayout.addWidget(self.protein_contrast_slider)
        self.horizontalLayout.addLayout(self.protein_vlayout)

