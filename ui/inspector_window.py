import sys

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QVBoxLayout, QHBoxLayout,
    QFrame, QPushButton, QListWidget, QTextEdit, QWidget, QSplitter, QGraphicsProxyWidget, QSpacerItem, QSizePolicy,
    QTableWidget
)
from PyQt5.QtCore import Qt, pyqtSignal
from qtpy import QtGui

from project.ui.basic_window import BasicWindow


class InspectorWindow(BasicWindow):
    """Class for representing an inspector of experiments"""
    def __init__(self):
        super().__init__()

        self.object_list = QListWidget()
        self.object_list.addItems(["Об'єкт 1", "Об'єкт 2", "Об'єкт 3", "Об'єкт 4"])
        self.layout.addWidget(self.object_list)
        self.layout.addStretch()


