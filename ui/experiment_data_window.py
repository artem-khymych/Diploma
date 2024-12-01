import sys

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QVBoxLayout, QHBoxLayout,
    QFrame, QPushButton, QListWidget, QTextEdit, QWidget, QSplitter, QGraphicsProxyWidget, QSpacerItem, QSizePolicy,
    QTableWidget
)
from PyQt5.QtCore import Qt, pyqtSignal

from inspector_window import InspectorWindow


class ExperimentDataWindow(InspectorWindow):
    """Class for representing an inspector of experiments"""
    changeSizeRequested = pyqtSignal()

    def __init__(self):
        super().__init__()
        self.object_list = QTableWidget()
        self.object_list.addItems(["Об'єкт 1", "Об'єкт 2", "Об'єкт 3", "Об'єкт 4"])
        self.layout.addWidget(self.object_list)
        self.changeSize.setIcon(self.style().standardIcon(QApplication.style().SP_ArrowRight))
