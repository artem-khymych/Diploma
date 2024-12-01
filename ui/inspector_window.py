import sys

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QVBoxLayout, QHBoxLayout,
    QFrame, QPushButton, QListWidget, QTextEdit, QWidget, QSplitter, QGraphicsProxyWidget, QSpacerItem, QSizePolicy,
    QTableWidget
)
from PyQt5.QtCore import Qt, pyqtSignal
from qtpy import QtGui


class InspectorWindow(QFrame):
    """Class for representing an inspector of experiments"""
    changeSizeRequested = pyqtSignal()

    def __init__(self):
        super().__init__()
        self.MINIMUM_SIZE = 20
        self.setWindowTitle("Experiments inspector")

        self.setMinimumSize(self.MINIMUM_SIZE, self.height())
        self.layout = QVBoxLayout(self)
        self.layout.setContentsMargins(0, 0, 0, 0)
        self.layout.setSpacing(0)

        # Список об'єктів
        self.object_list = QListWidget()
        self.object_list.addItems(["Об'єкт 1", "Об'єкт 2", "Об'єкт 3", "Об'єкт 4"])

        self.layout.addWidget(self.object_list)

        # Кнопки для керування зоною
        self.changeSize = QPushButton()
        self.changeSize.setIcon(self.style().standardIcon(QApplication.style().SP_ArrowLeft))
        self.changeSize.setFixedWidth(self.MINIMUM_SIZE)

        self.layout.addWidget(self.changeSize)
        self.layout.addWidget(self.object_list)

        self.changeSize.clicked.connect(self.changeSizeRequested.emit)

        self.layout.addStretch()


