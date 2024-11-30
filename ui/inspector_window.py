import sys

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QVBoxLayout, QHBoxLayout,
    QFrame, QPushButton, QListWidget, QTextEdit, QWidget, QSplitter
)
from PyQt5.QtCore import Qt, pyqtSignal


class InspectorWindow(QFrame):
    """Class for representing an inspector of experiments"""
    changeSizeRequested = pyqtSignal()  # Сигнал для зменшення

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Experiments inspector")
        self.setMinimumSize(10, self.height())
        layout = QVBoxLayout(self)

        # Список об'єктів
        self.object_list = QListWidget()
        self.object_list.addItems(["Об'єкт 1", "Об'єкт 2", "Об'єкт 3", "Об'єкт 4"])

        layout.addWidget(self.object_list)

        # Кнопки для керування зоною
        self.changeSize = QPushButton("Згорнути праву панель")

        layout.addWidget(self.changeSize)

        # Підключення кнопок до функцій
        self.changeSize.clicked.connect(self.changeSizeRequested.emit)

        layout.addStretch()


