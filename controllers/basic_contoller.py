from PyQt5.uic.properties import QtCore
from qtpy import QtGui
from PyQt5.QtWidgets import QApplication, QSizePolicy
from project.ui.inspector_window import InspectorWindow


class BasicController:
    def __init__(self, view):
        self.isMaximized = True
        self.view = view
        self.close_arrow = self.view.style().standardIcon(QApplication.style().SP_ArrowLeft)
        self.open_arrow = self.view.style().standardIcon(QApplication.style().SP_ArrowRight)

