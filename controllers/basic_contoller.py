from PyQt5.uic.properties import QtCore
from qtpy import QtGui
from PyQt5.QtWidgets import QApplication, QSizePolicy
from project.ui.inspector_window import InspectorWindow


class BasicController:
    def __init__(self, view):
        self.isMaximized = True
        self.view = view
        self.view.changeSizeRequested.connect(self.change_size)
        self.close_arrow = self.view.style().standardIcon(QApplication.style().SP_ArrowLeft)
        self.open_arrow = self.view.style().standardIcon(QApplication.style().SP_ArrowRight)
    def change_size(self):
        sender = self.view.changeSize
        parent_layout = self.view.parentWidget()
        if self.isMaximized:
            self.previousWidth = self.view.size().width()
            self.view.setFixedWidth(self.view.MINIMUM_SIZE)
            self.isMaximized = False
            self.view.changeSize.setIcon(self.open_arrow)

            for i in range(self.view.layout.count()):
                widget = self.view.layout.itemAt(i).widget()
                if widget and widget != sender:
                    widget.setVisible(False)

        else:
            self.view.setFixedWidth(self.previousWidth)
            self.view.setMinimumWidth(0)
            self.view.setMaximumWidth(1920)
            self.isMaximized = True
            self.view.changeSize.setIcon(self.close_arrow)

            for i in range(self.view.layout.count()):
                widget = self.view.layout.itemAt(i).widget()
                if widget and widget != sender:
                    widget.setVisible(True)

            parent_layout.update()
