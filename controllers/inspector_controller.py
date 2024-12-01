from qtpy import QtGui
from PyQt5.QtWidgets import QApplication, QSizePolicy
from project.ui.inspector_window import InspectorWindow


class InspectorController:
    def __init__(self, inspector: InspectorWindow):
        self.isMaximized = True
        self.view = inspector
        self.view.changeSizeRequested.connect(self.change_size)

    def change_size(self):
        #TODO fix button 
        """Приховати праву панель."""
        if self.isMaximized:
            self.previousWidth = self.view.size().width()
            self.view.setFixedWidth(self.view.MINIMUM_SIZE)
            self.isMaximized = False
            self.view.object_list.setVisible(False)
            self.view.changeSize.setIcon(self.view.style().standardIcon(QApplication.style().SP_ArrowRight))
        else:
            self.view.setFixedWidth(self.previousWidth)
            self.view.setMinimumWidth(0)
            self.view.setMaximumWidth(1920)
            self.isMaximized = True
            self.view.changeSize.setIcon(self.view.style().standardIcon(QApplication.style().SP_ArrowLeft))
            self.view.object_list.setVisible(True)
