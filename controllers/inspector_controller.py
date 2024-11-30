from project.ui.inspector_window import InspectorWindow


class InspectorController:
    def __init__(self, inspector: InspectorWindow):
        self.isMaximized = True
        self.view = inspector
        self.view.changeSizeRequested.connect(self.change_size)

    def change_size(self):
        """Приховати праву панель."""
        if self.isMaximized:
            self.previousWidth = self.view.size().width()
            self.view.setFixedWidth(50)
            self.isMaximized = False
        else:
            self.view.setFixedWidth(self.previousWidth)
            self.isMaximized = True
