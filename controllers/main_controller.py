from PyQt5.QtWidgets import QMainWindow
from project.ui.main_window import MainWindow
from project.ui.inspector_window import InspectorWindow

# from project.logic.calculations import calculate_sum

# ІМпорти логіки і юай одночасно
class MainController:
    def __init__(self):
        self.view = MainWindow()
        self.connect_signals()

    def connect_signals(self):
        # Підключення сигналів і слотів
        pass

    def show(self):
        self.view.show()

    def _update_experiment_list(self):
        pass