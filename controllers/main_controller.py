from PyQt5.QtWidgets import QMainWindow, QDialog

from project.controllers.experiment_settings_dialog.experiment_settings_controller import ExperimentSettingsController
from project.controllers.inspector_controller import InspectorController
from project.controllers.task_selector_controller import TaskSelectorController
from project.logic.experiment_manager import ExperimentManager
from project.logic.modules.models_manager import ModelsManager
from project.ui.experiment_settings_dialog.experiment_settings_dialog import ExperimentSettingsDialog
from project.ui.main_window import MainWindow
from project.ui.inspector_window import InspectorWindow


class MainController:
    def __init__(self):
        self.view = MainWindow()
        self.models_manager = ModelsManager()
        self.inspector_controller = InspectorController(self.view.inspector_frame,
                                                        self.view.scene,
                                                        self.view.graphics_view)
        self.task_selector_controller = TaskSelectorController(self.view)
        self.experiment_manager = ExperimentManager()
        self.connect_signals()

    def connect_signals(self):
        self.task_selector_controller.request_models_dict.connect(self.models_manager.create_models_dict)
        self.models_manager.models_dict_ready.connect(self.task_selector_controller.handle_models_dict_response)
        self.view.signals.add_new_experiment.connect(self.inspector_controller.node_controller.create_node)
        self.inspector_controller.node_controller.node_created.connect(self.experiment_manager.get_node)
        self.view.signals.add_new_experiment.connect(self.task_selector_controller.show_approach_selection)
        self.task_selector_controller.send_ml_model.connect(self.experiment_manager.get_ml_model)
        self.inspector_controller.node_controller.nodeInfoOpened.connect(self._show_experiment_settings_dialog)

    def _show_experiment_settings_dialog(self, node_id):
        """Функція для відображення діалогу налаштувань експерименту"""
        dialog = ExperimentSettingsDialog(self.view)
        experiment = self.experiment_manager.get_experiment(node_id)
        controller = ExperimentSettingsController(experiment, dialog)
        controller.show()

    def show(self):
        self.view.show()

    def _update_experiment_list(self):
        pass