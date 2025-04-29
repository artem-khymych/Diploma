from abc import ABC

from PyQt5.QtCore import pyqtSignal
from PyQt5.QtWidgets import QMessageBox

from project.controllers.experiment_settings_dialog.ml_tuning_dialog import show_param_tuning_dialog
from project.controllers.experiment_settings_dialog.nn_tuning_dialog import show_nn_tuning_dialog
from project.controllers.experiment_settings_dialog.tab_controller import TabController
from project.logic.experiment.experiment import Experiment
from project.logic.experiment.nn_experiment import NeuralNetworkExperiment
from project.ui.experiment_settings_dialog.hypeparams_tab import HyperparamsTabWidget


class HyperparamsTabController(TabController):
    """Контролер для вкладки параметрів моделі"""
    get_input_data_for_tuning = pyqtSignal()
    def __init__(self, experiment: Experiment, view: HyperparamsTabWidget):
        super().__init__(experiment, view)
        self.init_view()
        self.connect_signals()

    def connect_signals(self):
        self.view.tune_params.clicked.connect(self._tune_params_start)

    def init_view(self):
        self.view.params_widget.populate_table(self.experiment.params)

    def _tune_params_start(self):
        self.get_input_data_for_tuning.emit()
        model = self.experiment.model
        params = self.experiment.params
        X_train, y_train = self.experiment.get_params_for_tune()

        if not isinstance(X_train, type(None)):
            if isinstance(self.experiment, NeuralNetworkExperiment):
                compile_params, fit_params = show_nn_tuning_dialog(model, X_train, y_train)
                self.experiment.params["model_params"] = compile_params
                self.experiment.params["fit_params"] = fit_params
                self.view.params_widget.update_parameters(self.experiment.params)
            else:
                best_params = show_param_tuning_dialog(model, params, X_train, y_train)
                self.experiment.params = best_params
                self.view.params_widget.update_parameters(best_params)
        else:
            QMessageBox.critical(self.view, "Помилка",
                                 "Налаштуйте вхідний датасет")
            return None


    def _update_params(self):
        self.experiment.params = self.view.params_widget.get_current_parameters()
        print(self.view.params_widget.get_current_parameters())

    def update_model_from_view(self):
        self._update_params()
