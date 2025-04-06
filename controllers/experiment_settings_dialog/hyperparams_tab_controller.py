from abc import ABC

from project.controllers.experiment_settings_dialog.tab_controller import TabController
from project.logic.experiment.experiment import Experiment
from project.ui.experiment_settings_dialog.hypeparams_tab import HyperparamsTabWidget


class HyperparamsTabController(TabController, ABC):
    """Контролер для вкладки параметрів моделі"""

    def __init__(self, experiment: Experiment, view: HyperparamsTabWidget):
        super().__init__(experiment, view)
        self.init_view()
        self.connect_signals()

    def connect_signals(self):
        pass

    def init_view(self):
        self.view.params_widget.populate_table(self.experiment.params)

    def _update_params(self):
        self.experiment.params = self.view.params_widget.get_current_parameters()
        print(self.view.params_widget.get_current_parameters())

    def update_model_from_view(self):
        self._update_params()
