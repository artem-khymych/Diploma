import os
from abc import abstractmethod, ABC

from PyQt5.QtWidgets import QWidget, QMessageBox

from project.logic.experiment.experiment import Experiment
from project.ui.experiment_settings_dialog.general_tab import GeneralTabWidget
from project.ui.experiment_settings_dialog.hypeparams_tab import HyperparamsTabWidget


class TabController:
    def __init__(self, experiment: Experiment, view: QWidget):
        self.experiment = experiment
        self.view = view

    @abstractmethod
    def connect_signals(self):
        raise NotImplemented

    @abstractmethod
    def update_model_from_view(self):
        raise NotImplemented

    @abstractmethod
    def init_view(self):
        raise NotImplemented





