import inspect

from PyQt5.QtCore import pyqtSignal, QObject
from sklearn.base import ClassifierMixin, RegressorMixin, is_regressor, is_classifier

from .experiment.experiment import Experiment
from .modules import models_manager
from ..ui.node import Node


class ExperimentManager(QObject):

    def __init__(self):
        super().__init__()
        self.experiments = {}
        self.current_node = None
        self.current_model = None
        self.current_params = None

    def get_node(self, node: Node):
        self.current_node = node
        self._check_experiment_data()
    def get_ml_model(self,task, model, params):
        self.current_model = model
        self.current_params = params
        self.current_task = task
        self._check_experiment_data()

    def _check_experiment_data(self):
        if self.current_node is not None and self.current_model is not None and self.current_params is not None and self.current_task is not None:
            self.create_new_experiment()
    def create_new_experiment(self):
        experiment = Experiment(self.current_node.id, self.current_task, self.current_model, self.current_params)
        self.experiments[self.current_node.id] = experiment
        print(experiment)

    def get_experiment(self, experiment_id):
        if experiment_id in self.experiments:
            return self.experiments[experiment_id]


