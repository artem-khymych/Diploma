from typing import List, Set, Dict
from collections.abc import Iterator, Callable

from pandas import DataFrame
from sklearn.neural_network import MLPClassifier, MLPRegressor

from project.logic.evaluation.metric_strategy import (MetricStrategy, ClassificationMetric, RegressionMetric,
                                                      ClusteringMetric, AnomalyDetectionMetric, DensityEstimationMetric,
                                                      DimReduction, TimeSeriesMetric)
from project.logic.experiment.input_data_params import InputDataParams
from project.logic.modules import task_names


class Experiment:

    def __init__(self, id, task, model, params):
        self.id: int = id
        self.task: str = task
        self.model: Callable[[Dict]] = model
        self._params: Dict[str:any] = params
        self._name = f"Experiment {id}"
        self.description = ""
        self.input_data_params = InputDataParams()

        self.train_data: DataFrame
        self.test_data: DataFrame

        self.train_time: float
        self.is_finished: bool = False
        self.metric_strategy: MetricStrategy

    @property
    def name(self) -> str:
        return self._name

    @name.setter
    def name(self, new_name):
        if not isinstance(new_name, str):
            raise ValueError("Назва має бути рядком")
        self._name = new_name

    @property
    def params(self):
        return self._params

    @params.setter
    def params(self, new_params):
        self._params = new_params

    def _choose_metric_strategy(self) -> MetricStrategy:
        # get metric strategy for standart ML tasks
        if self.task == task_names.CLASSIFICATION:
            self.metric_strategy = ClassificationMetric()
        elif self.task == task_names.REGRESSION:
            self.metric_strategy = RegressionMetric()
        elif self.task == task_names.CLUSTERING:
            self.metric_strategy = ClusteringMetric()
        elif self.task == task_names.ANOMALY_DETECTION:
            self.metric_strategy = AnomalyDetectionMetric()
        elif self.task == task_names.DENSITY_ESTIMATION:
            self.metric_strategy = DensityEstimationMetric()
        elif self.task == task_names.DIMENSIONALITY_REDUCTION:
            self.metric_strategy = DimReduction()
        elif self.task == task_names.TIME_SERIES:
            self.metric_strategy = TimeSeriesMetric()

        # get metric strategy if we have nn from scikit task
        if self.task == task_names.MLP:
            self.metric_strategy = self.__get_metric_strategy_for_mlp()

    def __get_metric_strategy_for_mlp(self) -> MetricStrategy:
        if self.model == MLPClassifier():
            self.task = task_names.CLASSIFICATION
            return ClassificationMetric()
        elif self.model == MLPRegressor():
            self.task = task_names.REGRESSION
            return RegressionMetric()
        else:
            raise TypeError(f"Unrecognized model: {self.model}")
