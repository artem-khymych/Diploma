from typing import List, Set, Dict
from collections.abc import Iterator, Callable

from pandas import DataFrame
from sklearn.neural_network import MLPClassifier, MLPRegressor

from project.logic.evaluation.metric_strategy import (MetricStrategy, ClassificationMetric, RegressionMetric,
                                                      ClusteringMetric, AnomalyDetectionMetric, DensityEstimationMetric,
                                                      DimReduction, TimeSeriesMetric)
from project.logic.experiment.input_data_params import InputDataParams
from project.logic.modules import task_names
import time
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

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

    def run(self):
        """
        Виконує експеримент машинного навчання: завантажує дані, навчає модель,
        обчислює метрики та зберігає результати.
        """
        # Встановлення стратегії метрик відповідно до задачі
        self._choose_metric_strategy()

        # Завантаження даних відповідно до параметрів
        if self.input_data_params.mode == 'single_file':
            # Визначення формату файлу за розширенням
            file_path = self.input_data_params.single_file_path
            file_extension = file_path.split('.')[-1].lower()

            # Завантаження файлу відповідно до формату
            if file_extension == 'csv':
                data = pd.read_csv(
                    file_path,
                    sep=self.input_data_params.single_file_separator,
                    encoding=self.input_data_params.single_file_encoding
                )
            elif file_extension in ['xlsx', 'xls']:
                data = pd.read_excel(file_path)
            elif file_extension == 'json':
                data = pd.read_json(file_path)
            elif file_extension == 'parquet':
                data = pd.read_parquet(file_path)
            else:
                raise ValueError(f"Непідтримуваний формат файлу: {file_extension}")

            # Розділення даних на тренувальну та тестову вибірки
            if self.input_data_params.is_target_not_required():
                # Для задач без цільової змінної
                train_data, test_data = train_test_split(
                    data,
                    test_size=self.input_data_params.test_percent / 100,
                    train_size=self.input_data_params.train_percent / 100,
                    random_state=self.input_data_params.seed
                )
                X_train = train_data
                y_train = None
                X_test = test_data
                y_test = None
            else:
                # Для задач з цільовою змінною
                target = self.input_data_params.target_variable
                if target not in data.columns:
                    raise ValueError(f"Цільова змінна '{target}' не знайдена в даних")

                X = data.drop(columns=[target])
                y = data[target]

                X_train, X_test, y_train, y_test = train_test_split(
                    X, y,
                    test_size=self.input_data_params.test_percent / 100,
                    train_size=self.input_data_params.train_percent / 100,
                    random_state=self.input_data_params.seed
                )
        else:
            # Режим з окремими файлами для тренувальних та тестових даних
            # Завантаження X_train
            X_train = self._load_file(
                self.input_data_params.x_train_file_path,
                self.input_data_params.x_train_file_separator,
                self.input_data_params.x_train_file_encoding
            )

            # Завантаження X_test
            X_test = self._load_file(
                self.input_data_params.x_test_file_path,
                self.input_data_params.x_test_file_separator,
                self.input_data_params.x_test_file_encoding
            )

            # Для задач, які потребують цільові змінні
            if not self.input_data_params.is_target_not_required():
                # Завантаження y_train
                y_train = self._load_file(
                    self.input_data_params.y_train_file_path,
                    self.input_data_params.y_train_file_separator,
                    self.input_data_params.y_train_file_encoding
                )

                # Перетворення DataFrame на Series, якщо потрібно
                if isinstance(y_train, pd.DataFrame) and y_train.shape[1] == 1:
                    y_train = y_train.iloc[:, 0]

                # Завантаження y_test
                y_test = self._load_file(
                    self.input_data_params.y_test_file_path,
                    self.input_data_params.y_test_file_separator,
                    self.input_data_params.y_test_file_encoding
                )

                # Перетворення DataFrame на Series, якщо потрібно
                if isinstance(y_test, pd.DataFrame) and y_test.shape[1] == 1:
                    y_test = y_test.iloc[:, 0]
            else:
                y_train = None
                y_test = None

        # Збереження даних для подальшого використання
        if y_train is not None:
            self.train_data = pd.concat([X_train, pd.Series(y_train, name=self.input_data_params.target_variable)],
                                        axis=1)
            self.test_data = pd.concat([X_test, pd.Series(y_test, name=self.input_data_params.target_variable)], axis=1)
        else:
            self.train_data = X_train
            self.test_data = X_test

        # Створення та навчання моделі
        model_instance = type(self.model)(**self._params)
        self.model = model_instance
        # Вимірювання часу навчання
        start_time = time.time()

        # Навчання моделі відповідно до типу задачі
        if self.input_data_params.is_target_not_required():
            # Для задач без цільової змінної (кластеризація, зниження розмірності, виявлення аномалій, оцінка густини)
            model_instance.fit(X_train)
            y_pred = model_instance.predict(X_test)
        else:
            # Для задач з цільовою змінною (класифікація, регресія, часові ряди)
            model_instance.fit(X_train, y_train)
            y_pred = model_instance.predict(X_test)

        self.train_time = time.time() - start_time

        # Обчислення метрик відповідно до типу задачі
        metrics = {}
        if self.task == task_names.CLASSIFICATION:
            metrics = self.metric_strategy.evaluate(y_test, y_pred)
        elif self.task == task_names.REGRESSION:
            metrics = self.metric_strategy.evaluate(y_test, y_pred)
        elif self.task == task_names.CLUSTERING:
            metrics = self.metric_strategy.evaluate(X_test, y_pred)
        elif self.task == task_names.ANOMALY_DETECTION:
            metrics = self.metric_strategy.evaluate(X_test, y_pred)
        elif self.task == task_names.DENSITY_ESTIMATION:
            metrics = self.metric_strategy.evaluate(X_test, y_pred)
        elif self.task == task_names.DIMENSIONALITY_REDUCTION:
            # Для зниження розмірності часто порівнюють якість реконструкції або збереження дисперсії
            metrics = self.metric_strategy.evaluate(X_test, model_instance.transform(X_test))
        elif self.task == task_names.TIME_SERIES:
            metrics = self.metric_strategy.evaluate(y_test, y_pred)

        # Записати результати
        self.metrics = metrics
        self.trained_model = model_instance
        self.is_finished = True

        return metrics

    def _load_file(self, file_path, separator, encoding):
        """
        Допоміжний метод для завантаження файлів різних форматів
        """
        import pandas as pd

        file_extension = file_path.split('.')[-1].lower()

        if file_extension == 'csv':
            return pd.read_csv(file_path, sep=separator, encoding=encoding)
        elif file_extension in ['xlsx', 'xls']:
            return pd.read_excel(file_path)
        elif file_extension == 'json':
            return pd.read_json(file_path)
        elif file_extension == 'parquet':
            return pd.read_parquet(file_path)
        else:
            raise ValueError(f"Непідтримуваний формат файлу: {file_extension}")
