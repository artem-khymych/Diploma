from typing import List, Set, Dict
from collections.abc import Iterator, Callable

from PyQt5.QtCore import pyqtSignal, QObject
from pandas import DataFrame
from sklearn.neural_network import MLPClassifier, MLPRegressor

from project.logic.evaluation.anomaly_detection_metric import AnomalyDetectionMetric
from project.logic.evaluation.classification_metric import ClassificationMetric
from project.logic.evaluation.clustering_metric import ClusteringMetric
from project.logic.evaluation.density_estimation_metric import DensityEstimationMetric
from project.logic.evaluation.dim_reduction_metric import DimReduction
from project.logic.evaluation.metric_strategy import MetricStrategy, TimeSeriesMetric
from project.logic.evaluation.regression_metric import RegressionMetric
from project.logic.experiment.input_data_params import InputDataParams
from project.logic.modules import task_names
import time
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


class Experiment(QObject):
    experiment_finished = pyqtSignal(float, object, object)

    def __init__(self, id, task, model, params):
        super().__init__()
        self.id: int = id
        self.task: str = task
        self._name = f"Experiment {id}"
        self.description = ""

        self.model: Callable[[Dict]] = model
        self.trained_model = None
        self._params: Dict[str:any] = params
        self.input_data_params = InputDataParams()

        # Зберігання даних у класичному форматі
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None

        self.train_time: float = 0
        self.is_finished: bool = False
        self.metric_strategy: MetricStrategy

        # Для зберігання результатів
        self.train_predictions = None
        self.test_predictions = None
        self.train_actual = None
        self.test_actual = None

        # Для зберігання результатів трансформацій
        self.transformed_train = None
        self.transformed_test = None

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
        Запускає експеримент машинного навчання з урахуванням вхідних параметрів.
        Обробляє дані, навчає модель та оцінює результати на тренувальному та тестовому наборах.
        """
        # Завантаження та підготовка даних
        self._load_data()

        # Вибір відповідної метрики для оцінки моделі
        self._choose_metric_strategy()

        # Навчання моделі та вимірювання часу
        start_time = time.time()

        # Створення моделі з переданими параметрами
        model_instance = type(self.model)(**self._params)

        # Різна логіка навчання в залежності від типу завдання
        if self.task in [task_names.CLASSIFICATION, task_names.REGRESSION, task_names.MLP]:
            # Навчання моделі
            model_instance.fit(self.X_train, self.y_train)

            # Збереження результатів (як для тренувального, так і для тестового наборів)
            self.train_predictions = model_instance.predict(self.X_train)
            self.test_predictions = model_instance.predict(self.X_test)
            self.train_actual = self.y_train
            self.test_actual = self.y_test

        elif self.task in [task_names.CLUSTERING, task_names.ANOMALY_DETECTION, task_names.DENSITY_ESTIMATION]:
            # Для unsupervised завдань не потрібна цільова змінна
            model_instance.fit(self.X_train)

            # Отримання результатів для обох наборів
            if hasattr(model_instance, 'predict'):
                self.train_predictions = model_instance.predict(self.X_train)
                self.test_predictions = model_instance.predict(self.X_test)
            elif hasattr(model_instance, 'fit_predict') and self.task == task_names.CLUSTERING:
                # Для деяких кластеризаційних алгоритмів
                # Для тренувального набору використовуємо результати з fit
                self.train_predictions = model_instance.labels_ if hasattr(model_instance,
                                                                           'labels_') else model_instance.fit_predict(
                    self.X_train)
                # Для тестового набору
                self.test_predictions = model_instance.fit_predict(self.X_test)

        elif self.task == task_names.DIMENSIONALITY_REDUCTION:
            # Навчання моделі зниження розмірності
            #model_instance.fit(self.X_train, y=None)

            # Трансформація даних для обох наборів
            self.transformed_train = model_instance.fit_transform(self.X_train, y=None)
            self.transformed_test = model_instance.fit_transform(self.X_test, y=None)

        elif self.task == task_names.TIME_SERIES:
            # Специфічна обробка для часових рядів
            # Логіка залежить від конкретної реалізації та моделі
            pass

        # Збереження часу навчання
        self.train_time = time.time() - start_time

        # Зберігаємо навчену модель
        self.trained_model = model_instance

        # Позначаємо експеримент як завершений
        self.is_finished = True

        # Обчислення та виведення результатів метрик для обох наборів
        train_metrics, test_metrics = self._calculate_metrics()
        print("Train metrics:", train_metrics)
        print("Test metrics:", test_metrics)

        # Відправляємо сигнал з результатами експерименту
        self.experiment_finished.emit(self.train_time, train_metrics, test_metrics)
        return

    def _load_data(self):
        """
        Завантажує дані в залежності від параметрів вхідних даних.
        Обробляє категоріальні змінні та розділяє дані на X_train, X_test, y_train, y_test.
        """
        params = self.input_data_params

        if params.mode == 'single_file':
            # Завантаження з одного файлу
            data = self._load_file(params.single_file_path)

            # Обробка даних перед розділенням
            if not params.is_target_not_required():
                # Для supervised learning
                X = data.drop(params.target_variable, axis=1)
                y = data[params.target_variable]

                # Розділення на навчальну та тестову вибірки
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y,
                    test_size=params.test_percent / 100,
                    random_state=params.seed,
                    stratify=y if self.task == task_names.CLASSIFICATION else None
                )

                # Обробка категоріальних змінних для X_train та X_test
                X_combined = pd.concat([X_train, X_test])
                X_combined_encoded = self._encode_categorical_variables(X_combined)

                # Розділення назад після обробки
                self.X_train = X_combined_encoded.iloc[:len(X_train)]
                self.X_test = X_combined_encoded.iloc[len(X_train):]

                # Обробка y, якщо це категоріальна змінна і задача класифікації
                if self.task == task_names.CLASSIFICATION and y.dtype == 'object':
                    y_combined = pd.concat([y_train, y_test])
                    y_combined_encoded = self._encode_categorical_variables(pd.DataFrame(y_combined))

                    self.y_train = y_combined_encoded.iloc[:len(y_train)].values.ravel()
                    self.y_test = y_combined_encoded.iloc[len(y_train):].values.ravel()
                else:
                    self.y_train = y_train
                    self.y_test = y_test

            else:
                # Для unsupervised learning
                # Розділення на навчальну та тестову вибірки
                X_train, X_test = train_test_split(
                    data,
                    test_size=params.test_percent / 100,
                    random_state=params.seed
                )

                # Обробка категоріальних змінних для X_train та X_test
                X_combined = pd.concat([X_train, X_test])
                X_combined_encoded = self._encode_categorical_variables(X_combined)

                # Розділення назад після обробки
                self.X_train = X_combined_encoded.iloc[:len(X_train)]
                self.X_test = X_combined_encoded.iloc[len(X_train):]

                # Для unsupervised навчання y не потрібен
                self.y_train = None
                self.y_test = None
        else:
            # Завантаження з окремих файлів
            X_train = self._load_file(params.x_train_file_path)
            X_test = self._load_file(params.x_test_file_path)

            # Обробка категоріальних змінних (спільна обробка для узгодженості кодування)
            X_combined = pd.concat([X_train, X_test])
            X_combined_encoded = self._encode_categorical_variables(X_combined)

            # Розділення назад після обробки
            self.X_train = X_combined_encoded.iloc[:len(X_train)]
            self.X_test = X_combined_encoded.iloc[len(X_train):]

            if not params.is_target_not_required():
                # Для supervised learning
                y_train = self._load_file(params.y_train_file_path)
                y_test = self._load_file(params.y_test_file_path)

                # Перевірка, чи це DataFrame і обробка відповідно
                if isinstance(y_train, pd.DataFrame):
                    if len(y_train.columns) == 1:
                        y_train = y_train.iloc[:, 0]
                    else:
                        # Якщо кілька колонок, обробляємо першу
                        y_train = y_train.iloc[:, 0]

                if isinstance(y_test, pd.DataFrame):
                    if len(y_test.columns) == 1:
                        y_test = y_test.iloc[:, 0]
                    else:
                        # Якщо кілька колонок, обробляємо першу
                        y_test = y_test.iloc[:, 0]

                # Обробка категоріальних змінних для y, якщо необхідно
                if self.task == task_names.CLASSIFICATION and pd.api.types.is_object_dtype(y_train):
                    y_combined = pd.concat([y_train, y_test])
                    y_combined = pd.DataFrame(y_combined, columns=['target'])
                    y_combined_encoded = self._encode_categorical_variables(y_combined)

                    self.y_train = y_combined_encoded.iloc[:len(y_train)].values.ravel()
                    self.y_test = y_combined_encoded.iloc[len(y_train):].values.ravel()
                else:
                    self.y_train = y_train
                    self.y_test = y_test
            else:
                # Для unsupervised learning
                self.y_train = None
                self.y_test = None

    def _load_file(self, file_path):
        """
        Завантажує дані з файлу різних форматів.
        Підтримуються формати: CSV, Excel, JSON, Parquet.

        Args:
            file_path (str): Шлях до файлу з даними

        Returns:
            DataFrame: Завантажені дані у вигляді pandas DataFrame
        """
        file_extension = file_path.split('.')[-1].lower()

        if file_extension == 'csv':
            return pd.read_csv(
                file_path,
                encoding=self.input_data_params.file_encoding,
                sep=self.input_data_params.file_separator
            )
        elif file_extension in ['xlsx', 'xls']:
            return pd.read_excel(file_path)
        elif file_extension == 'json':
            return pd.read_json(file_path)
        elif file_extension == 'parquet':
            return pd.read_parquet(file_path)
        else:
            raise ValueError(f"Непідтримуваний формат файлу: {file_extension}")

    def _encode_categorical_variables(self, data):
        """
        Обробляє категоріальні змінні у DataFrame згідно з обраним методом кодування.

        Args:
            data (DataFrame): Дані для обробки

        Returns:
            DataFrame: Дані з обробленими категоріальними змінними
        """
        categorical_columns = data.select_dtypes(include=['object', 'category']).columns

        if len(categorical_columns) == 0:
            return data

        result_data = data.copy()

        # Вибір методу кодування категоріальних змінних
        encoding_method = self.input_data_params.categorical_encoding

        if encoding_method == 'one-hot':
            # One-hot кодування (всі категорії окремими стовпцями)
            for column in categorical_columns:
                # Створення one-hot кодування
                one_hot = pd.get_dummies(
                    result_data[column],
                    prefix=column,
                    drop_first=False
                )

                # Видалення оригінального стовпця і додавання закодованих стовпців
                result_data = pd.concat([result_data.drop(column, axis=1), one_hot], axis=1)

        elif encoding_method == 'to_categorical':
            from sklearn.preprocessing import LabelEncoder

            for column in categorical_columns:
                # Створюємо енкодер і навчаємо його на даних
                label_encoder = LabelEncoder()
                result_data[column] = label_encoder.fit_transform(result_data[column])

        return result_data

    def _calculate_metrics(self):
        """
        Обчислює метрики продуктивності моделі для тренувального та тестового наборів.

        Returns:
            tuple: (train_metrics, test_metrics) - кортеж зі словниками метрик для кожного набору
        """
        if not self.is_finished:
            raise BlockingIOError("The experiment isn't finished")

        train_metrics = {}
        test_metrics = {}

        if self.task in [task_names.CLASSIFICATION, task_names.REGRESSION, task_names.MLP]:
            # Метрики для тренувального набору
            train_metrics = self.metric_strategy.evaluate(self.train_actual, self.train_predictions)
            # Метрики для тестового набору
            test_metrics = self.metric_strategy.evaluate(self.test_actual, self.test_predictions)

        elif self.task == task_names.CLUSTERING:
            # Для кластеризації можуть використовуватись внутрішні метрики
            train_metrics = self.metric_strategy.evaluate(self.X_train, self.train_predictions)
            test_metrics = self.metric_strategy.evaluate(self.X_test, self.test_predictions)

        elif self.task == task_names.DIMENSIONALITY_REDUCTION:
            # Для зниження розмірності
            train_metrics = self.metric_strategy.evaluate(self.X_train, self.transformed_train)
            test_metrics = self.metric_strategy.evaluate(self.X_test, self.transformed_test)

        elif self.task == task_names.ANOMALY_DETECTION:
            # Для виявлення аномалій
            train_metrics = self.metric_strategy.evaluate(self.y_train, self.train_predictions)
            test_metrics = self.metric_strategy.evaluate(self.y_test, self.test_predictions)

        elif self.task == task_names.DENSITY_ESTIMATION:
            # Для оцінки щільності
            train_metrics = self.metric_strategy.evaluate(self.X_train, self.train_predictions)
            test_metrics = self.metric_strategy.evaluate(self.X_test, self.test_predictions)

        elif self.task == task_names.TIME_SERIES:
            # Для часових рядів (якщо можливо поділити на тренувальний та тестовий)
            if self.train_actual is not None and self.train_predictions is not None:
                train_metrics = self.metric_strategy.evaluate(self.train_actual, self.train_predictions)
            if self.test_actual is not None and self.test_predictions is not None:
                test_metrics = self.metric_strategy.evaluate(self.test_actual, self.test_predictions)

        # Якщо метрики порожні, додамо повідомлення про помилку
        if not train_metrics:
            train_metrics = {"error": "Неможливо обчислити метрики для тренувального набору"}
        if not test_metrics:
            test_metrics = {"error": "Неможливо обчислити метрики для тестового набору"}

        return train_metrics, test_metrics
