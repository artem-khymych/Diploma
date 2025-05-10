from enum import Enum
from dataclasses import dataclass
from typing import Dict, List, Set, Type


class NNModelType(Enum):
    """Типи нейронних мереж"""
    GENERIC = "GENERIC"  # Інші типи моделей


class TaskType(Enum):
    CLASSIFICATION = "Classification"
    REGRESSION = "Regression"
    #CLUSTERING = "Clustering"
    #DIMENSIONALITY_REDUCTION = "Dimensionality Reduction"
    #ANOMALY_DETECTION = "Anomaly Detection"
    #DENSITY_ESTIMATION = "Density Estimation"
    TIME_SERIES_FORECASTING = "Time Series Forecasting"


@dataclass
class TaskConfig:
    """Конфігурація завдання для нейронної мережі"""
    task_type: TaskType  # Тип завдання
    metric_class: Type  # Клас метрики для оцінки моделі
    description: str  # Опис завдання


class ModelTaskRegistry:
    """Реєстр, який зберігає інформацію про типи завдань для різних типів нейронних мереж"""

    def __init__(self):
        # Словник, що відображає тип моделі на список підтримуваних завдань
        self._model_to_tasks: Dict[NNModelType, List[TaskConfig]] = {NNModelType.GENERIC: []}

        # Словник, що відображає тип завдання на список типів моделей, які його підтримують
        self._task_to_models: Dict[TaskType, Set[NNModelType]] = {task_type: set() for task_type in TaskType}

        # Ініціалізація реєстру стандартними відображеннями
        self._initialize_registry()

    def _initialize_registry(self):
        """Ініціалізує реєстр стандартними відображеннями завдань і моделей"""
        # Імпортуємо всі необхідні класи метрик
        from project.logic.evaluation.classification_metric import ClassificationMetric
        from project.logic.evaluation.regression_metric import RegressionMetric
        from project.logic.evaluation.clustering_metric import ClusteringMetric
        from project.logic.evaluation.dim_reduction_metric import DimReduction
        from project.logic.evaluation.anomaly_detection_metric import AnomalyDetectionMetric
        from project.logic.evaluation.density_estimation_metric import DensityEstimationMetric
        from project.logic.evaluation.metric_strategy import TimeSeriesMetric

        # Додаємо всі завдання до GENERIC типу моделі з відповідними метриками
        self.register_task(
            NNModelType.GENERIC,
            TaskConfig(
                TaskType.CLASSIFICATION,
                ClassificationMetric,
                "Метрики для задач класифікації"
            )
        )

        self.register_task(
            NNModelType.GENERIC,
            TaskConfig(
                TaskType.REGRESSION,
                RegressionMetric,
                "Метрики для задач регресії"
            )
        )

        """self.register_task(
            NNModelType.GENERIC,
            TaskConfig(
                TaskType.CLUSTERING,
                ClusteringMetric,
                "Метрики для задач кластеризації"
            )
        )

        self.register_task(
            NNModelType.GENERIC,
            TaskConfig(
                TaskType.DIMENSIONALITY_REDUCTION,
                DimReduction,
                "Метрики для оцінки якості зниження розмірності"
            )
        )

        self.register_task(
            NNModelType.GENERIC,
            TaskConfig(
                TaskType.ANOMALY_DETECTION,
                AnomalyDetectionMetric,
                "Метрики для виявлення аномалій"
            )
        )

        self.register_task(
            NNModelType.GENERIC,
            TaskConfig(
                TaskType.DENSITY_ESTIMATION,
                DensityEstimationMetric,
                "Метрики для оцінки щільності розподілу"
            )
        )"""

        self.register_task(
            NNModelType.GENERIC,
            TaskConfig(
                TaskType.TIME_SERIES_FORECASTING,
                TimeSeriesMetric,
                "Метрики для аналізу часових рядів"
            )
        )

    def register_task(self, model_type, task_config):
        """Реєструє нове завдання для типу моделі"""
        self._model_to_tasks[model_type].append(task_config)
        self._task_to_models[task_config.task_type].add(model_type)

    def get_tasks_for_model(self, model_type: str):
        """Отримує список завдань, які підтримуються певним типом моделі"""
        model = NNModelType(model_type)
        return self._model_to_tasks.get(model, [])

    def get_models_for_task(self, task_type):
        """Отримує список типів моделей, які підтримують певне завдання"""
        return list(self._task_to_models.get(task_type, set()))

    def get_metric_class(self, model_type, task_type):
        """Отримує клас метрики для заданого типу моделі та завдання"""
        for task_config in self._model_to_tasks.get(model_type, []):
            if task_config.task_type == task_type:
                return task_config.metric_class
        return None


class NNMetricFactory:
    """Фабрика для створення відповідних метрик для різних завдань нейронних мереж"""

    def __init__(self, registry=None):
        self.registry = registry or ModelTaskRegistry()

    def create_metric(self, model_type: NNModelType, task_type: TaskType):
        """
        Створює відповідну метрику для оцінки нейронної мережі

        Parameters:
        -----------
        model_type : NNModelType
            Тип нейронної мережі
        task_type : TaskType
            Тип завдання

        Returns:
        -----------
        MetricStrategy
            Екземпляр відповідної стратегії метрики
        """
        metric_class = self.registry.get_metric_class(model_type, task_type)

        if metric_class is None:
            raise ValueError(f"Не знайдено відповідної метрики для моделі {model_type} і завдання {task_type}")

        return metric_class()


# Допоміжна функція для роботи зі строковими назвами
def get_nn_metric(model_type_str: str, task_type_str: str):
    """
    Функція-утиліта для отримання метрики за строковими назвами типу моделі та завдання

    Parameters:
    -----------
    model_type_str : str
        Назва типу моделі (наприклад, "GENERIC")
    task_type_str : str
        Назва типу завдання (наприклад, "Classification", "Regression")

    Returns:
    -----------
    MetricStrategy
        Екземпляр відповідної стратегії метрики
    """
    # Мапимо строкові значення до enum для завдань
    task_type_map = {
        "CLASSIFICATION": TaskType.CLASSIFICATION,
        "REGRESSION": TaskType.REGRESSION,
        #"CLUSTERING": TaskType.CLUSTERING,
        #"DIMENSIONALITY_REDUCTION": TaskType.DIMENSIONALITY_REDUCTION,
        #"ANOMALY_DETECTION": TaskType.ANOMALY_DETECTION,
        #"DENSITY_ESTIMATION": TaskType.DENSITY_ESTIMATION,
        "TIME_SERIES_FORECASTING": TaskType.TIME_SERIES_FORECASTING
    }

    try:
        model_type = NNModelType(model_type_str.upper())
    except ValueError:
        raise ValueError(f"Невідомий тип моделі: {model_type_str}")

    try:
        task_type = task_type_map[task_type_str.upper()]
    except KeyError:
        raise ValueError(f"Невідомий тип завдання: {task_type_str}")

    # Створюємо фабрику та отримуємо метрику
    factory = NNMetricFactory()
    return factory.create_metric(model_type, task_type)
########################################################################################################################
from enum import Enum, auto
from typing import Dict, List, Type
from dataclasses import dataclass
from project.logic.evaluation.metric_strategy import MetricStrategy


class MLTaskType(Enum):
    """Типи завдань машинного навчання"""
    CLASSIFICATION = "Classification"
    REGRESSION = "Regression"
    CLUSTERING = "Clustering"
    DIMENSIONALITY_REDUCTION = "Dimensionality Reduction"
    ANOMALY_DETECTION = "Anomaly Detection"
    DENSITY_ESTIMATION = "Density Estimation"
    MLP = "Scikit-learn MLP models"
    TIME_SERIES = "Time Series"
    OWN_NN = "Import own"


@dataclass
class TaskMetricConfig:
    """Конфігурація метрики для завдання машинного навчання"""
    metric_class: Type[MetricStrategy]  # Клас метрики для оцінки
    description: str  # Опис завдання та метрики


class MLTaskMetricRegistry:
    """Реєстр, який зберігає інформацію про метрики для різних завдань машинного навчання"""

    def __init__(self):
        # Словник, що відображає тип завдання на конфігурацію метрики
        self._task_to_metric: Dict[MLTaskType, TaskMetricConfig] = {}

        # Ініціалізація реєстру стандартними відображеннями
        self._initialize_registry()

    def _initialize_registry(self):
        """Ініціалізує реєстр стандартними відображеннями завдань і метрик"""
        # Імпортуємо всі необхідні класи метрик
        from project.logic.evaluation.classification_metric import ClassificationMetric
        from project.logic.evaluation.regression_metric import RegressionMetric
        from project.logic.evaluation.clustering_metric import ClusteringMetric
        from project.logic.evaluation.dim_reduction_metric import DimReduction
        from project.logic.evaluation.anomaly_detection_metric import AnomalyDetectionMetric
        from project.logic.evaluation.density_estimation_metric import DensityEstimationMetric
        from project.logic.evaluation.metric_strategy import TimeSeriesMetric

        # Реєструємо метрики для кожного типу завдання
        self.register_metric(
            MLTaskType.CLASSIFICATION,
            TaskMetricConfig(
                ClassificationMetric,
                "Метрики для задач класифікації"
            )
        )

        self.register_metric(
            MLTaskType.REGRESSION,
            TaskMetricConfig(
                RegressionMetric,
                "Метрики для задач регресії"
            )
        )

        self.register_metric(
            MLTaskType.CLUSTERING,
            TaskMetricConfig(
                ClusteringMetric,
                "Метрики для задач кластеризації"
            )
        )

        self.register_metric(
            MLTaskType.DIMENSIONALITY_REDUCTION,
            TaskMetricConfig(
                DimReduction,
                "Метрики для оцінки якості зниження розмірності"
            )
        )

        self.register_metric(
            MLTaskType.ANOMALY_DETECTION,
            TaskMetricConfig(
                AnomalyDetectionMetric,
                "Метрики для виявлення аномалій"
            )
        )

        self.register_metric(
            MLTaskType.DENSITY_ESTIMATION,
            TaskMetricConfig(
                DensityEstimationMetric,
                "Метрики для оцінки щільності розподілу"
            )
        )

        self.register_metric(
            MLTaskType.TIME_SERIES,
            TaskMetricConfig(
                TimeSeriesMetric,
                "Метрики для аналізу часових рядів"
            )
        )

    def register_metric(self, task_type: MLTaskType, metric_config: TaskMetricConfig):
        """Реєструє нову метрику для типу завдання"""
        self._task_to_metric[task_type] = metric_config

    def get_metric_config(self, task_type: MLTaskType) -> TaskMetricConfig:
        """Отримує конфігурацію метрики для заданого типу завдання"""
        return self._task_to_metric.get(task_type)

    def get_metric_class(self, task_type: MLTaskType) -> Type[MetricStrategy]:
        """Отримує клас метрики для заданого типу завдання"""
        config = self.get_metric_config(task_type)
        return config.metric_class if config else None


class MLMetricFactory:
    """Фабрика для створення відповідних метрик для різних завдань машинного навчання"""

    def __init__(self, registry=None):
        self.registry = registry or MLTaskMetricRegistry()

    def create_metric(self, task_type: MLTaskType) -> MetricStrategy:
        """
        Створює відповідну метрику для оцінки моделі машинного навчання

        Parameters:
        -----------
        task_type : MLTaskType
            Тип завдання машинного навчання

        Returns:
        -----------
        MetricStrategy
            Екземпляр відповідної стратегії метрики
        """
        metric_class = self.registry.get_metric_class(task_type)

        if metric_class is None:
            raise ValueError(f"Не знайдено відповідної метрики для завдання типу {task_type}")

        return metric_class()


# Допоміжна функція для роботи зі строковими назвами
def get_ml_metric(task_type_str: str) -> MetricStrategy:
    """
    Функція-утиліта для отримання метрики за строковою назвою типу завдання

    Parameters:
    -----------
    task_type_str : str
        Назва типу завдання (наприклад, "Classification", "Regression")

    Returns:
    -----------
    MetricStrategy
        Екземпляр відповідної стратегії метрики
    """
    # Мапимо строкові значення до enum
    task_type_map = {
        "CLASSIFICATION": MLTaskType.CLASSIFICATION,
        "REGRESSION": MLTaskType.REGRESSION,
        "CLUSTERING": MLTaskType.CLUSTERING,
        "DIMENSIONALITY_REDUCTION": MLTaskType.DIMENSIONALITY_REDUCTION,
        "ANOMALY_DETECTION": MLTaskType.ANOMALY_DETECTION,
        "DENSITY_ESTIMATION": MLTaskType.DENSITY_ESTIMATION,
        "TIME_SERIES": MLTaskType.TIME_SERIES
    }

    try:
        task_type = task_type_map[task_type_str.upper()]
    except KeyError:
        raise ValueError(f"Невідомий тип завдання: {task_type_str}")

    # Створюємо фабрику та отримуємо метрику
    factory = MLMetricFactory()
    return factory.create_metric(task_type)
