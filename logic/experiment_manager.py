import inspect
import copy

from PyQt5.QtCore import pyqtSignal, QObject
from sklearn.base import ClassifierMixin, RegressorMixin, is_regressor, is_classifier

from .experiment.experiment import Experiment
from .modules import models_manager
from ..ui.experiment_settings_dialog.experiment_comparison_dialog import ExperimentComparisonDialog
from ..ui.node import Node


class ExperimentManager(QObject):
    _instance = None  # Змінна класу для зберігання екземпляра

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        super().__init__()
        self.experiments = {}
        self.current_node = None
        self.current_model = None
        self.current_params = None
        self.current_task = None

    def get_node(self, node: Node):
        self.current_node = node
        self._check_experiment_data()

    def get_ml_model(self, task, model, params):
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
        print(f"Created new experiment with ID: {self.current_node.id}")

        # Скидаємо поточні дані після створення експерименту
        self.current_node = None
        self.current_model = None
        self.current_params = None
        self.current_task = None

        return experiment

    def inherit_experiment_from(self, parent_id, child_id):
        """Створює новий експеримент на основі батьківського, але без метрик оцінки"""
        if parent_id not in self.experiments:
            print(f"Error: Parent experiment with ID {parent_id} not found")
            return None

        parent_experiment = self.experiments[parent_id]

        # Створюємо новий експеримент з тими ж параметрами, але іншим ID
        child_experiment = Experiment(
            id=child_id,
            task=parent_experiment.task,
            model=parent_experiment.model,
            params=copy.deepcopy(parent_experiment._params),
            parent=parent_experiment
        )

        # Копіюємо дані про дані та параметри, але не копіюємо метрики
        child_experiment.input_data_params = copy.deepcopy(parent_experiment.input_data_params)
        child_experiment.description = f"Успадковано від '{parent_experiment.name}'"
        child_experiment._name = f"Успадкований {parent_experiment.name}"

        # Зберігаємо експеримент у словнику
        self.experiments[child_id] = child_experiment

        print(f"Inherited experiment created with ID: {child_id} from parent ID: {parent_id}")
        return child_experiment

    def get_experiment(self, experiment_id):
        if experiment_id in self.experiments:
            return self.experiments[experiment_id]
        return None

    def get_related_experiments(self, experiment_id):
        """
        Отримує всі пов'язані експерименти (батьки, нащадки та паралельні гілки)
        для вказаного експерименту.

        Args:
            experiment_id (int): ID експерименту для пошуку пов'язаних

        Returns:
            list: Список завершених експериментів, пов'язаних з вказаним ID
        """
        # Перевіряємо, чи існує експеримент з таким ID
        if experiment_id not in self.experiments:
            print(f"Experiment with ID {experiment_id} not found.")
            return []

        related_experiments = []
        main_experiment = self.experiments[experiment_id]

        # Додаємо сам експеримент, якщо він завершений
        if main_experiment.is_finished:
            related_experiments.append(main_experiment)

        # Додаємо всіх батьків до кореня
        parent = main_experiment.parent
        while parent is not None:
            if parent.is_finished:
                related_experiments.append(parent)
            parent = parent.parent

        # Функція для рекурсивного додавання всіх нащадків
        def add_children(experiment):
            for child in experiment.children:
                if child.is_finished:
                    related_experiments.append(child)
                add_children(child)

        # Додаємо всіх нащадків
        add_children(main_experiment)

        # Якщо є батько, додамо всіх його інших нащадків (паралельні гілки)
        if main_experiment.parent:
            for sibling in main_experiment.parent.children:
                if sibling.id != experiment_id and sibling.is_finished:
                    related_experiments.append(sibling)
                    add_children(sibling)

        return related_experiments

    def show_comparison_dialog(self, experiment_id):
        """
        Відображає діалогове вікно з порівнянням метрик усіх пов'язаних експериментів.

        Args:
            experiment_id (int): ID експерименту, для якого потрібно відобразити порівняння
        """
        # Отримуємо всі пов'язані експерименти
        related_experiments = self.get_related_experiments(experiment_id)

        if not related_experiments:
            print(f"No completed experiments related to ID {experiment_id} found.")
            return

        # Створюємо діалогове вікно
        dialog = ExperimentComparisonDialog(related_experiments)
        dialog.exec_()
