import os
import pickle
from typing import Dict
from PyQt5.QtCore import QObject, pyqtSignal, QPointF, pyqtSlot
from project.logic.experiment.experiment import Experiment
from project.logic.experiment.nn_experiment import NeuralNetworkExperiment
from project.logic.experiment_manager import ExperimentManager

class WorkspaceSerializer(QObject):
    """
    Класс для серіалізації та десеріалізації робочого простору експериментів.
    Зберігає всю інформацію про експерименти, вузли та зв'язки між ними.
    """

    # Сигнали для інформування про статус операцій
    workspace_saved = pyqtSignal(str)  # Signal emitted when workspace is saved (path)
    workspace_loaded = pyqtSignal(str)  # Signal emitted when workspace is loaded (path)
    save_error = pyqtSignal(str)  # Signal emitted when saving fails (error message)
    load_error = pyqtSignal(str)  # Signal emitted when loading fails (error message)

    def __init__(self, experiment_manager=None, node_controller=None):
        super().__init__()
        # Посилання на менеджер експериментів та контролер вузлів
        self.experiment_manager = experiment_manager or ExperimentManager()
        self.node_controller = node_controller

        # Версія формату файлу (для майбутньої сумісності)
        self.file_format_version = "1.0"

    def set_node_controller(self, node_controller):
        """Встановлює контролер вузлів для серіалізатора."""
        self.node_controller = node_controller

    def set_experiment_manager(self, experiment_manager):
        """Встановлює менеджер експериментів для серіалізатора."""
        self.experiment_manager = experiment_manager

    def save_workspace(self, filepath: str) -> bool:
        """
        Зберігає робочий простір у файл.

        Args:
            filepath: Шлях до файлу для збереження

        Returns:
            bool: True при успіху, False при помилці
        """
        try:
            # Перевіримо, чи встановлені необхідні залежності
            if not self.experiment_manager or not self.node_controller:
                self.save_error.emit("Не встановлено необхідні залежності")
                return False

            # Отримуємо дані для серіалізації
            serialized_data = self._prepare_serialization_data()

            # Зберігаємо дані в файл
            with open(filepath, 'wb') as file:
                pickle.dump(serialized_data, file, protocol=pickle.HIGHEST_PROTOCOL)

            # Повідомляємо про успішне збереження
            self.workspace_saved.emit(filepath)
            return True

        except Exception as e:
            error_message = f"Помилка при збереженні робочого простору: {str(e)}"
            print(error_message)
            self.save_error.emit(error_message)
            return False

    def load_workspace(self, filepath: str) -> bool:
        """
        Завантажує робочий простір з файлу.

        Args:
            filepath: Шлях до файлу для завантаження

        Returns:
            bool: True при успіху, False при помилці
        """
        try:
            # Перевіримо, чи встановлені необхідні залежності
            if not self.experiment_manager or not self.node_controller:
                self.load_error.emit("Не встановлено необхідні залежності")
                return False

            # Перевіримо, чи існує файл
            if not os.path.exists(filepath):
                self.load_error.emit(f"Файл {filepath} не існує")
                return False

            # Завантажуємо дані з файлу
            with open(filepath, 'rb') as file:
                loaded_data = pickle.load(file)

            # Відновлюємо дані
            self._restore_from_serialization_data(loaded_data)

            # Повідомляємо про успішне завантаження
            self.workspace_loaded.emit(filepath)
            return True

        except Exception as e:
            error_message = f"Помилка при завантаженні робочого простору: {str(e)}"
            print(error_message)
            self.load_error.emit(error_message)
            return False

    def _prepare_serialization_data(self) -> Dict:
        """
        Підготовлює дані для серіалізації.

        Returns:
            Dict: Словник з даними для серіалізації
        """
        # Головний словник для всіх даних
        serialized_data = {
            "version": self.file_format_version,
            "experiments": {},
            "nodes": [],
            "edges": [],
            "node_positions": {},
            "experiment_node_map": {}
        }

        # Серіалізуємо експерименти
        for exp_id, experiment in self.experiment_manager.experiments.items():
            # Створюємо структуру для збереження всіх необхідних даних експерименту
            exp_data = {
                "id": experiment.id,
                "name": experiment.name,
                "description": experiment.description,
                "task": experiment.task,
                "is_finished": experiment.is_finished,
                "train_time": experiment.train_time,
                "params": experiment._params,
                "input_data_params": vars(experiment.input_data_params),
                "parent_id": experiment.parent.id if experiment.parent else None,
                "type": "neural_network" if isinstance(experiment, NeuralNetworkExperiment) else "standard"
            }

            # Додаємо дані про метрики, якщо експеримент завершений
            if experiment.is_finished:
                exp_data["train_metrics"] = experiment.train_metrics
                exp_data["test_metrics"] = experiment.test_metrics

            # Зберігаємо дані експерименту
            serialized_data["experiments"][exp_id] = exp_data

        # Серіалізуємо дані вузлів
        for node in self.node_controller.nodes:
            node_data = {
                "id": node.id,
                "name": node.get_name(),
                "position": (node.pos().x(), node.pos().y())
            }
            serialized_data["nodes"].append(node_data)

            # Зберігаємо позицію окремо для зручності відновлення
            serialized_data["node_positions"][node.id] = (node.pos().x(), node.pos().y())

            # Зв'язуємо вузол з експериментом
            serialized_data["experiment_node_map"][node.id] = node.id  # В данному випадку ID вузла = ID експерименту

        # Серіалізуємо зв'язки між вузлами
        for edge in self.node_controller.edges:
            edge_data = {
                "source_id": edge.source_node.id,
                "target_id": edge.target_node.id
            }
            serialized_data["edges"].append(edge_data)

        return serialized_data

    def _restore_from_serialization_data(self, data: Dict) -> None:
        """
        Відновлює робочий простір з серіалізованих даних.

        Args:
            data: Словник з серіалізованими даними
        """
        # Перевірка версії формату
        if "version" not in data or data["version"] != self.file_format_version:
            print(f"Попередження: Версія формату файлу ({data.get('version', 'невідома')}) "
                  f"відрізняється від поточної ({self.file_format_version})")

        # Очистка поточних даних
        self._clear_current_workspace()

        # Відновлюємо вузли
        node_map = {}  # Для збереження відповідності: старий ID -> новий вузол
        for node_data in data["nodes"]:
            # Створюємо новий вузол
            node = self.node_controller.create_node(
                x=node_data["position"][0],
                y=node_data["position"][1]
            )
            node.set_name(node_data["name"])

            # Зберігаємо відповідність ID
            node_map[node_data["id"]] = node

            # Явно встановлюємо позицію вузла
            node.setPos(QPointF(node_data["position"][0], node_data["position"][1]))

        # Відновлюємо експерименти
        experiment_map = {}  # старий ID -> новий експеримент
        for exp_id, exp_data in data["experiments"].items():
            # Визначаємо тип експерименту
            if exp_data["type"] == "neural_network":
                # Створення нейромережевого експерименту
                experiment = NeuralNetworkExperiment(
                    id=node_map[exp_id].id,  # Використовуємо ID нового вузла
                    task=exp_data["task"],
                    model=None,  # Буде відновлено пізніше
                    params=exp_data["params"]
                )
            else:
                # Створення стандартного експерименту
                experiment = Experiment(
                    id=node_map[exp_id].id,  # Використовуємо ID нового вузла
                    task=exp_data["task"],
                    model=None,  # Буде відновлено пізніше
                    params=exp_data["params"]
                )

            # Відновлюємо основні атрибути
            experiment._name = exp_data["name"]
            experiment.description = exp_data["description"]
            experiment.is_finished = exp_data["is_finished"]
            experiment.train_time = exp_data["train_time"]

            # Відновлюємо параметри вхідних даних
            self._restore_input_data_params(experiment, exp_data["input_data_params"])

            # Відновлюємо метрики, якщо експеримент завершений
            if experiment.is_finished and "train_metrics" in exp_data:
                experiment.train_metrics = exp_data["train_metrics"]
                experiment.test_metrics = exp_data["test_metrics"]

            # Зберігаємо експеримент в менеджері
            self.experiment_manager.experiments[experiment.id] = experiment

            # Зберігаємо відповідність ID
            experiment_map[int(exp_id)] = experiment

        # Відновлюємо батьківські відносини експериментів
        for exp_id, exp_data in data["experiments"].items():
            if exp_data["parent_id"] is not None:
                # Знаходимо відповідний батьківський експеримент
                parent_exp = experiment_map.get(exp_data["parent_id"])
                if parent_exp:
                    # Встановлюємо батьківський експеримент
                    experiment_map[int(exp_id)].parent = parent_exp
                    # Додаємо до батька посилання на дочірній експеримент
                    parent_exp.children.append(experiment_map[int(exp_id)])

        # Відновлюємо зв'язки між вузлами
        for edge_data in data["edges"]:
            # Знаходимо відповідні вузли
            source_node = node_map.get(edge_data["source_id"])
            target_node = node_map.get(edge_data["target_id"])

            if source_node and target_node:
                # Створюємо зв'язок
                self.node_controller.create_edge(source_node, target_node)

        # Оновлюємо сцену, щоб всі елементи коректно відображалися
        if self.node_controller.scene:
            self.node_controller.scene.update()

    def _restore_input_data_params(self, experiment, params_data):
        """
        Відновлює параметри вхідних даних для експерименту.

        Args:
            experiment: Експеримент, для якого відновлюються параметри
            params_data: Словник з параметрами
        """
        # Визначаємо тип параметрів в залежності від типу експерименту
        if isinstance(experiment, NeuralNetworkExperiment):
            params = experiment.input_data_params
        else:
            params = experiment.input_data_params

        # Відновлюємо всі атрибути
        for key, value in params_data.items():
            if hasattr(params, key):
                setattr(params, key, value)

    def _clear_current_workspace(self):
        """Очищує поточний робочий простір"""
        # Очищуємо експерименти
        self.experiment_manager.experiments = {}

        # Очищуємо вузли та зв'язки зі сцени
        for node in list(self.node_controller.nodes):
            self.node_controller.delete_node(node)

        # Додаткова очистка, якщо потрібно
        self.node_controller.nodes = []
        self.node_controller.edges = []

