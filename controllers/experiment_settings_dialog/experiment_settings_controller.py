from PyQt5.QtWidgets import QMessageBox, QDialog

from project.controllers.experiment_settings_dialog.general_tab_controller import GeneralSettingsController
from project.controllers.experiment_settings_dialog.hyperparams_tab_controller import HyperparamsTabController
from project.controllers.experiment_settings_dialog.input_data_tab_controller import InputDataTabController
from project.controllers.experiment_settings_dialog.metrics_tab_controller import MetricsTabController
from project.logic.experiment.experiment import Experiment

from project.ui.experiment_settings_dialog.experiment_settings_dialog import ExperimentSettingsDialog


#TODO налаштувати діалогове вікно інформації про ноду(експеримент) контролери і метод для запуску, який би отримував айді ноди і експерименту
class ExperimentSettingsController:
    """Головний контролер для діалогу налаштувань експерименту"""
    def __init__(self, experiment: Experiment, dialog: ExperimentSettingsDialog):
        self.experiment = experiment
        self.dialog = dialog

        # Створення контролерів для вкладок
        self.general_controller = GeneralSettingsController(experiment, dialog.general_tab)

        self.hyperparams_controller = HyperparamsTabController(experiment, dialog.model_tab)

        self.input_data_controller = InputDataTabController(experiment, dialog.data_tab)
        self.metrics_controller = MetricsTabController(experiment, dialog.evaluation_tab)

        # Підключення сигналів
        self.dialog.ok_btn.setAutoDefault(False)
        self.connect_signals()

    def connect_signals(self):
        self.dialog.ok_btn.clicked.connect(self.on_accept)
        self.general_controller.experiment_started.connect(self.experiment.run)

    def on_accept(self):
        """Обробка натискання кнопки OK"""
        # Перевірка введених даних
        if self.validate_data():
            self.update_model_from_all_views()
            self.dialog.accept()
        else:
            self.dialog.reject()
            return



    def validate_data(self):
        """Перевірка валідності всіх даних"""
        # Перевірка даних з вкладки "Дані"
        if self.input_data_controller.input_data_params.mode == 'single_file' and not self.input_data_controller.input_data_params.single_file_path:
            QMessageBox.warning(self.dialog, "Помилка", "Будь ласка, виберіть файл даних на вкладці 'Дані'.")
            self.dialog.tab_widget.setCurrentIndex(2)
            return False
        elif self.input_data_controller.input_data_params.mode == 'two_files' and (
                not self.input_data_controller.input_data_params.train_file_path or not self.input_data_controller.input_data_params.test_file_path):
            QMessageBox.warning(self.dialog, "Помилка",
                                "Будь ласка, виберіть обидва файли для тренування і тестування на вкладці 'Дані'.")
            self.dialog.tab_widget.setCurrentIndex(1)  # Перехід на вкладку "Дані"
            return False

        # Перевірка даних з вкладки "Загальні налаштування"
        if not self.dialog.general_tab.experiment_name.text().strip():
            QMessageBox.warning(self.dialog, "Помилка", "Будь ласка, введіть назву експерименту.")
            self.dialog.tab_widget.setCurrentIndex(0)  # Перехід на вкладку "Загальні налаштування"
            return False
        #TODO clear print
        print(self.input_data_controller.input_data_params.to_dict())

        return True

    def update_model_from_all_views(self):
        """Оновлення моделі даними з усіх представлень"""
        self.general_controller.update_model_from_view()
        self.hyperparams_controller.update_model_from_view()
        self.input_data_controller.update_model_from_view()

        #self.metrics_controller.update_model_from_view()


    def show(self):
        while True:
            if self.dialog.exec_() == QDialog.Accepted:
                return self.input_data_controller.get_input_params()


