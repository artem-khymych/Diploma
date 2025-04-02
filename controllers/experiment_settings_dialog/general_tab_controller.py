from PyQt5.QtCore import QObject, pyqtSignal

from project.logic.experiment.experiment import Experiment
from project.ui.experiment_settings_dialog.general_tab import GeneralTabWidget


class GeneralSettingsController(QObject):
    """Контролер для вкладки загальних налаштувань"""
    # Сигнал для сповіщення про завершення експерименту
    #experiment_finished = pyqtSignal(float)  # Параметр - час навчання в секундах

    def __init__(self, experiment: Experiment, view: GeneralTabWidget) -> None:
        super().__init__()
        self.experiment = experiment
        self.view = view
        self.init_view()
        #self.connect_signals()

    def init_view(self):
        """Ініціалізація початкового стану представлення"""
        self.view.experiment_name.setText(self.experiment.name)
        self.view.description.setText(self.experiment.description)
        self.view.method_name.setText(type(self.experiment.model).__name__)

        # Встановлення статусу експерименту
        self.view.update_status(self.experiment.is_finished)

        if hasattr(self.experiment, 'training_time') and self.experiment.is_finished:
            self.view.set_experiment_finished(self.experiment.training_time)

    def update_model_from_view(self):
        """Оновлення моделі даними з представлення"""
        self.experiment.name = self.view.experiment_name.text()
        self.experiment.description = self.view.description.toPlainText()

    def set_experiment_name(self, name: str):
        self.view.experiment_name.setText(name)

    def connect_signals(self):
        pass

    def on_experiment_finished(self, training_time):
        """Обробник завершення експерименту"""
        print(f"Експеримент завершено за {training_time} секунд")
        self.experiment.is_finished = True
        self.experiment.training_time = training_time
        self.view.set_experiment_finished(training_time)