# Now let's update the controller class to work with the window instead of dialog
from PyQt5.QtCore import pyqtSignal, QObject
from PyQt5.QtWidgets import QMessageBox

from project.controllers.experiment_settings_dialog.general_tab_controller import GeneralSettingsController
from project.controllers.experiment_settings_dialog.hyperparams_tab_controller import HyperparamsTabController
from project.controllers.experiment_settings_dialog.input_data_tab_controller import InputDataTabController
from project.controllers.experiment_settings_dialog.metrics_tab_controller import MetricsTabController
from project.logic.experiment.experiment import Experiment
from project.ui.experiment_settings_dialog.experiment_settings_dialog import ExperimentSettingsWindow


class ExperimentSettingsController(QObject):
    """Головний контролер для вікна налаштувань експерименту"""
    experiment_inherited = pyqtSignal(int)
    window_closed = pyqtSignal(bool, object)

    def __init__(self, experiment: Experiment, window: ExperimentSettingsWindow):
        super().__init__()
        self.experiment = experiment
        self.window = window
        self.result_accepted = False
        self.input_params = None

        # Створення контролерів для вкладок
        self.general_controller = GeneralSettingsController(experiment, window.general_tab)
        self.hyperparams_controller = HyperparamsTabController(experiment, window.model_tab)
        self.input_data_controller = InputDataTabController(experiment, window.data_tab)
        self.metrics_controller = MetricsTabController(experiment, window.evaluation_tab)

        # Підключення сигналів
        self.window.ok_btn.setAutoDefault(False)
        self.connect_signals()

    def connect_signals(self):
        self.window.window_accepted.connect(self.on_accept)
        self.window.window_rejected.connect(self.on_cancel)
        self.general_controller.view.experiment_started.connect(self.check_settings_and_run_experiment)

        # evaluation started, update metrics and go to metrics tab
        self.experiment.experiment_evaluated.connect(self.metrics_controller.on_metrics_updated)
        self.experiment.experiment_evaluated.connect(lambda: self.window.tab_widget.setCurrentIndex(3))

        # Підключаємо кнопку успадкування до методу обробника
        self.general_controller.experiment_inherited.connect(self.on_experiment_inherited)

    def on_experiment_inherited(self, parent_id):
        """Передає сигнал успадкування далі"""
        # Створюємо сигнал для передачі вгору по ієрархії
        self.experiment_inherited.emit(parent_id)

    def check_settings_and_run_experiment(self):
        if self.check_settings():
            self.experiment.run()
        else:
            return

    def on_cancel(self):
        self.result_accepted = False
        self.window_closed.emit(False, None)

    def on_accept(self):
        self.update_model_from_all_views()
        self.result_accepted = True
        self.input_params = self.input_data_controller.get_input_params()
        self.window_closed.emit(True, self.input_params)

    def check_settings(self):
        """Перевірка валідності всіх даних"""
        # Перевірка даних з вкладки "Дані"
        if self.input_data_controller.input_data_params.mode == 'single_file' and not self.input_data_controller.input_data_params.single_file_path:
            QMessageBox.warning(self.window, "Помилка", "Будь ласка, виберіть файл даних на вкладці 'Дані'.")
            self.window.tab_widget.setCurrentIndex(2)
            return False
        elif self.input_data_controller.input_data_params.mode == 'multi_files' and (
                not self.input_data_controller.input_data_params.x_train_file_path
                or not self.input_data_controller.input_data_params.y_train_file_path
                or not self.input_data_controller.input_data_params.x_test_file_path
                or not self.input_data_controller.input_data_params.y_test_file_path
        ):
            QMessageBox.warning(self.window, "Помилка",
                                "Будь ласка, виберіть обидва файли для тренування і тестування на вкладці 'Дані'.")
            self.window.tab_widget.setCurrentIndex(2)  # Перехід на вкладку "Дані"
            return False

        # Перевірка даних з вкладки "Загальні налаштування"
        if not self.window.general_tab.experiment_name.text().strip():
            QMessageBox.warning(self.window, "Помилка", "Будь ласка, введіть назву експерименту.")
            self.window.tab_widget.setCurrentIndex(0)  # Перехід на вкладку "Загальні налаштування"
            return False

        try:
            model = type(self.experiment.model)().set_params(**self.experiment.params)
        except Exception as e:
            QMessageBox.warning(self.window, "Хибні параметри", f"Винила помилка у налаштованих параметрах:\n {e}")
            return False

        if self.validate_params_strict(model, self.experiment.params):
            return True
        else:
            return False

    def validate_params_strict(self, model_class, params):
        from sklearn.utils._param_validation import validate_parameter_constraints

        if not hasattr(model_class, "_parameter_constraints"):
            return True

        constraints = model_class._parameter_constraints
        try:
            validate_parameter_constraints(constraints, params, caller_name=model_class.__class__.__name__)
        except Exception as e:
            QMessageBox.warning(self.window, "Хибні параметри", f"Винила помилка у налаштованих параметрах:\n {e}")
            return False

        return True

    def update_model_from_all_views(self):
        """Оновлення моделі даними з усіх представлень"""
        self.general_controller.update_model_from_view()
        self.hyperparams_controller.update_model_from_view()
        self.input_data_controller.update_model_from_view()

    def show(self):
        """Show the window and wait for the result"""
        self.window.show()
        # The window_closed signal will be emitted when window is closed
        # The caller should connect to this signal to receive the result


"""
from PyQt5.QtCore import pyqtSignal, QObject
from PyQt5.QtWidgets import QMessageBox, QDialog

from project.controllers.experiment_settings_dialog.general_tab_controller import GeneralSettingsController
from project.controllers.experiment_settings_dialog.hyperparams_tab_controller import HyperparamsTabController
from project.controllers.experiment_settings_dialog.input_data_tab_controller import InputDataTabController
from project.controllers.experiment_settings_dialog.metrics_tab_controller import MetricsTabController
from project.logic.experiment.experiment import Experiment

from project.ui.experiment_settings_dialog.experiment_settings_dialog import ExperimentSettingsDialog


#TODO migarte to window not a dialog
class ExperimentSettingsController(QObject):
    experiment_inherited = pyqtSignal(int)

    def __init__(self, experiment: Experiment, dialog: ExperimentSettingsDialog):
        super().__init__()
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
        self.dialog.cancel_btn.clicked.connect(self.on_cancel)
        self.general_controller.view.experiment_started.connect(self.check_settings_and_run_experiment)
        # evaluation started, update metrics and go to metrics tab
        self.experiment.experiment_evaluated.connect(self.metrics_controller.on_metrics_updated)
        self.experiment.experiment_evaluated.connect(lambda: self.dialog.tab_widget.setCurrentIndex(3))
        # Підключаємо кнопку успадкування до методу обробника
        self.general_controller.experiment_inherited.connect(self.on_experiment_inherited)
        #self.metrics_controller.compare_experiments.connect(self.show_comparison_dialog)

    def on_experiment_inherited(self, parent_id):
        # Створюємо сигнал для передачі вгору по ієрархії

        self.experiment_inherited.emit(parent_id)

    def check_settings_and_run_experiment(self):

        if self.check_settings():
            self.experiment.run()
        else:
            return

    def on_cancel(self):
        self.dialog.reject()

    def on_accept(self):
        self.update_model_from_all_views()
        self.dialog.accept()

    def check_settings(self):
        # Перевірка даних з вкладки "Дані"
        if self.input_data_controller.input_data_params.mode == 'single_file' and not self.input_data_controller.input_data_params.single_file_path:
            QMessageBox.warning(self.dialog, "Помилка", "Будь ласка, виберіть файл даних на вкладці 'Дані'.")
            self.dialog.tab_widget.setCurrentIndex(2)
            return False
        elif self.input_data_controller.input_data_params.mode == 'multi_files' and (
                not self.input_data_controller.input_data_params.x_train_file_path
                or not self.input_data_controller.input_data_params.y_train_file_path
                or not self.input_data_controller.input_data_params.x_test_file_path
                or not self.input_data_controller.input_data_params.y_test_file_path
        ):
            QMessageBox.warning(self.dialog, "Помилка",
                                "Будь ласка, виберіть обидва файли для тренування і тестування на вкладці 'Дані'.")
            self.dialog.tab_widget.setCurrentIndex(2)  # Перехід на вкладку "Дані"
            return False

        # Перевірка даних з вкладки "Загальні налаштування"
        if not self.dialog.general_tab.experiment_name.text().strip():
            QMessageBox.warning(self.dialog, "Помилка", "Будь ласка, введіть назву експерименту.")
            self.dialog.tab_widget.setCurrentIndex(0)  # Перехід на вкладку "Загальні налаштування"
            return False

        try:
            model = type(self.experiment.model)().set_params(**self.experiment.params)
        except Exception as e:
            QMessageBox.warning(self.dialog, "Хибні параметри", f"Винила помилка у налаштованих параметрах:\n {e}")
            return False

        if self.validate_params_strict(model, self.experiment.params):
            return True
        else:
            return False

    def validate_params_strict(self, model_class, params):
        from sklearn.utils._param_validation import validate_parameter_constraints

        if not hasattr(model_class, "_parameter_constraints"):
            return True

        constraints = model_class._parameter_constraints
        try:
            validate_parameter_constraints(constraints, params, caller_name=model_class.__class__.__name__)
        except Exception as e:
            QMessageBox.warning(self.dialog, "Хибні параметри", f"Винила помилка у налаштованих параметрах:\n {e}")
            return False

        return True

    def update_model_from_all_views(self):
        self.general_controller.update_model_from_view()
        self.hyperparams_controller.update_model_from_view()
        self.input_data_controller.update_model_from_view()

        # self.metrics_controller.update_model_from_view()

    def show(self):
        while True:
            if self.dialog.exec_() == QDialog.Accepted:
                return self.input_data_controller.get_input_params()
            elif self.dialog.exec_() == QDialog.Rejected:
                return


"""



