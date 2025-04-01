from project.controllers.experiment_settings_dialog.tab_controller import TabController
from project.logic.experiment.experiment import Experiment


class MetricsTabController(TabController):
    """Контролер для вкладки параметрів оцінки"""

    def __init__(self, experiment: Experiment, view):
        super().__init__(experiment, view)
        self.connect_signals()
        # self.init_view()

    def connect_signals(self):
        """Підключення сигналів до слотів"""

    def init_view(self):
        pass

    def on_cv_toggled(self, checked):
        """Обробка перемикання крос-валідації"""
        return

    def on_cv_folds_changed(self, value):
        """Обробка зміни кількості фолдів"""
        return

    def update_metrics(self):
        """Оновлення списку метрик"""
        return

    def update_model_from_view(self):
        """Оновлення моделі даними з представлення"""
        return

