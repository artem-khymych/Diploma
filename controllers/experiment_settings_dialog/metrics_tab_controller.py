from project.controllers.experiment_settings_dialog.tab_controller import TabController
from project.logic.experiment.experiment import Experiment


class MetricsTabController(TabController):
    """Контролер для вкладки параметрів оцінки"""

    def __init__(self, experiment: Experiment, view):
        super().__init__(experiment, view)
        self.connect_signals()
        self.init_view()

    def connect_signals(self):
        """Підключення сигналів до слотів"""
        # Підключення сигналів експерименту, якщо вони доступні
        if hasattr(self.experiment, 'metrics_updated'):
            self.experiment.metrics_updated.connect(self.on_metrics_updated)

    def init_view(self):
        """Ініціалізація представлення"""
        # Отримати поточні метрики з експерименту, якщо вони є
        train_metrics = getattr(self.experiment, 'train_metrics', {})
        test_metrics = getattr(self.experiment, 'test_metrics', {})

        if train_metrics or test_metrics:
            self.update_view_metrics(train_metrics, test_metrics)

    def on_metrics_updated(self, time=None, train_metrics=None, test_metrics=None):
        """Обробник події оновлення метрик"""
        self.update_view_metrics(train_metrics, test_metrics)

    def update_view_metrics(self, train_metrics=None, test_metrics=None):
        """Оновлює відображення метрик у представленні"""
        if train_metrics is None:
            train_metrics = {}
        if test_metrics is None:
            test_metrics = {}

        # Об'єднуємо унікальні ключі з обох словників
        all_metrics = set(list(train_metrics.keys()) + list(test_metrics.keys()))
        metrics_data = {}

        for metric in all_metrics:
            metrics_data[metric] = {
                'train': train_metrics.get(metric, None),
                'test': test_metrics.get(metric, None)
            }

        self.view.update_metrics(metrics_data)