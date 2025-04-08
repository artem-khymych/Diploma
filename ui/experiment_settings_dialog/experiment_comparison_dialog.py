from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QHeaderView, QTableWidgetItem, QTableWidget, QPushButton, QVBoxLayout, QDialog


class ExperimentComparisonDialog(QDialog):
    """
    Діалогове вікно для порівняння метрик різних експериментів.
    """

    def __init__(self, experiments):
        super().__init__()
        self.experiments = experiments
        self.init_ui()

    def init_ui(self):
        self.setWindowTitle("Порівняння експериментів")
        self.setMinimumSize(1000, 900)

        layout = QVBoxLayout()

        # Створюємо таблицю для тренувальних метрик
        self.train_table = self.create_metric_table("Тренувальні метрики", "train_metrics")
        layout.addWidget(self.train_table)

        # Створюємо таблицю для тестових метрик
        self.test_table = self.create_metric_table("Тестові метрики", "test_metrics")
        layout.addWidget(self.test_table)

        # Кнопка закриття
        close_button = QPushButton("Закрити")
        close_button.clicked.connect(self.accept)
        layout.addWidget(close_button)

        self.setLayout(layout)

    def create_metric_table(self, title, metrics_attr):
        """
        Створює таблицю для відображення метрик експериментів.

        Args:
            title (str): Заголовок таблиці
            metrics_attr (str): Назва атрибуту з метриками ('train_metrics' або 'test_metrics')

        Returns:
            QTableWidget: Таблиця з метриками
        """
        # Збираємо всі унікальні метрики з усіх експериментів
        all_metrics = set()
        for exp in self.experiments:
            if hasattr(exp, metrics_attr) and getattr(exp, metrics_attr):
                all_metrics.update(getattr(exp, metrics_attr).keys())

        # Сортуємо метрики для послідовного відображення
        all_metrics = sorted(list(all_metrics))

        # Створюємо таблицю
        table = QTableWidget()
        table.setRowCount(len(all_metrics) + 1)  # +1 для заголовка з інформацією про експеримент
        table.setColumnCount(len(self.experiments) + 1)  # +1 для заголовків рядків

        # Заголовок таблиці
        table.setHorizontalHeaderItem(0, QTableWidgetItem("Метрика"))
        for i, exp in enumerate(self.experiments):
            header_text = f"{exp.name} (ID: {exp.id})"
            table.setHorizontalHeaderItem(i + 1, QTableWidgetItem(header_text))

        # Додаємо загальну інформацію про експерименти
        table.setItem(0, 0, QTableWidgetItem("Тип завдання"))
        for i, exp in enumerate(self.experiments):
            table.setItem(0, i + 1, QTableWidgetItem(exp.task))

        # Додаємо значення метрик
        for row, metric in enumerate(all_metrics):
            table.setItem(row + 1, 0, QTableWidgetItem(metric))

            for i, exp in enumerate(self.experiments):
                metrics = getattr(exp, metrics_attr, {})
                if metric in metrics:
                    value = metrics[metric]
                    # Форматуємо числові значення для кращого відображення
                    if isinstance(value, (int, float)):
                        value_str = f"{value:.4f}" if isinstance(value, float) else str(value)
                    else:
                        value_str = str(value)
                    table.setItem(row + 1, i + 1, QTableWidgetItem(value_str))
                else:
                    table.setItem(row + 1, i + 1, QTableWidgetItem("N/A"))

        # Налаштовуємо вигляд таблиці
        table.setAlternatingRowColors(True)
        table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        table.verticalHeader().setVisible(False)
        table.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        table.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)

        return table