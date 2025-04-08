from PyQt5.QtWidgets import QVBoxLayout, QTableWidget, QHeaderView, QTableWidgetItem, QWidget, QPushButton


class MetricsTabWidget(QWidget):
    """Представлення для вкладки параметрів оцінки"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.init_ui()

    def init_ui(self):
        """Ініціалізація інтерфейсу"""
        # Створюємо головний лейаут
        self.layout = QVBoxLayout(self)

        # Створюємо таблицю
        self.table = QTableWidget(1, 3)
        self.table.setHorizontalHeaderLabels(["Metric", "Train Value", "Test Value"])
        self.table.setEditTriggers(QTableWidget.NoEditTriggers)  # Забороняємо редагування
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)

        self.compare_button = QPushButton("Порівняти моделі")

        self.layout.addWidget(self.table)
        self.layout.addWidget(self.compare_button)

    def update_metrics(self, metrics_data: dict):
        """Оновлює значення метрик та активує таблицю

        Args:
            metrics_data: словник формату {metric_name: {'train': value, 'test': value}}
        """
        # Очищаємо таблицю
        self.table.clearContents()

        # Встановлюємо кількість рядків відповідно до кількості метрик
        self.table.setRowCount(len(metrics_data))

        for row, (metric_name, values) in enumerate(metrics_data.items()):
            # Додаємо назву метрики
            self.table.setItem(row, 0, QTableWidgetItem(metric_name))

            # Додаємо значення для тренувальних даних, якщо доступні
            if values['train'] is not None:
                self.table.setItem(row, 1, QTableWidgetItem(f"{values['train']:.4f}"))
            else:
                self.table.setItem(row, 1, QTableWidgetItem("N/A"))

            # Додаємо значення для тестових даних, якщо доступні
            if values['test'] is not None:
                self.table.setItem(row, 2, QTableWidgetItem(f"{values['test']:.4f}"))
            else:
                self.table.setItem(row, 2, QTableWidgetItem("N/A"))

        # Активуємо таблицю, якщо є хоча б одна метрика
        self.table.setEnabled(len(metrics_data) > 0)