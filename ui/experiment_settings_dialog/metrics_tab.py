from PyQt5.QtWidgets import QVBoxLayout, QTableWidget, QHeaderView, QTableWidgetItem, QWidget


class MetricsTabWidget(QWidget):
    """Представлення для вкладки параметрів оцінки (затичка)"""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.init_ui()

    def init_ui(self):
        """Ініціалізація інтерфейсу"""
        # Створюємо головний лейаут
        self.layout = QVBoxLayout(self)

        # Створюємо таблицю
        self.table = QTableWidget(1, 2)
        self.table.setHorizontalHeaderLabels(["Metric", "Value"])
        self.table.setEditTriggers(QTableWidget.NoEditTriggers)  # Забороняємо редагування
        self.table.setEnabled(False)  # Робимо неактивною
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.layout.addWidget(self.table)

    def update_metrics(self, new_metrics: dict):
        """Оновлює значення метрик та активує таблицю"""
        self.table.setRowCount(len(new_metrics))
        for row, (key, value) in enumerate(new_metrics.items()):
            self.table.setItem(row, 1, QTableWidgetItem(f"{value:.4f}"))

        self.table.setEnabled(True)  # Активуємо таблицю



