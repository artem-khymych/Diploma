from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (QHeaderView, QTableWidgetItem, QTableWidget,
                             QPushButton, QVBoxLayout, QDialog, QLabel,
                             QHBoxLayout, QFrame)
from PyQt5.QtGui import QFont, QColor, QBrush

from project.logic.experiment.nn_experiment import NeuralNetworkExperiment


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

        main_layout = QVBoxLayout()

        # Створюємо таблицю для тренувальних метрик з заголовком
        train_layout = QVBoxLayout()
        train_label = QLabel("Тренувальні метрики")
        train_label.setFont(QFont("Arial", 12, QFont.Bold))
        train_layout.addWidget(train_label)

        self.train_table = self.create_metric_table("train_metrics")
        train_layout.addWidget(self.train_table)

        # Додаємо розділювальну лінію
        line = QFrame()
        line.setFrameShape(QFrame.HLine)
        line.setFrameShadow(QFrame.Sunken)

        # Створюємо таблицю для тестових метрик з заголовком
        test_layout = QVBoxLayout()
        test_label = QLabel("Тестові метрики")
        test_label.setFont(QFont("Arial", 12, QFont.Bold))
        test_layout.addWidget(test_label)

        self.test_table = self.create_metric_table("test_metrics")
        test_layout.addWidget(self.test_table)

        # Кнопка закриття
        button_layout = QHBoxLayout()
        close_button = QPushButton("Закрити")
        close_button.setMinimumHeight(30)
        close_button.clicked.connect(self.accept)
        button_layout.addStretch()
        button_layout.addWidget(close_button)

        # Додаємо всі елементи до основного лейауту
        main_layout.addLayout(train_layout)
        main_layout.addWidget(line)
        main_layout.addLayout(test_layout)
        main_layout.addLayout(button_layout)

        # Додаємо легенду
        legend_layout = QHBoxLayout()

        best_indicator = QFrame()
        best_indicator.setFixedSize(20, 20)
        best_indicator.setAutoFillBackground(True)
        best_palette = best_indicator.palette()
        best_palette.setColor(best_indicator.backgroundRole(), QColor(180, 255, 180))
        best_indicator.setPalette(best_palette)

        worst_indicator = QFrame()
        worst_indicator.setFixedSize(20, 20)
        worst_indicator.setAutoFillBackground(True)
        worst_palette = worst_indicator.palette()
        worst_palette.setColor(worst_indicator.backgroundRole(), QColor(255, 180, 180))
        worst_indicator.setPalette(worst_palette)

        legend_layout.addWidget(best_indicator)
        legend_layout.addWidget(QLabel("Найкраще значення"))
        legend_layout.addSpacing(20)
        legend_layout.addWidget(worst_indicator)
        legend_layout.addWidget(QLabel("Найгірше значення"))
        legend_layout.addStretch()

        main_layout.addLayout(legend_layout)
        main_layout.addLayout(button_layout)

        self.setLayout(main_layout)

    def create_metric_table(self, metrics_attr):
        """
        Створює таблицю для відображення метрик експериментів.

        Args:
            metrics_attr (str): Назва атрибуту з метриками ('train_metrics' або 'test_metrics')

        Returns:
            QTableWidget: Таблиця з метриками
        """
        # Отримуємо інформацію про метрики (максимізувати чи мінімізувати)
        meta_info = {}
        if self.experiments and hasattr(self.experiments[0], 'metric_strategy'):
            meta_info = self.experiments[0].metric_strategy.get_metainformation()

        # Збираємо всі унікальні метрики з усіх експериментів
        all_metrics = set()
        for exp in self.experiments:
            if hasattr(exp, metrics_attr) and getattr(exp, metrics_attr):
                all_metrics.update(getattr(exp, metrics_attr).keys())

        # Розділяємо метрики на порівнювані та непорівнювані
        comparable_metrics = []
        non_comparable_metrics = []

        for metric in all_metrics:
            if metric in meta_info and meta_info[metric] is not None:
                comparable_metrics.append(metric)
            else:
                non_comparable_metrics.append(metric)

        # Сортуємо метрики для послідовного відображення
        comparable_metrics = sorted(comparable_metrics)
        non_comparable_metrics = sorted(non_comparable_metrics)

        # Створюємо таблицю
        table = QTableWidget()
        total_metrics = len(comparable_metrics) + len(non_comparable_metrics) + 1  # +1 для загальної інформації
        if non_comparable_metrics:
            total_metrics += 1  # Додаємо рядок для розділювача

        table.setRowCount(total_metrics)
        table.setColumnCount(len(self.experiments) + 1)  # +1 для заголовків рядків

        # Встановлюємо чіткий шрифт для таблиці
        table_font = QFont("Arial", 10)
        table.setFont(table_font)

        # Заголовок таблиці - явно встановлюємо шрифт і робимо жирним
        header_font = QFont("Arial", 10, QFont.Bold)

        # Назви стовпців
        header_item = QTableWidgetItem("Метрика")
        header_item.setFont(header_font)
        table.setHorizontalHeaderItem(0, header_item)

        for i, exp in enumerate(self.experiments):
            header_text = f"{exp.name} (ID: {exp.id})"
            header_item = QTableWidgetItem(header_text)
            header_item.setFont(header_font)
            table.setHorizontalHeaderItem(i + 1, header_item)

        # Додаємо загальну інформацію про експерименти
        info_item = QTableWidgetItem("Тип завдання")
        info_item.setFont(header_font)
        table.setItem(0, 0, info_item)

        for i, exp in enumerate(self.experiments):
            task_item = QTableWidgetItem(exp.task.value if isinstance(exp, NeuralNetworkExperiment) else exp.task)
            table.setItem(0, i + 1, task_item)

        # Додаємо порівнювані метрики і виділяємо найкращі/найгірші значення
        for row, metric in enumerate(comparable_metrics):
            metric_item = QTableWidgetItem(metric)
            metric_item.setFont(table_font)
            table.setItem(row + 1, 0, metric_item)

            # Збираємо всі значення метрики для порівняння
            values = []
            indices = []
            for i, exp in enumerate(self.experiments):
                metrics = getattr(exp, metrics_attr, {})
                if metric in metrics:
                    value = metrics[metric]
                    if isinstance(value, (int, float)):
                        values.append(value)
                        indices.append(i)
                    else:
                        values.append(None)
                        indices.append(i)
                else:
                    values.append(None)
                    indices.append(i)

            # Знаходимо найкращі та найгірші значення
            best_index = None
            worst_index = None
            valid_values = [v for v in values if v is not None]

            if valid_values:
                if meta_info.get(metric, True):  # True за замовчуванням, чим більше - тим краще
                    max_val = max([v for v in values if v is not None], default=None)
                    min_val = min([v for v in values if v is not None], default=None)
                    best_index = values.index(max_val) if max_val is not None else None
                    worst_index = values.index(min_val) if min_val is not None else None
                else:  # False, чим менше - тим краще
                    min_val = min([v for v in values if v is not None], default=None)
                    max_val = max([v for v in values if v is not None], default=None)
                    best_index = values.index(min_val) if min_val is not None else None
                    worst_index = values.index(max_val) if max_val is not None else None

            # Додаємо значення метрик
            for i, exp in enumerate(self.experiments):
                metrics = getattr(exp, metrics_attr, {})
                if metric in metrics:
                    value = metrics[metric]
                    # Форматуємо числові значення для кращого відображення
                    if isinstance(value, (int, float)):
                        value_str = f"{value:.4f}" if isinstance(value, float) else str(value)
                    else:
                        value_str = str(value)
                    value_item = QTableWidgetItem(value_str)

                    # Виділяємо найкраще значення зеленим
                    if i == best_index:
                        value_item.setBackground(QBrush(QColor(180, 255, 180)))  # Світло-зелений
                    # Виділяємо найгірше значення червоним
                    elif i == worst_index:
                        value_item.setBackground(QBrush(QColor(255, 180, 180)))  # Світло-червоний

                    table.setItem(row + 1, i + 1, value_item)
                else:
                    na_item = QTableWidgetItem("N/A")
                    table.setItem(row + 1, i + 1, na_item)

        # Додаємо роздільник, якщо є непорівнювані метрики
        current_row = len(comparable_metrics) + 1

        if non_comparable_metrics:
            # Додаємо розділювач
            for col in range(table.columnCount()):
                separator_item = QTableWidgetItem("")
                separator_item.setBackground(QBrush(QColor(200, 200, 200)))  # Сірий колір для розділювача
                table.setItem(current_row, col, separator_item)

            # Встановлюємо більшу висоту для рядка-розділювача
            table.setRowHeight(current_row, 5)
            current_row += 1

            # Додаємо заголовок для непорівнюваних метрик
            header_item = QTableWidgetItem("Непорівнювані метрики")
            header_item.setFont(header_font)
            table.setItem(current_row, 0, header_item)
            current_row += 1

            # Додаємо непорівнювані метрики
            for row_offset, metric in enumerate(non_comparable_metrics):
                row = current_row + row_offset
                metric_item = QTableWidgetItem(metric)
                metric_item.setFont(table_font)
                table.setItem(row, 0, metric_item)

                for i, exp in enumerate(self.experiments):
                    metrics = getattr(exp, metrics_attr, {})
                    if metric in metrics:
                        value = metrics[metric]
                        if isinstance(value, (int, float)):
                            value_str = f"{value:.4f}" if isinstance(value, float) else str(value)
                        else:
                            value_str = str(value)
                        value_item = QTableWidgetItem(value_str)
                        table.setItem(row, i + 1, value_item)
                    else:
                        na_item = QTableWidgetItem("N/A")
                        table.setItem(row, i + 1, na_item)

        # Налаштовуємо вигляд таблиці
        table.setAlternatingRowColors(True)
        table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        table.verticalHeader().setVisible(False)
        table.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        table.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)

        # Забезпечуємо видимість заголовків
        table.horizontalHeader().setVisible(True)

        return table