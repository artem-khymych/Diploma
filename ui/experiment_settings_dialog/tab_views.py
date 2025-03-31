import sys
import os
from PyQt5.QtWidgets import (QApplication, QDialog, QVBoxLayout, QHBoxLayout,
                             QLabel, QLineEdit, QPushButton, QFileDialog,
                             QRadioButton, QGroupBox, QSlider, QSpinBox,
                             QComboBox, QMessageBox, QTabWidget, QWidget,
                             QTextEdit, QCheckBox, QListWidget, QSizePolicy, QTableWidget, QTableWidgetItem,
                             QHeaderView)
from PyQt5.QtCore import Qt

from project.ui.parameter_editor_widget import ParameterEditorWidget


class InputDataTabWidget(QWidget):
    """Представлення для вкладки параметрів машинного навчання"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.init_ui()

    def init_ui(self):
        main_layout = QVBoxLayout()

        # Вибір режиму роботи з даними
        self.create_data_mode_group()
        main_layout.addWidget(self.data_mode_group)

        # --- Секція для одного файлу ---
        self.single_file_group = QGroupBox("Налаштування для одного файлу")
        single_file_layout = QVBoxLayout()

        # Шлях до файлу
        self.single_file_layout = QHBoxLayout()
        self.single_file_label = QLabel("Шлях до файлу:")
        self.single_file_path = QLineEdit()
        self.single_file_btn = QPushButton("Обрати...")
        self.single_file_layout.addWidget(self.single_file_label)
        self.single_file_layout.addWidget(self.single_file_path)
        self.single_file_layout.addWidget(self.single_file_btn)
        single_file_layout.addLayout(self.single_file_layout)

        # Кодування файлу
        self.single_encoding_layout = QHBoxLayout()
        self.single_encoding_label = QLabel("Кодування:")
        self.single_encoding_combo = QComboBox()
        self.single_encoding_combo.addItems(["utf-8", "cp1251", "latin-1", "iso-8859-1", "ascii"])
        self.single_encoding_layout.addWidget(self.single_encoding_label)
        self.single_encoding_layout.addWidget(self.single_encoding_combo)
        single_file_layout.addLayout(self.single_encoding_layout)

        # Роздільник для CSV (спочатку невидимий)
        self.single_separator_layout = QHBoxLayout()
        self.single_separator_label = QLabel("Роздільник CSV:")
        self.single_separator_combo = QComboBox()
        self.single_separator_combo.addItems([",", ";", "\\t", "|", " "])
        self.single_separator_layout.addWidget(self.single_separator_label)
        self.single_separator_layout.addWidget(self.single_separator_combo)
        single_file_layout.addLayout(self.single_separator_layout)

        # Вибір цільової змінної
        self.target_layout = QHBoxLayout()
        self.target_label = QLabel("Цільова змінна:")
        self.target_combo = QComboBox()
        self.target_layout.addWidget(self.target_label)
        self.target_layout.addWidget(self.target_combo)
        single_file_layout.addLayout(self.target_layout)

        self.single_file_group.setLayout(single_file_layout)
        main_layout.addWidget(self.single_file_group)

        # --- Секція для двох файлів ---
        self.two_files_group = QGroupBox("Налаштування для двох файлів")
        two_files_layout = QVBoxLayout()

        # Тренувальний файл
        self.train_file_layout = QVBoxLayout()

        # Шлях до тренувального файлу
        train_path_layout = QHBoxLayout()
        self.train_file_label = QLabel("Тренувальний набір:")
        self.train_file_path = QLineEdit()
        self.train_file_btn = QPushButton("Обрати...")
        train_path_layout.addWidget(self.train_file_label)
        train_path_layout.addWidget(self.train_file_path)
        train_path_layout.addWidget(self.train_file_btn)
        self.train_file_layout.addLayout(train_path_layout)

        # Кодування тренувального файлу
        train_encoding_layout = QHBoxLayout()
        self.train_encoding_label = QLabel("Кодування:")
        self.train_encoding_combo = QComboBox()
        self.train_encoding_combo.addItems(["utf-8", "cp1251", "latin-1", "iso-8859-1", "ascii"])
        train_encoding_layout.addWidget(self.train_encoding_label)
        train_encoding_layout.addWidget(self.train_encoding_combo)
        self.train_file_layout.addLayout(train_encoding_layout)

        # Роздільник для тренувального CSV
        self.train_separator_layout = QHBoxLayout()
        self.train_separator_label = QLabel("Роздільник CSV:")
        self.train_separator_combo = QComboBox()
        self.train_separator_combo.addItems([",", ";", "\\t", "|", " "])
        self.train_separator_layout.addWidget(self.train_separator_label)
        self.train_separator_layout.addWidget(self.train_separator_combo)
        self.train_file_layout.addLayout(self.train_separator_layout)

        two_files_layout.addLayout(self.train_file_layout)

        # Тестовий файл
        self.test_file_layout = QVBoxLayout()

        # Шлях до тестового файлу
        test_path_layout = QHBoxLayout()
        self.test_file_label = QLabel("Тестовий набір:")
        self.test_file_path = QLineEdit()
        self.test_file_btn = QPushButton("Обрати...")
        test_path_layout.addWidget(self.test_file_label)
        test_path_layout.addWidget(self.test_file_path)
        test_path_layout.addWidget(self.test_file_btn)
        self.test_file_layout.addLayout(test_path_layout)

        # Кодування тестового файлу
        test_encoding_layout = QHBoxLayout()
        self.test_encoding_label = QLabel("Кодування:")
        self.test_encoding_combo = QComboBox()
        self.test_encoding_combo.addItems(["utf-8", "cp1251", "latin-1", "iso-8859-1", "ascii"])
        test_encoding_layout.addWidget(self.test_encoding_label)
        test_encoding_layout.addWidget(self.test_encoding_combo)
        self.test_file_layout.addLayout(test_encoding_layout)

        two_files_layout.addLayout(self.test_file_layout)

        self.two_files_group.setLayout(two_files_layout)
        main_layout.addWidget(self.two_files_group)

        # Параметри розбиття тренувального та тестового наборів
        self.split_group = QGroupBox("Параметри розбиття")
        split_layout = QVBoxLayout()

        # Train/test split
        split_ratio_layout = QHBoxLayout()
        split_ratio_layout.addWidget(QLabel("Відсоток для тренування:"))

        self.train_percent = QSpinBox()
        self.train_percent.setRange(10, 90)
        self.train_percent.setValue(80)
        self.train_percent.setSuffix("%")
        split_ratio_layout.addWidget(self.train_percent)

        split_ratio_layout.addWidget(QLabel("Тестування:"))
        self.test_percent = QLabel("20%")
        split_ratio_layout.addWidget(self.test_percent)

        split_layout.addLayout(split_ratio_layout)

        # Seed значення
        seed_layout = QHBoxLayout()
        seed_layout.addWidget(QLabel("Seed значення:"))
        self.seed_spinbox = QSpinBox()
        self.seed_spinbox.setRange(0, 999999)
        self.seed_spinbox.setValue(42)
        seed_layout.addWidget(self.seed_spinbox)
        split_layout.addLayout(seed_layout)

        self.split_group.setLayout(split_layout)
        main_layout.addWidget(self.split_group)

        # Додаємо порожній простір внизу
        main_layout.addStretch()

        self.setLayout(main_layout)

        # Початкові налаштування видимості
        self.show_separator_fields(self.single_file_path.text(), True)
        self.show_separator_fields(self.train_file_path.text(), False)

    def create_data_mode_group(self):
        self.data_mode_group = QGroupBox("Режим роботи з даними")
        data_mode_layout = QHBoxLayout()

        self.single_file_radio = QRadioButton("Один файл (розбити на тренувальний та тестовий)")
        self.two_files_radio = QRadioButton("Окремі файли для тренування та тестування")

        data_mode_layout.addWidget(self.single_file_radio)
        data_mode_layout.addWidget(self.two_files_radio)

        self.data_mode_group.setLayout(data_mode_layout)

    def update_ui_state(self, single_file_mode=True):
        # Оновлення секцій інтерфейсу залежно від режиму
        self.single_file_group.setVisible(single_file_mode)
        self.two_files_group.setVisible(not single_file_mode)
        self.split_group.setVisible(single_file_mode)

    def update_test_percent(self, train_value):
        self.test_percent.setText(f"{100 - train_value}%")

    def get_file_dialog(self, title, file_filter):
        return QFileDialog.getOpenFileName(
            self, title, "", file_filter
        )

    def show_separator_fields(self, file_path, is_single_file):
        """Показує/приховує поля для вибору роздільника для CSV файлів"""
        if not file_path:
            if is_single_file:
                self.single_separator_label.setVisible(False)
                self.single_separator_combo.setVisible(False)
            else:
                self.train_separator_label.setVisible(False)
                self.train_separator_combo.setVisible(False)
            return

        # Визначаємо тип файлу за розширенням
        ext = os.path.splitext(file_path)[1].lower()
        is_csv = ext == '.csv'

        if is_single_file:
            self.single_separator_label.setVisible(is_csv)
            self.single_separator_combo.setVisible(is_csv)
        else:
            self.train_separator_label.setVisible(is_csv)
            self.train_separator_combo.setVisible(is_csv)

    def update_target_field_visibility(self, should_show):
        """Показує/приховує поля вибору цільової змінної"""
        self.target_label.setVisible(should_show)
        self.target_combo.setVisible(should_show)

class HyperparamsTabWidget(QWidget):
    """Представлення для вкладки параметрів моделі (затичка)"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.params_widget = None
        self.init_ui()

    def init_ui(self):
        """Ініціалізація інтерфейсу"""
        layout = QVBoxLayout()
        self.params_widget = ParameterEditorWidget()
        self.params_widget.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.save_button = QPushButton("Save")
        layout.addWidget(self.params_widget)
        layout.addWidget(self.save_button)
        self.setLayout(layout)





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


class GeneralTabWidget(QWidget):
    """Представлення для вкладки загальних налаштувань (затичка)"""
    def __init__(self,parent=None):
        super().__init__(parent)
        self.description = ""
        self.experiment_name = ""
        self.init_ui()

    def init_ui(self):
        """Ініціалізація інтерфейсу"""
        layout = QVBoxLayout()

        # Назва експерименту
        name_layout = QHBoxLayout()
        name_layout.addWidget(QLabel("Назва експерименту:"))
        self.experiment_name = QLineEdit("Новий експеримент")
        name_layout.addWidget(self.experiment_name)
        layout.addLayout(name_layout)

        model_layout = QHBoxLayout()
        model_layout.addWidget(QLabel("Method/Model:"))
        self.method_name = QLabel("")
        model_layout.addWidget(self.method_name)
        layout.addLayout(model_layout)

        # Опис експерименту
        layout.addWidget(QLabel("Опис експерименту:"))
        self.description = QTextEdit()
        layout.addWidget(self.description)

        layout.addStretch()

        self.setLayout(layout)


