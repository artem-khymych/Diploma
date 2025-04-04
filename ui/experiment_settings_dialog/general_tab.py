from PyQt5.QtCore import pyqtSignal
from PyQt5.QtGui import QIcon
from PyQt5.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit, QTextEdit, QPushButton


class GeneralTabWidget(QWidget):
    """Представлення для вкладки загальних налаштувань"""

    # Сигнал для запуску експерименту
    experiment_started = pyqtSignal()
    evaluate_clicked = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.description = ""
        self.experiment_name = ""
        self.is_finished = False
        self.training_time = None
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

        # Статус експерименту
        self.status_layout = QHBoxLayout()
        self.status_label = QLabel("Статус експерименту:")
        self.status_value = QLabel("Не запущено")
        self.status_layout.addWidget(self.status_label)
        self.status_layout.addWidget(self.status_value)
        layout.addLayout(self.status_layout)

        # Відображення часу навчання
        self.time_layout = QHBoxLayout()
        self.time_label = QLabel("Час навчання:")
        self.time_value = QLabel("-")
        self.time_layout.addWidget(self.time_label)
        self.time_layout.addWidget(self.time_value)
        layout.addLayout(self.time_layout)

        # Кнопка для запуску експерименту
        self.button_layout = QHBoxLayout()
        self.start_button = QPushButton("Розпочати")
        #self.evaluate_button = QPushButton("Оцінити")

        #self.evaluate_button.setEnabled(False)
        #self.button_layout.addWidget(self.evaluate_button)

        # Додаємо зелений трикутник як піктограму
        self.start_button.setIcon(QIcon.fromTheme("media-playback-start"))  # Стандартна іконка відтворення
        self.start_button.clicked.connect(self.on_start_clicked)
        self.button_layout.addWidget(self.start_button)
        self.button_layout.addStretch()
        layout.addLayout(self.button_layout)

        layout.addStretch()

        self.setLayout(layout)

    def on_start_clicked(self):
        """Обробник натискання кнопки запуску"""
        #self.evaluate_button.setEnabled(True)
        self.experiment_started.emit()

    def on_evaluate_clicked(self):
        pass


    def set_experiment_finished(self, training_time):
        """Встановлення статусу завершення експерименту"""
        self.is_finished = True
        self.training_time = training_time
        self.status_value.setText("Завершено")
        self.time_value.setText(f"{training_time} с")

    def update_status(self, is_finished, training_time=None):
        """Оновлення статусу експерименту"""
        self.is_finished = is_finished
        if is_finished:
            self.status_value.setText("Завершено")
            if training_time is not None:
                self.training_time = training_time
                self.time_value.setText(f"{training_time} с")
        else:
            self.status_value.setText("Не запущено")
            self.start_button.setEnabled(True)

