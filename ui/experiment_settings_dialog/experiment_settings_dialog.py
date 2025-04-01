from PyQt5.QtWidgets import QDialog, QVBoxLayout, QTabWidget, QHBoxLayout, QPushButton

from project.ui.experiment_settings_dialog.general_tab import GeneralTabWidget
from project.ui.experiment_settings_dialog.hypeparams_tab import HyperparamsTabWidget
from project.ui.experiment_settings_dialog.input_data_tab import InputDataTabWidget
from project.ui.experiment_settings_dialog.metrics_tab import MetricsTabWidget


class ExperimentSettingsDialog(QDialog):
    """Головне діалогове вікно налаштувань експерименту"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.model_tab = None
        self.init_ui()

    def init_ui(self):
        """Ініціалізація інтерфейсу"""
        self.setWindowTitle("Налаштування експерименту")
        self.setMinimumWidth(700)
        self.setMinimumHeight(900)

        main_layout = QVBoxLayout()

        # Створюємо віджет з вкладками
        self.tab_widget = QTabWidget()

        # Створюємо вкладки
        self.general_tab = GeneralTabWidget()
        self.data_tab = InputDataTabWidget()

        self.model_tab = HyperparamsTabWidget()
        self.evaluation_tab = MetricsTabWidget()

        # Додаємо вкладки до віджета
        self.tab_widget.addTab(self.general_tab, "Description")
        self.tab_widget.addTab(self.model_tab, "Params")
        self.tab_widget.addTab(self.data_tab, "Data")
        self.tab_widget.addTab(self.evaluation_tab, "Metrics")

        main_layout.addWidget(self.tab_widget)

        # Кнопки дій
        buttons_layout = QHBoxLayout()
        self.ok_btn = QPushButton("OK")
        self.ok_btn.clicked.connect(self.accept)
        self.cancel_btn = QPushButton("Скасувати")
        self.cancel_btn.clicked.connect(self.reject)
        buttons_layout.addWidget(self.ok_btn)
        buttons_layout.addWidget(self.cancel_btn)
        main_layout.addLayout(buttons_layout)

        self.setLayout(main_layout)

