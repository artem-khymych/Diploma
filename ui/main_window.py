import sys
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QVBoxLayout, QHBoxLayout,
    QFrame, QPushButton, QListWidget, QTextEdit, QWidget, QSplitter, QScrollArea, QGraphicsScene, QLabel
)
from PyQt5.QtCore import Qt, pyqtSignal, QObject

from project.ui.basic_window import BasicWindow
from project.ui.graphics_view import GraphicsView


# TODO нижня панель із текстом
# TODO бокова панель із результатами і даними


class MainWindow(QMainWindow):

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Програмне середовище тренування та тестування методів машинного навчання")
        self.setGeometry(100, 100, 800, 600)

        # Головний віджет (центральний)
        main_widget = QWidget()
        self.setCentralWidget(main_widget)

        # Основне вертикальне розташування
        main_layout = QVBoxLayout(main_widget)

        # Верхня панель для кнопок
        self.top_panel = QFrame()
        self.top_panel.setFrameShape(QFrame.StyledPanel)
        self.init_top_panel()
        main_layout.addWidget(self.top_panel, 1)

        # Створення спліттера для центральної зони та правої панелі
        self.splitter = QSplitter(Qt.Horizontal)
        self.central_area = QFrame()
        self.central_area.setFrameShape(QFrame.StyledPanel)

        self.init_central_area()
        self.init_inspector()
        self.splitter.addWidget(self.inspector_frame)
        self.splitter.addWidget(self.central_area)

        self.splitter.setStretchFactor(0, 1)  # Робоча зона займає більше місця
        self.splitter.setStretchFactor(1, 3)  # Права панель займає менше місця

        # Горизонтальне розташування з кнопками та спліттером
        center_layout = QHBoxLayout()
        center_layout.addWidget(self.splitter)

        main_layout.addLayout(center_layout, 10)

        class SignalEmitter(QObject):
            node_created = pyqtSignal(object)  # Сигнал створення вузла
            node_deleted = pyqtSignal(int)  # Сигнал видалення вузла (передаємо ID)
            node_renamed = pyqtSignal(object)  # Сигнал перейменування вузла
            add_new_experiment = pyqtSignal()
            open_settings = pyqtSignal()

        self.signals = SignalEmitter()

    def init_inspector(self):
        """Ініціалізація інспектора вузлів."""
        self.inspector_frame = BasicWindow()
        # Заголовок інспектора
        label = QLabel("Інспектор Експериментів")
        label.setStyleSheet("font-weight: bold; font-size: 14px;")
        # Список вузлів
        nodes_list = QListWidget()
        nodes_list.setObjectName("nodes_list")
        self.inspector_frame.layout.addWidget(label)
        self.inspector_frame.layout.addWidget(nodes_list)

    def init_top_panel(self):
        """Ініціалізація верхньої панелі з кнопками."""
        layout = QHBoxLayout(self.top_panel)

        # Додати кнопки
        self.settings_button = QPushButton("Налаштування")
        self.new_experiment_button = QPushButton("Створити експеримент")
        self.new_experiment_button.clicked.connect(self._add_new_experiment)

        layout.addWidget(self.settings_button)
        layout.addWidget(self.new_experiment_button)
        # Розтягнути панель
        layout.addStretch()

    def _add_new_experiment(self):
        self.signals.add_new_experiment.emit()

    def init_central_area(self):
        """Ініціалізація робочої зони."""
        layout = QVBoxLayout(self.central_area)
        self.scene = QGraphicsScene()
        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.graphics_view = GraphicsView()
        self.graphics_view.setScene(self.scene)

        self.central_widget = QWidget()
        self.central_layout = QVBoxLayout(self.central_widget)
        self.central_layout.addWidget(self.graphics_view)
        self.scroll_area.setWidget(self.central_widget)

        self.central_area.setLayout(QVBoxLayout())
        self.central_area.layout().addWidget(self.scroll_area)
        self.central_area.setStyleSheet("background-color: white;")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
