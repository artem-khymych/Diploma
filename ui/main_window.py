import sys
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QVBoxLayout, QHBoxLayout,
    QFrame, QPushButton, QListWidget, QTextEdit, QWidget, QSplitter
)
from PyQt5.QtCore import Qt

from project.controllers.inspector_controller import InspectorController
from project.ui.inspector_window import InspectorWindow


# TODO нижня панель із текстом
# TODO бокова панель із результатами і даними
# TODO маус зона і додавання ноди, промальовка та успадкування
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Робоче вікно з можливістю змінювати розміри зон")
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

        self.inspector = InspectorWindow()
        self.inspector.setFrameShape(QFrame.StyledPanel)

        # Додавання в спліттер
        self.splitter.addWidget(self.inspector)
        self.splitter.addWidget(self.central_area)

        self.splitter.setStretchFactor(1, 3)  # Робоча зона займає більше місця
        self.splitter.setStretchFactor(0, 1)  # Права панель займає менше місця

        # Горизонтальне розташування з кнопками та спліттером
        center_layout = QHBoxLayout()
        center_layout.addWidget(self.splitter)

        main_layout.addLayout(center_layout, 10)
        self.inspector_controller = InspectorController(self.inspector)

    def init_top_panel(self):
        """Ініціалізація верхньої панелі з кнопками."""
        layout = QHBoxLayout(self.top_panel)

        # Додати кнопки
        self.btn_settings = QPushButton("Налаштування")
        self.btn_start = QPushButton("Запуск")
        self.btn_stop = QPushButton("Зупинка")

        layout.addWidget(self.btn_settings)
        layout.addWidget(self.btn_start)
        layout.addWidget(self.btn_stop)

        # Розтягнути панель
        layout.addStretch()

    def init_central_area(self):
        """Ініціалізація робочої зони."""
        layout = QVBoxLayout(self.central_area)

        # Робоча зона - великий текстовий редактор
        self.text_editor = QTextEdit()
        self.text_editor.setPlaceholderText("Тут буде ваша робоча зона...")
        layout.addWidget(self.text_editor)
    def init_right_area(self):
        """Ініціалізація робочої зони."""
        layout = QVBoxLayout(self.central_area)

        # Робоча зона - великий текстовий редактор
        self.text_editor = QTextEdit()
        self.text_editor.setPlaceholderText("Тут буде ваша робоча зона...")
        layout.addWidget(self.text_editor)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
