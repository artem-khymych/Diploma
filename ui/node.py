from dataclasses import dataclass

from PyQt5.QtCore import Qt, QRectF
from PyQt5.QtGui import QPainter, QColor, QBrush, QFont
from PyQt5.QtWidgets import QLineEdit, QGraphicsItem, QGraphicsTextItem, QGraphicsProxyWidget



class Node(QGraphicsItem):
    """Вузол із унікальним ID, візуальним представленням та базовими властивостями."""

    @dataclass
    class NodeStyles:
        DEFAULT_CIRCLE_RADIUS = 50
        ACTIVE_CIRCLE_RADIUS = 70
        DEFAULT_COLOR = QColor(255, 0, 0)
        DEFAULT_NODE_SIZE = 70
        ACTIVE_NODE_SIZE = 100
        ACTIVE_COLOR = QColor(0, 0, 255, 200)

    id_counter = 0

    def __init__(self, parent=None):
        super().__init__(parent)
        Node.id_counter += 1
        self.id = Node.id_counter

        self._styles = self.NodeStyles()
        self._current_color = self._styles.DEFAULT_COLOR
        self._current_radius = self._styles.DEFAULT_CIRCLE_RADIUS
        self.is_active = False

        # Створення лейбла для назви вузла
        self.label = QGraphicsTextItem(f"Experiment {self.id}", self)
        self.label.setDefaultTextColor(Qt.black)
        self.label.setFont(QFont("Arial", 15))
        self._center_label()

        # Поле для редагування назви
        self.name_editor = QLineEdit()
        self.name_editor_proxy = QGraphicsProxyWidget(self)
        self.name_editor_proxy.setWidget(self.name_editor)
        self.name_editor.setGeometry(10, self._styles.DEFAULT_CIRCLE_RADIUS, self._styles.DEFAULT_NODE_SIZE,
                                     self._styles.DEFAULT_NODE_SIZE - self._styles.DEFAULT_CIRCLE_RADIUS)
        self.name_editor.setAlignment(Qt.AlignCenter)
        self.name_editor.setStyleSheet("font-size: 15px;")
        self.name_editor.setVisible(False)
        self.name_editor.editingFinished.connect(self._on_edit_finished)

        # Налаштування прапорців для забезпечення перетягування вузла
        self.setFlag(QGraphicsItem.ItemIsMovable)
        self.setFlag(QGraphicsItem.ItemSendsGeometryChanges)
        self.setFlag(QGraphicsItem.ItemIsSelectable)

    def itemChange(self, change, value):
        """Обробляє зміни властивостей вузла."""
        # Метод буде перевизначений в NodeController для кожного вузла
        # для оновлення зв'язків при переміщенні
        return super().itemChange(change, value)

    def _center_label(self):
        """Центрує текстову мітку відносно вузла."""
        text_width = self.label.boundingRect().width()
        self.label.setPos((self._current_radius - text_width) / 2 + 10, self._current_radius)

    def _on_edit_finished(self):
        """Обробник події завершення редагування назви."""
        self.set_name(self.name_editor.text())
        self.name_editor.setVisible(False)
        self.label.setVisible(True)

    def set_name(self, name):
        """Встановлює назву вузла."""
        self.label.setPlainText(name)
        self._center_label()

    def get_name(self):
        """Повертає назву вузла."""
        return self.label.toPlainText()

    def start_editing_name(self):
        """Починає редагування назви вузла."""
        self.label.setVisible(False)
        self.name_editor.setText(self.label.toPlainText())
        self.name_editor.setVisible(True)
        self.name_editor.setFocus()

    def set_active(self, active=True):
        """Встановлює активний стан вузла."""
        self.is_active = active
        if active:
            self._current_radius = self._styles.ACTIVE_CIRCLE_RADIUS
            self._current_color = self._styles.ACTIVE_COLOR
            # Оновлюємо розташування полів при збільшенні вузла
            self.name_editor.setGeometry(10, self._styles.ACTIVE_CIRCLE_RADIUS, self._styles.ACTIVE_NODE_SIZE,
                                         self._styles.ACTIVE_NODE_SIZE - self._styles.ACTIVE_CIRCLE_RADIUS)
        else:
            self._current_radius = self._styles.DEFAULT_CIRCLE_RADIUS
            self._current_color = self._styles.DEFAULT_COLOR
            # Оновлюємо розташування полів при зменшенні вузла
            self.name_editor.setGeometry(10, self._styles.DEFAULT_CIRCLE_RADIUS, self._styles.DEFAULT_NODE_SIZE,
                                         self._styles.DEFAULT_NODE_SIZE - self._styles.DEFAULT_CIRCLE_RADIUS)
        self._center_label()
        self.update()

        # Забезпечуємо повне оновлення сцени
        if self.scene():
            self.scene().update()

    def boundingRect(self):
        """Визначає межі вузла."""
        return QRectF(0, 0, self._current_radius + 20, self._current_radius + 20)

    def paint(self, painter, option, widget=None):
        """Малює круглу ноду."""
        painter.setRenderHint(QPainter.Antialiasing)
        brush = QBrush(self._current_color)
        painter.setBrush(brush)
        painter.setPen(Qt.NoPen)
        painter.drawEllipse(10, 0, self._current_radius, self._current_radius)