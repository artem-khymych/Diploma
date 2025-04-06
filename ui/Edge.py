from PyQt5.QtCore import QRectF, QLineF
from PyQt5.QtGui import QColor, QPen
from PyQt5.QtWidgets import QGraphicsItem


class Edge(QGraphicsItem):
    """Клас для представлення зв'язку між вузлами."""

    def __init__(self, source_node, target_node, parent=None):
        super().__init__(parent)
        self.source_node = source_node
        self.target_node = target_node
        self.setZValue(-1)  # Встановлюємо зв'язок під вузлами

        # Підключаємось до змін позицій вузлів
        self.source_node.setFlag(QGraphicsItem.ItemSendsScenePositionChanges)
        self.target_node.setFlag(QGraphicsItem.ItemSendsScenePositionChanges)

        self.update_position()

    def update_position(self):
        """Оновлює позицію лінії при переміщенні вузлів."""
        self.prepareGeometryChange()
        self.update()

    def boundingRect(self):
        """Визначає межі зв'язку для оптимізації перемальовування."""
        if not self.source_node or not self.target_node:
            return QRectF()

        # Отримуємо центри вузлів
        source_center = self.source_node.mapToScene(
            self.source_node.boundingRect().center().x(),
            self.source_node.boundingRect().center().y()
        )
        target_center = self.target_node.mapToScene(
            self.target_node.boundingRect().center().x(),
            self.target_node.boundingRect().center().y()
        )

        # Створюємо прямокутник, що охоплює обидва центри з невеликим запасом
        return QRectF(
            min(source_center.x(), target_center.x()) - 5,
            min(source_center.y(), target_center.y()) - 5,
            abs(source_center.x() - target_center.x()) + 10,
            abs(source_center.y() - target_center.y()) + 10
        )

    def paint(self, painter, option, widget=None):
        """Малює лінію між вузлами."""
        if not self.source_node or not self.target_node:
            return

        # Отримуємо центри вузлів
        source_center = self.source_node.mapToScene(
            self.source_node.boundingRect().center().x(),
            self.source_node.boundingRect().center().y()
        )
        target_center = self.target_node.mapToScene(
            self.target_node.boundingRect().center().x(),
            self.target_node.boundingRect().center().y()
        )

        # Налаштування пера для малювання
        pen = QPen(QColor(0, 0, 0))  # Чорний колір
        pen.setWidth(2)
        painter.setPen(pen)

        # Малюємо лінію
        line = QLineF(source_center, target_center)
        painter.drawLine(line)
