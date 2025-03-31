
from PyQt5.QtGui import QPainter, QWheelEvent
from PyQt5.QtWidgets import QGraphicsView
from PyQt5.QtCore import Qt, QRectF

class WorkArea(QGraphicsView):
    """Кастомний QGraphicsView для обробки прокрутки та масштабування."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setRenderHint(QPainter.Antialiasing)
        self.setRenderHint(QPainter.SmoothPixmapTransform)
        self.setRenderHint(QPainter.TextAntialiasing)
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.AnchorUnderMouse)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOn)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOn)
        self.setSceneRect(QRectF(0, 0, 2000, 2000))  # Розмір сцени

    def wheelEvent(self, event: QWheelEvent):
        """Обробка прокрутки коліщатка миші."""
        if event.modifiers() & Qt.ControlModifier:
            # Масштабування при затисненому Ctrl
            delta = event.angleDelta().y()
            if delta > 0:
                self.scale(1.1, 1.1)
            else:
                self.scale(0.9, 0.9)
        else:
            # Звичайна прокрутка
            super().wheelEvent(event)

