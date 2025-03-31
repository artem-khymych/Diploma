from PyQt5.QtCore import Qt, QPoint, QTimer
from PyQt5.QtWidgets import QGraphicsView, QMenu, QAction, QMessageBox

from project.ui.node import Node
from PyQt5.QtCore import pyqtSignal, QObject

class NodeController:
    """Контролер вузлів - відповідальний за створення, управління та взаємодію вузлів."""

    def __init__(self, scene, view):
        self.scene = scene
        self.view = view
        self.nodes = []
        self.active_node = None
        self.drag_timer = QTimer()
        self.drag_timer.setSingleShot(True)
        self.drag_timer.timeout.connect(self._start_dragging)
        self.current_press_node = None

        # Створюємо клас-посередник для сигналів
        class SignalEmitter(QObject):
            node_created = pyqtSignal(object)  # Сигнал створення вузла
            node_deleted = pyqtSignal(int)  # Сигнал видалення вузла (передаємо ID)
            node_renamed = pyqtSignal(object)  # Сигнал перейменування вузла
            nodeInfoOpened = pyqtSignal(int)

        self.signals = SignalEmitter()
        self.node_created = self.signals.node_created
        self.node_deleted = self.signals.node_deleted
        self.node_renamed = self.signals.node_renamed
        self.nodeInfoOpened = self.signals.nodeInfoOpened

        self.scene.installEventFilter(view)
        view.mousePressEvent = self._view_mouse_press
        view.mouseReleaseEvent = self._view_mouse_release
        view.mouseMoveEvent = self._view_mouse_move
        view.contextMenuEvent = self._view_context_menu

    def create_node(self):
        """Створює новий вузол в центрі видимої області."""
        node = Node()
        # Отримуємо видиму область перегляду
        viewport_rect = self.view.viewport().rect()

        # Перетворюємо центр видимої області в координати сцени
        viewport_center = QPoint(viewport_rect.width() // 2, viewport_rect.height() // 2)
        scene_center = self.view.mapToScene(viewport_center)

        # Враховуємо розмір вузла для точного розташування в центрі
        node_width = node.boundingRect().width()
        node_height = node.boundingRect().height()

        # Встановлюємо позицію вузла в центрі видимої області
        node.setPos(scene_center.x() - node_width / 2, scene_center.y() - node_height / 2)

        self.scene.addItem(node)
        self.nodes.append(node)

        self.node_created.emit(node)

        return node

    def open_node_info(self, node):
        """Відкриває діалогове вікно з інформацією про вузол."""
        self.nodeInfoOpened.emit(node.id)

    def edit_node_name(self, node):
        """Активує редагування назви вузла."""
        node.start_editing_name()
        # Підписуємось на сигнал завершення редагування
        node.name_editor.editingFinished.connect(lambda: self.node_renamed.emit(node))

    def delete_node(self, node):
        """Видаляє вузол зі сцени та зі списку вузлів."""
        if node in self.nodes:
            node_id = node.id
            self.scene.removeItem(node)
            self.nodes.remove(node)
            if self.active_node == node:
                self.active_node = None

            # Сигналізуємо про видалення вузла
            self.node_deleted.emit(node_id)

    def center_on_node(self, node_id):
        """Центрує вид на вузлі за його ID."""
        for node in self.nodes:
            if node.id == node_id:
                # Отримуємо центр вузла
                node_center = node.mapToScene(
                    node.boundingRect().center().x(),
                    node.boundingRect().center().y()
                )
                # Центруємо вид на вузлі
                self.view.centerOn(node_center)
                break

    def _view_mouse_press(self, event):
        """Обробник натискання кнопки миші на GraphicsView."""
        # Отримуємо елемент під мишею
        pos = self.view.mapToScene(event.pos())
        item = self.scene.itemAt(pos, self.view.transform())

        if event.button() == Qt.LeftButton and isinstance(item, Node):
            self.current_press_node = item
            # Запускаємо таймер для визначення затиснення
            self.drag_timer.start(200)
        else:
            # Передаємо стандартну обробку, якщо це не вузол
            QGraphicsView.mousePressEvent(self.view, event)

    def _view_mouse_release(self, event):
        """Обробник відпускання кнопки миші на GraphicsView."""
        if event.button() == Qt.LeftButton and self.current_press_node:
            if self.drag_timer.isActive():
                # Якщо таймер ще активний, це клік
                self.drag_timer.stop()
                self.open_node_info(self.current_press_node)
            else:
                # Якщо таймер неактивний, це закінчення перетягування
                if self.active_node:
                    self.active_node.set_active(False)
                    self.active_node = None

            self.current_press_node = None
        else:
            # Передаємо стандартну обробку для інших випадків
            QGraphicsView.mouseReleaseEvent(self.view, event)

    def _view_mouse_move(self, event):
        """Обробник руху миші на GraphicsView."""
        if self.active_node:
            # Якщо є активний вузол, оновлюємо його позицію
            pos = self.view.mapToScene(event.pos())
            node_width = self.active_node.boundingRect().width()
            node_height = self.active_node.boundingRect().height()
            self.active_node.setPos(pos.x() - node_width / 2, pos.y() - node_height / 2)
        else:
            # Для інших випадків - стандартна обробка
            QGraphicsView.mouseMoveEvent(self.view, event)

    def _view_context_menu(self, event):
        """Обробник контекстного меню."""
        # Отримуємо елемент під мишею
        pos = self.view.mapToScene(event.pos())
        item = self.scene.itemAt(pos, self.view.transform())

        if isinstance(item, Node):
            # Створюємо контекстне меню
            context_menu = QMenu(self.view)

            # Додаємо дії
            rename_action = QAction("Перейменувати", self.view)
            delete_action = QAction("Видалити", self.view)

            # Підключаємо обробники
            rename_action.triggered.connect(lambda: self.edit_node_name(item))
            delete_action.triggered.connect(lambda: self.delete_node(item))

            # Додаємо дії до меню
            context_menu.addAction(rename_action)
            context_menu.addAction(delete_action)

            # Показуємо меню
            context_menu.exec_(event.globalPos())
        else:
            # Стандартна обробка для інших випадків
            QGraphicsView.contextMenuEvent(self.view, event)

    def _start_dragging(self):
        """Активує режим перетягування вузла."""
        if self.current_press_node:
            self.active_node = self.current_press_node
            self.active_node.set_active(True)

