
from typing import Dict, List, Tuple, Any, Optional
from PyQt5.QtCore import QObject, pyqtSlot, Qt
from PyQt5.QtWidgets import QGraphicsView, QWidget, QMessageBox, QFileDialog
from project.controllers.node_controller import NodeController
from project.controllers.serializer import WorkspaceSerializer

from project.logic.experiment_manager import ExperimentManager

class WorkspaceManager(QObject):
    """
    Менеджер робочого простору, відповідальний за збереження та завантаження стану програми,
    а також за керування файлами проектів.
    """

    def __init__(self, parent=None):
        super().__init__(parent)

        # Шлях до поточного файлу проекту
        self.current_project_path: Optional[str] = None

        # Статус змін у проекті (чи були незбережені зміни)
        self.has_unsaved_changes: bool = False

        # Сериалізатор для збереження/завантаження
        self.serializer = WorkspaceSerializer()

        # Референси до інших компонентів системи
        self.experiment_manager: Optional[ExperimentManager] = None
        self.node_controller: Optional[NodeController] = None
        self.work_area: Optional[QGraphicsView] = None  # Додаємо посилання на робочу область

        # Підписуємось на сигнали серіалізатора
        self._connect_serializer_signals()

    def set_experiment_manager(self, manager: ExperimentManager):
        """Встановлює менеджер експериментів для WorkspaceManager."""
        self.experiment_manager = manager
        self.serializer.set_experiment_manager(manager)

    def set_node_controller(self, controller: NodeController):
        """Встановлює контролер вузлів для WorkspaceManager."""
        self.node_controller = controller
        self.serializer.set_node_controller(controller)

        # Підписуємось на сигнали контролера для відстеження змін
        self._connect_node_controller_signals()

    def set_work_area(self, work_area: QGraphicsView):
        """Встановлює робочу область для WorkspaceManager."""
        self.work_area = work_area

    def _connect_serializer_signals(self):
        """Підключає сигнали серіалізатора до відповідних слотів."""
        self.serializer.workspace_saved.connect(self._on_workspace_saved)
        self.serializer.workspace_loaded.connect(self._on_workspace_loaded)
        self.serializer.save_error.connect(self._on_save_error)
        self.serializer.load_error.connect(self._on_load_error)

    def _connect_node_controller_signals(self):
        """Підключає сигнали контролера вузлів для відстеження змін."""
        if self.node_controller:
            # Відстежуємо зміни у графі
            self.node_controller.node_created.connect(self._on_workspace_modified)
            self.node_controller.node_deleted.connect(self._on_workspace_modified)
            self.node_controller.node_renamed.connect(self._on_workspace_modified)
            self.node_controller.experiment_inherited.connect(self._on_workspace_modified)

    @pyqtSlot()
    def _on_workspace_modified(self):
        """Обробник події зміни робочого простору."""
        self.has_unsaved_changes = True

    @pyqtSlot(str)
    def _on_workspace_saved(self, path: str):
        """Обробник події успішного збереження робочого простору."""
        self.current_project_path = path
        self.has_unsaved_changes = False
        print(f"Робочий простір успішно збережено: {path}")

    @pyqtSlot(str)
    def _on_workspace_loaded(self, path: str):
        """Обробник події успішного завантаження робочого простору."""
        self.current_project_path = path
        self.has_unsaved_changes = False
        print(f"Робочий простір успішно завантажено: {path}")

        # Після завантаження підганяємо вигляд під всі елементи
        self.fit_view_to_content()

    @pyqtSlot(str)
    def _on_save_error(self, error_msg: str):
        """Обробник помилки збереження."""
        print(f"Помилка при збереженні: {error_msg}")
        QMessageBox.critical(None, "Помилка збереження", error_msg)

    @pyqtSlot(str)
    def _on_load_error(self, error_msg: str):
        """Обробник помилки завантаження."""
        print(f"Помилка при завантаженні: {error_msg}")
        QMessageBox.critical(None, "Помилка завантаження", error_msg)

    def new_project(self):
        """Створює новий проект."""
        # Перевіряємо наявність незбережених змін
        if self.has_unsaved_changes and not self._confirm_discard_changes():
            return False

        # Очищаємо поточні дані
        self._clear_workspace()

        # Скидаємо поточний шлях проекту
        self.current_project_path = None
        self.has_unsaved_changes = False

        return True

    def save_project(self, parent_widget: Optional[QWidget] = None) -> bool:
        """
        Зберігає поточний проект.

        Args:
            parent_widget: Батьківський віджет для діалогових вікон

        Returns:
            bool: True при успіху, False при помилці
        """
        # Перевіряємо наявність шляху до проекту
        if not self.current_project_path:
            return self.save_project_as(parent_widget)

        # Зберігаємо проект за поточним шляхом
        success = self.serializer.save_workspace(self.current_project_path)
        return success

    def save_project_as(self, parent_widget: Optional[QWidget] = None) -> bool:
        """
        Зберігає проект з вибором нового файлу.

        Args:
            parent_widget: Батьківський віджет для діалогових вікон

        Returns:
            bool: True при успіху, False при помилці
        """
        # Відкриваємо діалог вибору файлу для збереження
        options = QFileDialog.Options()
        filepath, _ = QFileDialog.getSaveFileName(
            parent_widget,
            "Зберегти проект",
            "",
            "ML Project Files (*.mlproj);;All Files (*)",
            options=options
        )

        if filepath:
            # Додаємо розширення, якщо користувач його не вказав
            if not filepath.endswith('.mlproj'):
                filepath += '.mlproj'

            # Зберігаємо проект у вибраний файл
            return self.serializer.save_workspace(filepath)

        return False

    def open_project(self, parent_widget: Optional[QWidget] = None) -> bool:
        """
        Відкриває існуючий проект.

        Args:
            parent_widget: Батьківський віджет для діалогових вікон

        Returns:
            bool: True при успіху, False при помилці
        """
        # Перевіряємо наявність незбережених змін
        if self.has_unsaved_changes and not self._confirm_discard_changes():
            return False

        # Відкриваємо діалог вибору файлу для завантаження
        options = QFileDialog.Options()
        filepath, _ = QFileDialog.getOpenFileName(
            parent_widget,
            "Відкрити проект",
            "",
            "ML Project Files (*.mlproj);;All Files (*)",
            options=options
        )

        if filepath:
            # Завантажуємо проект з вибраного файлу
            return self.serializer.load_workspace(filepath)

        return False

    def fit_view_to_content(self):
        """Підганяє вигляд під всі елементи на сцені."""
        if self.work_area and self.node_controller and self.node_controller.scene:
            # Якщо немає елементів, не робимо нічого
            if not self.node_controller.nodes:
                return

            # Обчислюємо прямокутник, що містить всі вузли
            rect = None
            for node in self.node_controller.nodes:
                node_rect = node.sceneBoundingRect()
                if rect is None:
                    rect = node_rect
                else:
                    rect = rect.united(node_rect)

            # Додаємо відступ навколо елементів
            if rect:
                margin = 50  # Відступ у пікселях
                rect = rect.adjusted(-margin, -margin, margin, margin)

                # Підганяємо вигляд під цей прямокутник
                self.work_area.fitInView(rect, Qt.KeepAspectRatio)

    def _confirm_discard_changes(self) -> bool:
        """
        Запитує користувача про підтвердження відхилення незбережених змін.

        Returns:
            bool: True, якщо користувач підтвердив відхилення змін, False інакше
        """
        reply = QMessageBox.question(
            None,
            "Незбережені зміни",
            "Є незбережені зміни. Бажаєте зберегти їх перед продовженням?",
            QMessageBox.Save | QMessageBox.Discard | QMessageBox.Cancel,
            QMessageBox.Save
        )

        if reply == QMessageBox.Save:
            # Зберігаємо зміни
            return self.save_project()
        elif reply == QMessageBox.Discard:
            # Відхиляємо зміни
            return True
        else:  # QMessageBox.Cancel
            # Скасовуємо дію
            return False

    def _clear_workspace(self):
        """Очищує робочий простір."""
        # Очищаємо експерименти в менеджері
        if self.experiment_manager:
            self.experiment_manager.experiments = {}

        # Очищаємо вузли на сцені
        if self.node_controller:
            for node in list(self.node_controller.nodes):
                self.node_controller.delete_node(node)
