from typing import Dict, Any, Union, List
from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtWidgets import (QLineEdit, QSpinBox, QDoubleSpinBox, QCheckBox,
                             QTableWidgetItem, QHBoxLayout, QPushButton, QTableWidget,
                             QVBoxLayout, QWidget, QComboBox, QDialog, QHeaderView,
                             QTabWidget, QDialogButtonBox, QListWidget, QAbstractItemView,
                             QMessageBox, QStackedWidget, QListWidgetItem)

import json
import ast
from typing import Optional, Union, Dict, List, Any


class SimpleParameterEditor(QWidget):
    """Редактор для простих словників параметрів"""
    parameterChanged = pyqtSignal(dict)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout(self)

        self.table = QTableWidget()
        self.table.setColumnCount(2)
        self.table.setHorizontalHeaderLabels(['Параметр', 'Значення'])
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.verticalHeader().setVisible(False)

        layout.addWidget(self.table)

        # Кнопки управління
        btn_layout = QHBoxLayout()
        self.save_btn = QPushButton("Зберегти")
        self.save_btn.clicked.connect(lambda: self.parameterChanged.emit(self.get_parameters()))
        btn_layout.addWidget(self.save_btn)

        layout.addLayout(btn_layout)

    def populate(self, params: dict):
        self.table.setRowCount(0)
        if not params:
            return

        self.table.setRowCount(len(params))
        for row, (key, value) in enumerate(params.items()):
            # Ключ параметра
            key_item = QTableWidgetItem(key)
            key_item.setFlags(key_item.flags() & ~Qt.ItemIsEditable)
            self.table.setItem(row, 0, key_item)

            # Значення параметра
            self._set_value_widget(row, value)

    def _set_value_widget(self, row: int, value: Any):
        # Перевіряємо на None перед перевіркою типу
        if value is None:
            widget = QLineEdit("None")
            self.table.setCellWidget(row, 1, widget)
            return

        # Далі обробка інших типів
        if not isinstance(value, str):
            if isinstance(value, bool):
                widget = QCheckBox()
                widget.setChecked(value)
                self.table.setCellWidget(row, 1, widget)
            elif isinstance(value, int):
                widget = QSpinBox()
                widget.setValue(value)
                self.table.setCellWidget(row, 1, widget)
            elif isinstance(value, float):
                widget = QDoubleSpinBox()
                widget.setValue(value)
                self.table.setCellWidget(row, 1, widget)
            elif isinstance(value, (dict, list)):
                btn = QPushButton("Редагувати...")
                btn.setProperty("original_value", value)
                btn.clicked.connect(lambda: self._edit_collection(btn))
                self.table.setCellWidget(row, 1, btn)
            return

        # Обробка рядків
        parsed_value = self.smart_value_parser(value)

        if not isinstance(parsed_value, str):
            self._set_value_widget(row, parsed_value)
        else:
            widget = QLineEdit(value)
            self.table.setCellWidget(row, 1, widget)

    def get_parameters(self) -> dict:
        params = {}
        for row in range(self.table.rowCount()):
            key_item = self.table.item(row, 0)
            if not key_item:
                continue

            key = key_item.text()
            widget = self.table.cellWidget(row, 1)

            if isinstance(widget, QCheckBox):
                value = widget.isChecked()
            elif isinstance(widget, (QSpinBox, QDoubleSpinBox)):
                value = widget.value()
            elif isinstance(widget, QPushButton):
                value = widget.property("original_value")
            elif isinstance(widget, QLineEdit):
                value = self.smart_value_parser(widget.text())
            else:
                value = None

            params[key] = value
        return params

    def _edit_collection(self, button: QPushButton):
        """Редагування списків/словників у простому редакторі"""
        original_value = button.property("original_value")
        dialog = EditableDictListEditor(original_value, self)
        if dialog.exec_() == QDialog.Accepted:
            new_value = dialog.get_data()
            button.setProperty("original_value", new_value)
            button.setText(self._format_collection_preview(new_value))

    def _format_collection_preview(self, data: Union[dict, list]) -> str:
        """Форматує короткий опис вмісту колекції"""
        if isinstance(data, dict):
            return f"Словник ({len(data)} елем.)"
        return f"Список ({len(data)} елем.)"

    def smart_value_parser(self, value_str: str) -> Any:
        """
        Намагається розпізнати тип даних у рядку:
        - None, True, False
        - Числа (int, float)
        - Списки, словники (у форматі JSON або Python)
        - Якщо не вдається - повертає оригінальний рядок
        """
        if not isinstance(value_str, str):
            return value_str

        value_str = value_str.strip()

        # Спеціальні значення
        if value_str.lower() == "none":
            return None
        if value_str.lower() == "true":
            return True
        if value_str.lower() == "false":
            return False

        # Спроба розпізнати число
        try:
            return int(value_str)
        except ValueError:
            try:
                return float(value_str)
            except ValueError:
                pass

        # Спроба розпізнати JSON
        try:
            return json.loads(value_str)
        except json.JSONDecodeError:
            pass

        # Спроба розпізнати Python literal (але обережно через безпеку)
        try:
            parsed = ast.literal_eval(value_str)
            if isinstance(parsed, (list, dict, tuple, set)):
                return parsed
        except (ValueError, SyntaxError):
            pass

        # Якщо нічого не спрацювало - повертаємо оригінальний рядок
        return value_str

    def _edit_collection(self, button: QPushButton):
        """Редагування списків/словників у простому редакторі"""
        original_value = button.property("original_value")
        dialog = EditableDictListEditor(original_value, self)
        if dialog.exec_() == QDialog.Accepted:
            new_value = dialog.get_data()
            button.setProperty("original_value", new_value)
            button.setText(self._format_collection_preview(new_value))

    def _format_collection_preview(self, data: Union[dict, list]) -> str:
        """Форматує короткий опис вмісту колекції"""
        if isinstance(data, dict):
            return f"Словник ({len(data)} елем.)"
        return f"Список ({len(data)} елем.)"
class EditableDictListEditor(QDialog):
    """Діалог для редагування словників/списків з можливістю додавання/видалення елементів"""

    def __init__(self, data: Union[Dict, List], parent=None, title="Редагування"):
        super().__init__(parent)
        self.original_data = data
        self.is_dict = isinstance(data, dict)
        self.setWindowTitle(title)
        self.setMinimumSize(400, 300)
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout(self)

        # Визначаємо тип елементів (для нових елементів)
        self.element_type = self._detect_element_type()

        # Таблиця для відображення даних
        self.table = QTableWidget()
        self.table.setColumnCount(2 if self.is_dict else 1)
        headers = ['Ключ', 'Значення'] if self.is_dict else ['Значення']
        self.table.setHorizontalHeaderLabels(headers)
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.verticalHeader().setVisible(False)

        self._populate_table()
        layout.addWidget(self.table)

        # Кнопки управління
        btn_layout = QHBoxLayout()
        self.add_btn = QPushButton("Додати")
        self.remove_btn = QPushButton("Видалити")
        btn_layout.addWidget(self.add_btn)
        btn_layout.addWidget(self.remove_btn)
        btn_layout.addStretch()

        layout.addLayout(btn_layout)

        # Діалогові кнопки
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

        # Підключення сигналів
        self.add_btn.clicked.connect(self._add_item)
        self.remove_btn.clicked.connect(self._remove_item)

    def _detect_element_type(self):
        """Визначає тип елементів у колекції"""
        if not self.original_data:
            return str  # Тип за замовчуванням для порожніх колекцій

        sample = next(iter(self.original_data.values() if self.is_dict else self.original_data))

        if isinstance(sample, bool):
            return bool
        elif isinstance(sample, int):
            return int
        elif isinstance(sample, float):
            return float
        return str

    def _populate_table(self):
        """Заповнює таблицю даними"""
        self.table.setRowCount(len(self.original_data))

        if self.is_dict:
            for row, (key, value) in enumerate(self.original_data.items()):
                self._set_row(row, key, value)
        else:
            for row, value in enumerate(self.original_data):
                self._set_row(row, None, value)

    def _set_row(self, row: int, key: Any, value: Any):
        """Додає рядок до таблиці"""
        if self.is_dict:
            key_item = QTableWidgetItem(str(key))
            self.table.setItem(row, 0, key_item)
            self._set_value_widget(row, 1, value)
        else:
            self._set_value_widget(row, 0, value)

    def _set_value_widget(self, row: int, col: int, value: Any):
        """Встановлює віджет для значення"""
        if value is None:
            widget = QLineEdit("None")
            self.table.setCellWidget(row, col, widget)
        elif isinstance(value, bool):
            widget = QCheckBox()
            widget.setChecked(value)
            self.table.setCellWidget(row, col, widget)
        elif isinstance(value, (int, float)):
            if isinstance(value, int):
                widget = QSpinBox()
                widget.setRange(-999999, 999999)
            else:
                widget = QDoubleSpinBox()
                widget.setRange(-999999.0, 999999.0)
                widget.setDecimals(6)
            widget.setValue(value)
            self.table.setCellWidget(row, col, widget)
        else:
            widget = QLineEdit(str(value))
            self.table.setCellWidget(row, col, widget)

    def _add_item(self):
        """Додає новий елемент"""
        row = self.table.rowCount()
        self.table.insertRow(row)

        if self.is_dict:
            # Для словника - додаємо ключ і значення
            key_item = QTableWidgetItem("")
            self.table.setItem(row, 0, key_item)
            self._set_value_widget(row, 1, self._get_default_value())
        else:
            # Для списку - тільки значення
            self._set_value_widget(row, 0, self._get_default_value())

    def _get_default_value(self):
        """Повертає значення за замовчуванням для нового елемента"""
        if self.element_type == bool:
            return False
        elif self.element_type == int:
            return 0
        elif self.element_type == float:
            return 0.0
        return ""

    def _remove_item(self):
        """Видаляє вибрані елементи"""
        # Отримуємо унікальні індекси вибраних рядків
        selected_rows = set()
        for index in self.table.selectedIndexes():
            selected_rows.add(index.row())

        if not selected_rows:
            QMessageBox.warning(self, "Попередження", "Виберіть елементи для видалення")
            return

        # Видаляємо рядки у зворотному порядку, щоб не зміщувати індекси
        for row in sorted(selected_rows, reverse=True):
            self.table.removeRow(row)

    def get_data(self) -> Union[Dict, List]:
        """Повертає оновлені дані"""
        if self.is_dict:
            result = {}
            for row in range(self.table.rowCount()):
                key = self.table.item(row, 0).text()
                widget = self.table.cellWidget(row, 1)
                result[key] = self._get_widget_value(widget)
            return result
        else:
            return [self._get_widget_value(self.table.cellWidget(row, 0))
                    for row in range(self.table.rowCount())]
    def _get_widget_value(self, widget) -> Any:
        """Отримує значення з віджета"""
        if isinstance(widget, QCheckBox):
            return widget.isChecked()
        elif isinstance(widget, (QSpinBox, QDoubleSpinBox)):
            return widget.value()
        elif isinstance(widget, QLineEdit):
            text = widget.text()
            # Спроба конвертації типів
            if text.lower() == "none":
                return None
            elif text.lower() == "true":
                return True
            elif text.lower() == "false":
                return False
            try:
                return int(text)
            except ValueError:
                try:
                    return float(text)
                except ValueError:
                    return text
        return None
class ComplexParameterEditor(QWidget):
    """Редактор для складних словників з вкладеними параметрами"""
    parameterChanged = pyqtSignal(dict)

    KNOWN_OPTIONS = {
        "optimizer": ["adam", "sgd", "rmsprop", "adagrad", "adadelta", "adamax", "nadam"],
        "activation": ["relu", "sigmoid", "tanh", "softmax", "softplus", "softsign", "elu", "selu", "linear"],
        "loss": ["binary_crossentropy", "sparse_categorical_crossentropy", "categorical_crossentropy", "mse", "mae", "mape", "cosine_similarity"],
        "metrics": ["accuracy", "precision", "recall", "auc", "mae", "mse"],
        "callbacks": ["EarlyStopping", "ModelCheckpoint", "ReduceLROnPlateau", "TensorBoard", "CSVLogger"]
    }

    MULTI_SELECT_PARAMS = ["metrics", "callbacks"]
    EDITABLE_DICT_PARAMS = ["class_weight"]

    def __init__(self, parent=None):
        super().__init__(parent)
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout(self)

        self.tab_widget = QTabWidget()

        # Створюємо вкладки
        self.main_table = self._create_table()
        self.model_params_table = self._create_table()
        self.fit_params_table = self._create_table()
        self.task_spec_params_table = self._create_table()

        self.tab_widget.addTab(self.model_params_table, "Параметри моделі")
        self.tab_widget.addTab(self.fit_params_table, "Параметри навчання")
        self.tab_widget.addTab(self.task_spec_params_table, "Параметри задачі")

        layout.addWidget(self.tab_widget)

        # Кнопка збереження
        self.save_btn = QPushButton("Зберегти")
        self.save_btn.clicked.connect(lambda: self.parameterChanged.emit(self.get_parameters()))
        layout.addWidget(self.save_btn)

    def _create_table(self) -> QTableWidget:
        table = QTableWidget()
        table.setColumnCount(2)
        table.setHorizontalHeaderLabels(['Параметр', 'Значення'])
        table.horizontalHeader().setStretchLastSection(True)
        table.verticalHeader().setVisible(False)
        return table

    def populate(self, params: dict):
        # Основні параметри
        main_params = {k: v for k, v in params.items()
                       if k not in ["model_params", "fit_params", "task_spec_params"]}
        self._populate_table(self.main_table, main_params)

        # Вкладені параметри
        if "model_params" in params:
            self._populate_table(self.model_params_table, params["model_params"])
            self.tab_widget.setTabVisible(0, True)
        else:
            self.tab_widget.setTabVisible(0, False)

        if "fit_params" in params:
            self._populate_table(self.fit_params_table, params["fit_params"])
            self.tab_widget.setTabVisible(1, True)
        else:
            self.tab_widget.setTabVisible(1, False)

        if "task_spec_params" in params:
            self._populate_table(self.task_spec_params_table, params["task_spec_params"])
            self.tab_widget.setTabVisible(2, True)
        else:
            self.tab_widget.setTabVisible(2, False)

    def _populate_table(self, table: QTableWidget, params: dict):
        table.setRowCount(0)
        if not params:
            return

        table.setRowCount(len(params))
        for row, (key, value) in enumerate(params.items()):
            # Ключ параметра
            key_item = QTableWidgetItem(key)
            key_item.setFlags(key_item.flags() & ~Qt.ItemIsEditable)
            table.setItem(row, 0, key_item)

            # Значення параметра
            self._set_value_widget(table, row, key, value)

    def _set_value_widget(self, table: QTableWidget, row: int, key: str, value: Any):
        # Перевіряємо на None перед іншими перевірками
        if value is None:
            line_edit = QLineEdit("None")
            table.setCellWidget(row, 1, line_edit)
            return

        if key in self.KNOWN_OPTIONS:
            if key in self.MULTI_SELECT_PARAMS and isinstance(value, list):
                # Множинний вибір
                btn = QPushButton(", ".join(value))
                btn.clicked.connect(lambda: self._edit_multi_select(key, btn))
                table.setCellWidget(row, 1, btn)
            else:
                # Одиночний вибір
                combo = QComboBox()
                combo.addItems(self.KNOWN_OPTIONS[key])
                if value in self.KNOWN_OPTIONS[key]:
                    combo.setCurrentText(value)
                table.setCellWidget(row, 1, combo)
        elif key in self.EDITABLE_DICT_PARAMS or (
                isinstance(value, (dict, list)) and not self._is_fixed_structure(value)):
            # Повністю редагуємі словники/списки
            btn = QPushButton("Редагувати...")
            btn.setProperty("original_value", value)
            btn.clicked.connect(lambda: self._edit_editable_collection(btn))
            table.setCellWidget(row, 1, btn)
        elif isinstance(value, (dict, list)):
            # Фіксовані вкладені структури
            btn = QPushButton("Переглянути...")
            btn.setProperty("original_value", value)
            btn.clicked.connect(lambda: self._view_nested_structure(btn))
            table.setCellWidget(row, 1, btn)
        elif isinstance(value, bool):
            checkbox = QCheckBox()
            checkbox.setChecked(value)
            table.setCellWidget(row, 1, checkbox)
        elif isinstance(value, int):
            spinbox = QSpinBox()
            spinbox.setValue(value)
            table.setCellWidget(row, 1, spinbox)
        elif isinstance(value, float):
            dspinbox = QDoubleSpinBox()
            dspinbox.setValue(value)
            table.setCellWidget(row, 1, dspinbox)
        else:
            line_edit = QLineEdit(str(value))
            table.setCellWidget(row, 1, line_edit)

    def _is_fixed_structure(self, value: Union[dict, list]) -> bool:
        if isinstance(value, list) and len(value) > 0:
            first_type = type(value[0])
            return not all(isinstance(x, first_type) for x in value)
        return False

    def _edit_multi_select(self, key: str, button: QPushButton):
        current = button.text().split(", ") if button.text() else []
        dialog = MultiSelectDialog(self.KNOWN_OPTIONS[key], current, self)
        if dialog.exec_() == QDialog.Accepted:
            selected = dialog.get_selected()
            button.setText(", ".join(selected))

    def _edit_editable_collection(self, button: QPushButton):
        original_value = button.property("original_value")
        dialog = EditableDictListEditor(original_value, self)
        if dialog.exec_() == QDialog.Accepted:
            new_value = dialog.get_data()
            button.setProperty("original_value", new_value)
            button.setText(self._format_collection_preview(new_value))

    def _view_nested_structure(self, button: QPushButton):
        value = button.property("original_value")
        if isinstance(value, dict):
            text = "\n".join(f"{k}: {v}" for k, v in value.items())
        else:
            text = "\n".join(str(v) for v in value)
        QMessageBox.information(self, "Вміст", text)

    def _format_collection_preview(self, data: Union[dict, list]) -> str:
        if isinstance(data, dict):
            return f"Словник ({len(data)} елем.)"
        return f"Список ({len(data)} елем.)"

    def get_parameters(self) -> dict:
        params = {
            "model_params": self._get_table_params(self.model_params_table),
            "fit_params": self._get_table_params(self.fit_params_table),
            "task_spec_params": self._get_table_params(self.task_spec_params_table)
        }
        #return {k: v for k, v in params.items() if v is not None}
        return params

    def _get_table_params(self, table: QTableWidget) -> dict:
        params = {}
        for row in range(table.rowCount()):
            key_item = table.item(row, 0)
            if not key_item:
                continue

            key = key_item.text()
            widget = table.cellWidget(row, 1)

            if isinstance(widget, QComboBox):
                value = widget.currentText()
            elif isinstance(widget, QCheckBox):
                value = widget.isChecked()
            elif isinstance(widget, (QSpinBox, QDoubleSpinBox)):
                value = widget.value()
            elif isinstance(widget, QPushButton):
                if hasattr(widget, "text") and "," in widget.text():
                    value = [x.strip() for x in widget.text().split(",")]
                else:
                    value = widget.property("original_value")
            elif isinstance(widget, QLineEdit):
                text = widget.text()
                if text.lower() == "none":
                    value = None
                elif text.lower() == "true":
                    value = True
                elif text.lower() == "false":
                    value = False
                else:
                    try:
                        value = int(text)
                    except ValueError:
                        try:
                            value = float(text)
                        except ValueError:
                            value = text
            else:
                value = None

            params[key] = value
        return params
class MultiSelectDialog(QDialog):
    """Діалог для множинного вибору значень"""

    def __init__(self, options: list, selected: list, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Вибір значень")
        self.setModal(True)

        layout = QVBoxLayout(self)
        self.list_widget = QListWidget()
        self.list_widget.setSelectionMode(QAbstractItemView.MultiSelection)

        for option in options:
            item = QListWidgetItem(option)
            item.setSelected(option in selected)
            self.list_widget.addItem(item)

        layout.addWidget(self.list_widget)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def get_selected(self) -> list:
        return [item.text() for item in self.list_widget.selectedItems()]

class ParameterEditorWidget(QWidget):
    """Універсальний редактор параметрів, який автоматично вибирає потрібний тип редактора"""
    parameterChanged = pyqtSignal(dict)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout(self)

        self.stack = QStackedWidget()

        # Створюємо обидва типи редакторів
        self.simple_editor = SimpleParameterEditor()
        self.complex_editor = ComplexParameterEditor()

        # Додаємо їх у стек
        self.stack.addWidget(self.simple_editor)
        self.stack.addWidget(self.complex_editor)

        layout.addWidget(self.stack)

        # Підключаємо сигнали
        self.simple_editor.parameterChanged.connect(self.parameterChanged.emit)
        self.complex_editor.parameterChanged.connect(self.parameterChanged.emit)

    def populate_table(self, params: Union[dict, list]):
        """Визначає тип параметрів і вибирає відповідний редактор"""
        if isinstance(params, list):
            # Якщо це список - обробляємо як простий словник з індексами
            params_dict = {str(i): v for i, v in enumerate(params)}
            self.simple_editor.populate(params_dict)
            self.stack.setCurrentWidget(self.simple_editor)
        elif any(k in params for k in ["model_params", "fit_params", "task_spec_params"]):
            # Якщо є вкладені словники - використовуємо складний редактор
            self.complex_editor.populate(params)
            self.stack.setCurrentWidget(self.complex_editor)
            self.simple_editor.hide()
        else:
            # Інакше - простий редактор
            self.simple_editor.populate(params)
            self.stack.setCurrentWidget(self.simple_editor)
            self.complex_editor.hide()

    def get_current_parameters(self) -> Union[dict, list]:
        """Повертає параметри у вихідній структурі"""
        current_editor = self.stack.currentWidget()
        params = current_editor.get_parameters()
        return params