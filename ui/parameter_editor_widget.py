"""from typing import Dict

from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtWidgets import QLineEdit, QSpinBox, QDoubleSpinBox, QCheckBox, QTableWidgetItem, QHBoxLayout, QPushButton, \
    QTableWidget, QVBoxLayout, QWidget


class ParameterEditorWidget(QWidget):

    def __init__(self, parent=None):
        super().__init__(parent)
        self.init_ui()

    def init_ui(self):
        main_layout = QVBoxLayout(self)

        # Створюємо таблицю
        self.table = QTableWidget()
        self.table.setColumnCount(2)
        self.table.setHorizontalHeaderLabels(['Параметр', 'Значення'])
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.verticalHeader().setVisible(False)

        main_layout.addWidget(self.table)

        # Кнопки для збереження
        btn_layout = QHBoxLayout()

        main_layout.addLayout(btn_layout)

        self.setLayout(main_layout)

    def populate_table(self, params_dict):
        # Очищаємо таблицю
        self.table.setRowCount(0)

        # Встановлюємо кількість рядків, що відповідає кількості параметрів
        self.table.setRowCount(len(params_dict))

        # Заповнюємо таблицю даними
        for row, (key, value) in enumerate(params_dict.items()):
            # Ключ (назва параметра)
            param_item = QTableWidgetItem(key)
            param_item.setFlags(param_item.flags() & ~Qt.ItemIsEditable)  # Робимо неможливим редагування ключа
            self.table.setItem(row, 0, param_item)

            # Підбираємо відповідний віджет для значення
            if isinstance(value, bool):
                widget = QCheckBox()
                widget.setChecked(value)
                self.table.setCellWidget(row, 1, widget)
            elif isinstance(value, int):
                widget = QSpinBox()
                widget.setRange(-999999, 999999)
                widget.setValue(value)
                self.table.setCellWidget(row, 1, widget)
            elif isinstance(value, float):
                widget = QDoubleSpinBox()
                widget.setRange(-999999.0, 999999.0)
                widget.setDecimals(6)
                widget.setValue(value)
                self.table.setCellWidget(row, 1, widget)
            elif value is None:
                # Для None використовуємо текстове поле з "None"
                widget = QLineEdit("None")
                self.table.setCellWidget(row, 1, widget)
            else:
                # Для всіх інших типів (включаючи рядки) використовуємо текстовий редактор
                widget = QLineEdit(str(value))
                self.table.setCellWidget(row, 1, widget)

        # Підганяємо розмір таблиці під контент
        self.table.resizeColumnsToContents()

    def get_current_parameters(self):
        current_params = {}

        for row in range(self.table.rowCount()):
            key = self.table.item(row, 0).text()
            widget = self.table.cellWidget(row, 1)

            # Отримуємо значення в залежності від типу віджета
            if isinstance(widget, QCheckBox):
                value = widget.isChecked()
            elif isinstance(widget, QSpinBox):
                value = widget.value()
            elif isinstance(widget, QDoubleSpinBox):
                value = widget.value()
            elif isinstance(widget, QLineEdit):
                text = widget.text()
                # Спроба перетворення в відповідний тип
                if text.lower() == "none":
                    value = None
                elif text.lower() == "true":
                    value = True
                elif text.lower() == "false":
                    value = False
                else:
                    # Спроба перетворення в числовий тип
                    try:
                        if '.' in text:
                            value = float(text)
                        else:
                            value = int(text)
                    except ValueError:
                        value = text
            else:
                value = None

            current_params[key] = value

        return current_params
"""
from typing import Dict, List, Any
from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtWidgets import (QLineEdit, QSpinBox, QDoubleSpinBox, QCheckBox, QTableWidgetItem, QHBoxLayout, QPushButton,
                             QTableWidget, QVBoxLayout, QWidget, QHeaderView, QTreeWidget, QTreeWidgetItem, QComboBox,
                             QDialog, QLabel, QGroupBox, QFormLayout, QDialogButtonBox, QTabWidget)
import json


class DictEditorDialog(QDialog):
    def __init__(self, dict_data=None, parent=None):
        super().__init__(parent)
        self.dict_data = dict_data or {}
        self.init_ui()

    def init_ui(self):
        self.setWindowTitle("Редагування словника")
        self.resize(500, 400)

        layout = QVBoxLayout(self)

        # Таблиця для відображення ключів і значень
        self.table = QTableWidget()
        self.table.setColumnCount(3)
        self.table.setHorizontalHeaderLabels(['Ключ', 'Значення', 'Дії'])
        self.table.horizontalHeader().setSectionResizeMode(1, QHeaderView.Stretch)

        layout.addWidget(self.table)

        # Кнопки для додавання елементів
        btn_layout = QHBoxLayout()
        self.add_button = QPushButton("Додати елемент")
        self.add_button.clicked.connect(self.add_item)
        btn_layout.addWidget(self.add_button)

        layout.addLayout(btn_layout)

        # Кнопки Ok / Cancel
        button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)
        layout.addWidget(button_box)

        self.populate_table()

    def populate_table(self):
        self.table.setRowCount(0)
        self.table.setRowCount(len(self.dict_data))

        for row, (key, value) in enumerate(self.dict_data.items()):
            # Ключ (назва параметра)
            key_item = QTableWidgetItem(str(key))
            self.table.setItem(row, 0, key_item)

            # Значення
            value_item = QTableWidgetItem(str(value))
            self.table.setItem(row, 1, value_item)

            # Кнопка для видалення
            delete_btn = QPushButton("Видалити")
            delete_btn.clicked.connect(lambda checked, r=row: self.delete_item(r))
            self.table.setCellWidget(row, 2, delete_btn)

    def add_item(self):
        row = self.table.rowCount()
        self.table.setRowCount(row + 1)

        key_item = QTableWidgetItem("новий_ключ")
        self.table.setItem(row, 0, key_item)

        value_item = QTableWidgetItem("")
        self.table.setItem(row, 1, value_item)

        delete_btn = QPushButton("Видалити")
        delete_btn.clicked.connect(lambda checked, r=row: self.delete_item(r))
        self.table.setCellWidget(row, 2, delete_btn)

    def delete_item(self, row):
        self.table.removeRow(row)

    def get_dict_data(self):
        result = {}
        for row in range(self.table.rowCount()):
            key = self.table.item(row, 0).text()
            value_text = self.table.item(row, 1).text()

            # Спроба конвертувати значення в потрібний тип
            value = self._convert_value(value_text)
            result[key] = value

        return result

    def _convert_value(self, value_text):
        # Конвертація значення у відповідний тип
        if value_text.lower() == "none":
            return None
        elif value_text.lower() == "true":
            return True
        elif value_text.lower() == "false":
            return False
        else:
            try:
                if '.' in value_text:
                    return float(value_text)
                else:
                    return int(value_text)
            except ValueError:
                return value_text

    def accept(self):
        self.dict_data = self.get_dict_data()
        super().accept()


class ListEditorDialog(QDialog):
    def __init__(self, list_data=None, parent=None):
        super().__init__(parent)
        self.list_data = list_data or []
        self.init_ui()

    def init_ui(self):
        self.setWindowTitle("Редагування списку")
        self.resize(500, 400)

        layout = QVBoxLayout(self)

        # Таблиця для відображення елементів списку
        self.table = QTableWidget()
        self.table.setColumnCount(2)
        self.table.setHorizontalHeaderLabels(['Значення', 'Дії'])
        self.table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)

        layout.addWidget(self.table)

        # Кнопки для додавання елементів
        btn_layout = QHBoxLayout()
        self.add_button = QPushButton("Додати елемент")
        self.add_button.clicked.connect(self.add_item)
        btn_layout.addWidget(self.add_button)

        layout.addLayout(btn_layout)

        # Кнопки Ok / Cancel
        button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)
        layout.addWidget(button_box)

        self.populate_table()

    def populate_table(self):
        self.table.setRowCount(0)
        self.table.setRowCount(len(self.list_data))

        for row, value in enumerate(self.list_data):
            # Значення
            value_item = QTableWidgetItem(str(value))
            self.table.setItem(row, 0, value_item)

            # Кнопка для видалення
            delete_btn = QPushButton("Видалити")
            delete_btn.clicked.connect(lambda checked, r=row: self.delete_item(r))
            self.table.setCellWidget(row, 1, delete_btn)

    def add_item(self):
        row = self.table.rowCount()
        self.table.setRowCount(row + 1)

        value_item = QTableWidgetItem("")
        self.table.setItem(row, 0, value_item)

        delete_btn = QPushButton("Видалити")
        delete_btn.clicked.connect(lambda checked, r=row: self.delete_item(r))
        self.table.setCellWidget(row, 1, delete_btn)

    def delete_item(self, row):
        self.table.removeRow(row)

    def get_list_data(self):
        result = []
        for row in range(self.table.rowCount()):
            value_text = self.table.item(row, 0).text()

            # Спроба конвертувати значення в потрібний тип
            value = self._convert_value(value_text)
            result.append(value)

        return result

    def _convert_value(self, value_text):
        # Конвертація значення у відповідний тип
        if value_text.lower() == "none":
            return None
        elif value_text.lower() == "true":
            return True
        elif value_text.lower() == "false":
            return False
        else:
            try:
                if '.' in value_text:
                    return float(value_text)
                else:
                    return int(value_text)
            except ValueError:
                return value_text

    def accept(self):
        self.list_data = self.get_list_data()
        super().accept()


class ComplexValueEditor(QWidget):
    valueChanged = pyqtSignal(object)

    def __init__(self, value=None, parent=None):
        super().__init__(parent)
        self.value = value
        self.init_ui()

    def init_ui(self):
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        # Відображення типу та скороченого значення
        self.value_display = QLineEdit()
        self.value_display.setReadOnly(True)
        self.update_display()

        self.edit_button = QPushButton("Редагувати")
        self.edit_button.clicked.connect(self.edit_value)

        layout.addWidget(self.value_display)
        layout.addWidget(self.edit_button)

    def update_display(self):
        if isinstance(self.value, dict):
            self.value_display.setText(f"Словник: {len(self.value)} елементів")
        elif isinstance(self.value, list):
            self.value_display.setText(f"Список: {len(self.value)} елементів")
        else:
            self.value_display.setText(str(self.value))

    def edit_value(self):
        if isinstance(self.value, dict):
            dialog = DictEditorDialog(self.value, self)
            if dialog.exec_() == QDialog.Accepted:
                self.value = dialog.dict_data
                self.update_display()
                self.valueChanged.emit(self.value)
        elif isinstance(self.value, list):
            dialog = ListEditorDialog(self.value, self)
            if dialog.exec_() == QDialog.Accepted:
                self.value = dialog.list_data
                self.update_display()
                self.valueChanged.emit(self.value)

    def get_value(self):
        return self.value


class NestedParameterEditorWidget(QWidget):
    parameterChanged = pyqtSignal(dict)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.params_dict = {}
        self.init_ui()

    def init_ui(self):
        main_layout = QVBoxLayout(self)

        # Створюємо табвіджет для різних категорій параметрів
        self.tab_widget = QTabWidget()

        # Створюємо дерево для відображення вкладеної структури
        self.tree = QTreeWidget()
        self.tree.setHeaderLabels(['Параметр', 'Значення'])
        self.tree.header().setSectionResizeMode(0, QHeaderView.ResizeToContents)
        self.tree.header().setSectionResizeMode(1, QHeaderView.Stretch)

        main_layout.addWidget(self.tree)

        # Кнопки для збереження
        btn_layout = QHBoxLayout()
        self.save_button = QPushButton("Зберегти зміни")
        self.save_button.clicked.connect(self.save_parameters)
        btn_layout.addWidget(self.save_button)

        main_layout.addLayout(btn_layout)

        self.setLayout(main_layout)

    def populate_tree(self, params_dict, parent_item=None):
        for key, value in params_dict.items():
            if parent_item is None:
                item = QTreeWidgetItem(self.tree)
            else:
                item = QTreeWidgetItem(parent_item)

            item.setText(0, str(key))

            # Якщо значення - вкладений словник, рекурсивно обробляємо його
            if isinstance(value, dict):
                item.setExpanded(True)
                self.populate_tree(value, item)
            # Якщо значення - список або словник, створюємо спеціальний віджет для редагування
            elif isinstance(value, (list, dict)):
                complex_editor = ComplexValueEditor(value)
                complex_editor.valueChanged.connect(lambda v, i=item, k=key: self.update_complex_value(i, k, v))
                self.tree.setItemWidget(item, 1, complex_editor)
            # Інакше, створюємо віджет в залежності від типу значення
            else:
                self.create_value_widget(item, value)

    def create_value_widget(self, item, value):
        if isinstance(value, bool):
            widget = QCheckBox()
            widget.setChecked(value)
            widget.stateChanged.connect(lambda state, i=item: self.update_boolean_value(i, state))
            self.tree.setItemWidget(item, 1, widget)
        elif isinstance(value, int):
            widget = QSpinBox()
            widget.setRange(-999999, 999999)
            widget.setValue(value)
            widget.valueChanged.connect(lambda val, i=item: self.update_numeric_value(i, val))
            self.tree.setItemWidget(item, 1, widget)
        elif isinstance(value, float):
            widget = QDoubleSpinBox()
            widget.setRange(-999999.0, 999999.0)
            widget.setDecimals(6)
            widget.setValue(value)
            widget.valueChanged.connect(lambda val, i=item: self.update_numeric_value(i, val))
            self.tree.setItemWidget(item, 1, widget)
        elif value is None:
            widget = QComboBox()
            widget.addItems(["None", "True", "False", "0", "0.0", "''"])
            widget.setCurrentText("None")
            widget.currentTextChanged.connect(lambda text, i=item: self.update_text_value(i, text))
            self.tree.setItemWidget(item, 1, widget)
        else:
            widget = QLineEdit(str(value))
            widget.textChanged.connect(lambda text, i=item: self.update_text_value(i, text))
            self.tree.setItemWidget(item, 1, widget)

    def update_boolean_value(self, item, state):
        value = bool(state == Qt.Checked)
        self.update_item_value(item, value)

    def update_numeric_value(self, item, value):
        self.update_item_value(item, value)

    def update_text_value(self, item, text):
        # Обробка спеціальних значень
        if text == "None":
            value = None
        elif text == "True":
            value = True
        elif text == "False":
            value = False
        else:
            # Спроба перетворення на числовий тип
            try:
                if '.' in text:
                    value = float(text)
                elif text.isdigit() or (text.startswith('-') and text[1:].isdigit()):
                    value = int(text)
                else:
                    value = text
            except ValueError:
                value = text

        self.update_item_value(item, value)

    def update_complex_value(self, item, key, value):
        self.update_item_value(item, value)

    def update_item_value(self, item, value):
        # Знаходимо повний шлях до елемента
        path = []
        current = item
        while current:
            path.insert(0, current.text(0))
            current = current.parent()

        # Оновлюємо значення в params_dict
        current_dict = self.params_dict
        for i, key in enumerate(path):
            if i == len(path) - 1:
                current_dict[key] = value
            else:
                if key not in current_dict:
                    current_dict[key] = {}
                current_dict = current_dict[key]

    def populate_with_tensorflow_params(self, params_dict):
        self.params_dict = params_dict
        self.tree.clear()

        # Створюємо кореневі елементи для кожної категорії параметрів
        if 'model_params' in params_dict:
            model_root = QTreeWidgetItem(self.tree)
            model_root.setText(0, 'model_params')
            model_root.setExpanded(True)
            self.populate_tree(params_dict['model_params'], model_root)

        if 'fit_params' in params_dict:
            fit_root = QTreeWidgetItem(self.tree)
            fit_root.setText(0, 'fit_params')
            fit_root.setExpanded(True)
            self.populate_tree(params_dict['fit_params'], fit_root)

        if 'task_spec_params' in params_dict:
            task_root = QTreeWidgetItem(self.tree)
            task_root.setText(0, 'task_spec_params')
            task_root.setExpanded(True)
            self.populate_tree(params_dict['task_spec_params'], task_root)

        # Якщо є інші параметри, які не входять в ці категорії
        for key, value in params_dict.items():
            if key not in ['model_params', 'fit_params', 'task_spec_params']:
                if isinstance(value, dict):
                    root = QTreeWidgetItem(self.tree)
                    root.setText(0, key)
                    root.setExpanded(True)
                    self.populate_tree(value, root)
                else:
                    item = QTreeWidgetItem(self.tree)
                    item.setText(0, key)
                    self.create_value_widget(item, value)

    def populate_table(self, params_dict):
        # Зворотна сумісність з оригінальним методом
        self.params_dict = params_dict
        self.tree.clear()
        self.populate_tree(params_dict)

    def save_parameters(self):
        self.parameterChanged.emit(self.params_dict)

    def get_current_parameters(self):
        return self.params_dict


class ParameterEditorWidget(QWidget):
    parameterChanged = pyqtSignal(dict)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.init_ui()

    def init_ui(self):
        main_layout = QVBoxLayout(self)

        # Створюємо таблицю
        self.table = QTableWidget()
        self.table.setColumnCount(2)
        self.table.setHorizontalHeaderLabels(['Параметр', 'Значення'])
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.verticalHeader().setVisible(False)

        main_layout.addWidget(self.table)

        # Кнопки для збереження
        btn_layout = QHBoxLayout()
        self.save_button = QPushButton("Зберегти зміни")
        self.save_button.clicked.connect(self.save_parameters)
        btn_layout.addWidget(self.save_button)

        main_layout.addLayout(btn_layout)

        self.setLayout(main_layout)

    def populate_table(self, params_dict):
        # Перевіряємо, чи є це структурований словник для TensorFlow
        if any(key in params_dict for key in ['model_params', 'fit_params', 'task_spec_params']):
            # Створюємо вкладений редактор, якщо його ще не створено
            if not hasattr(self, 'nested_editor'):
                self.nested_editor = NestedParameterEditorWidget(self)
                self.nested_editor.parameterChanged.connect(self.on_nested_parameters_changed)
                self.layout().replaceWidget(self.table, self.nested_editor)
                self.table.hide()

            self.nested_editor.populate_with_tensorflow_params(params_dict)
            return

        # Очищаємо таблицю
        self.table.setRowCount(0)

        # Встановлюємо кількість рядків, що відповідає кількості параметрів
        self.table.setRowCount(len(params_dict))

        # Заповнюємо таблицю даними
        for row, (key, value) in enumerate(params_dict.items()):
            # Ключ (назва параметра)
            param_item = QTableWidgetItem(key)
            param_item.setFlags(param_item.flags() & ~Qt.ItemIsEditable)  # Робимо неможливим редагування ключа
            self.table.setItem(row, 0, param_item)

            # Підбираємо відповідний віджет для значення
            if isinstance(value, bool):
                widget = QCheckBox()
                widget.setChecked(value)
                self.table.setCellWidget(row, 1, widget)
            elif isinstance(value, int):
                widget = QSpinBox()
                widget.setRange(-999999, 999999)
                widget.setValue(value)
                self.table.setCellWidget(row, 1, widget)
            elif isinstance(value, float):
                widget = QDoubleSpinBox()
                widget.setRange(-999999.0, 999999.0)
                widget.setDecimals(6)
                widget.setValue(value)
                self.table.setCellWidget(row, 1, widget)
            elif value is None:
                # Для None використовуємо текстове поле з "None"
                widget = QLineEdit("None")
                self.table.setCellWidget(row, 1, widget)
            elif isinstance(value, (list, dict)):
                # Для списків і словників використовуємо ComplexValueEditor
                widget = ComplexValueEditor(value)
                self.table.setCellWidget(row, 1, widget)
            else:
                # Для всіх інших типів (включаючи рядки) використовуємо текстовий редактор
                widget = QLineEdit(str(value))
                self.table.setCellWidget(row, 1, widget)

        # Підганяємо розмір таблиці під контент
        self.table.resizeColumnsToContents()

    def get_current_parameters(self):
        # Якщо маємо вкладений редактор, використовуємо його
        if hasattr(self, 'nested_editor') and self.nested_editor.isVisible():
            return self.nested_editor.get_current_parameters()

        current_params = {}

        for row in range(self.table.rowCount()):
            key = self.table.item(row, 0).text()
            widget = self.table.cellWidget(row, 1)

            # Отримуємо значення в залежності від типу віджета
            if isinstance(widget, QCheckBox):
                value = widget.isChecked()
            elif isinstance(widget, QSpinBox):
                value = widget.value()
            elif isinstance(widget, QDoubleSpinBox):
                value = widget.value()
            elif isinstance(widget, ComplexValueEditor):
                value = widget.get_value()
            elif isinstance(widget, QLineEdit):
                text = widget.text()
                # Спроба перетворення в відповідний тип
                if text.lower() == "none":
                    value = None
                elif text.lower() == "true":
                    value = True
                elif text.lower() == "false":
                    value = False
                else:
                    # Спроба перетворення в числовий тип
                    try:
                        if '.' in text:
                            value = float(text)
                        else:
                            value = int(text)
                    except ValueError:
                        value = text
            else:
                value = None

            current_params[key] = value

        return current_params

    def save_parameters(self):
        params = self.get_current_parameters()
        self.parameterChanged.emit(params)

    def on_nested_parameters_changed(self, params):
        self.parameterChanged.emit(params)