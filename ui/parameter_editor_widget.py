from typing import Dict, List, Any, Optional
from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtWidgets import (QLineEdit, QSpinBox, QDoubleSpinBox, QCheckBox,
                             QTableWidgetItem, QHBoxLayout, QPushButton, QTableWidget,
                             QVBoxLayout, QWidget, QComboBox, QTabWidget, QFrame,
                             QHeaderView, QLabel, QScrollArea, QGridLayout, QDialog)
from PyQt5.QtGui import QColor


class NestedDictEditor(QDialog):
    """Діалог для редагування вкладених словників та списків"""

    def __init__(self, value, parent=None, title="Редагування"):
        super().__init__(parent)
        self.value = value
        self.setWindowTitle(title)
        self.setMinimumSize(600, 400)
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout(self)

        # Створюємо редактор параметрів для вкладеної структури
        self.editor = ParameterEditorWidget()

        if isinstance(self.value, dict):
            self.editor.populate_table(self.value)
        elif isinstance(self.value, list):
            # Перетворюємо список у словник з індексами як ключами
            dict_value = {str(i): v for i, v in enumerate(self.value)}
            self.editor.populate_table(dict_value, is_list=True)

        layout.addWidget(self.editor)

        # Кнопки
        btn_layout = QHBoxLayout()
        ok_btn = QPushButton("OK")
        cancel_btn = QPushButton("Скасувати")

        btn_layout.addWidget(ok_btn)
        btn_layout.addWidget(cancel_btn)

        layout.addLayout(btn_layout)

        # Підключаємо обробники подій
        ok_btn.clicked.connect(self.accept)
        cancel_btn.clicked.connect(self.reject)

    def get_value(self):
        params = self.editor.get_current_parameters()
        if hasattr(self.editor, 'is_list') and self.editor.is_list:
            # Якщо це список, перетворюємо словник назад у список
            return [params[str(i)] for i in range(len(params))]
        return params


class ListItemEditor(QWidget):
    """Віджет для редагування окремого елемента списку"""
    valueChanged = pyqtSignal()

    def __init__(self, value, parent=None):
        super().__init__(parent)
        self.value = value
        self.init_ui()

    def init_ui(self):
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        # Створюємо відповідний віджет для типу значення
        self.widget = self.create_widget_for_value(self.value)
        layout.addWidget(self.widget)

        # Якщо це складений тип, додаємо кнопку для розширеного редагування
        if isinstance(self.value, (dict, list)):
            edit_btn = QPushButton("...")
            edit_btn.setMaximumWidth(30)
            edit_btn.clicked.connect(self.edit_complex_value)
            layout.addWidget(edit_btn)

    def create_widget_for_value(self, value):
        if isinstance(value, bool):
            widget = QCheckBox()
            widget.setChecked(value)
            widget.stateChanged.connect(lambda: self.valueChanged.emit())
            return widget
        elif isinstance(value, int):
            widget = QSpinBox()
            widget.setRange(-999999, 999999)
            widget.setValue(value)
            widget.valueChanged.connect(lambda: self.valueChanged.emit())
            return widget
        elif isinstance(value, float):
            widget = QDoubleSpinBox()
            widget.setRange(-999999.0, 999999.0)
            widget.setDecimals(6)
            widget.setValue(value)
            widget.valueChanged.connect(lambda: self.valueChanged.emit())
            return widget
        elif isinstance(value, (dict, list)):
            widget = QLineEdit(self.get_complex_type_description(value))
            widget.setReadOnly(True)
            return widget
        elif value is None:
            widget = QLineEdit("None")
            widget.textChanged.connect(lambda: self.valueChanged.emit())
            return widget
        else:
            widget = QLineEdit(str(value))
            widget.textChanged.connect(lambda: self.valueChanged.emit())
            return widget

    def get_complex_type_description(self, value):
        if isinstance(value, dict):
            return f"Словник ({len(value)} елементів)"
        elif isinstance(value, list):
            return f"Список ({len(value)} елементів)"
        return str(value)

    def edit_complex_value(self):
        dialog = NestedDictEditor(self.value, self)
        if dialog.exec_():
            self.value = dialog.get_value()
            if isinstance(self.widget, QLineEdit):
                self.widget.setText(self.get_complex_type_description(self.value))
            self.valueChanged.emit()

    def get_value(self):
        if isinstance(self.widget, QCheckBox):
            return self.widget.isChecked()
        elif isinstance(self.widget, QSpinBox):
            return self.widget.value()
        elif isinstance(self.widget, QDoubleSpinBox):
            return self.widget.value()
        elif isinstance(self.widget, QLineEdit):
            if isinstance(self.value, (dict, list)):
                return self.value

            text = self.widget.text()
            if text.lower() == "none":
                return None
            elif text.lower() == "true":
                return True
            elif text.lower() == "false":
                return False
            else:
                try:
                    if '.' in text:
                        return float(text)
                    else:
                        return int(text)
                except ValueError:
                    return text
        return self.value


class ParameterEditorWidget(QWidget):
    """Віджет для редагування параметрів, включаючи вкладені структури"""
    parameterChanged = pyqtSignal(dict)

    # Налаштування відомих параметрів, для яких можна використати комбобокс
    KNOWN_OPTIONS = {
        "optimizer": ["adam", "sgd", "rmsprop", "adagrad", "adadelta", "adamax", "nadam"],
        "activation": ["relu", "sigmoid", "tanh", "softmax", "softplus", "softsign", "elu", "selu", "linear"],
        "loss": ["binary_crossentropy", "categorical_crossentropy", "mse", "mae", "mape", "cosine_similarity"],
        "metrics": ["accuracy", "precision", "recall", "auc", "mae", "mse"],
        "callbacks": ["EarlyStopping", "ModelCheckpoint", "ReduceLROnPlateau", "TensorBoard", "CSVLogger"]
    }

    def __init__(self, parent=None):
        super().__init__(parent)
        self.init_ui()
        self.is_list = False

    def init_ui(self):
        main_layout = QVBoxLayout(self)

        # Створюємо TabWidget для різних груп параметрів
        self.tab_widget = QTabWidget()

        # Основна таблиця параметрів
        self.main_table = self.create_table_widget()

        # Додаткові таблиці для вкладених параметрів
        self.model_params_table = self.create_table_widget()
        self.fit_params_table = self.create_table_widget()
        self.task_spec_params_table = self.create_table_widget()

        # Додаємо таблиці до TabWidget
        self.tab_widget.addTab(self.main_table, "Основні параметри")
        self.tab_widget.addTab(self.model_params_table, "Параметри моделі")
        self.tab_widget.addTab(self.fit_params_table, "Параметри навчання")
        self.tab_widget.addTab(self.task_spec_params_table, "Параметри задачі")

        main_layout.addWidget(self.tab_widget)
        self.setLayout(main_layout)

    def create_table_widget(self):
        table = QTableWidget()
        table.setColumnCount(2)
        table.setHorizontalHeaderLabels(['Параметр', 'Значення'])
        table.horizontalHeader().setStretchLastSection(True)
        table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeToContents)
        table.verticalHeader().setVisible(False)
        table.setAlternatingRowColors(True)
        return table

    def populate_table(self, params_dict:Dict):
        # Визначаємо, чи є це вхідний параметр списком
        if params_dict.get("model_params", False):

            model_params = params_dict.get('model_params', {})
            fit_params = params_dict.get('fit_params', {})
            task_spec_params = params_dict.get('task_spec_params', {})

            # Заповнюємо відповідні таблиці
            self._populate_single_table(self.model_params_table, model_params)
            self._populate_single_table(self.fit_params_table, fit_params)
            self._populate_single_table(self.task_spec_params_table, task_spec_params)

            # Приховуємо вкладки, якщо вони порожні
            self.tab_widget.setTabVisible(0, False)
            self.tab_widget.setTabVisible(1, bool(model_params))
            self.tab_widget.setTabVisible(2, bool(fit_params))
            self.tab_widget.setTabVisible(3, bool(task_spec_params))
        else:
            # Якщо немає спеціальних словників, заповнюємо лише основну таблицю
            self._populate_single_table(self.main_table, params_dict)
            self.tab_widget.setTabVisible(1, False)
            self.tab_widget.setTabVisible(2, False)
            self.tab_widget.setTabVisible(3, False)

    def _populate_single_table(self, table, params_dict):
        # Очищаємо таблицю
        table.setRowCount(0)

        if not params_dict:
            return

        # Встановлюємо кількість рядків
        table.setRowCount(len(params_dict))

        # Заповнюємо таблицю даними
        for row, (key, value) in enumerate(params_dict.items()):
            # Ключ (назва параметра)
            param_item = QTableWidgetItem(key)
            param_item.setFlags(param_item.flags() & ~Qt.ItemIsEditable)  # Робимо неможливим редагування ключа
            table.setItem(row, 0, param_item)

            # Значення параметра
            self._set_value_widget(table, row, key, value)

        # Підганяємо розмір таблиці
        table.resizeColumnsToContents()

    def _set_value_widget(self, table, row, key, value):
        # Визначаємо, чи є цей параметр одним з тих, для яких ми маємо фіксований список опцій
        if key in self.KNOWN_OPTIONS:
            if isinstance(value, list):
                # Для списків відомих опцій створюємо спеціальний віджет
                widget = self._create_multiselect_widget(key, value)
            else:
                # Для одиночного вибору створюємо комбобокс
                widget = QComboBox()
                widget.addItems(self.KNOWN_OPTIONS[key])
                current_index = widget.findText(str(value)) if value else 0
                if current_index >= 0:
                    widget.setCurrentIndex(current_index)
            table.setCellWidget(row, 1, widget)
        else:
            # Для інших типів створюємо відповідний віджет
            if isinstance(value, bool):
                widget = QCheckBox()
                widget.setChecked(value)
                table.setCellWidget(row, 1, widget)
            elif isinstance(value, int):
                widget = QSpinBox()
                widget.setRange(-999999, 999999)
                widget.setValue(value)
                table.setCellWidget(row, 1, widget)
            elif isinstance(value, float):
                widget = QDoubleSpinBox()
                widget.setRange(-999999.0, 999999.0)
                widget.setDecimals(6)
                widget.setValue(value)
                table.setCellWidget(row, 1, widget)
            elif isinstance(value, (list, dict)):
                # Для складених типів використовуємо спеціальний редактор
                widget = ListItemEditor(value)
                table.setCellWidget(row, 1, widget)
            elif value is None:
                widget = QLineEdit("None")
                table.setCellWidget(row, 1, widget)
            else:
                widget = QLineEdit(str(value))
                table.setCellWidget(row, 1, widget)

    def _create_multiselect_widget(self, key, selected_values):
        widget = QWidget()
        layout = QHBoxLayout(widget)
        layout.setContentsMargins(0, 0, 0, 0)

        # Створюємо текстове поле для відображення вибраних опцій
        line_edit = QLineEdit(", ".join(str(x) for x in selected_values))
        line_edit.setReadOnly(True)

        # Кнопка для відкриття діалогу вибору
        edit_btn = QPushButton("...")
        edit_btn.setMaximumWidth(30)

        layout.addWidget(line_edit)
        layout.addWidget(edit_btn)

        # Зберігаємо дані про вибрані значення
        widget.selected_values = selected_values
        widget.line_edit = line_edit

        # Функція для відкриття діалогу вибору
        def open_selection_dialog():
            dialog = QDialog(self)
            dialog.setWindowTitle(f"Вибрати {key}")
            dialog_layout = QVBoxLayout(dialog)

            checkboxes = []
            for option in self.KNOWN_OPTIONS[key]:
                checkbox = QCheckBox(option)
                checkbox.setChecked(option in widget.selected_values)
                checkboxes.append(checkbox)
                dialog_layout.addWidget(checkbox)

            btn_layout = QHBoxLayout()
            ok_btn = QPushButton("OK")
            cancel_btn = QPushButton("Скасувати")

            btn_layout.addWidget(ok_btn)
            btn_layout.addWidget(cancel_btn)

            dialog_layout.addLayout(btn_layout)

            ok_btn.clicked.connect(dialog.accept)
            cancel_btn.clicked.connect(dialog.reject)

            if dialog.exec_():
                widget.selected_values = [cb.text() for cb in checkboxes if cb.isChecked()]
                widget.line_edit.setText(", ".join(widget.selected_values))

        edit_btn.clicked.connect(open_selection_dialog)

        return widget

    def get_current_parameters(self):
        # Основний словник параметрів
        if self.is_list:
            # Якщо це список, повертаємо одразу значення з основної таблиці
            params = self._get_parameters_from_table(self.main_table)
            # Перетворюємо словник на список
            return [params[str(i)] for i in range(len(params))]
        else:
            # Якщо це словник, обробляємо всі вкладені структури
            main_params = self._get_parameters_from_table(self.main_table)

            # Додаємо вкладені словники, якщо вони не порожні
            model_params = self._get_parameters_from_table(self.model_params_table)
            if model_params:
                main_params['model_params'] = model_params

            fit_params = self._get_parameters_from_table(self.fit_params_table)
            if fit_params:
                main_params['fit_params'] = fit_params

            task_spec_params = self._get_parameters_from_table(self.task_spec_params_table)
            if task_spec_params:
                main_params['task_spec_params'] = task_spec_params

            return main_params

    def _get_parameters_from_table(self, table):
        params = {}

        for row in range(table.rowCount()):
            if not table.item(row, 0):
                continue

            key = table.item(row, 0).text()
            widget = table.cellWidget(row, 1)

            if not widget:
                continue

            # Отримуємо значення в залежності від типу віджета
            if isinstance(widget, QCheckBox):
                value = widget.isChecked()
            elif isinstance(widget, QSpinBox):
                value = widget.value()
            elif isinstance(widget, QDoubleSpinBox):
                value = widget.value()
            elif isinstance(widget, QComboBox):
                value = widget.currentText()
            elif isinstance(widget, ListItemEditor):
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
            elif hasattr(widget, 'selected_values'):
                # Спеціальний випадок для мультивибору
                value = widget.selected_values
            else:
                value = None

            params[key] = value

        return params