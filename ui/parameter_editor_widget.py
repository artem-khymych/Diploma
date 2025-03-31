from typing import Dict

from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtWidgets import QLineEdit, QSpinBox, QDoubleSpinBox, QCheckBox, QTableWidgetItem, QHBoxLayout, QPushButton, \
    QTableWidget, QVBoxLayout, QWidget


class ParameterEditorWidget(QWidget):
    """Віджет для редагування параметрів"""

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
        """Заповнює таблицю параметрами"""
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
        """Отримує поточні параметри з таблиці без збереження"""
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
