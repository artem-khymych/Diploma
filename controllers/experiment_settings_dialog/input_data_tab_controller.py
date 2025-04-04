import os

from PyQt5.QtWidgets import QMessageBox

from project.controllers.experiment_settings_dialog.tab_controller import TabController
from project.logic.experiment.experiment import Experiment


class InputDataTabController(TabController):
    """Контролер для вкладки параметрів вводу даних"""

    def __init__(self, experiment: Experiment, view):
        super().__init__(experiment, view)
        self.input_data_params = self.experiment.input_data_params
        self.input_data_params.current_task = self.experiment.task
        self.connect_signals()
        self.init_view()

    def connect_signals(self):
        # Режим роботи з даними
        self.view.single_file_radio.toggled.connect(self.on_mode_changed)
        self.view.multi_files_radio.toggled.connect(self.on_mode_changed)

        # Вибір файлів
        self.view.single_file_btn.clicked.connect(self.browse_single_file)
        self.view.x_train_file_btn.clicked.connect(lambda: self.browse_file('x_train'))
        self.view.y_train_file_btn.clicked.connect(lambda: self.browse_file('y_train'))
        self.view.x_test_file_btn.clicked.connect(lambda: self.browse_file('x_test'))
        self.view.y_test_file_btn.clicked.connect(lambda: self.browse_file('y_test'))

        # Відстеження зміни шляху в полі вводу
        self.view.single_file_path.textChanged.connect(lambda text: self.on_file_path_changed(text, 'single'))
        self.view.x_train_file_path.textChanged.connect(lambda text: self.on_file_path_changed(text, 'x_train'))
        self.view.y_train_file_path.textChanged.connect(lambda text: self.on_file_path_changed(text, 'y_train'))
        self.view.x_test_file_path.textChanged.connect(lambda text: self.on_file_path_changed(text, 'x_test'))
        self.view.y_test_file_path.textChanged.connect(lambda text: self.on_file_path_changed(text, 'y_test'))

        # Відсоток розбиття
        self.view.train_percent.valueChanged.connect(self.on_train_percent_changed)

        # Цільова змінна
        self.view.target_combo.currentTextChanged.connect(self.on_target_changed)

        # Чекбокси для ручного налаштування кодування
        self.view.single_manual_encoding_checkbox.toggled.connect(self.on_manual_encoding_toggled)
        self.view.multi_manual_encoding_checkbox.toggled.connect(self.on_manual_encoding_toggled)

        # Кодування для одиночного файла
        self.view.single_encoding_combo.currentTextChanged.connect(
            lambda text: self.on_encoding_changed('single', text))

        # Кодування та роздільники для мульти-файлового режиму (загальні)
        self.view.multi_encoding_combo.currentTextChanged.connect(
            lambda text: self.on_multi_encoding_changed(text))
        self.view.multi_separator_combo.currentTextChanged.connect(
            lambda text: self.on_multi_separator_changed(text))

        # Роздільник для одиночного файла
        self.view.single_separator_combo.currentTextChanged.connect(
            lambda text: self.on_separator_changed('single', text))

        # Seed значення
        self.view.seed_spinbox.valueChanged.connect(lambda value: setattr(self.input_data_params, 'seed', value))
        # зміна режиму обробки категоріальних
        self.view.one_hot_radio.clicked.connect(self.on_categorical_encoding_changed)
        self.view.to_categorical_radio.clicked.connect(self.on_categorical_encoding_changed)

    def init_view(self):
        # Встановлення режиму одного файлу за замовчуванням
        self.view.single_file_radio.setChecked(True)

        # Оновлюємо відображення полів цільової змінної відповідно до поточного завдання
        supervised_learning = not self.input_data_params.is_target_not_required()
        self.view.update_ui_state(single_file_mode=True, supervised_learning=supervised_learning)
        self.view.update_target_field_visibility(supervised_learning)

        # Встановлюємо початковий стан для полів кодування
        self.view.update_encoding_fields_visibility()
        self.view.load_input_data_params(self.input_data_params)

    def on_mode_changed(self):
        single_file_mode = self.view.single_file_radio.isChecked()
        self.input_data_params.mode = 'single_file' if single_file_mode else 'multi_files'

        supervised_learning = not self.input_data_params.is_target_not_required()
        self.view.update_ui_state(single_file_mode, supervised_learning)

    def on_train_percent_changed(self, value):
        self.input_data_params.train_percent = value
        self.input_data_params.test_percent = 100 - value
        self.view.update_test_percent(value)

    def on_target_changed(self, value):
        self.input_data_params.target_variable = value

    def on_manual_encoding_toggled(self):
        # Оновлюємо видимість полів кодування та роздільників
        self.view.update_encoding_fields_visibility()

    def on_encoding_changed(self, file_type, value):
        if file_type == 'single':
            self.input_data_params.file_encoding = value
            if self.input_data_params.single_file_path:
                self.load_column_names(self.input_data_params.single_file_path)

    def on_multi_encoding_changed(self, value):
        # Оновлюємо значення кодування для всіх файлів в режимі multi_files
        self.input_data_params.file_encoding = value

    def on_separator_changed(self, file_type, value):
        if file_type == 'single':
            self.input_data_params.file_separator = value
            if self.input_data_params.single_file_path and os.path.splitext(self.input_data_params.single_file_path)[
                1].lower() == '.csv':
                self.load_column_names(self.input_data_params.single_file_path)

    def on_categorical_encoding_changed(self):
        self.experiment.input_data_params.categorical_encoding = self.view.get_categorical_encoding_method()

    def on_multi_separator_changed(self, value):
        # Оновлюємо значення роздільника для всіх файлів в режимі multi_files
        self.input_data_params.file_separator = value

    def on_file_path_changed(self, path, file_type):
        """Обробник зміни шляху до файлу (для відображення/приховування полів роздільника)"""
        self.view.show_separator_fields(path, file_type)

        # Оновлюємо відповідне поле в моделі
        if file_type == 'single':
            self.input_data_params.single_file_path = path
            # Автоматично визначаємо кодування і роздільник, якщо не ввімкнено ручний режим
            if not self.view.single_manual_encoding_checkbox.isChecked() and path:
                self.auto_detect_format(path, 'single')
        elif file_type == 'x_train':
            self.input_data_params.x_train_file_path = path
            if not self.view.multi_manual_encoding_checkbox.isChecked() and path:
                self.auto_detect_format(path, 'multi')
        elif file_type == 'y_train':
            self.input_data_params.y_train_file_path = path
        elif file_type == 'x_test':
            self.input_data_params.x_test_file_path = path
        elif file_type == 'y_test':
            self.input_data_params.y_test_file_path = path

    def browse_single_file(self):
        file, _ = self.view.get_file_dialog(
            "Вибрати файл даних",
            "Усі файли (*.csv *.xlsx *.xls *.json *.parquet);;CSV файли (*.csv);;Excel файли (*.xlsx *.xls);;JSON файли (*.json);;Parquet файли (*.parquet)"
        )
        if file:
            self.input_data_params.single_file_path = file
            self.view.single_file_path.setText(file)

            # Автоматично визначаємо кодування і роздільник, якщо не ввімкнено ручний режим
            if not self.view.single_manual_encoding_checkbox.isChecked():
                success = self.auto_detect_format(file, 'single')
                if success:
                    self.load_column_names(file)
            else:
                self.load_column_names(file)

    def browse_file(self, file_type):
        title_map = {
            'x_train': "Вибрати файл для тренувальних даних (X_train)",
            'y_train': "Вибрати файл для тренувальних міток (y_train)",
            'x_test': "Вибрати файл для тестових даних (X_test)",
            'y_test': "Вибрати файл для тестових міток (y_test)"
        }
        file, _ = self.view.get_file_dialog(
            title_map.get(file_type, "Вибрати файл"),
            "Усі файли (*.csv *.xlsx *.xls *.json *.parquet);;CSV файли (*.csv);;Excel файли (*.xlsx *.xls);;JSON файли (*.json);;Parquet файли (*.parquet)"
        )
        if file:
            if file_type == 'x_train':
                self.input_data_params.x_train_file_path = file
                self.view.x_train_file_path.setText(file)
                # Автоматично визначаємо кодування і роздільник для всіх файлів
                if not self.view.multi_manual_encoding_checkbox.isChecked():
                    self.auto_detect_format(file, 'multi')
            elif file_type == 'y_train':
                self.input_data_params.y_train_file_path = file
                self.view.y_train_file_path.setText(file)
            elif file_type == 'x_test':
                self.input_data_params.x_test_file_path = file
                self.view.x_test_file_path.setText(file)
            elif file_type == 'y_test':
                self.input_data_params.y_test_file_path = file
                self.view.y_test_file_path.setText(file)

    def auto_detect_format(self, file_path, mode='single'):
        """Автоматично визначає кодування і роздільник CSV файлу"""
        ext = os.path.splitext(file_path)[1].lower()
        if ext != '.csv':
            return False

        encodings = ['utf-8', 'cp1251', 'latin-1', 'iso-8859-1', 'ascii']
        separators = [',', ';', '\t', '|', ' ']

        import pandas as pd

        for encoding in encodings:
            for separator in separators:
                try:
                    df = pd.read_csv(file_path, nrows=5, encoding=encoding, sep=separator)
                    if len(df.columns) > 1:  # Перевіряємо, чи було успішно розпізнано кілька стовпців
                        # Оновлюємо значення в моделі та інтерфейсі в залежності від режиму
                        if mode == 'single':
                            self.input_data_params.file_encoding = encoding
                            self.view.single_encoding_combo.setCurrentText(encoding)
                            self.input_data_params.file_separator = self.get_separator_display(separator)
                            self.view.single_separator_combo.setCurrentText(self.get_separator_display(separator))
                        else:  # multi
                            # Оновлюємо значення для всіх файлів в режимі multi_files
                            self.input_data_params.file_encoding = encoding
                            self.view.multi_encoding_combo.setCurrentText(encoding)

                            self.input_data_params.file_separator = self.get_separator_display(separator)
                            self.view.multi_separator_combo.setCurrentText(self.get_separator_display(separator))

                        # Якщо це одиночний файл, оновлюємо випадаючий список цільових змінних
                        if mode == 'single':
                            self.view.target_combo.clear()
                            self.view.target_combo.addItems([str(col) for col in df.columns.tolist()])
                            if df.columns.tolist():
                                self.input_data_params.target_variable = str(df.columns.tolist()[0])

                        return True
                except Exception:
                    continue

        # Якщо автоматичне визначення не вдалося, показуємо повідомлення
        QMessageBox.warning(
            self.view,
            "Не вдалося визначити формат",
            "Не вдалося автоматично визначити кодування та роздільник файлу. Будь ласка, виберіть їх вручну."
        )

        # Ввімкнути чекбокс ручного налаштування
        if mode == 'single':
            self.view.single_manual_encoding_checkbox.setChecked(True)
        else:
            self.view.multi_manual_encoding_checkbox.setChecked(True)

        return False

    def load_column_names(self, file_path):
        """Завантажує імена стовпців з файлу і встановлює їх у випадаючий список"""
        try:
            # Очистить поточний список
            self.view.target_combo.clear()

            # Визначити формат файлу і завантажити заголовки
            ext = os.path.splitext(file_path)[1].lower()
            column_names = []

            if ext == '.csv':
                try:
                    import pandas as pd
                    # Використовуємо вибране кодування та роздільник
                    df = pd.read_csv(
                        file_path,
                        nrows=1,
                        encoding=self.input_data_params.file_encoding,
                        sep=self.convert_separator(self.input_data_params.file_separator)
                    )
                    column_names = [str(col) for col in df.columns.tolist()]
                except Exception as e:
                    print(f"Помилка читання CSV: {e}")
                    # Не викликаємо try_different_encodings, бо тепер у нас є auto_detect_format

            elif ext in ['.xlsx', '.xls']:
                try:
                    import pandas as pd
                    df = pd.read_excel(file_path, nrows=1, header=0)
                    column_names = [str(col) for col in df.columns.tolist()]

                    if all(col.startswith('Unnamed:') or col.startswith('Column') for col in column_names):
                        df = pd.read_excel(file_path, nrows=1, header=None)
                        column_names = [f"Column {i + 1}" for i in range(len(df.columns))]
                        first_row_values = [str(val) for val in df.iloc[0].tolist()]
                        if any(val and not val.isspace() for val in first_row_values):
                            column_names = first_row_values
                except Exception as e:
                    print(f"Помилка читання Excel: {e}")

            elif ext == '.json':
                try:
                    import pandas as pd
                    df = pd.read_json(file_path, encoding=self.input_data_params.file_encoding)
                    column_names = [str(col) for col in df.columns.tolist()]
                except Exception as e:
                    print(f"Помилка читання JSON: {e}")
                    if not self.view.single_manual_encoding_checkbox.isChecked():
                        self.auto_detect_format(file_path, 'single')

            elif ext == '.parquet':
                try:
                    import pandas as pd
                    df = pd.read_parquet(file_path)
                    column_names = [str(col) for col in df.columns.tolist()]
                except Exception as e:
                    print(f"Помилка читання Parquet: {e}")

            # Додати імена стовпців у випадаючий список
            if column_names:
                self.view.target_combo.addItems(column_names)
                # Встановити перший стовпець як цільову змінну за замовчуванням
                if column_names:
                    self.input_data_params.target_variable = column_names[0]

        except Exception as e:
            print(f"Помилка при читанні заголовків файлу: {e}")
            QMessageBox.warning(
                self.view,
                "Помилка читання файлу",
                f"Не вдалося прочитати заголовки файлу. Перевірте формат, кодування та роздільник.\nПомилка: {str(e)}"
            )

    def convert_separator(self, separator):
        """Конвертує відображуваний роздільник у фактичний для pandas"""
        if separator == "\\t":
            return "\t"
        return separator

    def get_separator_display(self, separator):
        """Конвертує фактичний роздільник у відображуваний для інтерфейсу"""
        if separator == "\t":
            return "\\t"
        return separator

    def update_model_from_view(self):
        self.input_data_params.mode = 'single_file' if self.view.single_file_radio.isChecked() else 'multi_files'

        # Одинарний файл
        self.input_data_params.single_file_path = self.view.single_file_path.text()
        self.input_data_params.file_encoding = self.view.single_encoding_combo.currentText()
        self.input_data_params.file_separator = self.view.single_separator_combo.currentText()

        # Для режиму multiple files - використовуємо спільні налаштування
        if self.input_data_params.mode == 'multi_files':
            common_encoding = self.view.multi_encoding_combo.currentText()
            common_separator = self.view.multi_separator_combo.currentText()

            # Встановлюємо спільні налаштування для всіх файлів
            self.input_data_params.file_encoding = common_encoding
            self.input_data_params.file_separator = common_separator

        # Шляхи до файлів в режимі multiple
        self.input_data_params.x_train_file_path = self.view.x_train_file_path.text()
        self.input_data_params.y_train_file_path = self.view.y_train_file_path.text()
        self.input_data_params.x_test_file_path = self.view.x_test_file_path.text()
        self.input_data_params.y_test_file_path = self.view.y_test_file_path.text()

        # Параметри розбиття
        self.input_data_params.train_percent = self.view.train_percent.value()
        self.input_data_params.test_percent = 100 - self.view.train_percent.value()
        self.input_data_params.seed = self.view.seed_spinbox.value()

        # Цільова змінна (лише якщо поле видиме і навчання з учителем)
        if not self.input_data_params.is_target_not_required() and self.view.target_combo.isVisible():
            self.input_data_params.target_variable = self.view.target_combo.currentText()

        self.input_data_params.categorical_encoding = self.view.get_categorical_encoding_method()

    def get_input_params(self):
        self.update_model_from_view()
        return self.input_data_params.to_dict()
