import os
from abc import abstractmethod, ABC

from PyQt5.QtWidgets import QWidget, QMessageBox

from project.logic.experiment.experiment import Experiment
from project.ui.experiment_settings_dialog.tab_views import HyperparamsTabWidget, GeneralTabWidget


class TabController:
    def __init__(self, experiment: Experiment, view: QWidget):
        self.experiment = experiment
        self.view = view

    @abstractmethod
    def connect_signals(self):
        raise NotImplemented

    @abstractmethod
    def update_model_from_view(self):
        raise NotImplemented

    @abstractmethod
    def init_view(self):
        raise NotImplemented


class InputDataTabController(TabController):
    """Контролер для вкладки параметрів машинного навчання"""

    def __init__(self, experiment: Experiment, view):
        super().__init__(experiment, view)
        self.model = self.experiment.input_data_params
        self.model.current_task = self.experiment.task
        self.connect_signals()
        self.init_view()

    def connect_signals(self):
        # Режим роботи з даними
        self.view.single_file_radio.toggled.connect(self.on_mode_changed)
        self.view.two_files_radio.toggled.connect(self.on_mode_changed)

        # Вибір файлів
        self.view.single_file_btn.clicked.connect(self.browse_single_file)
        self.view.train_file_btn.clicked.connect(self.browse_train_file)
        self.view.test_file_btn.clicked.connect(self.browse_test_file)

        # Відстеження зміни шляху в полі вводу
        self.view.single_file_path.textChanged.connect(lambda text: self.on_file_path_changed(text, True))
        self.view.train_file_path.textChanged.connect(lambda text: self.on_file_path_changed(text, False))

        # Відсоток розбиття
        self.view.train_percent.valueChanged.connect(self.on_train_percent_changed)

        # Цільова змінна
        self.view.target_combo.currentTextChanged.connect(self.on_target_changed)

        # Кодування
        self.view.single_encoding_combo.currentTextChanged.connect(self.on_single_encoding_changed)
        self.view.train_encoding_combo.currentTextChanged.connect(self.on_train_encoding_changed)
        self.view.test_encoding_combo.currentTextChanged.connect(self.on_test_encoding_changed)

        # Роздільники
        self.view.single_separator_combo.currentTextChanged.connect(self.on_single_separator_changed)
        self.view.train_separator_combo.currentTextChanged.connect(self.on_train_separator_changed)

    def init_view(self):
        # Встановлення режиму одного файлу за замовчуванням
        self.view.single_file_radio.setChecked(True)
        self.view.update_ui_state(True)

        # Оновлюємо відображення полів цільової змінної відповідно до поточного завдання
        should_show_target = not self.model.is_target_not_required()
        self.view.update_target_field_visibility(should_show_target)

    def on_mode_changed(self):
        single_file_mode = self.view.single_file_radio.isChecked()
        self.model.mode = 'single_file' if single_file_mode else 'two_files'
        self.view.update_ui_state(single_file_mode)

    def on_train_percent_changed(self, value):
        self.model.train_percent = value
        self.model.test_percent = 100 - value
        self.view.update_test_percent(value)

    def on_target_changed(self, value):
        self.model.target_variable = value

    def on_single_encoding_changed(self, value):
        self.model.single_file_encoding = value
        # Перезавантажити дані з новим кодуванням, якщо файл вже вибрано
        if self.model.single_file_path:
            self.load_column_names(self.model.single_file_path)

    def on_train_encoding_changed(self, value):
        self.model.train_file_encoding = value

    def on_test_encoding_changed(self, value):
        self.model.test_file_encoding = value

    def on_single_separator_changed(self, value):
        self.model.single_file_separator = value
        # Перезавантажити дані з новим роздільником, якщо файл вже вибрано
        if self.model.single_file_path and os.path.splitext(self.model.single_file_path)[1].lower() == '.csv':
            self.load_column_names(self.model.single_file_path)

    def on_train_separator_changed(self, value):
        self.model.train_file_separator = value

    def on_file_path_changed(self, path, is_single_file):
        """Обробник зміни шляху до файлу (для відображення/приховування полів роздільника)"""
        self.view.show_separator_fields(path, is_single_file)

    def browse_single_file(self):
        file, _ = self.view.get_file_dialog(
            "Вибрати файл даних",
            "Усі файли (*.csv *.xlsx *.xls *.json *.parquet);;CSV файли (*.csv);;Excel файли (*.xlsx *.xls);;JSON файли (*.json);;Parquet файли (*.parquet)"
        )
        if file:
            self.model.single_file_path = file
            self.view.single_file_path.setText(file)
            self.load_column_names(file)

    def browse_train_file(self):
        file, _ = self.view.get_file_dialog(
            "Вибрати файл для тренування",
            "Усі файли (*.csv *.xlsx *.xls *.json *.parquet);;CSV файли (*.csv);;Excel файли (*.xlsx *.xls);;JSON файли (*.json);;Parquet файли (*.parquet)"
        )
        if file:
            self.model.train_file_path = file
            self.view.train_file_path.setText(file)

    def browse_test_file(self):
        file, _ = self.view.get_file_dialog(
            "Вибрати файл для тестування",
            "Усі файли (*.csv *.xlsx *.xls *.json *.parquet);;CSV файли (*.csv);;Excel файли (*.xlsx *.xls);;JSON файли (*.json);;Parquet файли (*.parquet)"
        )
        if file:
            self.model.test_file_path = file
            self.view.test_file_path.setText(file)

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
                        encoding=self.model.single_file_encoding,
                        sep=self.convert_separator(self.model.single_file_separator)
                    )
                    column_names = [str(col) for col in df.columns.tolist()]
                except Exception as e:
                    print(f"Помилка читання CSV: {e}")
                    self.try_different_encodings(file_path)

            elif ext in ['.xlsx', '.xls']:
                try:
                    import pandas as pd
                    # Попробуем сначала стандартный метод - первая строка как заголовки
                    df = pd.read_excel(file_path, nrows=1, header=0)
                    column_names = [str(col) for col in df.columns.tolist()]

                    # Если заголовки получились как числа (Column0, Column1...),
                    # возможно Excel файл не имеет заголовков, попробуем прочитать первую строку как данные
                    if all(col.startswith('Unnamed:') or col.startswith('Column') for col in column_names):
                        # Читаем без заголовков, используя первую строку как данные
                        df = pd.read_excel(file_path, nrows=1, header=None)
                        # Используем индексы столбцов как имена
                        column_names = [f"Column {i + 1}" for i in range(len(df.columns))]

                        # Альтернативно, можно использовать первую строку данных как имена столбцов
                        first_row_values = [str(val) for val in df.iloc[0].tolist()]
                        if any(val and not val.isspace() for val in first_row_values):
                            column_names = first_row_values
                except Exception as e:
                    print(f"Помилка читання Excel: {e}")

            elif ext == '.json':
                try:
                    import pandas as pd
                    df = pd.read_json(file_path, encoding=self.model.single_file_encoding)
                    column_names = [str(col) for col in df.columns.tolist()]
                except Exception as e:
                    print(f"Помилка читання JSON: {e}")
                    self.try_different_encodings(file_path)

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
                    self.model.target_variable = column_names[0]

        except Exception as e:
            print(f"Помилка при читанні заголовків файлу: {e}")
            # Можна додати сповіщення для користувача про помилку
            QMessageBox.warning(
                self.view,
                "Помилка читання файлу",
                f"Не вдалося прочитати заголовки файлу. Перевірте формат, кодування та роздільник.\nПомилка: {str(e)}"
            )

    def try_different_encodings(self, file_path):
        """Спроба автоматично визначити кодування файлу"""
        encodings = ['utf-8', 'cp1251', 'latin-1', 'iso-8859-1', 'ascii']
        separators = [',', ';', '\t', '|', ' ']

        ext = os.path.splitext(file_path)[1].lower()
        if ext != '.csv':
            return False

        import pandas as pd

        for encoding in encodings:
            for separator in separators:
                try:
                    df = pd.read_csv(file_path, nrows=1, encoding=encoding, sep=separator)
                    if len(df.columns) > 1:  # Перевіряємо, чи було успішно розпізнано кілька стовпців
                        # Оновлюємо значення в моделі та інтерфейсі
                        self.model.single_file_encoding = encoding
                        self.view.single_encoding_combo.setCurrentText(encoding)
                        self.model.single_file_separator = self.get_separator_display(separator)
                        self.view.single_separator_combo.setCurrentText(self.get_separator_display(separator))

                        # Оновлюємо випадаючий список цільових змінних
                        self.view.target_combo.clear()
                        self.view.target_combo.addItems(df.columns.tolist())
                        if df.columns.tolist():
                            self.model.target_variable = df.columns.tolist()[0]

                        QMessageBox.information(
                            self.view,
                            "Кодування визначено",
                            f"Автоматично визначено кодування: {encoding}, роздільник: {self.get_separator_display(separator)}"
                        )

                        return True
                except Exception:
                    continue

        return False

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
        self.model.mode = 'single_file' if self.view.single_file_radio.isChecked() else 'two_files'
        self.model.single_file_path = self.view.single_file_path.text()
        self.model.train_file_path = self.view.train_file_path.text()
        self.model.test_file_path = self.view.test_file_path.text()
        self.model.train_percent = self.view.train_percent.value()
        self.model.test_percent = 100 - self.view.train_percent.value()
        self.model.seed = self.view.seed_spinbox.value()

        # Кодування
        self.model.single_file_encoding = self.view.single_encoding_combo.currentText()
        self.model.train_file_encoding = self.view.train_encoding_combo.currentText()
        self.model.test_file_encoding = self.view.test_encoding_combo.currentText()

        # Роздільники
        self.model.single_file_separator = self.view.single_separator_combo.currentText()
        self.model.train_file_separator = self.view.train_separator_combo.currentText()

        # Цільова змінна (лише якщо поле видиме)
        if self.view.target_combo.isVisible():
            self.model.target_variable = self.view.target_combo.currentText()

    def get_input_params(self):
        print(self.experiment.input_data_params.to_dict())
        return self.model.to_dict()


class HyperparamsTabController(TabController, ABC):
    """Контролер для вкладки параметрів моделі"""

    def __init__(self, experiment: Experiment, view: HyperparamsTabWidget):
        super().__init__(experiment, view)
        self.connect_signals()
        self.init_view()

    def connect_signals(self):
        self.view.save_button.clicked.connect(self._update_params)

    def init_view(self):
        self.view.params_widget.populate_table(self.experiment.params)

    def _update_params(self):
        self.experiment.params = self.view.params_widget.get_current_parameters()
        print(self.view.params_widget.get_current_parameters())

    def update_model_from_view(self):
        self._update_params()


class MetricsTabController(TabController):
    """Контролер для вкладки параметрів оцінки"""

    def __init__(self, experiment: Experiment, view):
        super().__init__(experiment, view)
        self.connect_signals()
        # self.init_view()

    def connect_signals(self):
        """Підключення сигналів до слотів"""

    def init_view(self):
        pass

    def on_cv_toggled(self, checked):
        """Обробка перемикання крос-валідації"""
        return

    def on_cv_folds_changed(self, value):
        """Обробка зміни кількості фолдів"""
        return

    def update_metrics(self):
        """Оновлення списку метрик"""
        return

    def update_model_from_view(self):
        """Оновлення моделі даними з представлення"""
        return


class GeneralSettingsController:
    """Контролер для вкладки загальних налаштувань"""

    def __init__(self, experiment: Experiment, view: GeneralTabWidget) -> None:
        self.experiment = experiment
        self.view = view
        self.init_view()

    def init_view(self):
        """Ініціалізація початкового стану представлення"""
        self.view.experiment_name.setText(self.experiment.name)
        self.view.description.setText(self.experiment.description)
        self.view.method_name.setText(type(self.experiment.model).__name__)

    def update_model_from_view(self):
        """Оновлення моделі даними з представлення"""
        self.experiment.name = self.view.experiment_name.text()
        self.experiment.description = self.view.description.toPlainText()

    def set_experiment_name(self, name: str):
        self.view.experiment_name.setText(name)
