import os

from PyQt5.QtWidgets import QFileDialog, QGroupBox, QRadioButton, QHBoxLayout, QSpinBox, QLabel, QVBoxLayout, QComboBox, \
    QPushButton, QLineEdit, QWidget


class InputDataTabWidget(QWidget):

    def __init__(self, parent=None):
        super().__init__(parent)
        self.init_ui()

    def init_ui(self):
        main_layout = QVBoxLayout()

        self.create_data_mode_group()
        main_layout.addWidget(self.data_mode_group)

        self.single_file_group = QGroupBox("Налаштування для одного файлу")
        single_file_layout = QVBoxLayout()

        self.single_file_layout = QHBoxLayout()
        self.single_file_label = QLabel("Шлях до файлу:")
        self.single_file_path = QLineEdit()
        self.single_file_btn = QPushButton("Обрати...")
        self.single_file_layout.addWidget(self.single_file_label)
        self.single_file_layout.addWidget(self.single_file_path)
        self.single_file_layout.addWidget(self.single_file_btn)
        single_file_layout.addLayout(self.single_file_layout)

        self.single_encoding_layout = QHBoxLayout()
        self.single_encoding_label = QLabel("Кодування:")
        self.single_encoding_combo = QComboBox()
        self.single_encoding_combo.addItems(["utf-8", "cp1251", "latin-1", "iso-8859-1", "ascii"])
        self.single_encoding_layout.addWidget(self.single_encoding_label)
        self.single_encoding_layout.addWidget(self.single_encoding_combo)
        single_file_layout.addLayout(self.single_encoding_layout)

        self.single_separator_layout = QHBoxLayout()
        self.single_separator_label = QLabel("Роздільник CSV:")
        self.single_separator_combo = QComboBox()
        self.single_separator_combo.addItems([",", ";", "\\t", "|", " "])
        self.single_separator_layout.addWidget(self.single_separator_label)
        self.single_separator_layout.addWidget(self.single_separator_combo)
        single_file_layout.addLayout(self.single_separator_layout)

        self.target_layout = QHBoxLayout()
        self.target_label = QLabel("Цільова змінна:")
        self.target_combo = QComboBox()
        self.target_layout.addWidget(self.target_label)
        self.target_layout.addWidget(self.target_combo)
        single_file_layout.addLayout(self.target_layout)

        self.single_file_group.setLayout(single_file_layout)
        main_layout.addWidget(self.single_file_group)

        self.multi_files_group = QGroupBox("Налаштування для декількох файлів")
        multi_files_layout = QVBoxLayout()

        # X_train секція
        self.x_train_group = QGroupBox("Тренувальні дані для навчання (X_train)")
        x_train_layout = QVBoxLayout()

        x_train_path_layout = QHBoxLayout()
        self.x_train_file_label = QLabel("Шлях до файлу:")
        self.x_train_file_path = QLineEdit()
        self.x_train_file_btn = QPushButton("Обрати...")
        x_train_path_layout.addWidget(self.x_train_file_label)
        x_train_path_layout.addWidget(self.x_train_file_path)
        x_train_path_layout.addWidget(self.x_train_file_btn)
        x_train_layout.addLayout(x_train_path_layout)

        x_train_encoding_layout = QHBoxLayout()
        self.x_train_encoding_label = QLabel("Кодування:")
        self.x_train_encoding_combo = QComboBox()
        self.x_train_encoding_combo.addItems(["utf-8", "cp1251", "latin-1", "iso-8859-1", "ascii"])
        x_train_encoding_layout.addWidget(self.x_train_encoding_label)
        x_train_encoding_layout.addWidget(self.x_train_encoding_combo)
        x_train_layout.addLayout(x_train_encoding_layout)

        x_train_separator_layout = QHBoxLayout()
        self.x_train_separator_label = QLabel("Роздільник CSV:")
        self.x_train_separator_combo = QComboBox()
        self.x_train_separator_combo.addItems([",", ";", "\\t", "|", " "])
        x_train_separator_layout.addWidget(self.x_train_separator_label)
        x_train_separator_layout.addWidget(self.x_train_separator_combo)
        x_train_layout.addLayout(x_train_separator_layout)

        self.x_train_group.setLayout(x_train_layout)
        multi_files_layout.addWidget(self.x_train_group)

        # Y_train секція (для задач з учителем)
        self.y_train_group = QGroupBox("Тренувальні дані для тестування (y_train)")
        y_train_layout = QVBoxLayout()

        y_train_path_layout = QHBoxLayout()
        self.y_train_file_label = QLabel("Шлях до файлу:")
        self.y_train_file_path = QLineEdit()
        self.y_train_file_btn = QPushButton("Обрати...")
        y_train_path_layout.addWidget(self.y_train_file_label)
        y_train_path_layout.addWidget(self.y_train_file_path)
        y_train_path_layout.addWidget(self.y_train_file_btn)
        y_train_layout.addLayout(y_train_path_layout)

        y_train_encoding_layout = QHBoxLayout()
        self.y_train_encoding_label = QLabel("Кодування:")
        self.y_train_encoding_combo = QComboBox()
        self.y_train_encoding_combo.addItems(["utf-8", "cp1251", "latin-1", "iso-8859-1", "ascii"])
        y_train_encoding_layout.addWidget(self.y_train_encoding_label)
        y_train_encoding_layout.addWidget(self.y_train_encoding_combo)
        y_train_layout.addLayout(y_train_encoding_layout)

        y_train_separator_layout = QHBoxLayout()
        self.y_train_separator_label = QLabel("Роздільник CSV:")
        self.y_train_separator_combo = QComboBox()
        self.y_train_separator_combo.addItems([",", ";", "\\t", "|", " "])
        y_train_separator_layout.addWidget(self.y_train_separator_label)
        y_train_separator_layout.addWidget(self.y_train_separator_combo)
        y_train_layout.addLayout(y_train_separator_layout)

        self.y_train_group.setLayout(y_train_layout)
        multi_files_layout.addWidget(self.y_train_group)

        # X_test секція
        self.x_test_group = QGroupBox("Тестові дані для навчання (X_test)")
        x_test_layout = QVBoxLayout()

        x_test_path_layout = QHBoxLayout()
        self.x_test_file_label = QLabel("Шлях до файлу:")
        self.x_test_file_path = QLineEdit()
        self.x_test_file_btn = QPushButton("Обрати...")
        x_test_path_layout.addWidget(self.x_test_file_label)
        x_test_path_layout.addWidget(self.x_test_file_path)
        x_test_path_layout.addWidget(self.x_test_file_btn)
        x_test_layout.addLayout(x_test_path_layout)

        x_test_encoding_layout = QHBoxLayout()
        self.x_test_encoding_label = QLabel("Кодування:")
        self.x_test_encoding_combo = QComboBox()
        self.x_test_encoding_combo.addItems(["utf-8", "cp1251", "latin-1", "iso-8859-1", "ascii"])
        x_test_encoding_layout.addWidget(self.x_test_encoding_label)
        x_test_encoding_layout.addWidget(self.x_test_encoding_combo)
        x_test_layout.addLayout(x_test_encoding_layout)

        x_test_separator_layout = QHBoxLayout()
        self.x_test_separator_label = QLabel("Роздільник CSV:")
        self.x_test_separator_combo = QComboBox()
        self.x_test_separator_combo.addItems([",", ";", "\\t", "|", " "])
        x_test_separator_layout.addWidget(self.x_test_separator_label)
        x_test_separator_layout.addWidget(self.x_test_separator_combo)
        x_test_layout.addLayout(x_test_separator_layout)

        self.x_test_group.setLayout(x_test_layout)
        multi_files_layout.addWidget(self.x_test_group)

        # Y_test секція (для задач з учителем)
        self.y_test_group = QGroupBox("Тестові дані для тестування (y_test)")
        y_test_layout = QVBoxLayout()

        y_test_path_layout = QHBoxLayout()
        self.y_test_file_label = QLabel("Шлях до файлу:")
        self.y_test_file_path = QLineEdit()
        self.y_test_file_btn = QPushButton("Обрати...")
        y_test_path_layout.addWidget(self.y_test_file_label)
        y_test_path_layout.addWidget(self.y_test_file_path)
        y_test_path_layout.addWidget(self.y_test_file_btn)
        y_test_layout.addLayout(y_test_path_layout)

        y_test_encoding_layout = QHBoxLayout()
        self.y_test_encoding_label = QLabel("Кодування:")
        self.y_test_encoding_combo = QComboBox()
        self.y_test_encoding_combo.addItems(["utf-8", "cp1251", "latin-1", "iso-8859-1", "ascii"])
        y_test_encoding_layout.addWidget(self.y_test_encoding_label)
        y_test_encoding_layout.addWidget(self.y_test_encoding_combo)
        y_test_layout.addLayout(y_test_encoding_layout)

        y_test_separator_layout = QHBoxLayout()
        self.y_test_separator_label = QLabel("Роздільник CSV:")
        self.y_test_separator_combo = QComboBox()
        self.y_test_separator_combo.addItems([",", ";", "\\t", "|", " "])
        y_test_separator_layout.addWidget(self.y_test_separator_label)
        y_test_separator_layout.addWidget(self.y_test_separator_combo)
        y_test_layout.addLayout(y_test_separator_layout)

        self.y_test_group.setLayout(y_test_layout)
        multi_files_layout.addWidget(self.y_test_group)

        self.multi_files_group.setLayout(multi_files_layout)
        main_layout.addWidget(self.multi_files_group)

        self.split_group = QGroupBox("Параметри розбиття")
        split_layout = QVBoxLayout()

        split_ratio_layout = QHBoxLayout()
        split_ratio_layout.addWidget(QLabel("Відсоток для тренування:"))

        self.train_percent = QSpinBox()
        self.train_percent.setRange(10, 90)
        self.train_percent.setValue(80)
        self.train_percent.setSuffix("%")
        split_ratio_layout.addWidget(self.train_percent)

        split_ratio_layout.addWidget(QLabel("Тестування:"))
        self.test_percent = QLabel("20%")
        split_ratio_layout.addWidget(self.test_percent)

        split_layout.addLayout(split_ratio_layout)

        seed_layout = QHBoxLayout()
        seed_layout.addWidget(QLabel("Seed значення:"))
        self.seed_spinbox = QSpinBox()
        self.seed_spinbox.setRange(0, 999999)
        self.seed_spinbox.setValue(42)
        seed_layout.addWidget(self.seed_spinbox)
        split_layout.addLayout(seed_layout)

        self.split_group.setLayout(split_layout)
        main_layout.addWidget(self.split_group)

        main_layout.addStretch()

        self.setLayout(main_layout)

        self.show_separator_fields(self.single_file_path.text(), 'single')
        self.show_separator_fields(self.x_train_file_path.text(), 'x_train')
        self.show_separator_fields(self.x_test_file_path.text(), 'x_test')
        self.show_separator_fields(self.y_train_file_path.text(), 'y_train')
        self.show_separator_fields(self.y_test_file_path.text(), 'y_test')

    def create_data_mode_group(self):
        self.data_mode_group = QGroupBox("Режим роботи з даними")
        data_mode_layout = QHBoxLayout()

        self.single_file_radio = QRadioButton("Один файл (розбити на тренувальний та тестовий)")
        self.multi_files_radio = QRadioButton("Окремі файли для навчання та тестування")

        data_mode_layout.addWidget(self.single_file_radio)
        data_mode_layout.addWidget(self.multi_files_radio)

        self.data_mode_group.setLayout(data_mode_layout)

    def update_ui_state(self, single_file_mode=True, supervised_learning=True):
        self.single_file_group.setVisible(single_file_mode)
        self.multi_files_group.setVisible(not single_file_mode)
        self.split_group.setVisible(single_file_mode)

        if not single_file_mode:
            self.y_train_group.setVisible(supervised_learning)
            self.y_test_group.setVisible(supervised_learning)

    def update_test_percent(self, train_value):
        self.test_percent.setText(f"{100 - train_value}%")

    def get_file_dialog(self, title, file_filter):
        return QFileDialog.getOpenFileName(
            self, title, "", file_filter
        )

    def show_separator_fields(self, file_path, file_type):
        if not file_path:
            if file_type == 'single':
                self.single_separator_label.setVisible(False)
                self.single_separator_combo.setVisible(False)
            elif file_type == 'x_train':
                self.x_train_separator_label.setVisible(False)
                self.x_train_separator_combo.setVisible(False)
            elif file_type == 'y_train':
                self.y_train_separator_label.setVisible(False)
                self.y_train_separator_combo.setVisible(False)
            elif file_type == 'x_test':
                self.x_test_separator_label.setVisible(False)
                self.x_test_separator_combo.setVisible(False)
            elif file_type == 'y_test':
                self.y_test_separator_label.setVisible(False)
                self.y_test_separator_combo.setVisible(False)
            return

        ext = os.path.splitext(file_path)[1].lower()
        is_csv = ext == '.csv'

        if file_type == 'single':
            self.single_separator_label.setVisible(is_csv)
            self.single_separator_combo.setVisible(is_csv)
        elif file_type == 'x_train':
            self.x_train_separator_label.setVisible(is_csv)
            self.x_train_separator_combo.setVisible(is_csv)
        elif file_type == 'y_train':
            self.y_train_separator_label.setVisible(is_csv)
            self.y_train_separator_combo.setVisible(is_csv)
        elif file_type == 'x_test':
            self.x_test_separator_label.setVisible(is_csv)
            self.x_test_separator_combo.setVisible(is_csv)
        elif file_type == 'y_test':
            self.y_test_separator_label.setVisible(is_csv)
            self.y_test_separator_combo.setVisible(is_csv)

    def update_target_field_visibility(self, should_show):
        self.target_label.setVisible(should_show)
        self.target_combo.setVisible(should_show)

