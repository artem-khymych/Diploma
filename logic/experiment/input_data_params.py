from project.logic.modules import task_names
class InputDataParams:

    def __init__(self):
        # Налаштування режиму роботи з даними
        self.mode = 'single_file'  # 'single_file' або 'two_files'

        # Шляхи до файлів
        self.single_file_path = ''
        self.train_file_path = ''
        self.test_file_path = ''

        # Параметри розбиття
        self.train_percent = 80
        self.test_percent = 20
        self.seed = 42

        # Цільова змінна
        self.target_variable = ''

        # Кодування та роздільник для CSV
        self.single_file_encoding = 'utf-8'
        self.train_file_encoding = 'utf-8'
        self.test_file_encoding = 'utf-8'
        self.single_file_separator = ','
        self.train_file_separator = ','

        # Поточне завдання
        self.current_task = ''

    def to_dict(self):
        data = {
            'mode': self.mode,
            'current_task': self.current_task
        }

        if self.mode == 'single_file':
            data.update({
                'single_file_path': self.single_file_path,
                'train_percent': self.train_percent,
                'test_percent': self.test_percent,
                'seed': self.seed,
                'single_file_encoding': self.single_file_encoding,
                'single_file_separator': self.single_file_separator
            })

            # Додаємо цільову змінну лише якщо вона потрібна для завдання
            if self.target_variable and not self.is_target_not_required():
                data['target_variable'] = self.target_variable
        else:
            data.update({
                'train_file_path': self.train_file_path,
                'test_file_path': self.test_file_path,
                'train_file_encoding': self.train_file_encoding,
                'test_file_encoding': self.test_file_encoding,
                'train_file_separator': self.train_file_separator
            })

        return data

    def is_target_not_required(self):
        """Перевіряє, чи потрібен вибір цільової змінної для поточного завдання"""
        return self.current_task in InputDataParams.tasks_without_target

    # Список завдань, для яких не потрібен вибір цільової змінної
    tasks_without_target = [
        task_names.CLUSTERING,
        task_names.DIMENSIONALITY_REDUCTION,
        task_names.ANOMALY_DETECTION,
        task_names.DENSITY_ESTIMATION
    ]
