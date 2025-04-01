from project.logic.modules import task_names


class InputDataParams:

    def __init__(self):
        self.mode = 'single_file'

        self.single_file_path = ''
        self.x_train_file_path = ''
        self.y_train_file_path = ''
        self.x_test_file_path = ''
        self.y_test_file_path = ''

        self.train_percent = 80
        self.test_percent = 20
        self.seed = 42

        self.target_variable = ''

        self.single_file_encoding = 'utf-8'
        self.x_train_file_encoding = 'utf-8'
        self.y_train_file_encoding = 'utf-8'
        self.x_test_file_encoding = 'utf-8'
        self.y_test_file_encoding = 'utf-8'

        self.single_file_separator = ','
        self.x_train_file_separator = ','
        self.y_train_file_separator = ','
        self.x_test_file_separator = ','
        self.y_test_file_separator = ','

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

            if self.target_variable and not self.is_target_not_required():
                data['target_variable'] = self.target_variable
        else:
            if self.is_target_not_required():
                data.update({
                    'x_train_file_path': self.x_train_file_path,
                    'x_test_file_path': self.x_test_file_path,
                    'x_train_file_encoding': self.x_train_file_encoding,
                    'x_test_file_encoding': self.x_test_file_encoding,
                    'x_train_file_separator': self.x_train_file_separator,
                    'x_test_file_separator': self.x_test_file_separator
                })
            else:
                data.update({
                    'x_train_file_path': self.x_train_file_path,
                    'y_train_file_path': self.y_train_file_path,
                    'x_test_file_path': self.x_test_file_path,
                    'y_test_file_path': self.y_test_file_path,
                    'x_train_file_encoding': self.x_train_file_encoding,
                    'y_train_file_encoding': self.y_train_file_encoding,
                    'x_test_file_encoding': self.x_test_file_encoding,
                    'y_test_file_encoding': self.y_test_file_encoding,
                    'x_train_file_separator': self.x_train_file_separator,
                    'y_train_file_separator': self.y_train_file_separator,
                    'x_test_file_separator': self.x_test_file_separator,
                    'y_test_file_separator': self.y_test_file_separator
                })

        return data

    def is_target_not_required(self):
        return self.current_task in InputDataParams.tasks_without_target

    def is_filled(self):
        if self.mode == "single_file":
            if self.single_file_path != "":
                return True
        else:
            if self.x_test_file_path and self.x_train_file_path and self.y_train_file_path and self.y_test_file_path != "":
                return True
        return False

    tasks_without_target = [
        task_names.CLUSTERING,
        task_names.DIMENSIONALITY_REDUCTION,
        task_names.ANOMALY_DETECTION,
        task_names.DENSITY_ESTIMATION
    ]
