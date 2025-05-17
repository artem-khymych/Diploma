import time
from typing import Dict, Any, Optional
import tensorflow as tf
from tensorflow.keras.callbacks import History
import numpy as np

from project.logic.experiment.experiment import Experiment
from project.logic.experiment.nn_input_data_params import NeuralNetInputDataParams
from project.logic.evaluation.task_register import TaskType, NNMetricFactory, get_nn_metric, \
    NNModelType


class NeuralNetworkExperiment(Experiment):
    """
    Клас експерименту для нейронних мереж з підтримкою різних архітектур
    та збереженням специфічних гіперпараметрів для кожного типу завдання.
    """

    def __init__(self, id: int, task, model: Any, params: Dict[str, Any], parent=None):
        super().__init__(id, task, model, params, parent)

        # Тип завдання нейронної мережі (для використання з ModelTaskRegistry)
        self.task: Optional[TaskType] = task
        # Історія навчання
        self.history: Optional[History] = None
        # Шляхи до файлів моделі
        self.model_file_path: str = ''
        self.weights_file_path: str = ''
        self.load_type: str = ''
        self.input_data_params: NeuralNetInputDataParams = NeuralNetInputDataParams()

        # Реєстр завдань моделей
        self.metric_factory = NNMetricFactory()

        # Для зберігання інформації про дані
        self.data_info = {}

    def load_model_from_file(self) -> None:
        """
        Завантажити модель з файлу.
        """
        try:
            if not self.model_file_path:
                raise ValueError("Шлях до файлу моделі не вказано")

            if self.load_type == 'Keras (.h5)':
                self.model = tf.keras.models.load_model(self.model_file_path)
            elif self.load_type == 'TensorFlow SavedModel':
                self.model = tf.keras.models.load_model(self.model_file_path)
            elif self.load_type == 'JSON + Weights':
                with open(self.model_file_path, 'r') as json_file:
                    model_json = json_file.read()
                self.model = tf.keras.models.model_from_json(model_json)
                if self.weights_file_path:
                    self.model.load_weights(self.weights_file_path)
            else:
                raise ValueError(f"Непідтримуваний тип моделі: {self.load_type}")

        except Exception as e:
            raise ValueError(f"Помилка при завантаженні моделі: {str(e)}")

    def get_params_for_tune(self):
        self._load_data()

        self._validate_data()
        try:
            self.X_train = self._convert_to_tensorflow_compatible(self.X_train)
            self.X_test = self._convert_to_tensorflow_compatible(self.X_test)
            if self.y_train is not None:
                self.y_train = self._convert_to_tensorflow_compatible(self.y_train)
            if self.y_test is not None:
                self.y_test = self._convert_to_tensorflow_compatible(self.y_test)
        except Exception as e:
            raise ValueError(f"Помилка при перетворенні даних в формат TensorFlow: {str(e)}")

        return self.X_train, self.y_train
    def run(self) -> None:
        """
        Запуск експерименту з нейронною мережею.
        """
        self._load_data()

        self._validate_data()
        try:
            self.X_train = self._convert_to_tensorflow_compatible(self.X_train)
            self.X_test = self._convert_to_tensorflow_compatible(self.X_test)
            if self.y_train is not None:
                self.y_train = self._convert_to_tensorflow_compatible(self.y_train)
            if self.y_test is not None:
                self.y_test = self._convert_to_tensorflow_compatible(self.y_test)
        except Exception as e:
            raise ValueError(f"Помилка при перетворенні даних в формат TensorFlow: {str(e)}")

        # Початок вимірювання часу
        start_time = time.time()

        if self.model_file_path:
            # Якщо модель завантажується з файлу
            self.load_model_from_file()
        model = self.model
        """else:
            # Інакше створюємо модель з переданими параметрами
            model = self.model(**self._params)"""

        # Компіляція моделі (якщо це Keras модель)
        if isinstance(model, tf.keras.Model) or isinstance(model, tf.keras.Sequential):
            self._compile_model(model)

        # Навчання моделі (якщо потрібно)
        if self.X_train is not None:
            try:
                self.history = self._train_model(model)
            except Exception as e:
                error_msg = str(e)
                if "SparseSoftmaxCrossEntropyWithLogits" in error_msg and "valid range" in error_msg:
                    raise ValueError(
                        f"Помилка при навчанні: значення міток виходять за межі допустимого діапазону.\n"
                        f"Перевірте формат міток та їх відповідність кількості класів у моделі.\n"
                        f"Деталі: {error_msg}"
                    )
                else:
                    raise

        # Передбачення
        if self.task == TaskType.REGRESSION:
            self.train_predictions = model.predict(self.X_train)
            self.test_predictions = model.predict(self.X_test)
            self.train_actual = self.y_train
            self.test_actual = self.y_test
        elif self.task == TaskType.CLASSIFICATION:
            train_probabilities = model.predict(self.X_train)
            test_probabilities = model.predict(self.X_test)

            self.train_predictions = self._convert_probabilities_to_classes(train_probabilities)
            self.test_predictions = self._convert_probabilities_to_classes(test_probabilities)

            self.train_actual = self.y_train
            self.test_actual = self.y_test
        elif self.task == TaskType.TIME_SERIES_FORECASTING:
            # Для часових рядів потрібна спеціальна обробка
            self._process_time_series_predictions(model)
        """elif self.task == TaskType.DIMENSIONALITY_REDUCTION:
            # Для зниження розмірності
            self.transformed_train = model.predict(self.X_train)
            self.transformed_test = model.predict(self.X_test)
        elif self.task == TaskType.ANOMALY_DETECTION:
            # Для виявлення аномалій
            train_probabilities = model.predict(self.X_train)
            test_probabilities = model.predict(self.X_test)

            self.train_predictions = self._convert_probabilities_to_classes(train_probabilities)
            self.test_predictions = self._convert_probabilities_to_classes(test_probabilities)

            self.train_actual = self.y_train
            self.test_actual = self.y_test"""

        # Збереження навченої моделі
        self.trained_model = model
        self.train_time = time.time() - start_time
        self.is_finished = True
        self.experiment_finished.emit(self.train_time)

    def _process_time_series_predictions(self, model):

        try:
            x_train = self._convert_to_tensorflow_compatible(self.X_train)
            x_test = self._convert_to_tensorflow_compatible(self.X_test)

            if isinstance(x_train, (np.ndarray, tf.Tensor)) and len(x_train.shape) == 3:
                self.train_predictions = model.predict(x_train)
                self.test_predictions = model.predict(x_test)
                self.train_actual = self.y_train
                self.test_actual = self.y_test
            else:
                raise NotImplementedError("Обробка для даного типу даних не передбачена")
        except Exception as e:
            print(f"Помилка  при обробці часового ряду:: {str(e)}")

    def _validate_data(self):
        """
        Перевірка відповідності даних типу задачі та моделі.
        """
        if self.task == TaskType.CLASSIFICATION:
            # Перевірка для задач класифікації
            self._validate_classification_data()
        elif self.task == TaskType.REGRESSION:
            # Перевірка для задач регресії
            self._validate_regression_data()
        elif self.task == TaskType.TIME_SERIES_FORECASTING:
            # Перевірка для задач прогнозування часових рядів
            self._validate_time_series_data()

        # Зберігаємо інформацію про дані
        self._store_data_info()

    def _validate_classification_data(self):
        """
        Перевірка даних для задачі класифікації.
        """
        if self.y_train is None or self.y_test is None:
            raise ValueError("Для класифікації необхідні мітки класів")

        # Перевірка, що мітки є цілими числами для sparse_categorical_crossentropy
        if not isinstance(self.y_train, np.ndarray):
            self.y_train = np.array(self.y_train)
        if not isinstance(self.y_test, np.ndarray):
            self.y_test = np.array(self.y_test)

        # Перевірка міток
        unique_labels = np.unique(np.concatenate([self.y_train, self.y_test]))
        num_classes = len(unique_labels)

        if num_classes == 2:
            # Бінарна класифікація
            if np.min(unique_labels) < 0 or np.max(unique_labels) > 1:
                # Перетворення міток для бінарної класифікації (0 та 1)
                self._transform_binary_labels()
        elif num_classes > 2:
            # Багатокласова класифікація
            if np.min(unique_labels) < 0:
                raise ValueError(f"Виявлено від'ємні значення міток: {unique_labels}")

            # Перевірка на послідовність міток (починаючись з 0)
            if not np.array_equal(unique_labels, np.arange(num_classes)):
                # Перетворення міток для послідовної нумерації від 0
                self._transform_multiclass_labels()
        else:
            raise ValueError(f"Надто мало класів у даних: {num_classes}")

    def _transform_binary_labels(self):
        """
        Перетворення міток для бінарної класифікації.
        """
        unique_labels = np.unique(np.concatenate([self.y_train, self.y_test]))
        print(f"Перетворення міток бінарної класифікації: {unique_labels} -> [0, 1]")

        # Перетворення до 0 та 1
        label_map = {unique_labels[0]: 0, unique_labels[1]: 1}

        # Зберігання для використання в оцінці
        self.data_info['original_labels'] = unique_labels
        self.data_info['label_map'] = label_map

        # Перетворення міток
        self.y_train = np.array([label_map[label] for label in self.y_train])
        self.y_test = np.array([label_map[label] for label in self.y_test])

    def _transform_multiclass_labels(self):
        """
        Перетворення міток для багатокласової класифікації.
        """
        unique_labels = np.unique(np.concatenate([self.y_train, self.y_test]))
        print(f"Перетворення міток багатокласової класифікації: {unique_labels} -> [0...{len(unique_labels) - 1}]")

        # Створюємо відображення для міток
        label_map = {label: i for i, label in enumerate(unique_labels)}

        # Зберігання для використання в оцінці
        self.data_info['original_labels'] = unique_labels
        self.data_info['label_map'] = label_map
        self.data_info['num_classes'] = len(unique_labels)

        # Перетворення міток
        self.y_train = np.array([label_map[label] for label in self.y_train])
        self.y_test = np.array([label_map[label] for label in self.y_test])

    def _validate_regression_data(self):
        """
        Перевірка даних для задачі регресії.
        """
        if self.y_train is None or self.y_test is None:
            raise ValueError("Для регресії необхідні цільові значення")

        # Перетворення до numpy масивів
        if not isinstance(self.y_train, np.ndarray):
            self.y_train = np.array(self.y_train)
        if not isinstance(self.y_test, np.ndarray):
            self.y_test = np.array(self.y_test)

        # Перевірка на наявність невідповідних значень
        if np.isnan(self.y_train).any() or np.isnan(self.y_test).any():
            raise ValueError("Виявлено NaN значення в цільових змінних")

        # Зберігаємо інформацію про дані
        self.data_info['y_min'] = min(np.min(self.y_train), np.min(self.y_test))
        self.data_info['y_max'] = max(np.max(self.y_train), np.max(self.y_test))

    def _validate_time_series_data(self):
        """
        Перевірка даних для задачі прогнозування часових рядів.
        """
        # Базова перевірка
        if self.X_train is None or self.X_test is None:
            raise ValueError("Відсутні дані для часових рядів")

        # Перевірка форми даних
        if isinstance(self.X_train, np.ndarray):
            if len(self.X_train.shape) != 3:
                raise ValueError(
                    f"Неправильна форма вхідних даних для часових рядів. Очікується 3D масив, отримано: {self.X_train.shape}")
        elif isinstance(self.X_train, tf.keras.utils.Sequence):
            # Для генераторів послідовностей перевірки проводяться під час завантаження
            pass
        else:
            raise ValueError(f"Непідтримуваний тип даних для часових рядів: {type(self.X_train)}")

    def _store_data_info(self):
        """
        Зберігання інформації про дані для використання в інших методах.
        """
        # Базова інформація про дані
        try:
            if isinstance(self.X_train, np.ndarray):
                self.data_info['x_shape'] = self.X_train.shape
            if isinstance(self.y_train, np.ndarray):
                self.data_info['y_shape'] = self.y_train.shape

            if self.task == TaskType.CLASSIFICATION:
                unique_train = np.unique(self.y_train)
                unique_test = np.unique(self.y_test)
                all_unique = np.unique(np.concatenate([unique_train, unique_test]))

                self.data_info['num_classes'] = len(all_unique)
                self.data_info['unique_labels'] = all_unique
                print(f"Виявлено {self.data_info['num_classes']} класів з мітками: {all_unique}")

                # Перевірка на однопрохідність класів у тренувальних даних
                if len(unique_train) != len(all_unique):
                    print(f"Увага: Деякі класи відсутні в тренувальних даних. Тренувальні мітки: {unique_train}")

            elif self.task == TaskType.REGRESSION:
                # Для регресії додаткова інформація
                if isinstance(self.y_train, np.ndarray) and isinstance(self.y_test, np.ndarray):
                    self.data_info['y_mean'] = np.mean(np.concatenate([self.y_train, self.y_test]))
                    self.data_info['y_std'] = np.std(np.concatenate([self.y_train, self.y_test]))

        except Exception as e:
            print(f"Помилка при зборі інформації про дані: {str(e)}")

    def _compile_model(self, model) -> None:
        """
        Компіляція моделі Keras з параметрами, специфічними для завдання.

        Args:
            model: Модель Keras для компіляції
        """
        # Параметри компіляції за замовчуванням
        default_compile_params = {
            'optimizer': 'adam',
            'metrics': ['accuracy']
        }

        # Визначення функції втрат на основі типу завдання та характеристик даних
        if 'loss' not in self._params:
            if self.task == TaskType.CLASSIFICATION:
                # Для класифікації вибір залежить від кількості класів та формату міток
                num_classes = self.data_info.get('num_classes', 0)

                # Перевірка вихідного шару моделі
                try:
                    output_units = model.layers[-1].output_shape[-1]
                    if output_units != num_classes:
                        print(f"Попередження: Кількість нейронів у вихідному шарі ({output_units}) "
                              f"не відповідає кількості класів ({num_classes})")
                except:
                    output_units = None

                if num_classes == 2:
                    # Для бінарної класифікації
                    default_compile_params['loss'] = 'binary_crossentropy'
                    # Переконуємося, що останній шар має sigmoid активацію
                    try:
                        last_layer_activation = model.layers[-1].activation.__name__
                        if last_layer_activation != 'sigmoid':
                            print(f"Попередження: Для бінарної класифікації рекомендується "
                                  f"використовувати sigmoid активацію у вихідному шарі, а не {last_layer_activation}")
                    except:
                        pass
                else:
                    # Для багатокласової класифікації
                    default_compile_params['loss'] = 'sparse_categorical_crossentropy'
                    # Переконуємося, що останній шар має softmax активацію
                    try:
                        last_layer_activation = model.layers[-1].activation.__name__
                        if last_layer_activation != 'softmax':
                            print(f"Попередження: Для багатокласової класифікації рекомендується "
                                  f"використовувати softmax активацію у вихідному шарі, а не {last_layer_activation}")
                    except:
                        pass
            elif self.task == TaskType.REGRESSION:
                default_compile_params['loss'] = 'mse'
                default_compile_params['metrics'] = ['mae', 'mse']
            elif self.task == TaskType.ANOMALY_DETECTION:
                default_compile_params['loss'] = 'binary_crossentropy'
            elif self.task == TaskType.TIME_SERIES_FORECASTING:
                default_compile_params['loss'] = 'mse'
                default_compile_params['metrics'] = ['mae']
            else:
                default_compile_params['loss'] = 'mse'  # За замовчуванням

        # Оновлення параметрів за замовчуванням з переданими параметрами
        compile_params = {**default_compile_params}

        # Оновлення з параметрів моделі в self._params
        if 'model_params' in self._params:
            model_params = self._params.get('model_params', {})
            for key in ['loss', 'metrics']:
                if key in model_params and model_params[key] is not None:
                    compile_params[key] = model_params[key]

            # Обробка оптимізатора з урахуванням learning_rate
            if 'optimizer' in model_params and model_params['optimizer'] is not None:
                optimizer_name = model_params['optimizer']
                learning_rate = model_params.get('learning_rate')

                # Створення об'єкта оптимізатора з вказаним learning_rate
                if learning_rate is not None:
                    if optimizer_name == 'adam':
                        compile_params['optimizer'] = tf.keras.optimizers.Adam(learning_rate=learning_rate)
                    elif optimizer_name == 'sgd':
                        compile_params['optimizer'] = tf.keras.optimizers.SGD(learning_rate=learning_rate)
                    elif optimizer_name == 'rmsprop':
                        compile_params['optimizer'] = tf.keras.optimizers.RMSprop(learning_rate=learning_rate)
                    elif optimizer_name == 'adagrad':
                        compile_params['optimizer'] = tf.keras.optimizers.Adagrad(learning_rate=learning_rate)
                    elif optimizer_name == 'adadelta':
                        compile_params['optimizer'] = tf.keras.optimizers.Adadelta(learning_rate=learning_rate)
                    elif optimizer_name == 'adamax':
                        compile_params['optimizer'] = tf.keras.optimizers.Adamax(learning_rate=learning_rate)
                    elif optimizer_name == 'nadam':
                        compile_params['optimizer'] = tf.keras.optimizers.Nadam(learning_rate=learning_rate)
                    else:
                        # Якщо назва оптимізатора не розпізнана, використовуємо просту рядкову назву
                        compile_params['optimizer'] = optimizer_name
                        print(
                            f"Попередження: Нерозпізнаний оптимізатор '{optimizer_name}'. Learning rate буде проігноровано.")
                else:
                    # Якщо learning_rate не вказано, використовуємо просто назву оптимізатора
                    compile_params['optimizer'] = optimizer_name

        # Компіляція моделі
        try:
            model.compile(**compile_params)
            print(f"Модель скомпільована з параметрами: {compile_params}")
        except Exception as e:
            raise ValueError(f"Помилка при компіляції моделі: {str(e)}")

    def _train_model(self, model) -> History:
        """
        Навчання моделі Keras з параметрами, специфічними для завдання.

        Args:
            model: Модель Keras для навчання

        Returns:
            History: Історія навчання
        """
        # Параметри навчання за замовчуванням
        default_fit_params = {
            'x': self.X_train,
            'batch_size': 32,
            'epochs': 10,
            'verbose': 1,
            'validation_split': 0.2,
            'shuffle': True,
            'y': self.y_train
        }

        # Додавання цільових змінних залежно від типу завдання
        if self.task in [TaskType.CLASSIFICATION, TaskType.REGRESSION]:
            default_fit_params['y'] = self.y_train

        # Для задач з генераторами
        if isinstance(self.X_train, tf.keras.utils.Sequence):
            # Для генераторів не потрібно вказувати y
            default_fit_params = {
                'x': self.X_train,
                'batch_size': None,  # Генератор сам визначає розмір батчу
                'epochs': 10,
                'verbose': 1,
                'validation_data': self.X_test,
                'shuffle': False  # Генератор сам може перемішувати дані
            }
            if 'y' in default_fit_params:
                del default_fit_params['y']
            if 'validation_split' in default_fit_params:
                del default_fit_params['validation_split']

        # Оновлення параметрів навчання
        fit_params = {**default_fit_params}

        # Оновлення з параметрів навчання в self._params
        if 'fit_params' in self._params:
            for key, value in self._params.get('fit_params', {}).items():
                if value is not None:
                    fit_params[key] = value

        # Видалення None параметрів
        for key in list(fit_params.keys()):
            if fit_params[key] is None:
                del fit_params[key]

        # Перевірка на порожній словник class_weight
        if 'class_weight' in fit_params and fit_params['class_weight'] == {}:
            del fit_params['class_weight']

        # Додавання збалансованих ваг класів для незбалансованих даних класифікації
        if self.task == TaskType.CLASSIFICATION and 'class_weight' not in fit_params:
            try:
                unique_train_labels, counts = np.unique(self.y_train, return_counts=True)
                if len(unique_train_labels) > 1 and np.max(counts) / np.min(counts) > 5:
                    print("Виявлено незбалансований датасет. Застосування автоматичних ваг класів.")
                    class_weights = {}
                    for i, count in enumerate(counts):
                        class_weights[i] = len(self.y_train) / (len(unique_train_labels) * count)
                    fit_params['class_weight'] = class_weights
            except Exception as e:
                print(f"Не вдалося автоматично визначити ваги класів: {str(e)}")

        if 'x' in fit_params and not isinstance(fit_params['x'], tf.keras.utils.Sequence):
            fit_params['x'] = self._convert_to_tensorflow_compatible(fit_params['x'])
        if 'y' in fit_params and not isinstance(fit_params['y'], tf.keras.utils.Sequence):
            fit_params['y'] = self._convert_to_tensorflow_compatible(fit_params['y'])
        if 'validation_data' in fit_params and not isinstance(fit_params['validation_data'], tf.keras.utils.Sequence):
            if isinstance(fit_params['validation_data'], tuple):
                # Если validation_data - это кортеж (x_val, y_val)
                x_val, y_val = fit_params['validation_data']
                x_val = self._convert_to_tensorflow_compatible(x_val)
                y_val = self._convert_to_tensorflow_compatible(y_val)
                fit_params['validation_data'] = (x_val, y_val)
            else:
                fit_params['validation_data'] = self._convert_to_tensorflow_compatible(fit_params['validation_data'])

        # Навчання моделі
        try:
            history = model.fit(**fit_params)
            return history
        except Exception as e:
            error_msg = str(e)
            if "SparseSoftmaxCrossEntropyWithLogits" in error_msg and "valid range" in error_msg:
                # Детальне повідомлення про помилку з мітками класів
                unique_labels = np.unique(self.y_train)
                num_classes = model.layers[-1].output_shape[-1]

                error_details = (
                    f"Помилка при навчанні моделі класифікації:\n"
                    f"Виявлено мітки: {unique_labels}\n"
                    f"Кількість класів у вихідному шарі моделі: {num_classes}\n"
                    f"Функція втрат очікує мітки в діапазоні [0, {num_classes - 1}]\n"
                    f"Оригінальна помилка: {error_msg}"
                )
                raise ValueError(error_details)
            else:
                raise

    def evaluate(self) -> None:
        """
        Спрощена оцінка результатів - отримує стратегію метрики для даної мережі та задачі,
        і викликає її з відповідними параметрами
        """
        if not self.is_finished:
            raise RuntimeError("Експеримент повинен бути завершений перед оцінкою")

        # Ініціалізація словників для метрик
        self.train_metrics = {}
        self.test_metrics = {}

        try:
            # Отримуємо стратегію метрики для даної комбінації моделі та задачі
            self.metric_strategy = get_nn_metric(NNModelType.GENERIC.value, self.task.value)

            # Визначаємо вхідні дані для оцінки в залежності від типу задачі
            if self.task in [TaskType.CLASSIFICATION, TaskType.REGRESSION, TaskType.TIME_SERIES_FORECASTING]:
                # Для класифікації, регресії та прогнозування - використовуємо actual та predictions
                train_input = (self.train_actual, self.train_predictions)
                test_input = (self.test_actual, self.test_predictions)
                """elif self.task == TaskType.ANOMALY_DETECTION:
                    # Для аномалій - використовуємо вхідні дані та predictions
                    train_input = (self.X_train, self.train_predictions)
                    test_input = (self.X_test, self.test_predictions)
                elif self.task == TaskType.DIMENSIONALITY_REDUCTION:
                    # Для зниження розмірності - оригінальні та трансформовані дані
                    train_input = (self.X_train, self.transformed_train)
                    test_input = (self.X_test, self.transformed_test)"""
            else:
                # Універсальний підхід для інших задач
                train_input = (self.X_train, self.train_predictions) if hasattr(self, 'train_predictions') else None
                test_input = (self.X_test, self.test_predictions) if hasattr(self, 'test_predictions') else None

            # Виконуємо оцінку для тренувальних даних, якщо вони доступні
            if train_input and all(x is not None for x in train_input):
                self.train_metrics.update(
                    self.metric_strategy.evaluate(*train_input)
                )

            # Виконуємо оцінку для тестових даних, якщо вони доступні
            if test_input and all(x is not None for x in test_input):
                self.test_metrics.update(
                    self.metric_strategy.evaluate(*test_input)
                )

        except Exception as e:
            print(f"Помилка при оцінці: {str(e)}")
            self.train_metrics = {"error": f"Помилка оцінки: {str(e)}"}
            self.test_metrics = {"error": f"Помилка оцінки: {str(e)}"}

        # Сигнал про завершення оцінки
        self.experiment_evaluated.emit(self.train_metrics, self.test_metrics)

    def _convert_probabilities_to_classes(self, predictions, threshold=0.5):
        """
        Преобразует вероятности в метки классов.

        Args:
            predictions: Предсказания модели (вероятности классов)
            threshold: Порог для бинарной классификации (по умолчанию 0.5)

        Returns:
            Метки классов
        """
        if predictions is None:
            return None

        # Проверка формы предсказаний
        if not isinstance(predictions, np.ndarray):
            try:
                predictions = np.array(predictions)
            except Exception as e:
                raise ValueError(f"Не удалось преобразовать предсказания в numpy array: {str(e)}")

        # Для бинарной классификации с одним выходом (sigmoid)
        if len(predictions.shape) == 1 or (len(predictions.shape) == 2 and predictions.shape[1] == 1):
            return (predictions > threshold).astype(int)

        # Для многоклассовой классификации (softmax)
        elif len(predictions.shape) == 2 and predictions.shape[1] > 1:
            return np.argmax(predictions, axis=1)

        # Для других форматов предсказаний
        else:
            raise ValueError(f"Неподдерживаемый формат предсказаний с формой {predictions.shape}")

    def _load_data(self):
        """
        Завантажує дані в залежності від типу нейромережі та завдання.
        Розширений метод, який обробляє різні типи даних: табличні, зображення,
        послідовності, текст і спеціалізовані дані для автоенкодерів.
        """
        super()._load_data()
        print(f"Завантажено дані для загальної моделі: X_train={self.X_train.shape if hasattr(self.X_train, 'shape') else 'N/A'}")

        # Перевірка даних після завантаження
        self._check_loaded_data()

    def _check_loaded_data(self):
        """
        Перевірка правильності завантажених даних перед використанням.
        """
        # Перевірка наявності даних
        if self.X_train is None:
            raise ValueError("Дані для навчання не завантажені (X_train is None)")
        # Перевірка відповідності X та y
        if self.task in [TaskType.CLASSIFICATION, TaskType.REGRESSION]:
            if self.y_train is None:
                raise ValueError("Цільові змінні для навчання не завантажені (y_train is None)")

            if isinstance(self.X_train, np.ndarray) and isinstance(self.y_train, np.ndarray):
                if len(self.X_train) != len(self.y_train):
                    raise ValueError(
                        f"Розмірності X_train ({len(self.X_train)}) та y_train ({len(self.y_train)}) не співпадають")

                if len(self.X_test) != len(self.y_test):
                    raise ValueError(
                        f"Розмірності X_test ({len(self.X_test)}) та y_test ({len(self.y_test)}) не співпадають")

    def _convert_to_tensorflow_compatible(self, data):

        if data is None:
            return None

        if isinstance(data, tf.Tensor):
            return data

        # Якщо дані - це pandas DataFrame чи Series
        if hasattr(data, 'values'):
            data = data.values

        # Якщо дані ітерований об'єкт
        if not isinstance(data, np.ndarray):
            try:
                data = np.array(data)
            except Exception as e:
                raise ValueError(f"Не вдалось перетворити дані типу {type(data)} в numpy array: {str(e)}")

        # перетворення даних у підходящий тип
        if data.dtype == np.int32:
            data = data.astype(np.int32)
        elif data.dtype == np.float32:
            data = data.astype(np.float32)
        elif data.dtype == bool:
            data = data.astype(np.bool_)
        elif np.issubdtype(data.dtype, np.object_):
            # знайти підходящий тип даних для масиву
            if all(isinstance(x, (int, np.integer)) for x in data.flatten() if x is not None):
                data = data.astype(np.int32)
            elif all(isinstance(x, (float, np.floating)) for x in data.flatten() if x is not None):
                data = data.astype(np.float32)
            elif all(isinstance(x, str) for x in data.flatten() if x is not None):
                pass
            else:
                try:
                    data = data.astype(np.float32)
                except Exception as e:
                    raise ValueError(f"Не вдалось перетворити масив у чисельний формат: {str(e)}")

        try:
            tensor = tf.convert_to_tensor(data)
            return tensor
        except Exception as e:
            raise ValueError(
                f"Помилка при перетворенні в тензор TensorFlow: {str(e)}\nТип даних: {data.dtype}, Форма: {data.shape}")

