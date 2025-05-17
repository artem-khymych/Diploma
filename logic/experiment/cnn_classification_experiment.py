import tensorflow as tf
import numpy as np
import h5py
import os
from typing import Dict, Any, Optional, Tuple, List

from project.logic.evaluation.task_register import TaskType, NNModelType
from project.logic.experiment.nn_experiment import NeuralNetworkExperiment


class CNNExperiment(NeuralNetworkExperiment):
    """
    Клас експерименту для згорткових нейронних мереж з підтримкою роботи
    з даними зображень у форматах .npz, .h5 або parquet.
    """

    def __init__(self, id: int, task, model: Any, params: Dict[str, Any], parent=None):
        super().__init__(id, task, model, params, parent)

        # Специфічні параметри для CNN
        self.image_height = None
        self.image_width = None
        self.channels = None
        self.task_spec_params = params.get("task_spec_params", {})

        # Ініціалізація параметрів зображення з task_spec_params
        self._init_image_params()

    def _init_image_params(self) -> None:
        """
        Ініціалізація параметрів зображення з task_spec_params.
        """
        if self.task_spec_params:
            self.image_height = self.task_spec_params.get("image_height", None)
            self.image_width = self.task_spec_params.get("image_width", None)
            self.channels = self.task_spec_params.get("channels", None)

            # Логування параметрів
            if all([self.image_height, self.image_width, self.channels]):
                print(f"Ініціалізовано параметри зображення: {self.image_height}x{self.image_width}x{self.channels}")
            else:
                print("Попередження: Не всі параметри зображення визначені в task_spec_params")

    def _load_data(self) -> None:
        """
        Розширений метод для завантаження даних зображень у різних форматах.
        """
        # Отримання шляхів до файлів з параметрів
        train_data_path = self.input_data_params.train_data_path
        test_data_path = self.input_data_params.test_data_path

        if not train_data_path or not os.path.exists(train_data_path):
            raise ValueError(f"Неправильний шлях до навчальних даних: {train_data_path}")

        if not test_data_path or not os.path.exists(test_data_path):
            raise ValueError(f"Неправильний шлях до тестових даних: {test_data_path}")

        # Визначення формату файлу і вибір відповідного способу завантаження
        train_ext = os.path.splitext(train_data_path)[1].lower()
        test_ext = os.path.splitext(test_data_path)[1].lower()

        if train_ext != test_ext:
            print(f"Попередження: Різні формати файлів даних для навчання ({train_ext}) та тестування ({test_ext})")

        # Завантаження даних відповідно до формату
        if train_ext == '.npz':
            self.X_train, self.y_train = self._load_npz_data(train_data_path)
            self.X_test, self.y_test = self._load_npz_data(test_data_path)
        elif train_ext == '.h5':
            self.X_train, self.y_train = self._load_h5_data(train_data_path)
            self.X_test, self.y_test = self._load_h5_data(test_data_path)
        elif train_ext == '.parquet':
            self.X_train, self.y_train = self._load_parquet_data(train_data_path)
            self.X_test, self.y_test = self._load_parquet_data(test_data_path)
        else:
            # Спробувати використати базовий метод завантаження
            super()._load_data()

        # Перевірка і перетворення даних для CNN
        self._preprocess_image_data()

        print(f"Завантажено дані для CNN моделі: "
              f"X_train={self.X_train.shape if hasattr(self.X_train, 'shape') else 'N/A'}, "
              f"X_test={self.X_test.shape if hasattr(self.X_test, 'shape') else 'N/A'}")

    def _load_npz_data(self, file_path: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        Завантаження даних з файлу .npz.

        Args:
            file_path: Шлях до файлу .npz

        Returns:
            Tuple[np.ndarray, np.ndarray]: Кортеж (X, y) з даними та мітками
        """
        try:
            data = np.load(file_path)

            # Спроба знайти дані та мітки за стандартними ключами
            x_keys = ['x', 'data', 'features', 'images', 'X']
            y_keys = ['y', 'labels', 'targets', 'Y']

            X = None
            y = None

            # Пошук даних
            for key in x_keys:
                if key in data:
                    X = data[key]
                    break

            # Пошук міток
            for key in y_keys:
                if key in data:
                    y = data[key]
                    break

            # Якщо не знайдено за стандартними ключами, спробувати використати перші два масиви
            if X is None or y is None:
                keys = list(data.keys())
                if len(keys) >= 2:
                    X = data[keys[0]]
                    y = data[keys[1]]

            if X is None or y is None:
                raise ValueError(f"Не вдалось знайти дані або мітки в файлі {file_path}")

            return X, y

        except Exception as e:
            raise ValueError(f"Помилка при завантаженні даних з NPZ файлу {file_path}: {str(e)}")

    def _load_h5_data(self, file_path: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        Завантаження даних з файлу .h5.

        Args:
            file_path: Шлях до файлу .h5

        Returns:
            Tuple[np.ndarray, np.ndarray]: Кортеж (X, y) з даними та мітками
        """
        try:
            with h5py.File(file_path, 'r') as hf:
                # Спроба знайти дані та мітки за стандартними ключами
                x_keys = ['x', 'data', 'features', 'images', 'X']
                y_keys = ['y', 'labels', 'targets', 'Y']

                X = None
                y = None

                # Пошук даних
                for key in x_keys:
                    if key in hf:
                        X = np.array(hf[key])
                        break

                # Пошук міток
                for key in y_keys:
                    if key in hf:
                        y = np.array(hf[key])
                        break

                # Якщо не знайдено за стандартними ключами, спробувати використати перші два набори даних
                if X is None or y is None:
                    keys = list(hf.keys())
                    if len(keys) >= 2:
                        X = np.array(hf[keys[0]])
                        y = np.array(hf[keys[1]])

                if X is None or y is None:
                    raise ValueError(f"Не вдалось знайти дані або мітки в файлі {file_path}")

                return X, y

        except Exception as e:
            raise ValueError(f"Помилка при завантаженні даних з H5 файлу {file_path}: {str(e)}")

    def _load_parquet_data(self, file_path: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        Завантаження даних з файлу .parquet.

        Args:
            file_path: Шлях до файлу .parquet

        Returns:
            Tuple[np.ndarray, np.ndarray]: Кортеж (X, y) з даними та мітками
        """
        try:
            # Потрібен pandas для роботи з parquet
            import pandas as pd

            df = pd.read_parquet(file_path)

            # Спроба знайти колонку з мітками
            # Типові назви для колонок з мітками
            label_columns = ['label', 'target', 'class', 'y', 'Y']

            # Пошук колонки з мітками
            y_column = None
            for col in label_columns:
                if col in df.columns:
                    y_column = col
                    break

            if y_column is None:
                # Якщо не знайдено, припускаємо, що остання колонка - мітки
                y_column = df.columns[-1]
                print(f"Попередження: Не знайдено стандартну колонку з мітками, використовуємо {y_column}")

            # Розділення на дані та мітки
            y = df[y_column].values
            X = df.drop(y_column, axis=1).values

            # Для зображень потрібно перетворити плоский формат в 3D/4D
            if len(X.shape) == 2:
                # Плоский формат, потрібно перетворити в формат зображень
                expected_size = self.image_height * self.image_width * self.channels
                if X.shape[1] == expected_size:
                    X = X.reshape(-1, self.image_height, self.image_width, self.channels)
                else:
                    print(f"Попередження: Розмірність даних ({X.shape[1]}) не відповідає очікуваній "
                          f"({expected_size}) для зображень {self.image_height}x{self.image_width}x{self.channels}")

            return X, y

        except ImportError:
            raise ImportError("Для роботи з parquet файлами потрібен пакет pandas. Встановіть його: pip install pandas")
        except Exception as e:
            raise ValueError(f"Помилка при завантаженні даних з Parquet файлу {file_path}: {str(e)}")

    def _preprocess_image_data(self) -> None:
        """
        Передобробка даних зображень для CNN.
        """
        # Перевірка, чи дані вже у відповідному форматі
        if self.X_train is None or self.X_test is None:
            raise ValueError("Дані не завантажені")

        # Перевірка розмірності даних і перетворення, якщо потрібно
        self._check_and_reshape_data()

        # Нормалізація даних зображень
        self._normalize_images()

        # Перетворення міток для класифікації
        if self.task == TaskType.CLASSIFICATION:
            self._process_labels()

    def _check_and_reshape_data(self) -> None:
        """
        Перевіряє і перетворює розмірність даних для CNN, якщо потрібно.
        """
        # Прочитати параметри зображення, якщо не встановлені
        if not all([self.image_height, self.image_width, self.channels]):
            # Спроба визначити з форми даних
            if len(self.X_train.shape) == 4:  # Припускаємо формат (samples, height, width, channels)
                self.image_height = self.X_train.shape[1]
                self.image_width = self.X_train.shape[2]
                self.channels = self.X_train.shape[3]
                print(f"Визначено параметри зображення з даних: {self.image_height}x{self.image_width}x{self.channels}")
            else:
                raise ValueError(
                    "Неможливо визначити параметри зображення. Задайте їх в task_spec_params або переконайтеся, "
                    "що дані мають правильний формат"
                )

        # Перевірка розмірності і перетворення, якщо потрібно
        if len(self.X_train.shape) == 2:  # Плоский формат (samples, features)
            expected_size = self.image_height * self.image_width * self.channels
            if self.X_train.shape[1] == expected_size:
                # Перетворення з плоского формату в формат зображень
                self.X_train = self.X_train.reshape(-1, self.image_height, self.image_width, self.channels)
                self.X_test = self.X_test.reshape(-1, self.image_height, self.image_width, self.channels)
                print(f"Дані перетворені з плоского формату в формат зображень: {self.X_train.shape}")
            else:
                raise ValueError(
                    f"Розмірність даних ({self.X_train.shape[1]}) не відповідає очікуваній "
                    f"({expected_size}) для зображень {self.image_height}x{self.image_width}x{self.channels}"
                )
        elif len(self.X_train.shape) == 3:  # Можливо, пропущена розмірність каналів
            # Додати розмірність каналів
            if self.channels == 1:
                self.X_train = self.X_train.reshape(-1, self.X_train.shape[1], self.X_train.shape[2], 1)
                self.X_test = self.X_test.reshape(-1, self.X_test.shape[1], self.X_test.shape[2], 1)
                print(f"Додано розмірність каналів до даних: {self.X_train.shape}")
            else:
                raise ValueError(
                    f"Дані мають 3 розмірності, але очікується {self.channels} каналів. "
                    f"Перевірте формат даних або параметри зображення."
                )
        elif len(self.X_train.shape) == 4:  # Очікуваний формат (samples, height, width, channels)
            # Перевірка відповідності розмірів
            if (self.X_train.shape[1] != self.image_height or
                    self.X_train.shape[2] != self.image_width or
                    self.X_train.shape[3] != self.channels):
                print(
                    f"Попередження: Розмірність даних ({self.X_train.shape[1]}x{self.X_train.shape[2]}x{self.X_train.shape[3]}) "
                    f"не відповідає заявленій ({self.image_height}x{self.image_width}x{self.channels}). "
                    f"Оновлення параметрів зображення.")
                self.image_height = self.X_train.shape[1]
                self.image_width = self.X_train.shape[2]
                self.channels = self.X_train.shape[3]
        else:
            raise ValueError(f"Непідтримувана розмірність даних: {self.X_train.shape}")

    def _normalize_images(self) -> None:
        """
        Нормалізація даних зображень.
        """
        # Перевірка типу даних
        if self.X_train.dtype != np.float32 and self.X_train.dtype != np.float64:
            # Нормалізація для uint8 (типовий формат зображень 0-255)
            if self.X_train.dtype == np.uint8:
                self.X_train = self.X_train.astype(np.float32) / 255.0
                self.X_test = self.X_test.astype(np.float32) / 255.0
                print("Дані зображень нормалізовані від 0-255 до 0-1")
            else:
                # Загальна нормалізація
                self.X_train = self.X_train.astype(np.float32)
                self.X_test = self.X_test.astype(np.float32)

                # Нормалізація до діапазону [0, 1]
                x_min = np.min(self.X_train)
                x_max = np.max(self.X_train)

                if x_min != 0 or x_max != 1:
                    if x_max > x_min:
                        self.X_train = (self.X_train - x_min) / (x_max - x_min)
                        self.X_test = (self.X_test - x_min) / (x_max - x_min)
                        print(f"Дані зображень нормалізовані з діапазону [{x_min}, {x_max}] до [0, 1]")

    def _process_labels(self) -> None:
        """
        Обробка міток для задач класифікації.
        """
        if self.task != TaskType.CLASSIFICATION:
            return

        # Перевірка наявності міток
        if self.y_train is None or self.y_test is None:
            raise ValueError("Мітки для класифікації не завантажені")

        # Перетворення міток в одновимірний масив, якщо потрібно
        if len(self.y_train.shape) > 1 and self.y_train.shape[1] == 1:
            self.y_train = self.y_train.flatten()
            self.y_test = self.y_test.flatten()

        # Перетворення до категоріальних міток, якщо це вказано в параметрах
        if self.task_spec_params.get("categorical_labels", False):
            # Перевірка, чи мітки вже в one-hot форматі
            if len(self.y_train.shape) == 1 or self.y_train.shape[1] == 1:
                # Перетворення до one-hot формату
                num_classes = self.task_spec_params.get("num_classes", None)
                if num_classes is None:
                    # Визначення числа класів з даних
                    num_classes = len(np.unique(np.concatenate([self.y_train, self.y_test])))

                # Перетворення до one-hot
                self.y_train = tf.keras.utils.to_categorical(self.y_train, num_classes)
                self.y_test = tf.keras.utils.to_categorical(self.y_test, num_classes)
                print(f"Мітки перетворені до one-hot формату з {num_classes} класами")

    def _compile_model(self, model) -> None:
        """
        Розширений метод компіляції моделі з додатковими параметрами для CNN.
        """
        # Викликаємо базовий метод
        super()._compile_model(model)

        # Перевірка моделі на відповідність CNN архітектурі (для переконання, що це CNN)
        has_conv_layer = False
        for layer in model.layers:
            if 'conv' in layer.__class__.__name__.lower():
                has_conv_layer = True
                break

        if not has_conv_layer:
            print(
                "Попередження: Модель не має згорткових шарів. Переконайтеся, що архітектура відповідає завданню CNN.")

    def _train_model(self, model) -> tf.keras.callbacks.History:
        """
        Розширений метод навчання моделі з додатковими параметрами для CNN.
        """
        # Додаємо специфічні для CNN параметри

        # Автоматичне визначення пакетів даних (batch size) в залежності від розміру зображень
        if 'fit_params' in self._params and 'batch_size' not in self._params['fit_params']:
            img_size = self.image_height * self.image_width * self.channels
            if img_size > 1000000:  # Великі зображення
                batch_size = 8
            elif img_size > 250000:  # Середні зображення
                batch_size = 16
            elif img_size > 50000:  # Малі зображення
                batch_size = 32
            else:  # Дуже малі зображення
                batch_size = 64

            if 'fit_params' not in self._params:
                self._params['fit_params'] = {}

            self._params['fit_params']['batch_size'] = batch_size
            print(f"Автоматично встановлено batch_size={batch_size} на основі розміру зображень")

        # Додавання колбеків для CNN моделей
        if 'fit_params' in self._params and 'callbacks' not in self._params['fit_params']:
            callbacks = []

            # EarlyStopping для запобігання перенавчання
            early_stopping = tf.keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True
            )
            callbacks.append(early_stopping)

            # ReduceLROnPlateau для адаптивного зменшення learning rate
            reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.2,
                patience=5,
                min_lr=1e-6
            )
            callbacks.append(reduce_lr)

            # ModelCheckpoint для збереження найкращої моделі
            self._params['fit_params']['callbacks'] = callbacks
            print("Додано стандартні колбеки для CNN: EarlyStopping, ReduceLROnPlateau")

        # Викликаємо базовий метод навчання
        return super()._train_model(model)

    def _validate_data(self) -> None:
        """
        Розширена перевірка даних для CNN.
        """
        # Викликаємо базовий метод
        super()._validate_data()

        # Додаткові перевірки для CNN
        if self.X_train is not None and len(self.X_train.shape) != 4:
            raise ValueError(f"Неправильна форма даних для CNN. Очікується 4D масив, отримано: {self.X_train.shape}")

        # Перевірка відповідності розмірів зображень
        if self.X_train is not None:
            if self.X_train.shape[1] != self.image_height or self.X_train.shape[2] != self.image_width or \
                    self.X_train.shape[3] != self.channels:
                raise ValueError(
                    f"Розмірність даних ({self.X_train.shape[1]}x{self.X_train.shape[2]}x{self.X_train.shape[3]}) "
                    f"не відповідає заявленій ({self.image_height}x{self.image_width}x{self.channels})"
                )