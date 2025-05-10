from typing import Dict, Any

from project.logic.experiment.input_data_params import InputDataParams
from project.logic.modules.nn_model_types import NNModelType, TaskType


class NeuralNetInputDataParams(InputDataParams):
    """
    Параметри вхідних даних для нейронних мереж.
    Зберігає посилання на файли моделі та додаткові параметри.
    """

    def __init__(self):
        super().__init__()

        # Шляхи до файлів моделі
        self.model_file_path = ''
        self.weights_file_path = ''
        self.model_config_path = ''

        # Додаткові параметри для нейронних мереж
        self.load_type = ''

        self.text_directory = ''
        self.image_directory = ''

    def to_dict(self) -> Dict[str, Any]:
        """
        Конвертує параметри у словник, включаючи параметри батьківського класу.
        """
        data = super().to_dict()

        # Додаємо параметри, специфічні для нейронних мереж
        data.update({
            'model_file_path': self.model_file_path,
            'weights_file_path': self.weights_file_path,
            'model_config_path': self.model_config_path,
            'load_type': self.load_type,
        })

        return data
