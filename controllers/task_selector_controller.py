from PyQt5.QtCore import Qt, pyqtSignal, QObject
from PyQt5.QtGui import QFont
from PyQt5.QtWidgets import QPushButton, QLabel, QVBoxLayout, QDialog, QRadioButton, QButtonGroup, QListWidget

from project.ui.task_selector.dynamic_button_dialog import DynamicButtonDialog
from project.ui.parameter_editor_widget import ParameterEditorWidget
from project.controllers.parameter_editor_controller import ParameterEditorController


class TaskSelectorController(QObject):
    request_models_dict = pyqtSignal(str)

    own_nn_selected = pyqtSignal()
    send_ml_model = pyqtSignal(str, object, object)
    def __init__(self, sender):
        super().__init__()
        self.sender = sender
        self.selected_task: str = ""
        #self.parameter_editor_controller = ParameterEditorController()
        self.editor = ParameterEditorWidget()

    def show_approach_selection(self):
        approach_dict = {
            "Classical ML": 1,
            "Neural Networks": 2
        }
        approach_dialog = DynamicButtonDialog("Select ML Approach", approach_dict, self.sender)
        result = approach_dialog.exec_()

        if result == 1:  # Classical ML
            self.show_learning_type_selection()
        elif result == 2:  # Neural Networks
            self.show_neural_network_selection()

    def show_learning_type_selection(self):
        learning_type_dict = {
            "Supervised Learning": 1,
            "Unsupervised Learning": 2
        }
        learning_type_dialog = DynamicButtonDialog("Select Learning Type", learning_type_dict, self.sender)
        result = learning_type_dialog.exec_()

        if result == 1:  # Supervised Learning
            self.show_supervised_task_selection()
        elif result == 2:  # Unsupervised Learning
            self.show_unsupervised_task_selection()

    def show_supervised_task_selection(self):
        tasks = {
            "Classification": 1,
            "Regression": 2
        }
        self.show_task_selection_dialog("Select Supervised Learning Task", tasks)

    def show_unsupervised_task_selection(self):
        tasks = {
            "Clustering": 1,
            "Dimensionality Reduction": 2,
            "Anomaly Detection": 3,
            "Density estimation": 4
        }
        self.show_task_selection_dialog("Select Unsupervised Learning Task", tasks)

    def show_neural_network_selection(self):
        nn_types = {
            "Scikit-learn MLP models": 1,
            "Import own": 2
        }
        self.show_task_selection_dialog("Select Neural Network Type", nn_types)

    def show_task_selection_dialog(self, title, tasks):
        dialog = QDialog(self.sender)
        dialog.setWindowTitle(title)
        dialog.setGeometry(200, 200, 400, 500)

        layout = QVBoxLayout(dialog)

        # Button Group for Tasks
        task_group = QButtonGroup(dialog)

        for task_text, task_value in tasks.items():
            radio_btn = QRadioButton(task_text)
            radio_btn.setMinimumHeight(100)
            radio_btn.setFont(QFont('Arial', 12))
            task_group.addButton(radio_btn, task_value)
            layout.addWidget(radio_btn)

        # Confirm Button
        confirm_btn = QPushButton("Confirm")
        confirm_btn.clicked.connect(lambda: self.handle_task_selection(task_group, dialog))
        layout.addWidget(confirm_btn)

        dialog.exec_()

    def handle_task_selection(self, group, dialog):
        selected_button = group.checkedButton()
        selected_task = group.checkedButton().text()
        self.selected_task = selected_task
        #TODO handle own nn selection
        if selected_button:
            # Create final placeholder page
            placeholder_dialog = QDialog(self.sender)
            placeholder_dialog.setWindowTitle("Selection Confirmation")
            placeholder_dialog.setGeometry(200, 200, 400, 300)

            layout = QVBoxLayout(placeholder_dialog)

            # Title with selected option

            title_label = QLabel(f"Selected Option: {selected_task}")

            title_label.setAlignment(Qt.AlignCenter)
            title_label.setFont(QFont('Arial', 14))
            layout.addWidget(title_label)
            dialog.accept()
            self.request_models_dict.emit(selected_task)


    def show_model_selection_dialog(self, models_dict):
        """
            Show a dialog with all models from the dictionary and return the selected model

            Args:
                models_dict: Dictionary with model names as keys and model classes as values

            Returns:
                Instance of the selected model or None if canceled
            """
        # Create dialog
        dialog = QDialog(self.sender)
        dialog.setWindowTitle("Select Model")
        dialog.setGeometry(200, 200, 500, 600)

        # Create layout
        layout = QVBoxLayout(dialog)

        # Add title
        title = QLabel("Select a Model")
        title.setAlignment(Qt.AlignCenter)
        title.setFont(QFont('Arial', 14, QFont.Bold))
        layout.addWidget(title)

        # Create list widget for models
        models_list = QListWidget()
        models_list.setFont(QFont('Arial', 12))

        # Sort model names for better usability
        model_names = sorted(models_dict.keys())

        # Add models to list
        for model_name in model_names:
            models_list.addItem(model_name)

        # Set minimum height for better visibility
        models_list.setMinimumHeight(400)
        layout.addWidget(models_list)

        # Add buttons
        buttons_layout = QVBoxLayout()

        select_btn = QPushButton("Select")
        select_btn.setFont(QFont('Arial', 12))
        select_btn.setMinimumHeight(40)

        buttons_layout.addWidget(select_btn)
        layout.addLayout(buttons_layout)

        # Initialize result
        selected_model = None
        def on_select():
            nonlocal selected_model
            current_item = models_list.currentItem()
            if current_item:
                model_name = current_item.text()
                model_class = models_dict[model_name]
                selected_model = model_class()
                dialog.accept()

        select_btn.clicked.connect(on_select)

        # Double click to select
        models_list.itemDoubleClicked.connect(lambda item: on_select())

        # Show dialog
        result = dialog.exec_()

        # Return selected model or None if canceled
        return selected_model if result == QDialog.Accepted else None


    def handle_models_dict_response(self, models_dict):
        choosen_model = (self.show_model_selection_dialog(models_dict))

        dialog = QDialog(self.sender)
        dialog.setWindowTitle("Set model parameters")
        dialog.setGeometry(200, 200, 400, 600)
        layout = QVBoxLayout(dialog)
        layout.addWidget(self.editor)

        self.editor.populate_table(choosen_model.get_params())

        save_btn = QPushButton('Save parameters')
        layout.addWidget(save_btn)

        params = {}
        def on_get_parameters():
            nonlocal params
            params = self.editor.get_current_parameters()
            dialog.accept()

        save_btn.clicked.connect(on_get_parameters)
        dialog.exec_()

        self.send_data_to_experiment_manager(choosen_model, params)

    def send_data_to_experiment_manager(self, model, params):
        print(model, params)
        self.send_ml_model.emit(self.selected_task, model, params)


#TODO maybe experiment data handler?