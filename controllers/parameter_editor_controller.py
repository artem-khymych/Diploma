class ParameterEditorController:
    """Контролер для віджета ParameterEditorWidget"""
    def __init__(self):
        self.view = None
    def set_view(self, view):
        self.view = view
        #self.view.parametersChanged.connect(self.on_parameters_changed)
    def show(self, params_dict):
        """Завантажує параметри у віджет"""
        if self.view:
            self.view.populate_table(params_dict)

    def on_parameters_changed(self, updated_params):
        """Обробник події зміни параметрів"""
        # Тут можна додати додаткову логіку перед подальшою обробкою
        print("Контролер: Параметри оновлено:", updated_params)
        return updated_params

    def get_current_parameters(self):
        """Повертає поточні параметри з віджета"""
        if self.view:
            return self.view.get_current_parameters()
        return {}
