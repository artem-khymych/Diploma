import sys
from PyQt5.QtWidgets import QApplication
from project.controllers.main_controller import MainController
from project.ui.styles.styles import setup_app_style


def main():
    app = QApplication(sys.argv)
    controller = MainController()
    #AppStyle.apply_base_style(app)
    setup_app_style(app)
    controller.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
