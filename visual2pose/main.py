from PyQt5.QtWidgets import QApplication
import sys
from handpose.ui.qt_window import ControlWindow


if __name__ == "__main__":
    app = QApplication(sys.argv)
    main = ControlWindow()
    main.show()
    sys.exit(app.exec_())

