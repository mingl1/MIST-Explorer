from PyQt5 import  QtWidgets
import sys, app



if __name__ == "__main__":
    import sys
    app_ = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = app.Ui_MainWindow(MainWindow)
    MainWindow.show()
    sys.exit(app_.exec_())