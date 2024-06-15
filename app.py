from PyQt6 import QtCore, QtGui, QtWidgets
import tool, protein_selection, canvas, Dialogs, tifffile as tiff
class Ui_MainWindow(object):
    def __init__(self, MainWindow:QtWidgets.QMainWindow):
        QtGui.QImageReader.setAllocationLimit(0)
        self.MainWindow = MainWindow
        self.setupUi(MainWindow)

    def setupUi(self, MainWindow):

        self.MainWindow.setObjectName("MainWindow")
        self.MainWindow.resize(1280, 800)
        self.MainWindow.setMinimumSize(QtCore.QSize(1024, 768))
        self.centralwidget = QtWidgets.QWidget(self.MainWindow) # central widget inside main window
        self.centralwidget.setObjectName("centralwidget")
        self.central_widget_layout = QtWidgets.QHBoxLayout(self.centralwidget) # add a layout to centralwidget so window can resize properly
        self.central_widget_layout.setObjectName("central_widget_layout")
        self.main_layout = QtWidgets.QHBoxLayout() # main layout to add align canvas and the tab
        self.main_layout.setObjectName("main_layout") 
        
        # initialize tabs
        self.tabWidget = QtWidgets.QTabWidget(self.centralwidget)
        self.tabWidget.setMinimumSize(QtCore.QSize(400, 500))
        self.tabWidget.setMaximumSize(QtCore.QSize(500, 2000))
        self.tabWidget.setAutoFillBackground(False)
        self.tabWidget.setObjectName("tabWidget")

        ####### preprocess tab ###################################
        self.preprocessUISetup()

        ####### view tab #######################################
        self.tabWidget.addTab(self.preprocessing_tab, "")
        self.view_tab = QtWidgets.QWidget()
        self.view_tab.setObjectName("view_tab")
        self.protein_hlayout = QtWidgets.QHBoxLayout(self.view_tab)
        self.protein_hlayout.setObjectName("horizontalLayout_3")

        
        # add the protein selection boxes
        self.proteinWidget_main_vlayout = QtWidgets.QVBoxLayout()
        self.proteinWidget_main_vlayout.setObjectName("proteinWidget_main_vlayout")
        self.protein1_groupbox = protein_selection.Protein_Selector(self.view_tab)
        self.protein2_groupbox = protein_selection.Protein_Selector(self.view_tab)
        self.protein3_groupbox = protein_selection.Protein_Selector(self.view_tab)
        self.proteinWidget_main_vlayout.addWidget(self.protein1_groupbox)
        self.proteinWidget_main_vlayout.addWidget(self.protein2_groupbox)
        self.proteinWidget_main_vlayout.addWidget(self.protein3_groupbox)
        self.protein_hlayout.addLayout(self.proteinWidget_main_vlayout) # allow expansion of the groupbox when you resize
        self.tabWidget.addTab(self.view_tab, "")
        self.main_layout.addWidget(self.tabWidget)

        ########################################################

        # canvas
        self.canvas = canvas.ImageGraphicsView(self.centralwidget)
        self.canvas.setMinimumSize(QtCore.QSize(800, 500))
        self.canvas.setObjectName("canvas")


        self.small_view = canvas.ReferenceGraphicsView(self.MainWindow)
        self.small_view.setGeometry(600, 200, 200, 150)  # Position over the large view


        self.main_layout.addWidget(self.canvas) 
        self.central_widget_layout.addLayout(self.main_layout)
        self.MainWindow.setCentralWidget(self.centralwidget)


        # menubar
        self.menubar = QtWidgets.QMenuBar(self.MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 1061, 22))
        self.menubar.setObjectName("menubar")
        self.menuFile = QtWidgets.QMenu(self.menubar)
        self.menuFile.setObjectName("menuFile")
        self.menuOpen = QtWidgets.QMenu(self.menuFile)
        self.menuOpen.setObjectName("menuOpen")
        self.MainWindow.setMenuBar(self.menubar)


        # toolbar
        self.toolBar = QtWidgets.QToolBar(self.MainWindow)
        self.toolBar.setObjectName("toolBar")
        self.MainWindow.addToolBar(QtCore.Qt.ToolBarArea.TopToolBarArea, self.toolBar)
        self.actionOpen = tool.Tool(self.MainWindow, "actionOpen", "icons/folder.png")
        self.actionSaveAs = tool.Tool(self.MainWindow, "actionSaveAs", "icons/save-as.png")
        self.actionRotate = tool.Tool(self.MainWindow, "actionRotate", "icons/rotate-right.png")
        self.actionSelect_ROI = tool.Tool(self.MainWindow, "actionSelect_ROI",  "icons/rectangle.png")
        self.actionReset = tool.Tool(self.MainWindow, "actionReset", "icons/home.png")
        self.action_BC = tool.Tool(self.MainWindow, "actionBC", "icons/brightness.png")
        self.action_reference = tool.Tool(self.MainWindow, "action_reference")

        self.menuOpen.addAction(self.action_reference)
        self.menuOpen.addAction(self.actionOpen)
        self.menuFile.addAction(self.actionSaveAs)

        self.menubar.addAction(self.menuFile.menuAction())
        self.menuFile.addAction(self.menuOpen.menuAction())
        self.toolBar.addAction(self.actionSelect_ROI)
        self.toolBar.addAction(self.actionRotate)
        self.toolBar.addAction(self.actionReset)
        self.toolBar.addAction(self.action_BC)
        
        # status bar
        self.statusbar = QtWidgets.QStatusBar(self.MainWindow)
        self.statusbar.setObjectName("statusbar")
        self.MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        self.tabWidget.setCurrentIndex(0) # start from preprocessing tab


        self.rotation_dialog=None


        self.actionRotate.triggered.connect(self.createRotateDialog)
        self.actionReset.triggered.connect(self.canvas.resetImage)
        self.action_BC.triggered.connect(self.createBCDialog)
        self.action_reference.triggered.connect(self.on_action_reference_triggered)
        self.actionOpen.triggered.connect(self.on_actionOpen_triggered)

        QtCore.QMetaObject.connectSlotsByName(self.MainWindow)

    def preprocessUISetup(self):
        self.preprocessing_tab = QtWidgets.QWidget()
        self.preprocessing_tab.setMinimumSize(QtCore.QSize(0, 0))
        self.preprocessing_tab.setObjectName("preprocessing_tab")
        self.horizontalLayout = QtWidgets.QHBoxLayout(self.preprocessing_tab)
        self.horizontalLayout.setObjectName("horizontalLayout") 
        self.preprocessing_dockwidget_main_vlayout = QtWidgets.QVBoxLayout()
        self.preprocessing_dockwidget_main_vlayout.setObjectName("preprocessing_dockwidget_main_vlayout")
        self.thresholding_groupBox = QtWidgets.QGroupBox(self.preprocessing_tab)
        self.thresholding_groupBox.setMinimumSize(QtCore.QSize(300, 300))
        self.thresholding_groupBox.setObjectName("thresholding_groupBox")
        self.horizontalLayout_9 = QtWidgets.QHBoxLayout(self.thresholding_groupBox)
        self.horizontalLayout_9.setObjectName("horizontalLayout_9")
        self.thresholding_components_vlayout = QtWidgets.QVBoxLayout()
        self.thresholding_components_vlayout.setObjectName("thresholding_components_vlayout")
        self.thresholding_label1 = QtWidgets.QLabel(self.thresholding_groupBox)
        self.thresholding_label1.setObjectName("thresholding_label1")
        self.thresholding_components_vlayout.addWidget(self.thresholding_label1)
        self.thresholding_slider1 = QtWidgets.QSlider(self.thresholding_groupBox)
        self.thresholding_slider1.setMinimumSize(QtCore.QSize(100, 0))
        self.thresholding_slider1.setOrientation(QtCore.Qt.Orientation.Horizontal)
        self.thresholding_slider1.setObjectName("thresholding_slider1")
        self.thresholding_components_vlayout.addWidget(self.thresholding_slider1)
        self.thresholding_label2 = QtWidgets.QLabel(self.thresholding_groupBox)
        self.thresholding_label2.setObjectName("thresholding_label2")
        self.thresholding_components_vlayout.addWidget(self.thresholding_label2)
        self.thresholding_slider2 = QtWidgets.QSlider(self.thresholding_groupBox)
        self.thresholding_slider2.setMinimumSize(QtCore.QSize(100, 0))
        self.thresholding_slider2.setOrientation(QtCore.Qt.Orientation.Horizontal)
        self.thresholding_slider2.setObjectName("thresholding_slider2")
        self.thresholding_components_vlayout.addWidget(self.thresholding_slider2)
        self.thresholding_run_button = QtWidgets.QPushButton(self.thresholding_groupBox)
        self.thresholding_run_button.setObjectName("thresholding_run_button")
        self.thresholding_components_vlayout.addWidget(self.thresholding_run_button)
        self.horizontalLayout_9.addLayout(self.thresholding_components_vlayout)
        self.preprocessing_dockwidget_main_vlayout.addWidget(self.thresholding_groupBox)
        self.stardist_groupbox = QtWidgets.QGroupBox(self.preprocessing_tab)
        self.stardist_groupbox.setMinimumSize(QtCore.QSize(300, 300))
        self.stardist_groupbox.setObjectName("stardist_groupbox")
        self.horizontalLayout_4 = QtWidgets.QHBoxLayout(self.stardist_groupbox)
        self.horizontalLayout_4.setObjectName("horizontalLayout_4")
        self.stardist_components_vlayout = QtWidgets.QVBoxLayout()
        self.stardist_components_vlayout.setObjectName("stardist_components_vlayout")
        self.stardist_hlayout1 = QtWidgets.QHBoxLayout()
        self.stardist_hlayout1.setObjectName("stardist_hlayout1")
        self.stardist_label1 = QtWidgets.QLabel(self.stardist_groupbox)
        self.stardist_label1.setObjectName("stardist_label1")
        self.stardist_hlayout1.addWidget(self.stardist_label1)
        self.stardist_combobox1 = QtWidgets.QComboBox(self.stardist_groupbox)
        self.stardist_combobox1.setObjectName("stardist_combobox1")
        self.stardist_hlayout1.addWidget(self.stardist_combobox1)
        self.stardist_components_vlayout.addLayout(self.stardist_hlayout1)
        self.stardist_hlayout2 = QtWidgets.QHBoxLayout()
        self.stardist_hlayout2.setObjectName("stardist_hlayout2")
        self.stardist_label2 = QtWidgets.QLabel(self.stardist_groupbox)
        self.stardist_label2.setObjectName("stardist_label2")
        self.stardist_hlayout2.addWidget(self.stardist_label2)
        self.stardist_spinbox1 = QtWidgets.QDoubleSpinBox(self.stardist_groupbox)
        self.stardist_spinbox1.setProperty("value", 0.5)
        self.stardist_spinbox1.setObjectName("stardist_spinbox1")
        self.stardist_hlayout2.addWidget(self.stardist_spinbox1)
        self.stardist_components_vlayout.addLayout(self.stardist_hlayout2)
        self.stardist_hlayout3 = QtWidgets.QHBoxLayout()
        self.stardist_hlayout3.setObjectName("stardist_hlayout3")
        self.stardist_label3 = QtWidgets.QLabel(self.stardist_groupbox)
        self.stardist_label3.setObjectName("stardist_label3")
        self.stardist_hlayout3.addWidget(self.stardist_label3)
        self.stardist_spinbox2 = QtWidgets.QDoubleSpinBox(self.stardist_groupbox)
        self.stardist_spinbox2.setProperty("value", 0.5)
        self.stardist_spinbox2.setObjectName("stardist_spinbox2")
        self.stardist_hlayout3.addWidget(self.stardist_spinbox2)
        self.stardist_components_vlayout.addLayout(self.stardist_hlayout3)
        self.stardist_hlayout4 = QtWidgets.QHBoxLayout()
        self.stardist_hlayout4.setObjectName("stardist_hlayout4")
        self.stardist_label4 = QtWidgets.QLabel(self.stardist_groupbox)
        self.stardist_label4.setObjectName("stardist_label4")
        self.stardist_hlayout4.addWidget(self.stardist_label4)
        self.stardist_spinbox3 = QtWidgets.QDoubleSpinBox(self.stardist_groupbox)
        self.stardist_spinbox3.setProperty("value", 0.5)
        self.stardist_spinbox3.setObjectName("stardist_spinbox3")
        self.stardist_hlayout4.addWidget(self.stardist_spinbox3)
        self.stardist_components_vlayout.addLayout(self.stardist_hlayout4)
        self.stardist_hlayout5 = QtWidgets.QHBoxLayout()
        self.stardist_hlayout5.setObjectName("stardist_hlayout5")
        self.stardist_label5 = QtWidgets.QLabel(self.stardist_groupbox)
        self.stardist_label5.setObjectName("stardist_label5")
        self.stardist_hlayout5.addWidget(self.stardist_label5)
        self.stardist_spinbox4 = QtWidgets.QDoubleSpinBox(self.stardist_groupbox)
        self.stardist_spinbox4.setProperty("value", 0.4)
        self.stardist_spinbox4.setObjectName("stardist_spinbox4")
        self.stardist_hlayout5.addWidget(self.stardist_spinbox4)
        self.stardist_components_vlayout.addLayout(self.stardist_hlayout5)
        self.stardist_hlayout6 = QtWidgets.QHBoxLayout()
        self.stardist_hlayout6.setObjectName("stardist_hlayout6")
        self.stardist_label6 = QtWidgets.QLabel(self.stardist_groupbox)
        self.stardist_label6.setObjectName("stardist_label6")
        self.stardist_hlayout6.addWidget(self.stardist_label6)
        self.stardist_spinbox5 = QtWidgets.QSpinBox(self.stardist_groupbox)
        self.stardist_spinbox5.setObjectName("stardist_spinbox5")
        self.stardist_hlayout6.addWidget(self.stardist_spinbox5)
        self.stardist_components_vlayout.addLayout(self.stardist_hlayout6)
        self.stardist_run_button = QtWidgets.QPushButton(self.stardist_groupbox)
        self.stardist_run_button.setObjectName("stardist_run_button")
        self.stardist_components_vlayout.addWidget(self.stardist_run_button)
        self.horizontalLayout_4.addLayout(self.stardist_components_vlayout)
        self.preprocessing_dockwidget_main_vlayout.addWidget(self.stardist_groupbox)
        self.horizontalLayout.addLayout(self.preprocessing_dockwidget_main_vlayout)

    def createRotateDialog(self):
        self.rotation_dialog = Dialogs.RotateDialog(self.MainWindow, self.canvas, self.canvas.pixmap)

    def createBCDialog(self):
        self.BC_dialog = Dialogs.BrightnessContrastDialog(self.MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        self.MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.thresholding_groupBox.setTitle(_translate("MainWindow", "Thresholding"))
        self.thresholding_label1.setText(_translate("MainWindow", "Block Size"))
        self.thresholding_label2.setText(_translate("MainWindow", "Offset"))
        self.thresholding_run_button.setText(_translate("MainWindow", "Confirm"))
        self.stardist_groupbox.setTitle(_translate("MainWindow", "Stardist"))
        self.stardist_label1.setText(_translate("MainWindow", "Pre-trained Model"))
        self.stardist_label2.setText(_translate("MainWindow", "Percentile Low"))
        self.stardist_label3.setText(_translate("MainWindow", "Percentile High"))
        self.stardist_label4.setText(_translate("MainWindow", "Probability/ Score Threshold"))
        self.stardist_label5.setText(_translate("MainWindow", "Overlap Threshold"))
        self.stardist_label6.setText(_translate("MainWindow", "Number of Tiles"))
        self.stardist_run_button.setText(_translate("MainWindow", "Run"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.preprocessing_tab), _translate("MainWindow", "Preprocessing"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.view_tab), _translate("MainWindow", "View"))
        self.menuFile.setTitle(_translate("MainWindow", "File"))
        self.toolBar.setWindowTitle(_translate("MainWindow", "toolBar"))
        self.action_reference.setText(_translate("MainWindow", "Open Reference"))
        self.menuOpen.setTitle(_translate("MainWindow", "Open"))
        self.actionOpen.setText(_translate("MainWindow", "Open Image"))
        self.actionSaveAs.setText(_translate("MainWindow", "Save As..."))
        self.actionRotate.setText(_translate("MainWindow", "Rotate"))
        self.actionRotate.setToolTip(_translate("MainWindow", "Rotate"))
        self.actionSelect_ROI.setText(_translate("MainWindow", "Select ROI"))
        self.actionSelect_ROI.setToolTip(_translate("MainWindow", "Select ROI"))
        self.actionReset.setText(_translate("MainWindow", "Reset"))
        self.actionReset.setToolTip(_translate("MainWindow", "Reset Image"))


    def openFileDialog(self, view):
        fileName, _ = QtWidgets.QFileDialog.getOpenFileName(None, "Open Image File", "", "Images (*.png *.xpm *.jpg *.bmp *.gif *.tif);;All Files (*)")
        
        if fileName:
            # if fileName.endswith((".tiff", ".tif")):
            #     image = tiff.imread(fileName)
            #     num_channels, _, _ = image.shape
            pixmap = QtGui.QPixmap(fileName)
            if not pixmap.isNull():
                view.addImage(pixmap)
    def on_action_reference_triggered(self):
       self.openFileDialog(self.small_view)
    def on_actionOpen_triggered(self):
       self.openFileDialog(self.canvas)
       