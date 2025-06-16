"""Class to handle signal connections"""

import ui.Dialogs as Dialogs, numpy as np, cv2, core.canvas, core.stardist, core.cell_intensity, core.register
from ui.app import Ui_MainWindow
from PyQt6.QtWidgets import QFileDialog, QMessageBox
from PyQt6.QtGui import QPixmap
from PyQt6.QtCore import pyqtSignal
from PIL import Image
import uuid


class Controller:

    _instance = None
    controllerSignal = pyqtSignal(object)

    def __new__(cls, app):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self, app: Ui_MainWindow):
        if hasattr(self, "_initialized") and self._initialized:
            return
        self._initialized = True
        self.image_count = 0
        self.model_canvas = core.canvas.ImageGraphicsView()
        self.model_stardist = core.stardist.StarDist()
        self.model_register = core.register.Register()
        self.model_cellIntensity = core.cell_intensity.CellIntensity()
        self.reference_view = core.canvas.ReferenceGraphicsView()
        self.view = app
        self.openFilesDialog = None
        self.view.images_tab.set_model_canvas(self.model_canvas)
        self.signal_manager = SignalConnectionManager(self)
        self.signal_manager.setup_all_connections()

        # Handle initial arguments
        initial_args = vars(app.args)
        if initial_args["image"] is not None:
            self.model_canvas.add_to_canvas(initial_args["image"])
        if initial_args["reference"] is not None:
            self.reference_view.add_to_canvas(initial_args["reference"])

    def handleError(self, error_message):
        QMessageBox.critical(self.view, "Error", error_message)

    def save_pixmap_as_image(self, pixmap: QPixmap, filename: str):
        qimage = pixmap.toImage()
        # Convert QImage to numpy array
        width = qimage.width()
        height = qimage.height()
        ptr = qimage.bits()
        ptr.setsize(height * width * 4)
        arr = np.array(ptr).reshape(height, width, 4)  # 4 for RGBA

        # Save numpy array as an image file using OpenCV
        cv2.imwrite(filename, cv2.cvtColor(arr, cv2.COLOR_BGRA2BGRA))

    def pixmap_to_image(self, pixmap: QPixmap):

        if pixmap == None:
            return None
        qimage = pixmap.toImage()
        width = qimage.width()
        height = qimage.height()
        ptr = qimage.bits()
        ptr.setsize(height * width * 4)
        arr = np.array(ptr).reshape(height, width, 4)  # 4 for RGBA

        # Convert from BGRA to RGB by dropping alpha channel and reversing BGR
        if arr.shape[2] == 4:  # If the image has an alpha channel
            arr = arr[:, :, :3]  # Remove the alpha channel

        # Convert BGR to RGB (OpenCV uses BGR, but most other libraries use RGB)
        arr = arr[:, :, ::-1]

        return arr

    def controlSave(self):

        pm = self.model_canvas.pixmap
        print(pm)
        # qimage = pm.toImage()
        if pm != None:
            im = self.pixmap_to_image(pm)

            file_name, _ = QFileDialog.getSaveFileName(
                None, "Save File", "image.png", "*.png;;*.jpg;;*.tif;; All Files(*)"
            )
            if file_name:
                print(file_name)
                Image.fromarray(im).save(file_name)

            else:
                return False

        else:
            self.handleError("No image in canvas, please load image")

    def openFileDialog(self, viewer):
        file_name, _ = QFileDialog.getOpenFileName(
            None, "Open Image File", "", "Images (*.png *.jpg *.tif);;All Files (*)"
        )
        if file_name:
            viewer.add_to_canvas(file_name)

    def on_action_reference_triggered(self):
        self.openFileDialog(self.reference_view)

    def on_actionOpen_triggered(self):
        self.openFileDialog(self.model_canvas)

    def handle_new_image(self, data, file_name):
        storage_item = {}
        storage_item["name"] = file_name.split("/")[-1]
        self.image_count += 1
        storage_item["data"] = data
        my_uuid = str(uuid.uuid4())
        self.view.images_tab.add_to_storage(my_uuid, storage_item)
        self.view.images_tab.add_item(my_uuid)


class SignalConnectionManager:
    """Manages signal connections"""

    def __init__(self, controller: Controller):
        self.c = controller

    def setup_all_connections(self):
        """Set up all signal-slot connections with full IntelliSense support"""
        self._setup_alignment_connections()
        self._setup_menubar_connections()
        self._setup_toolbar_connections()
        self._setup_image_handling_connections()
        self._setup_canvas_connections()
        self._setup_crop_connections()
        self._setup_transform_connections()
        self._setup_stardist_connections()
        self._setup_registration_connections()
        self._setup_cell_intensity_connections()
        self._setup_image_broadcast_connections()
        self._setup_misc_connections()

    def _setup_alignment_connections(self):
        """Alignment section signal connections"""
        self.c.view.images_tab.tissue_target_selected.connect(
            self.c.view.cell_layer_alignment.set_target_image
        )
        self.c.view.images_tab.tissue_unaligned_selected.connect(
            self.c.view.cell_layer_alignment.set_unaligned_image
        )
        self.c.view.cell_layer_alignment.alignmentCompleteSignal.connect(
            self.c.handle_new_image
        )
        self.c.view.cell_layer_alignment.loadOnCanvasSignal.connect(
            self.c.model_canvas.add_to_canvas
        )
        self.c.view.cell_layer_alignment.aligner.progress.connect(
            self.c.view.update_progress_bar
        )

    def _setup_menubar_connections(self):
        """Menu bar signal connections"""
        self.c.view.menuBarUI.actionOpenReference.triggered.connect(
            self.c.on_action_reference_triggered
        )
        self.c.view.menuBarUI.actionOpen.triggered.connect(
            self.c.on_actionOpen_triggered
        )
        self.c.view.menuBarUI.actionSaveAs.triggered.connect(self.c.view.save)

    def _setup_toolbar_connections(self):
        """Toolbar signal connections"""
        self.c.view.toolBarUI.actionReset.triggered.connect(
            self.c.model_canvas.reset_image
        )
        self.c.view.toolBarUI.channelChanged.connect(self.c.model_canvas.swap_channel)
        self.c.view.toolBarUI.contrastSlider.valueChanged.connect(
            self.c.model_canvas.update_contrast
        )
        self.c.view.toolBarUI.cmapChanged.connect(self.c.model_canvas.update_image)
        self.c.view.toolBarUI.auto_contrast_button.clicked.connect(
            self.c.model_canvas.auto_contrast
        )

    def _setup_image_handling_connections(self):
        """Image loading and display connections"""
        self.c.view.canvas.imageDropped.connect(self.c.model_canvas.add_to_canvas)
        self.c.view.small_view.imageDropped.connect(self.c.reference_view.add_to_canvas)
        self.c.reference_view.update_reference.connect(self.c.view.small_view.display)
        self.c.model_canvas.newImageAdded.connect(self.c.view.canvas.addNewImage)
        self.c.view.view_tab.changePix.connect(self.c.view.canvas.addNewImage)
        self.c.model_canvas.canvasUpdated.connect(self.c.view.canvas.updateCanvas)
        self.c.model_canvas.update_manager.connect(self.c.handle_new_image)
        self.c.reference_view.update_manager.connect(self.c.handle_new_image)

    def _setup_canvas_connections(self):
        """Canvas-related signal connections"""
        self.c.model_canvas.updateProgress.connect(self.c.view.update_progress_bar)
        self.c.model_canvas.errorSignal.connect(self.c.handleError)
        self.c.view.canvas.showCrop.connect(self.c.model_canvas.crop)
        self.c.model_canvas.cropSignal.connect(self.c.view.canvas.set_crop_status)
        self.c.model_canvas.update_cmap.connect(
            self.c.view.toolBarUI.update_cmap_selector
        )
        self.c.model_canvas.changeSlider.connect(
            self.c.view.toolBarUI.update_contrast_slider
        )
        self.c.model_canvas.fill_metadata.connect(self.c.view.get_metadata)

        # Crop visibility toggle
        self.c.model_canvas.cropSignal.connect(
            lambda x: self.c.view.small_view.setVisible(not x)
        )

    def _setup_crop_connections(self):
        """Crop operation connections"""
        self.c.view.crop_groupbox.crop_button.triggered.connect(
            lambda: self.c.view.canvas.set_crop_status(True)
        )
        self.c.view.crop_groupbox.crop_button.triggered.connect(
            lambda: self.c.view.small_view.setVisible(False)
        )
        self.c.view.crop_groupbox.cancel_crop_button.triggered.connect(
            lambda: self.c.view.canvas.set_crop_status(False)
        )
        self.c.view.crop_groupbox.cancel_crop_button.triggered.connect(
            lambda: self.c.view.small_view.setVisible(True)
        )

    def _setup_transform_connections(self):
        """Image transformation connections"""
        # Flip operations
        self.c.view.canvas.requestFlipHorizontal.connect(
            self.c.model_canvas.flip_horizontal
        )
        self.c.view.canvas.requestFlipVertical.connect(
            self.c.model_canvas.flip_vertical
        )

        # Rotation
        self.c.view.rotate_groupbox.rotate_confirm.pressed.connect(
            lambda: self.c.model_canvas.rotateImage(
                self.c.view.rotate_groupbox.rotate_line_edit.text()
            )
        )

        # Gaussian blur
        self.c.view.gaussian_blur.confirm.clicked.connect(
            lambda: self.c.model_canvas.blur_layer(0, confirm=True)
        )
        self.c.view.gaussian_blur.slider.doubleValueChanged.connect(
            self.c.view.gaussian_blur.update_slider_label
        )
        self.c.view.gaussian_blur.slider.doubleValueChanged.connect(
            self.c.model_canvas.blur_layer
        )

    def _setup_stardist_connections(self):
        """StarDist-related connections"""
        # Parameter connections
        self.c.view.stardist_groupbox.stardist_channel_selector.currentTextChanged.connect(
            self.c.model_stardist.setChannel
        )
        self.c.view.stardist_groupbox.stardist_pretrained_models.currentTextChanged.connect(
            self.c.model_stardist.setModel
        )
        self.c.view.stardist_groupbox.percentile_high.valueChanged.connect(
            self.c.model_stardist.setPercentileHigh
        )
        self.c.view.stardist_groupbox.percentile_low.valueChanged.connect(
            self.c.model_stardist.setPercentileLow
        )
        self.c.view.stardist_groupbox.prob_threshold.valueChanged.connect(
            self.c.model_stardist.setProbThresh
        )
        self.c.view.stardist_groupbox.nms_threshold.valueChanged.connect(
            self.c.model_stardist.setNMSThresh
        )
        self.c.view.stardist_groupbox.n_tiles.valueChanged.connect(
            self.c.model_stardist.setNumberTiles
        )
        self.c.view.stardist_groupbox.radius.valueChanged.connect(
            self.c.model_stardist.setDilationRadius
        )

        # Execution and results
        self.c.view.stardist_groupbox.stardist_run_button.pressed.connect(
            self.c.model_stardist.runStarDist
        )
        self.c.model_stardist.stardistDone.connect(
            self.c.model_canvas.loadStardistLabels
        )
        self.c.model_stardist.stardistDone.connect(
            self.c.model_cellIntensity.loadStardistLabels
        )
        self.c.model_stardist.errorSignal.connect(self.c.handleError)
        self.c.model_stardist.progress.connect(self.c.view.update_progress_bar)
        self.c.view.stardist_groupbox.save_button.clicked.connect(
            self.c.model_stardist.saveImage
        )
        self.c.view.stardist_groupbox.cancel_button.clicked.connect(
            self.c.model_stardist.cancel
        )

    def _setup_registration_connections(self):
        """Registration-related connections"""
        # Parameters
        self.c.view.register_groupbox.alignment_layer.currentTextChanged.connect(
            self.c.model_register.setAlignmentLayer
        )
        self.c.view.register_groupbox.protein_cell_layer.currentTextChanged.connect(
            self.c.model_register.setCellLayer
        )
        self.c.view.register_groupbox.intensity_layer.currentTextChanged.connect(
            self.c.model_register.setProteinDetectionLayer
        )
        self.c.view.register_groupbox.overlap.valueChanged.connect(
            self.c.model_register.setOverlap
        )
        self.c.view.register_groupbox.max_size.valueChanged.connect(
            self.c.model_register.setMaxSize
        )
        self.c.view.register_groupbox.num_tiles.valueChanged.connect(
            self.c.model_register.setNumTiles
        )
        self.c.view.register_groupbox.has_blue_color.currentTextChanged.connect(
            self.c.model_register.hasBlueColor
        )

        # Execution
        self.c.view.register_groupbox.run_button.clicked.connect(
            self.c.model_register.run_registration
        )
        self.c.model_register.cell_image_signal.connect(
            self.c.model_stardist.loadCellImage
        )
        self.c.model_register.protein_signal_arr_signal.connect(
            self.c.model_cellIntensity.loadProteinSignalArray
        )
        self.c.model_register.progress.connect(self.c.view.update_progress_bar)
        self.c.view.register_groupbox.cancel_button.clicked.connect(
            self.c.model_register.cancel
        )

    def _setup_cell_intensity_connections(self):
        """Cell intensity-related connections"""
        self.c.view.cellIntensity_groupbox.bead_data.clicked.connect(
            self.c.view.cellIntensity_groupbox.loadBeadData
        )
        self.c.view.cellIntensity_groupbox.color_code.clicked.connect(
            self.c.view.cellIntensity_groupbox.loadColorCode
        )
        self.c.view.cellIntensity_groupbox.emitBeadData.connect(
            self.c.model_cellIntensity.getBeadData
        )
        self.c.view.cellIntensity_groupbox.emitColorCode.connect(
            self.c.model_cellIntensity.getColorCode
        )
        self.c.view.cellIntensity_groupbox.num_cycles.valueChanged.connect(
            self.c.model_cellIntensity.setNumDecodingCycles
        )
        self.c.view.cellIntensity_groupbox.num_layers_each.valueChanged.connect(
            self.c.model_cellIntensity.setNumDecodingColors
        )
        self.c.view.cellIntensity_groupbox.radius_fg.valueChanged.connect(
            self.c.model_cellIntensity.setRadiusFG
        )
        self.c.view.cellIntensity_groupbox.radius_bg.valueChanged.connect(
            self.c.model_cellIntensity.setRadiusBG
        )
        self.c.view.cellIntensity_groupbox.run_button.clicked.connect(
            self.c.model_cellIntensity.generateCellIntensityTable
        )
        self.c.model_cellIntensity.errorSignal.connect(self.c.handleError)
        self.c.view.cellIntensity_groupbox.save_button.clicked.connect(
            self.c.model_cellIntensity.save_cell_data
        )
        self.c.model_cellIntensity.progress.connect(self.c.view.update_progress_bar)
        self.c.view.cellIntensity_groupbox.cancel_button.clicked.connect(
            self.c.model_cellIntensity.cancel
        )

    def _setup_image_broadcast_connections(self):
        """Image signal broadcast to multiple targets"""

        image_signal = self.c.model_canvas.image_signal
        image_signal.connect(self.c.view.toolBarUI.updateChannelSelector)
        image_signal.connect(self.c.view.stardist_groupbox.updateChannelSelector)
        image_signal.connect(self.c.view.register_groupbox.updateChannelSelector)
        image_signal.connect(self.c.view.canvas.loadChannels)
        image_signal.connect(self.c.model_stardist.updateChannels)
        image_signal.connect(self.c.view.gaussian_blur.updateChannelSelector)
        image_signal.connect(self.c.model_register.update_protein_channels)

        ref_image = self.c.reference_view.image_signal
        ref_image.connect(self.c.model_register.update_reference_channels)
        ref_image.connect(self.c.view.small_view.load_channels)

    def _setup_misc_connections(self):
        """Miscellaneous connections"""
        self.c.view.saveSignal.connect(self.c.controlSave)
        self.c.view.view_tab.progress.connect(self.c.view.update_progress_bar)
        self.c.view.stackedWidget.currentChanged.connect(
            lambda x: self.c.view.small_view.setVisible(x == 1)
        )
