'''Class to handle signal connections'''

import ui.Dialogs as Dialogs, numpy as np, cv2, core.canvas, core.stardist, core.cell_intensity, core.register
from ui.app import Ui_MainWindow
from PyQt6.QtWidgets import QFileDialog, QMessageBox
from PyQt6.QtGui import QPixmap
from PyQt6.QtCore import pyqtSignal
from PIL import Image
import uuid
class Controller:
    controllerSignal = pyqtSignal(object)
    def __init__(self, app: Ui_MainWindow):
        # temporary, can change later to be based on reference's and model's image count
        # to get "Image 0", "Reference 0" differentiation
        self.image_count = 0
        self.model_canvas = core.canvas.ImageGraphicsView()
        self.model_stardist = core.stardist.StarDist()
        self.model_register = core.register.Register()
        self.model_cellIntensity = core.cell_intensity.CellIntensity()
        self.reference_view = core.canvas.ReferenceGraphicsView()
        self.view = app


        # Alignment Section signals
        self.view.images_tab.tissue_target_selected.connect(self.view.cell_layer_alignment.set_target_image)
        self.view.images_tab.tissue_unaligned_selected.connect(self.view.cell_layer_alignment.set_unaligned_image)
        self.view.cell_layer_alignment.alignmentCompleteSignal.connect(self.handle_new_image)
        self.view.cell_layer_alignment.replaceLayerSignal.connect(self.view.replace_layer_in_canvas)
        self.view.cell_layer_alignment.loadOnCanvasSignal.connect(self.view.canvas.addNewImage)
        self.view.cell_layer_alignment.aligner.progress.connect(self.view.update_progress_bar)

        self.model_canvas.update_manager.connect(self.handle_new_image)
        self.reference_view.update_manager.connect(self.handle_new_image)
        self.openFilesDialog = None
        #menubar signals
        # self.view.menubar.actionOpenFiles.triggered.connect(self.on_action_openFiles_triggered)
        self.view.menubar.actionOpenReference.triggered.connect(self.on_action_reference_triggered)
        self.view.menubar.actionOpen.triggered.connect(self.on_actionOpen_triggered)
        
        # self.view.connect()
        self.view.register_groupbox.has_blue_color.currentTextChanged.connect(self.model_register.hasBlueColor)
        #toolbar signals
        self.view.toolBar.actionReset.triggered.connect(self.model_canvas.reset_image)
        self.view.toolBar.channelChanged.connect(self.model_canvas.swap_channel)
        self.view.toolBar.contrastSlider.valueChanged.connect(self.model_canvas.update_contrast)
        self.view.toolBar.cmapChanged.connect(self.model_canvas.update_image) # change cmap in model_canvas then send to view.canvas for display


        self.view.toolBar.auto_contrast_button.clicked.connect(self.model_canvas.auto_contrast)

        self.view.canvas.imageDropped.connect(self.model_canvas.addImage)
        self.view.small_view.imageDropped.connect(self.reference_view.addImage)
        self.reference_view.update_reference.connect(self.view.small_view.display)
        self.model_canvas.newImageAdded.connect(self.view.canvas.addNewImage) # loading a new image
        self.view.view_tab.changePix.connect(self.view.canvas.addNewImage) # loading a new image
        self.model_canvas.canvasUpdated.connect(self.view.canvas.updateCanvas) # operation done on current image
        self.reference_view.multi_layer.connect(self.model_register.update_reference_channels)
        self.model_canvas.multi_layer.connect(self.model_register.update_protein_channels)
        self.reference_view.multi_layer.connect(self.view.small_view.load_channels)

        self.model_canvas.multi_layer.connect(self.view.toolBar.updateChannelSelector) # update toolbar channel combobox
        self.model_canvas.multi_layer.connect(self.view.stardist_groupbox.updateChannelSelector) #update stardist channel combobox
        self.model_canvas.multi_layer.connect(self.view.register_groupbox.updateChannelSelector) #update register
        self.model_canvas.multi_layer.connect(self.view.canvas.loadChannels) #this is for cropping because cropping function is in canvas ui
        self.model_canvas.multi_layer.connect(self.model_stardist.updateChannels) #pass the channels for stardist processing
        self.model_canvas.multi_layer.connect(self.view.gaussian_blur.updateChannelSelector)

        self.model_canvas.single_layer.connect(self.view.toolBar.clearChannelSelector) #if new image loaded is not multilayer, then we clear the channel selector since there's only one channel
        self.model_canvas.single_layer.connect(self.view.stardist_groupbox.clearChannelSelector)
        self.model_canvas.single_layer.connect(self.model_stardist.setImageToProcess) 
        
        self.model_canvas.updateProgress.connect(self.view.update_progress_bar) # loading image progress bar
        self.model_canvas.errorSignal.connect(self.handleError)
        self.view.canvas.showCrop.connect(self.model_canvas.crop)
        self.model_canvas.cropSignal.connect(self.view.canvas.set_crop_status)
        self.model_canvas.cropSignal.connect(lambda x: self.view.small_view.setVisible(not x))

        self.model_canvas.update_cmap.connect(self.view.toolBar.update_cmap_selector)
        self.model_canvas.changeSlider.connect(self.view.toolBar.update_contrast_slider)
        self.model_canvas.fill_metadata.connect(self.view.get_metadata)
        # crop signals
        self.view.crop_groupbox.crop_button.triggered.connect(lambda: self.view.canvas.set_crop_status(True)) 
        self.view.crop_groupbox.crop_button.triggered.connect(lambda: self.view.small_view.setVisible(False)) 

        self.view.crop_groupbox.cancel_crop_button.triggered.connect(lambda: self.view.canvas.set_crop_status(False))
        self.view.crop_groupbox.cancel_crop_button.triggered.connect(lambda: self.view.small_view.setVisible(True))

        # flip signals
        self.view.canvas.requestFlipHorizontal.connect(self.model_canvas.flip_horizontal)
        self.view.canvas.requestFlipVertical.connect(self.model_canvas.flip_vertical)

        # confirm rotate signal

        self.view.saveSignal.connect(self.controlSave)

        self.view.rotate_groupbox.rotate_confirm.pressed.connect(lambda: self.model_canvas.rotateImage(self.view.rotate_groupbox.rotate_line_edit.text()))

        self.view.gaussian_blur.confirm.clicked.connect(lambda: self.model_canvas.blur_layer(0, confirm=True))
        self.view.gaussian_blur.slider.doubleValueChanged.connect(self.view.gaussian_blur.update_slider_label)
        self.view.gaussian_blur.slider.doubleValueChanged.connect(self.model_canvas.blur_layer)

        #stardist signals
        #change params
        self.view.stardist_groupbox.stardist_channel_selector.currentTextChanged.connect(self.model_stardist.setChannel)
        self.view.stardist_groupbox.stardist_pretrained_models.currentTextChanged.connect(self.model_stardist.setModel)
        self.view.stardist_groupbox.percentile_high.valueChanged.connect(self.model_stardist.setPercentileHigh)
        self.view.stardist_groupbox.percentile_low.valueChanged.connect(self.model_stardist.setPercentileLow)
        self.view.stardist_groupbox.prob_threshold.valueChanged.connect(self.model_stardist.setProbThresh)
        self.view.stardist_groupbox.nms_threshold.valueChanged.connect(self.model_stardist.setNMSThresh)
        self.view.stardist_groupbox.n_tiles.valueChanged.connect(self.model_stardist.setNumberTiles)

        self.view.stardist_groupbox.radius.valueChanged.connect(self.model_stardist.setDilationRadius)


        #run stardist
        self.view.stardist_groupbox.stardist_run_button.pressed.connect(self.model_stardist.runStarDist)

        # display stardist result
        self.model_stardist.stardistDone.connect(self.model_canvas.loadStardistLabels) #probably better to use a super class for all model classes so we don't repeat this code
        self.model_stardist.stardistDone.connect(self.model_cellIntensity.loadStardistLabels)
        self.model_stardist.errorSignal.connect(self.handleError)
        self.model_stardist.progress.connect(self.view.update_progress_bar)
        self.view.view_tab.progress.connect(self.view.update_progress_bar)
        self.view.stardist_groupbox.save_button.clicked.connect(self.model_stardist.saveImage)


        # registration
        # change params
        self.view.register_groupbox.alignment_layer.currentTextChanged.connect(self.model_register.setAlignmentLayer)
        self.view.register_groupbox.protein_cell_layer.currentTextChanged.connect(self.model_register.setCellLayer)
        self.view.register_groupbox.intensity_layer.currentTextChanged.connect(self.model_register.setProteinDetectionLayer)
        self.view.register_groupbox.overlap.valueChanged.connect(self.model_register.setOverlap)
        self.view.register_groupbox.max_size.valueChanged.connect(self.model_register.setMaxSize)
        self.view.register_groupbox.num_tiles.valueChanged.connect(self.model_register.setNumTiles)
        self.view.register_groupbox.run_button.clicked.connect(self.model_register.run_registration)
        self.model_register.cell_image_signal.connect(self.model_stardist.loadCellImage)
        self.model_register.protein_signal_arr_signal.connect(self.model_cellIntensity.loadProteinSignalArray)

        # generate cell data
        self.view.cellIntensity_groupbox.bead_data.clicked.connect(self.view.cellIntensity_groupbox.loadBeadData)
        self.view.cellIntensity_groupbox.color_code.clicked.connect(self.view.cellIntensity_groupbox.loadColorCode)
        self.view.cellIntensity_groupbox.emitBeadData.connect(self.model_cellIntensity.getBeadData)
        self.view.cellIntensity_groupbox.emitColorCode.connect(self.model_cellIntensity.getColorCode)
        self.view.cellIntensity_groupbox.num_cycles.valueChanged.connect(self.model_cellIntensity.setNumDecodingCycles)
        self.view.cellIntensity_groupbox.num_layers_each.valueChanged.connect(self.model_cellIntensity.setNumDecodingColors)
        self.view.cellIntensity_groupbox.radius_fg.valueChanged.connect(self.model_cellIntensity.setRadiusFG)
        self.view.cellIntensity_groupbox.radius_bg.valueChanged.connect(self.model_cellIntensity.setRadiusBG)
        self.view.cellIntensity_groupbox.run_button.clicked.connect(self.model_cellIntensity.generateCellIntensityTable)

        self.model_cellIntensity.errorSignal.connect(self.handleError)
        self.view.cellIntensity_groupbox.save_button.clicked.connect(self.model_cellIntensity.save_cell_data)        

        self.model_register.progress.connect(self.view.update_progress_bar)
        self.model_cellIntensity.progress.connect(self.view.update_progress_bar)
        
        # tab switched - update small view visibility
        self.view.stackedWidget.currentChanged.connect(lambda x: self.view.small_view.setVisible(x == 1))

        # cancel process
        self.view.register_groupbox.cancel_button.clicked.connect(self.model_register.cancel)
        self.view.cellIntensity_groupbox.cancel_button.clicked.connect(self.model_cellIntensity.cancel)
        self.view.stardist_groupbox.cancel_button.clicked.connect(self.model_stardist.cancel)
        
        initial_args = vars(app.args) # Converts namespace to dict
        if initial_args['image'] is not None:
            self.model_canvas.addImage(initial_args['image'])
        if initial_args['reference'] is not None:
            self.reference_view.addImage(initial_args['reference'])
    def handleError(self, error_message):
        QMessageBox.critical(self.view,"Error", error_message)

    def on_action_openFiles_triggered(self):
        self.openFilesDialog = Dialogs.OpenFilesDialog(self.view)
        self.openFilesDialog.show()

    def save_pixmap_as_image(self, pixmap: QPixmap, filename: str):
        
        
        # Convert QPixmap to QImage
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
    
    def is_grayscale(image: np.ndarray) -> bool:

        if len(image.shape) == 3 and image.shape[2] == 3:
            return False
        elif len(image.shape) == 2 or (len(image.shape) == 3 and image.shape[2] == 1):
            return True
        else:
            raise ValueError("Image format not recognized")
    
    def controlSave(self):
        
        pm = self.model_canvas.pixmap
        print(pm)
        # qimage = pm.toImage()
        if pm != None:
            im = self.pixmap_to_image(pm)

            file_name, _ = QFileDialog.getSaveFileName(None, "Save File", "image.png", "*.png;;*.jpg;;*.tif;; All Files(*)")
            if file_name:
                print(file_name)
                Image.fromarray(im).save(file_name)
                
            else:
                return False
            
        else:
            self.handleError("No image in canvas, please load image")
            
    def openFileDialog(self, viewer):
        file_name, _ = QFileDialog.getOpenFileName(None, "Open Image File", "", "Images (*.png *.jpg *.tif);;All Files (*)")
        if file_name:
            viewer.addImage(file_name)

    def on_action_reference_triggered(self):
       self.openFileDialog(self.reference_view)

    def on_actionOpen_triggered(self):
       self.openFileDialog(self.model_canvas)

    def handle_new_image(self, data, file_name):
        storage_item = {}
        storage_item['name'] = f"Image {self.image_count}"
        self.image_count += 1
        storage_item['data'] = data
        my_uuid = str(uuid.uuid4())
        self.view.images_tab.add_to_storage(my_uuid,storage_item)
        self.view.images_tab.add_item(my_uuid)
        
