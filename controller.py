"""Class to handle signal connections"""

import copy
import logging
import os
import typing
import uuid

import cv2
import numpy as np
from PIL import Image
from PyQt6.QtCore import QTimer
from PyQt6.QtGui import QPixmap
from PyQt6.QtWidgets import QFileDialog, QMessageBox

from core import (
    CellIntensity,
    ImageGraphicsView,
    ImageWrapper,
    ReferenceGraphicsView,
    Register,
    StarDist,
)
from ui.alignment.alignment_preview_dialog import AlignmentPreviewDialog

logger = logging.getLogger(__name__)

if typing.TYPE_CHECKING:
    from ui.app import MainWindow


# pylint: disable=too-many-instance-attributes
class Controller:
    """
    Controller class implements the Singleton pattern for managing the MIST-Explorer application.
    This class serves as the central controller for the image analysis application, coordinating
    between the UI, image processing models, and data storage. It handles image loading, saving,
    alignment operations, and user interactions.
    Attributes:
        _instance (Controller): Singleton instance of the Controller class
        image_count (int): Counter for loaded images, not really used in the current implementation
        model_canvas (ImageGraphicsView): Main image display canvas
        model_stardist (StarDist): StarDist model for image analysis
        model_register (Register): Registration model for image alignment
        model_cellIntensity (CellIntensity): Cell intensity analysis model
        reference_view (ReferenceGraphicsView): Reference image display view
        view (Ui_MainWindow): Main application UI window
        openFilesDialog (QFileDialog): File dialog for opening images
        signal_manager (SignalConnectionManager): Manages signal connections
        storage: Image data storage system
    Methods:
        __new__(cls, app): Creates or returns the singleton instance
        __init__(app): Initializes the controller with the main application window (app)
        handleError(error_message): Displays error messages to the user
        pixmap_to_image(pixmap): Converts QPixmap to numpy array for image processing
        controlSave(): Handles saving the current canvas image to file
        openFileDialog(viewer): Opens file dialog for image selection
        on_action_reference_triggered(): Handles reference image selection
        on_actionOpen_triggered(): Handles main image opening
        handle_new_image(data, file_name): Processes newly loaded images
        handle_new_reference_image(data, file_name): Processes newly loaded reference images
        _handle_aligned_image(aligned_data, target_small, aligned_small): Processes aligned images
        _show_preview_dialog(target_small, aligned_small): Shows alignment preview dialog
    Note:
        This class uses the Singleton pattern to ensure only one controller instance exists
        throughout the application lifecycle. It connects most UI interactions to function calls,
        there are some exceptions where the UI directly calls methods on the model. Perhaps those
        instances should also be handled by the controller.
    """

    _instance = None

    @classmethod
    def init(cls, app):
        """Initializes the controller."""
        if cls._instance is None:
            cls._instance = cls(app)

    @classmethod
    def get(cls):
        """Returns the singleton instance of the controller."""
        if cls._instance is None:
            raise RuntimeError("Controller not initialized")
        return cls._instance

    def __init__(self, app: "MainWindow"):
        self._initialized = True
        self.image_count = 0
        self.prev_tab_index = 0
        self.model_canvas = ImageGraphicsView(self)
        self.model_stardist = StarDist()
        self.model_register = Register()
        self.model_cell_intensity = CellIntensity()
        self.model_reference_canvas = ReferenceGraphicsView()
        self.view = app
        self.open_files_dialog = None
        self.view.images_tab.set_model_canvas(self.model_canvas)
        self.view.images_tab.set_model_stardist(self.model_stardist)
        self.view.images_tab.set_model_reference_canvas(self.model_reference_canvas)
        self.signal_manager = SignalConnectionManager(self)
        self.signal_manager.setup_all_connections()
        self.storage = self.view.images_tab.storage

        # Handle initial arguments
        initial_args = app.args
        if initial_args.image is not None:
            self.model_canvas.add_to_canvas(initial_args.image)
        if initial_args.reference is not None:
            self.model_reference_canvas.add_to_canvas(initial_args.reference)

        # self.need_preview_alignment.connect(self._handle_aligned_image)
        self.view.cell_layer_alignment.aligner.aligned_image_signal.connect(
            self._handle_aligned_image
        )
        self.model_register.alignment_complete.connect(self._handle_aligned_image)
        self.model_register.error.connect(self.handle_error)

    def handle_error(self, error_message):
        """Displays an error message."""
        QMessageBox.critical(self.view, "Error", error_message)

    def pixmap_to_image(self, pixmap: QPixmap):
        """Converts a QPixmap to a numpy array."""
        if pixmap is None:
            raise ValueError("No pixmap provided")
        qimage = pixmap.toImage()
        width = qimage.width()
        height = qimage.height()
        ptr = qimage.bits()
        assert ptr is not None, "QImage bits() returned None"
        ptr.setsize(height * width * 4)
        arr = np.array(ptr).reshape((height, width, 4))  # 4 for RGBA

        # Convert from BGRA to RGB by dropping alpha channel and reversing BGR
        if arr.shape[2] == 4:  # If the image has an alpha channel
            arr = arr[:, :, :3]  # Remove the alpha channel

        # Convert BGR to RGB (OpenCV uses BGR, but most other libraries use RGB)
        arr = arr[:, :, ::-1]

        return arr

    def control_save(self):
        """Saves the current canvas image."""
        img = self.model_canvas.image_wrapper.data
        if img.dtype != np.uint8:
            img = (img * 255).astype(np.uint8)
        # qimage = pm.toImage()
        if img is not None:
            file_name, _ = QFileDialog.getSaveFileName(
                None, "Save File", "image.png", "*.png;;*.jpg;;*.tif;; All Files(*)"
            )
            if file_name:
                logger.debug(file_name)
                Image.fromarray(img).save(file_name)
                return True

            return False

        self.handle_error("No image in canvas, please load image")
        return False

    def open_file_dialog(self, viewer):
        """Opens a file dialog to select an image."""
        file_name, _ = QFileDialog.getOpenFileName(
            None, "Open Image File", "", "Images (*.png *.jpg *.tif);;All Files (*)"
        )
        if file_name:
            viewer.add_to_canvas(file_name)

    def on_action_reference_triggered(self):
        """Handles the reference image open action."""
        self.open_file_dialog(self.model_reference_canvas)

    def on_action_open_triggered(self):
        """Handles the main image open action."""
        self.open_file_dialog(self.model_canvas)

    # add new image to storage
    def handle_new_image(self, data, file_name, metadata=None):
        """Handles a new image by adding it to storage."""
        storage_item = {}
        storage_item["name"] = os.path.basename(file_name)
        storage_item["metadata"] = metadata
        self.image_count += 1

        storage_item["data"] = data
        my_uuid = str(uuid.uuid4())
        self.view.images_tab.add_to_storage(my_uuid, storage_item)
        self.model_canvas.set_uuid(my_uuid)
        self.view.images_tab.add_item(my_uuid)

    def handle_new_reference_image(self, data, file_name):
        """Handles a new reference image by adding it to storage."""
        storage_item = {}
        storage_item["name"] = os.path.basename(file_name)
        self.image_count += 1

        storage_item["data"] = data
        my_uuid = str(uuid.uuid4())
        self.view.images_tab.add_to_storage(my_uuid, storage_item)
        self.model_reference_canvas.set_uuid(my_uuid)
        self.view.images_tab.add_item(my_uuid)

    def _handle_aligned_image(self, aligned_data, target_small, aligned_small):
        """Handle the aligned image result"""
        # Show the preview dialog
        snapshot = {
            "target_image": target_small,
            "aligned_image": aligned_small,
        }
        is_align_arrays = isinstance(aligned_data["layer"], list)

        def handle_accepted_image(moving_image, is_manual=False):
            """Handle the moving image change in the preview dialog"""
            aligned_image = moving_image
            item_uuid = aligned_data["uuid"]
            layer = aligned_data["layer"]
            item = self.storage.get_data(item_uuid)
            assert item is not None, "Aligned image data not found in storage"
            data = copy.deepcopy(item["data"])
            if is_manual:
                layer = list(data.keys())
                aligned_data["data"] = {}
                # treat moving_image as transformation matrix
                transf_matrix = moving_image
                # print(transf_matrix)
                for layer_key in layer:
                    # print(L)
                    height, width = data[layer_key].data.shape[-2:]
                    # print(h,w)
                    aligned_data["data"][layer_key] = cv2.warpAffine(  # pylint: disable=no-member
                        item["data"][layer_key].data, transf_matrix, (width, height)
                    )
            filename = item["name"]
            if isinstance(layer, list):
                # handles register.py
                assert len(aligned_data["data"].keys()) == len(layer), (
                    "Aligned data keys do not match the expected layers"
                )
                data = {}
                for layer_key in layer:
                    img_data = (
                        aligned_data["data"][layer_key]
                        if aligned_data["data"][layer_key] is not None
                        else moving_image
                    )
                    wrapped_image = ImageWrapper(img_data, layer_key)
                    # data[L].data = aligned_data["data"][L]
                    data[layer_key] = wrapped_image
                aligned_name = "Registered_" + filename
            else:
                wrapped_image = ImageWrapper(aligned_image, layer)
                aligned_name = f"Aligned_{filename}"
                data[layer].data = aligned_image
            if is_manual:
                aligned_name = "Manual_" + filename

            self.model_canvas.add_to_canvas(data, True, aligned_name)

        # manual alignment
        if not np.any(aligned_small) and not np.any(target_small):
            handle_accepted_image(aligned_data["data"], True)
        else:
            if is_align_arrays:
                preview_dialog = AlignmentPreviewDialog(
                    snapshot, can_edit=False, can_emit=True
                )
            else:
                preview_dialog = AlignmentPreviewDialog(snapshot, can_edit=True)
            preview_dialog.moving_image_changed.connect(handle_accepted_image)
            preview_dialog.exec()

    def _show_preview_dialog(self, target_small, aligned_small):
        """Show the preview dialog with red/green overlay"""
        # Use the downscaled images for the preview dialog
        snapshot = {
            "target_image": target_small,
            "aligned_image": aligned_small,
            "metadata": {
                "preview_target_shape": target_small.shape,
                "preview_aligned_shape": aligned_small.shape,
            },
        }
        preview_dialog = AlignmentPreviewDialog(snapshot, can_edit=True)
        result = preview_dialog.exec()
        return result == 1 and preview_dialog.result_accepted

    def need_canvas_change(self, new_index):
        """Handles the canvas change when the tab is changed."""
        if self.prev_tab_index != 0 and new_index == 0:
            # self.view.canvas.pixmap_item.show()
            self.view.canvas.show_images_tab_image()
            if self.model_canvas.uuid:
                logger.debug("swapping channel")
                self.model_canvas.swap_channel(self.model_canvas.current_channel)
            else:
                logger.debug("clearing canvas")
                self.model_canvas.clear_canvas()
        elif self.prev_tab_index == 0 and new_index != 0:
            self.view.canvas.show_view_tab_image()
            self.view.view_tab.process_images()
            # self.view.canvas.pixmap_item.hide()

    def handle_tab_change(self, index):
        """Handles the tab change."""
        self.need_canvas_change(index)
        self.prev_tab_index = index

        # Update ROI button visibility based on tab
        # Show on View (1) and Analyze (2) tabs, hide on Extract (0)
        if hasattr(self.view, 'canvas'):
            should_show_buttons = index in (1, 2)
            self.view.canvas.set_buttons_visible(should_show_buttons)

    def update_small_view_visibility(self, *args):
        """Updates the visibility of the small view (reference view)."""
        # args can be ignored (either index from tab change or pixmap from update_reference)
        index = self.view.stacked_widget.currentIndex()
        is_images_tab = index == 0
        has_content = not self.view.small_view.is_empty()
        self.view.small_view.setVisible(is_images_tab and has_content)

    def handle_cancel_registration(self):
        """Cancels the registration."""
        self.model_register.cancel()


# pylint: disable=too-few-public-methods
class SignalConnectionManager:
    """Manages signal connections"""

    def __init__(self, controller: Controller):
        self.c = controller
        self.blur_timer = None

    def setup_all_connections(self):
        """Set up all signal-slot connections with full IntelliSense support"""
        self._setup_alignment_connections()
        self._setup_menubar_connections()
        self._setup_toolbar_connections()
        self._setup_img_handling_conns()
        self._setup_canvas_connections()
        self._setup_crop_connections()
        self._setup_transform_connections()
        self._setup_stardist_connections()
        self._setup_registration_connections()
        self._setup_cell_intensity_conns()
        self._setup_img_broadcast_conns()
        self._setup_misc_connections()

    def _setup_alignment_connections(self):
        """Alignment section signal connections"""
        self.c.view.cell_layer_alignment.alignmentCompleteSignal.connect(
            self.c.handle_new_image
        )
        self.c.view.cell_layer_alignment.loadOnCanvasSignal.connect(
            self.c.model_canvas.add_to_canvas
        )
        self.c.view.cell_layer_alignment.progress.connect(
            self.c.view.update_progress_bar
        )

    def _setup_menubar_connections(self):
        """Menu bar signal connections"""
        self.c.view.menu_bar.open_reference.triggered.connect(
            self.c.on_action_reference_triggered
        )
        self.c.view.menu_bar.open_image.triggered.connect(
            self.c.on_action_open_triggered
        )

    def _setup_toolbar_connections(self):
        """Toolbar signal connections"""
        self.c.view.tool_bar.actionReset.triggered.connect(
            self.c.model_canvas.reset_image
        )
        self.c.view.tool_bar.channelChanged.connect(self.c.model_canvas.swap_channel)
        self.c.view.tool_bar.contrast_slider.valueChanged.connect(
            self.c.model_canvas.update_contrast
        )
        self.c.view.tool_bar.cmapChanged.connect(self.c.model_canvas.update_image)
        self.c.view.tool_bar.auto_contrast_button.clicked.connect(
            self.c.model_canvas.auto_contrast
        )
        self.c.view.tool_bar.tabChanged.connect(
            self.c.view.stacked_widget.setCurrentIndex
        )
        self.c.view.tool_bar.tabChanged.connect(self.c.handle_tab_change)

    def _setup_img_handling_conns(self):
        """Image loading and display connections"""
        self.c.view.canvas.image_dropped.connect(self.c.model_canvas.add_to_canvas)
        self.c.view.small_view.image_dropped.connect(
            self.c.model_reference_canvas.add_to_canvas
        )
        self.c.model_reference_canvas.update_reference.connect(
            self.c.view.small_view.display
        )
        self.c.model_reference_canvas.update_reference.connect(
            self.c.update_small_view_visibility
        )
        self.c.model_canvas.update_canvas.connect(self.c.view.canvas.update_canvas)
        self.c.model_canvas.update_channel.connect(
            self.c.view.tool_bar.setChannelSelector
        )

        self.c.model_canvas.update_sidebar.connect(
            self.c.view.images_tab.set_channel_icon
        )
        self.c.view.view_tab.change_pix.connect(
            self.c.view.canvas.update_view_tab_canvas
        )
        self.c.view.view_tab.update_contrast_sig.connect(
            self.c.view.canvas.update_layer_levels
        )
        self.c.view.view_tab.update_layer_cmap_sig.connect(
            self.c.view.canvas.update_layer_cmap
        )

        self.c.view.view_tab.update_layer_opacity_sig.connect(
            self.c.view.canvas.update_layer_opacity
        )
        self.c.view.view_tab.update_layer_visible_sig.connect(
            self.c.view.canvas.update_layer_visibility
        )
        self.c.view.view_tab.reset_view_tab.connect(self.c.view.canvas.reset_view_tab)

        self.c.view.view_tab.export_png_sig.connect(self.c.view.canvas.save_as_png)
        self.c.view.view_tab.export_tif_sig.connect(self.c.view.canvas.save_as_tif)
        # self.c.model_canvas.canvas_updated.connect(self.c.view.canvas.update_canvas)
        self.c.model_canvas.update_manager.connect(self.c.handle_new_image)
        self.c.model_reference_canvas.update_manager.connect(
            self.c.handle_new_reference_image
        )

    def _setup_canvas_connections(self):
        """Canvas-related signal connections"""
        self.c.model_canvas.update_progress.connect(self.c.view.update_progress_bar)
        self.c.model_canvas.error_signal.connect(self.c.handle_error)
        self.c.view.canvas.show_crop.connect(self.c.model_canvas.crop)
        self.c.model_canvas.update_cmap.connect(
            self.c.view.tool_bar.update_cmap_selector
        )
        self.c.model_canvas.change_slider.connect(
            self.c.view.tool_bar.update_contrast_slider
        )
        self.c.model_canvas.fill_metadata.connect(self.c.view.get_metadata)

    def _setup_crop_connections(self):
        """Crop operation connections"""
        self.c.view.crop_groupbox.crop_button.triggered.connect(
            self.c.view.canvas.start_crop_mode
        )
        self.c.view.crop_groupbox.cancel_crop_button.triggered.connect(
            self.c.view.canvas.cancel_crop_mode
        )

    def _setup_transform_connections(self):
        """Image transformation connections"""
        # Flip operations
        self.c.view.canvas.horizontal_flip.connect(self.c.model_canvas.flip_horizontal)
        self.c.view.canvas.vertical_flip.connect(self.c.model_canvas.flip_vertical)

        # Rotation
        self.c.view.rotate_groupbox.rotate_confirm.pressed.connect(
            lambda: self.c.model_canvas.rotate_image(
                self.c.view.rotate_groupbox.rotate_line_edit.text()
            )
        )

        # Gaussian blur
        self.c.view.gaussian_blur.confirm.clicked.connect(
            lambda: self.c.model_canvas.blur_layer(
                self.c.view.gaussian_blur.slider.value(), confirm=True
            )
        )
        self.c.view.gaussian_blur.slider.doubleValueChanged.connect(
            self.c.view.gaussian_blur.update_slider_label
        )
        self.blur_timer = QTimer()
        self.blur_timer.setSingleShot(True)
        self.blur_timer.setInterval(300)  # ms of inactivity before applying

        # Connect the timer timeout to the actual blur update
        self.blur_timer.timeout.connect(
            lambda: self.c.model_canvas.blur_layer(
                self.c.view.gaussian_blur.slider.value()
            )
        )
        self.c.view.gaussian_blur.slider.doubleValueChanged.connect(
            lambda _: self.blur_timer.start()
        )

    def _setup_stardist_connections(self):
        """StarDist-related connections"""
        # Parameter connections
        self.c.view.stardist_groupbox.stardist_channel_selector.currentTextChanged.connect(
            self.c.model_stardist.set_channel
        )
        self.c.model_stardist.cell_image_set.connect(
            self.c.view.stardist_groupbox.set_groupbox_title
        )
        self.c.model_stardist.cell_channel.connect(
            self.c.view.stardist_groupbox.stardist_channel_selector.setCurrentText
        )
        self.c.view.stardist_groupbox.stardist_pretrained_models.currentTextChanged.connect(
            self.c.model_stardist.set_model
        )
        self.c.view.stardist_groupbox.percentile_high.valueChanged.connect(
            self.c.model_stardist.set_percentile_high
        )
        self.c.view.stardist_groupbox.percentile_low.valueChanged.connect(
            self.c.model_stardist.set_percentile_low
        )
        self.c.view.stardist_groupbox.prob_threshold.valueChanged.connect(
            self.c.model_stardist.set_prob_thresh
        )
        self.c.view.stardist_groupbox.nms_threshold.valueChanged.connect(
            self.c.model_stardist.set_nms_thresh
        )
        self.c.view.stardist_groupbox.n_tiles.valueChanged.connect(
            self.c.model_stardist.set_num_tiles
        )
        self.c.view.stardist_groupbox.radius.valueChanged.connect(
            self.c.model_stardist.set_dialation_radisu
        )

        # Execution and results
        self.c.view.stardist_groupbox.stardist_run_button.pressed.connect(
            self.c.model_stardist.start
        )
        self.c.model_stardist.stardist_done.connect(self.c.model_canvas.add_to_canvas)

        self.c.model_stardist.stardist_done.connect(
            lambda x, y, z: self.c.model_cell_intensity.load_stardist_labels(x)
        )
        self.c.model_stardist.error_signal.connect(self.c.handle_error)
        self.c.model_stardist.progress.connect(self.c.view.update_progress_bar)
        self.c.view.stardist_groupbox.cancel_button.clicked.connect(
            self.c.model_stardist.cancel
        )

    def _setup_registration_connections(self):
        """Registration-related connections"""
        # Parameters
        self.c.view.register_groupbox.alignment_layer.currentTextChanged.connect(
            self.c.model_register.set_alignment_layer
        )
        self.c.view.register_groupbox.protein_cell_layer.currentTextChanged.connect(
            self.c.model_register.set_cell_layer
        )
        self.c.view.register_groupbox.intensity_layer.currentTextChanged.connect(
            self.c.model_register.set_protein_detection_layer
        )
        self.c.view.register_groupbox.overlap.valueChanged.connect(
            self.c.model_register.set_overlap
        )
        self.c.view.register_groupbox.max_size.valueChanged.connect(
            self.c.model_register.set_max_size
        )
        self.c.view.register_groupbox.num_tiles.valueChanged.connect(
            self.c.model_register.set_num_tiles
        )
        self.c.view.register_groupbox.ncc_threshold.valueChanged.connect(
            lambda v: self.c.model_register.set_ncc_threshold(v / 100)
        )

        # Execution
        self.c.view.register_groupbox.run_button.clicked.connect(
            self.c.model_register.run_registration
        )
        self.c.model_register.cell_image_signal.connect(
            self.c.model_stardist.load_cell_image
        )
        self.c.model_register.protein_signal_arr_signal.connect(
            self.c.model_cell_intensity.load_protein_signal_array
        )
        self.c.model_register.progress.connect(self.c.view.update_progress_bar)
        self.c.view.register_groupbox.cancel_button.clicked.connect(
            self.c.handle_cancel_registration
        )

        # Results

    def _setup_cell_intensity_conns(self):
        """Cell intensity-related connections"""

        self.c.view.images_tab.image_tree_view.protein_data.connect(
            self.c.model_cell_intensity.load_protein_signal_array_from_storage
        )
        self.c.view.images_tab.image_tree_view.stardist_label.connect(
            self.c.model_cell_intensity.load_stardist_labels_from_storage
        )

        self.c.view.cell_intensity_groupbox.emitBeadData.connect(
            self.c.model_cell_intensity.set_bead_data
        )
        self.c.view.cell_intensity_groupbox.emitColorCodes.connect(
            self.c.model_cell_intensity.set_color_codes
        )
        self.c.view.cell_intensity_groupbox.radius_fg.valueChanged.connect(
            self.c.model_cell_intensity.set_radius_fg
        )
        self.c.view.cell_intensity_groupbox.radius_bg.valueChanged.connect(
            self.c.model_cell_intensity.set_radius_bg
        )
        self.c.view.cell_intensity_groupbox.generate_cell_data.connect(
            self.c.model_cell_intensity.generate_cell_intensity_table
        )
        self.c.model_cell_intensity.error_signal.connect(self.c.handle_error)
        self.c.view.cell_intensity_groupbox.save_button.clicked.connect(
            self.c.model_cell_intensity.save_cell_data
        )
        self.c.model_cell_intensity.progress.connect(self.c.view.update_progress_bar)
        self.c.view.cell_intensity_groupbox.cancel_button.clicked.connect(
            self.c.model_cell_intensity.cancel
        )
        self.c.view.cell_intensity_groupbox.requestFilteredStats.connect(
            self.c.model_cell_intensity.get_filtered_bead_count
        )
        self.c.model_cell_intensity.filtered_stats_ready[float].connect(
            self.c.view.cell_intensity_groupbox.show_filtered_stats
        )

    def _setup_img_broadcast_conns(self):
        """Image signal broadcast to multiple targets"""
        # ensure all the relevant UI components are not None

        image_signal = self.c.model_canvas.image_signal
        image_signal.connect(self.c.view.tool_bar.updateChannelSelector)
        image_signal.connect(self.c.view.register_groupbox.updateChannelSelector)
        image_signal.connect(self.c.view.canvas.load_channels)
        image_signal.connect(self.c.model_stardist.update_channels)
        image_signal.connect(self.c.view.stardist_groupbox.updateChannelSelector)
        image_signal.connect(self.c.view.gaussian_blur.updateChannelSelector)
        image_signal.connect(self.c.model_register.update_moving_image)
        image_signal.connect(self.c.view.cell_intensity_groupbox.update_channels)

        ref_image = self.c.model_reference_canvas.image_signal
        ref_image.connect(self.c.model_register.update_reference_channels)
        ref_image.connect(self.c.view.small_view.load_channels)

        deleted_uuid_signal = self.c.view.images_tab.image_tree_view.item_deleted
        deleted_uuid_signal.connect(self.c.model_canvas.remove_from_canvas)
        deleted_uuid_signal.connect(self.c.model_reference_canvas.remove_from_canvas)

    def _setup_misc_connections(self):
        """Miscellaneous connections"""
        self.c.view.view_tab.progress.connect(self.c.view.update_progress_bar)
        self.c.view.stacked_widget.currentChanged.connect(
            self.c.update_small_view_visibility
        )
