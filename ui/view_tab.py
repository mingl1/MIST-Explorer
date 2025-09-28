import os
import threading

import numpy as np
import pandas as pd
import tifffile as tiff
from numpy.typing import NDArray
from PyQt6.QtCore import QEvent, QTimer

from controller import Controller
from core.canvas import ImageWrapper
from utils import auto_contrast_helper, scale_adjust, create_lut, grayscale_to_agrb

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
import os

import cv2
from numba import njit
from PIL import Image
from PyQt6.QtWidgets import QFileDialog, QMessageBox

from core import Worker

Image.MAX_IMAGE_PIXELS = None

color_dict = {
    "None": [255, 255, 255],
    "red": [255, 0, 0],
    "blue": [0, 0, 255],
    "green": [0, 255, 0],
    "yellow": [255, 255, 0],
    "purple": [128, 0, 128],
    "orange": [255, 165, 0],
    "pink": [255, 192, 203],
    "brown": [165, 42, 42],
    "black": [0, 0, 0],
    "white": [255, 255, 255],
    "cyan": [0, 255, 255],
    "magenta": [255, 0, 255],
    "silver": [192, 192, 192],
    "gray": [128, 128, 128],
    "maroon": [128, 0, 0],
    "olive": [128, 128, 0],
    "dark green": [0, 128, 0],
    "teal": [0, 128, 128],
    "navy": [0, 0, 128],
    "indigo": [75, 0, 130],
    "deep pink": [255, 20, 147],
    "chocolate": [210, 105, 30],
    "moccasin": [255, 228, 181],
    "steel blue": [70, 130, 180],
    "light sky blue": [135, 206, 250],
    "sandy brown": [244, 164, 96],
    "green yellow": [173, 255, 47],
    "khaki": [240, 230, 140],
    "tomato": [255, 99, 71],
    "dark slate blue": [72, 61, 139],
}


def create_contrast_lut(min_val, max_val):
    """Creates a Look-Up Table for contrast adjustment."""
    if min_val >= max_val:
        return np.zeros(256, dtype=np.uint8)

    lut = np.zeros(256, dtype=np.uint8)
    lut[min_val : max_val + 1] = np.linspace(
        start=0,
        stop=255,
        num=(max_val - min_val + 1),
        endpoint=True,
        dtype=np.uint8,
    )
    lut[max_val + 1 :] = 255
    return lut


class ControlsBox:
    def __init__(self):
        self.name = ""
        self.image = np.array([[]])
        self.q_image = None
        self.cell_image = np.array([[]])
        self.contrast_cache = {}

        self.current_opacity = 1.0
        self.current_contrast = [0, 255]
        self.current_visibility = True
        self.current_tint = QColor(255, 255, 255)

        # actual components that we just want to keep track of
        self.tint_label = None
        self.opacity_slider = None

        # entire component layout
        self.layout = None

        # self.tint_yn
        self.tint_yn = True


import time


# def write_protein(protein_data, reduced_cell_img):
#     t = time.time()
#     print(protein_data)
#     cnv = write_protein_sub(protein_data, reduced_cell_img)
#     print(time.time() - t)

#     return cnv


# @njit(cache=True)
# def write_protein_sub(protein_data=np.array([]), reduced_cell_img=np.array([[]])):
#     # Copy the image
#     cnv = reduced_cell_img.copy()

#     # Extract protein data, winsorize, and rescale
#     protein_1 = protein_data
#     lower, upper = np.percentile(protein_1, [0, 100])
#     protein_1 = np.clip(protein_1, lower, upper)
#     # protein_1 = 60 + (protein_1 - lower) * (255 - 60) / (upper - lower)

#     for i in range(cnv.shape[0]):
#         for j in range(cnv.shape[1]):
#             id = reduced_cell_img[i, j]
#             if id > 0 and id <= len(protein_1):
#                 cnv[i, j] = protein_1[id - 1]

    # return cnv
def write_protein(protein_data, reduced_cell_img):
    """
    Generates a protein intensity image using vectorized numpy indexing.
    Intensity is scaled absolutely based on a 16-bit range (0-65535).

    Args:
        protein_data (np.ndarray): 1D array of intensity values for each cell.
        reduced_cell_img (np.ndarray): 2D array where pixel values are cell IDs (integers).

    Returns:
        np.ndarray: 2D array representing the protein intensity image.
    """
    # Ensure protein_data is a numpy array
    if not isinstance(protein_data, np.ndarray):
        protein_data = np.array(protein_data, dtype=np.float32)
    else:
        # Ensure float for division
        protein_data = protein_data.astype(np.float32)

    # Scale data relative to the max value of uint16 (65535).
    # This provides a consistent, absolute scaling across all proteins.
    scaled_data = (protein_data /65535.0*255.0).astype(np.float32)

    # Replace any NaN or inf values before casting to integer type.
    # NaNs are converted to 0, which is appropriate for missing data.
    safe_data = np.nan_to_num(scaled_data)

    # Clip values to ensure they are in the 0-65535 range and convert to uint8
    normalized_data = np.clip(safe_data, 0, 255).astype(np.uint8)
    # normalized_data[normalized_data==0] = np.nan

    # Create a lookup table (LUT) for protein intensities.
    # The +1 is for the background (cell ID 0), which will have an intensity of 0.
    num_cells = len(normalized_data)
    intensity_lut = np.zeros(num_cells + 1, dtype=np.uint8)
    cell_data_image = np.zeros(num_cells + 1, dtype=np.uint16)
    # intensity_lut.fill(np.nan)

    # Cell ID 1 maps to index 0 in protein_data, so it goes into index 1 of our LUT.
    intensity_lut[1:] = normalized_data
    cell_data_image[1:] = protein_data

    # Use the cell ID image to index into the LUT to create the final image.
    protein_image = intensity_lut[reduced_cell_img]
    cell_image = cell_data_image[reduced_cell_img]
    
    # protein_image[protein_image==0] = np.nan

    return protein_image,cell_image


import os

import numpy as np
import qtrangeslider
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QColor, QImage, QPixmap
from PyQt6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSlider,
    QVBoxLayout,
    QWidget,
)


def scale_image_to_255(image_array):

    try:
        if image_array.dtype == np.uint8:
            return image_array
    except:
        pass

    scaled_image = (
        255
        * (image_array - image_array.min())
        / (image_array.max() - image_array.min())
    )
    return scaled_image.astype(np.uint8)


def scale(val, src, dst):
    return int(((val - src[0]) / float(src[1] - src[0])) * (dst[1] - dst[0]) + dst[0])


class LayerDialog(QDialog):
    def __init__(self, layers, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Select Layer to Add")
        self.setGeometry(100, 100, 400, 300)

        self.layers = layers
        my_layout = QHBoxLayout()
        assert my_layout is not None

        self.layer_list = QListWidget()
        for i, layer in enumerate(layers):
            item = QListWidgetItem(layer["name"])
            self.layer_list.addItem(item)
        my_layout.addWidget(self.layer_list)

        self.buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        self.buttons.accepted.connect(self.accept)
        self.buttons.rejected.connect(self.reject)
        my_layout.addWidget(self.buttons)

        self.setLayout(my_layout)

    def get_selected_layer_index(self):
        print([item.text() for item in self.layer_list.selectedItems()])
        print([l["name"] for l in self.layers])
        print(
            [l["name"] for l in self.layers].index(
                [item.text() for item in self.layer_list.selectedItems()][0]
            )
        )

        selected_items = self.layer_list.selectedItems()
        if selected_items:
            selected_index = [l["name"] for l in self.layers].index(
                [item.text() for item in self.layer_list.selectedItems()][0]
            )

            print("selected index", selected_index)
            return selected_index
        return None


class ColorDialog(QDialog):
    def __init__(self, colors, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Select Tint Color")
        self.setGeometry(100, 100, 400, 300)

        self.colors = colors
        my_layout = QVBoxLayout()

        self.color_list = QListWidget()
        for color_name in colors.keys():
            item = QListWidgetItem(color_name)
            self.color_list.addItem(item)
        my_layout.addWidget(self.color_list)

        self.buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        self.buttons.accepted.connect(self.accept)
        self.buttons.rejected.connect(self.reject)
        my_layout.addWidget(self.buttons)

        self.setLayout(my_layout)

    def get_selected_color_name(self):
        selected_items = self.color_list.selectedItems()
        if selected_items:
            return selected_items[0].text()
        return None


# changes made
def adjust_contrast(image, min=5, max=100):
    t0 = time.perf_counter()

    image = scale_adjust(image)
    t1 = time.perf_counter()
    print(f"Time for scale_adjust: {t1 - t0:.6f} sec")

    lut = create_lut(min, max)
    t2 = time.perf_counter()
    print(f"Time for create_lut: {t2 - t1:.6f} sec")

    res = np.clip(cv2.LUT(image, lut), 0, 254, dtype=np.uint8)
    t3 = time.perf_counter()
    print(f"Time for LUT application: {t3 - t2:.6f} sec")

    print(f"Total adjust_contrast time: {t3 - t0:.6f} sec")
    return res

class ImageOverlay(QWidget):
    update_contrast_sig = pyqtSignal(int, tuple)
    update_layer_cmap_sig = pyqtSignal(int, np.ndarray)
    change_pix = pyqtSignal(np.ndarray, int)
    progress = pyqtSignal(int, str)

    def __init__(self, pixmap_label, enc):
        super().__init__()

        self.enc = enc

        self.pixmap_label = pixmap_label

        # self.df_path =  "/Users/clark/Downloads/cell_data_8_8_Full_Dataset_Biopsy.xlsx"
        # self.im_path = "/Users/clark/Downloads/new_sd.png"

        self.df_path = None
        self.im_path = None
        self.overlay_path = None

        self.controls = []

        self.loaded_df = None

        self.controller = None
        self.model_canvas = None

        o_timer = QTimer()
        o_timer.setInterval(300)
        o_timer.setSingleShot(True)
        self.opacity_timer = o_timer
        self.contrast_sliders = []
        self.tint_lut_cache = {}
        # Connect timer to the actual work
        self.grayscale = True
        self.initUI()
        self.pending_contrast_idx = 0
        self.contrast_timer = QTimer()
        self.contrast_timer.setInterval(10)  # 300 ms of no movement triggers update
        self.contrast_timer.setSingleShot(True)

    def load_stardist_image(self):
        if self.im_path is None:
            raise ValueError("im_path is None")

        stardist_labels = Image.open(self.im_path)
        stardist_labels = np.array(stardist_labels)

        # return stardist_labels

        # reduced_cell_img = cv2.resize(stardist_labels.astype("float32"), (3000, 3000), interpolation = cv2.INTER_NEAREST_EXACT)

        self.scale_down_factor = 1 / self.scale_down.value()

        reduced_cell_img = cv2.resize(
            stardist_labels,
            (0, 0),
            fx=self.scale_down_factor,
            fy=self.scale_down_factor,
            interpolation=cv2.INTER_NEAREST_EXACT,
        )
        return reduced_cell_img

    def load_df(self):
        if self.req_df() == "":
            raise ValueError("Need to load protein data first.")
        else:
            assert self.df_path is not None
        df = None
        if self.df_path.endswith("csv"):
            df = pd.read_csv(self.df_path)
        elif self.df_path.endswith("xlsx"):
            df = pd.read_excel(self.df_path)
        else:
            raise ValueError("Unsupported file format. Please use .csv or .xlsx")

        # print("df raw", df)
        df = df[df.columns.drop(list(df.filter(regex="N/A")))]

        self.loaded_df = df
        return df

    def generate_image(self, index):
        if self.reduced_cell_img.size == 0:
            raise ValueError("Please load an image first.")
        if self.df is None:
            raise ValueError("Dataframe not processed. Please click 'Apply' first.")

        # Get the name of the protein to generate from the layers list
        protein_name = self.layers[index]["name"]

        # The dataframe is now indexed by CellID.
        # We need to construct a 1D array that can be used as a LUT for write_protein.
        # The LUT must be indexed from 0 to max_cell_id.
        max_id_in_image = self.reduced_cell_img.max()

        # Get the sparse series of intensities for the current protein
        protein_series = self.df[protein_name]

        # Create a dense array for the LUT, with 0 for cells not in the data
        # The +1 is because cell IDs are 1-based.
        lut_data = np.zeros(max_id_in_image + 1)

        # Place the protein intensities at the correct index (cell ID)
        # This handles cases where the df doesn't contain all cell IDs present in the image
        valid_indices = protein_series.index[protein_series.index <= max_id_in_image]
        lut_data[valid_indices] = protein_series.loc[valid_indices].values

        # The vectorized write_protein function expects a simple 0-indexed array
        # corresponding to cell IDs 1, 2, 3...
        # So we pass the LUT data, but skip the 0th element.
        im,cell_data = write_protein(lut_data[1:], self.reduced_cell_img)

        # The call to adjust_contrast is removed to show the absolute scaled intensity initially.
        # User can adjust contrast manually with the layer slider.
        # result = tint_grayscale_image(im, [255, 255, 255])

        return im, cell_data

    def build_all(self):
        if not self.controller:
            self.controller = Controller.get()
            self.model_canvas = self.controller.model_canvas
        import time

        if not hasattr(self, "im_path"):
            import ui.app

            QMessageBox.critical(
                ui.app.Ui_MainWindow(), "Error", "Please an load image first!"
            )
            return

        if not hasattr(self, "df_path"):
            import ui.app

            QMessageBox.critical(
                ui.app.Ui_MainWindow(), "Error", "Please load data first!"
            )
            return

        start = time.time()
        self.progress.emit(10, "Loading images and data...")

        self.reduced_cell_img = self.load_stardist_image()

        df = self.loaded_df
        if df is None:
            # This should have been loaded by the user action before build_all is called
            return

        self.progress.emit(30, "Associating cells with data...")
        # --- NEW: Associate DF with Cell IDs ---
        if "CellID" not in df.columns:  # Perform this expensive operation only once
            scale_factor = 1 / self.scale_down.value()

            # Vectorized coordinate scaling
            scaled_x = (df["Global X"] * scale_factor).astype(int)
            scaled_y = (df["Global Y"] * scale_factor).astype(int)

            # Clip coordinates to be within image bounds
            h, w = self.reduced_cell_img.shape
            scaled_x = np.clip(scaled_x, 0, w - 1)
            scaled_y = np.clip(scaled_y, 0, h - 1)

            # Look up all cell IDs at once using vectorized indexing
            cell_ids = self.reduced_cell_img[scaled_y, scaled_x]
            df["CellID"] = cell_ids

        # Filter out rows that are in the background (don't map to a cell)
        df = df[df["CellID"] > 0].copy()

        # Set CellID as the index for fast lookups later
        self.df = df.set_index("CellID")
        # --- END NEW ---

        self.progress.emit(70, "Preparing layers...")

        # Get protein names (assuming they are all columns after the first few)
        protein_columns = self.loaded_df.columns.drop(
            ["Global X", "Global Y", "CellID"], errors="ignore"
        )
        protein_names = [col for col in protein_columns if col in self.df.columns]

        # Prepare layer structure, but don't generate images yet (on-demand is fast now)
        self.ims = [None for _ in protein_names]
        self.layers = [{"name": name, "image": None} for name in protein_names]

        end = time.time()
        print(f"Build time: {end - start:.2f} seconds")
        self.progress.emit(100, "Done")

        # Update UI
        self.scroll_area.setVisible(True)
        self.add_layer_button.setVisible(True)
        self.add_other_image_button.setVisible(True)
        self.open_image.setVisible(False)
        self.open_image_label.setVisible(False)
        self.open_df.setVisible(False)
        self.open_df_label.setVisible(False)
        self.scale_down_label.setVisible(False)
        self.scale_down.setVisible(False)
        self.apply_button.setVisible(False)
        self.cancel_reset.setVisible(True)
        self.export_tif_button.setVisible(True)
        self.export_png_button.setVisible(True)

        return (self.ims, protein_names)

    def get_layer_values_at(self, x, y):
        if len(self.controls) == 0:
            return None

        layer_values = []
        for c in self.controls:
            value = c.cell_image[y, x]
            layer_values.append((c.name, value))

        return layer_values

    def cancel_reset_first(self):
        reply = QMessageBox()
        reply.setText("Are you sure you want to reset?")
        reply.setStandardButtons(
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )

        resp = reply.exec()

        for i in range(len(self.controls)):
            self.delete_layer(0)

        self.enc.analysis_tab.view_index = 0

        # while self.enc.analysis_tab.deleteLater():
        #     pass

        # for i in range(len(self. )):
        #     self.delete_layer(0)

        if resp == QMessageBox.StandardButton.Yes:
            self.open_image.setVisible(True)
            self.open_image_label.setVisible(False)

            self.open_df.setVisible(True)
            self.open_df_label.setVisible(False)

            self.apply_button.setVisible(True)

            self.scale_down_label.setVisible(True)
            self.scale_down.setVisible(True)

            self.add_layer_button.setVisible(False)
            self.cancel_reset.setVisible(False)
            self.add_other_image_button.setVisible(False)
            self.export_tif_button.setVisible(False)
            self.export_png_button.setVisible(False)

    def initUI(self):
        main_layout = QVBoxLayout()

        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
        )
        self.scroll_area.setMinimumHeight(450)  # Set a reasonable minimum height
        self.scroll_area.setHorizontalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAlwaysOff
        )  # Disable horizontal scroll

        self.scroll_content = QWidget()
        self.scroll_content.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
        )
        self.scroll_layout = QVBoxLayout(self.scroll_content)
        self.scroll_layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        self.scroll_layout.setSpacing(10)
        self.scroll_layout.setContentsMargins(10, 10, 10, 20)

        self.scroll_content.setLayout(self.scroll_layout)
        self.scroll_area.setWidget(self.scroll_content)

        main_layout.addWidget(self.scroll_area)
        main_layout.setStretch(0, 1)  # Make the scroll area take up available space

        self.add_layer_button = QPushButton("Add Layer")
        self.add_layer_button.clicked.connect(self.show_layer_dialog)
        main_layout.addWidget(self.add_layer_button)
        self.add_layer_button.setVisible(False)

        self.add_other_image_button = QPushButton("Add Other Image")
        self.add_other_image_button.clicked.connect(self.open_other_image)
        main_layout.addWidget(self.add_other_image_button)
        self.add_other_image_button.setVisible(False)

        self.cancel_reset = QPushButton("Cancel And Upload New")
        self.cancel_reset.clicked.connect(self.cancel_reset_first)
        main_layout.addWidget(self.cancel_reset)
        self.cancel_reset.setVisible(False)

        self.open_image = QPushButton("Open Image")
        self.open_image.clicked.connect(self.load_image)
        main_layout.addWidget(self.open_image)

        self.open_image_label = QLabel("Path: ")
        self.open_image_label.setVisible(False)
        main_layout.addWidget(self.open_image_label)

        self.open_df = QPushButton("Open Cell Data")
        self.open_df.clicked.connect(self.load_df)
        main_layout.addWidget(self.open_df)

        self.open_df_label = QLabel("Path: ")
        self.open_df_label.setVisible(False)
        main_layout.addWidget(self.open_df_label)

        ### scale slider
        self.scale_down_label = QLabel("Scale Down Factor: ")
        main_layout.addWidget(self.scale_down_label)

        self.scale_down = QSlider(Qt.Orientation.Horizontal)
        self.scale_down.setTickPosition(QSlider.TickPosition.TicksAbove)
        self.scale_down.valueChanged.connect(self.scale_slider_update)

        self.scale_down.setRange(1, 10)
        self.scale_down.setValue(4)

        main_layout.addWidget(self.scale_down)
        ### scale slider

        self.apply_button = QPushButton("Apply")
        self.apply_button.clicked.connect(self.start_build_all_worker)
        main_layout.addWidget(self.apply_button)

        self.export_tif_button = QPushButton("Export to TIF")
        self.export_tif_button.clicked.connect(self.export_to_tif)
        self.export_tif_button.setVisible(False)
        main_layout.addWidget(self.export_tif_button)

        self.export_png_button = QPushButton("Export to PNG")
        self.export_png_button.clicked.connect(self.export_to_png)
        self.export_png_button.setVisible(False)
        main_layout.addWidget(self.export_png_button)

        # Add a spacer to ensure content can scroll all the way down
        main_layout.addStretch(1)  # Add stretch at the end to push content up

        self.setLayout(main_layout)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.process_images()

    def open_other_image(self):

        file_name, _ = QFileDialog.getOpenFileName(
            None,
            "Open Image File",
            "",
            "Images (*.png *.xpm *.jpg *.bmp *.gif *.tif);;All Files (*)",
        )

        # file_name = self.overlay_path

        if file_name:
            stardist_labels = Image.open(file_name)
            stardist_labels = np.array(stardist_labels)

            reduced_cell_img = cv2.resize(
                stardist_labels,
                (0, 0),
                fx=self.scale_down_factor,
                fy=self.scale_down_factor,
                interpolation=cv2.INTER_NEAREST_EXACT,
            )

            c = ControlsBox()

            if len(reduced_cell_img.shape) == 2:
                reduced_cell_img = np.stack((reduced_cell_img,) * 3, axis=-1)
            c.image = reduced_cell_img
            c.name = os.path.basename(file_name)
            c.tint_yn = False
            self.add_layer(c)

    def start_build_all_worker(self):
        self.build_all_worker = Worker(self.build_all)
        self.build_all_worker.start()
        self.build_all_worker.finished.connect(self.build_all_worker.quit)
        self.build_all_worker.finished.connect(self.build_all_worker.deleteLater)
        # self.threadpool.start(self.build_all_worker)

    def scale_slider_update(self, value):
        if value == 1:
            self.scale_down_label.setText(f"Scale Down Factor: original size")
            return
        self.scale_down_label.setText(f"Scale Down Factor: 1/{value} of original size")

    def less_than_15_chars(self, string):
        if len(string) > 50:
            return string[:49] + "..."

        return string

    def load_image(self):
        # print("Yo")

        file_name, _ = QFileDialog.getOpenFileName(
            None,
            "Open Image File",
            "",
            "Images (*.png *.xpm *.jpg *.bmp *.gif *.tif);;All Files (*)",
        )
        if file_name:
            # combined_image = np.array(Image.open(file_name))

            self.open_image_label.setText(
                f"File: {self.less_than_15_chars(os.path.basename(file_name))}"
            )
            self.open_image_label.setVisible(True)
            self.im_path = file_name

    def req_df(self):
        file_name, _ = QFileDialog.getOpenFileName(
            None, "Open Image File", "", "Spreadsheets (*.csv *.xlsx);;All Files (*)"
        )

        if file_name:
            # print()
            self.open_df_label.setText(
                f"File: {self.less_than_15_chars(os.path.basename(file_name))}"
            )
            self.open_df_label.setVisible(True)
            self.df_path = file_name
            return file_name
        return ""

    def show_layer_dialog(self):
        if not hasattr(self, "layers"):

            import ui.app

            QMessageBox.critical(
                ui.app.Ui_MainWindow(),
                "Error",
                "Empty canvas, please an load image first",
            )
            return

        dialog = LayerDialog(self.layers, self)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            try:
                selected_index = dialog.get_selected_layer_index()
            except IndexError:
                selected_index = 0
            print("selected indexxxx is ", selected_index)
            if selected_index is not None:
                c = ControlsBox()

                print("potential error", selected_index)

                try:
                    if self.layers[selected_index]["image"] == None:
                        self.layers[selected_index]["image"],self.layers[selected_index]["cell_data"] = self.generate_image(
                            selected_index
                        )
                except:
                    pass
                reduced_cell_img = np.array(self.layers[selected_index]["image"])
                print(reduced_cell_img.min(),reduced_cell_img.max(),reduced_cell_img.dtype)
                # reduced_cellImg = reduced_cell_img/255.0
                c.image = reduced_cell_img
                c.cell_image =  np.array(self.layers[selected_index]["cell_data"])
                c.name = self.layers[selected_index]["name"]

                self.add_layer(c)

    def show_color_dialog(self, idx):
        dialog = ColorDialog(color_dict, self)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            selected_color_name = dialog.get_selected_color_name()
            if selected_color_name:
                selected_color = color_dict[selected_color_name]
                selected_color = QColor(*selected_color)
                self.controls[idx].tint_label.setText(selected_color_name)
                self.update_layer_cmap_sig.emit(idx, np.array(self.get_lut(selected_color)))

    def add_layer(self, c):
        self.controls.append(c)
        # Initialize the display image
        # contrasted = self.contrasted_image(c.image, c.current_contrast)
        # c.display_image = c.image
        self.add_layer_controls(c)
        self.change_pix.emit(c.image,len(self.controls)-1)
        # self.update_layer_display(len(self.controls)-1)
        # self.process_images()

    def update_current_image(self, image):
        last_index = len(self.controls) - 1
        self.controls[last_index].image = image
        self.process_images()

    def delete_layer(self, index):
        c = self.controls.pop(index)
        self.contrast_sliders.pop(index)

        layer = c.layout
        self.scroll_layout.removeWidget(layer)
        layer.deleteLater()
        layer = None

        self.process_images()

        if len(self.controls) == 0:
            combined_image = np.zeros((10, 10, 10))

            height, width, _ = combined_image.shape
            bytes_per_line = 3

            q_image = QImage(
                combined_image.tobytes(), width, height, QImage.Format.Format_RGB888
            )  # interesting image.tobytes() works well, maybe you don't need to do
            q_pixmap = QPixmap(q_image)
            self.change_pix.emit(np.ndarray(0), 0)

    def restart_contrast_timer(self):
        if self.contrast_timer.isActive():
            return  # Don't restart while still moving
        self.contrast_timer.start()

    def add_layer_controls(self, c):
        idx = len(self.controls) - 1

        group_box = QGroupBox(f"Layer {idx + 1}: {c.name}")
        group_box.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred
        )

        group_layout = QFormLayout()
        group_layout.setSpacing(8)  # Add spacing between form rows
        auto_contrast_button = QPushButton("Auto Contrast")
        auto_contrast_button.clicked.connect(lambda: self.auto_contrast(idx))
        opacity_slider = QSlider(Qt.Orientation.Horizontal)
        opacity_slider.setMinimum(0)
        opacity_slider.setMaximum(100)
        opacity_slider.setValue(100)
        opacity_slider.sliderReleased.connect(
            lambda: self.update_opacity(opacity_slider.value(), idx)
        )
        self.opacity_timer.timeout.connect(
            lambda: self.update_opacity(opacity_slider.value(), idx)
        )
        group_layout.addRow("Opacity:", opacity_slider)

        # --- Existing Contrast Slider ---
        contrast_slider = qtrangeslider.QLabeledDoubleRangeSlider(
            Qt.Orientation.Horizontal
        )
        contrast_slider.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        contrast_slider.setMinimumWidth(300)
        contrast_slider.valueChanged.connect(lambda _: self.restart_contrast_timer())

        self.contrast_timer.timeout.connect(lambda: self.set_contrast_from_slider(idx))

        contrast_slider.setMaximum(255)
        contrast_slider.setValue((0, 255))
        contrast_slider.setDecimals(0)
        self.contrast_sliders.append(contrast_slider)
        contrast_slider.installEventFilter(self)

        contrast_label = QLabel("Contrast:")
        contrast_label.setAlignment(Qt.AlignmentFlag.AlignVCenter)
        group_layout.addRow(contrast_label, contrast_slider)

        # --- New: Numeric Inputs for Min/Max ---
        min_input = QLineEdit()
        min_input.setPlaceholderText("Min")
        min_input.setFixedWidth(50)

        max_input = QLineEdit()
        max_input.setPlaceholderText("Max")
        max_input.setFixedWidth(50)

        apply_contrast_button = QPushButton("Set Contrast")

        # Layout for inputs
        contrast_input_layout = QHBoxLayout()
        contrast_input_layout.addWidget(QLabel("Min:"))
        contrast_input_layout.addWidget(min_input)
        contrast_input_layout.addWidget(QLabel("Max:"))
        contrast_input_layout.addWidget(max_input)
        contrast_input_layout.addWidget(apply_contrast_button)
        group_layout.addRow("", contrast_input_layout)

        # --- Button Logic ---
        def apply_contrast_values():
            try:
                min_val = int(min_input.text())
                max_val = int(max_input.text())
                if 0 <= min_val < max_val <= 255:
                    contrast_slider.setValue((min_val, max_val))
                    self.update_contrast((min_val, max_val), idx)
                else:
                    print("Invalid contrast range!")
            except ValueError:
                print("Enter valid integers for contrast.")

        apply_contrast_button.clicked.connect(apply_contrast_values)

        group_layout.addRow(auto_contrast_button)
        visibility_button = QPushButton("Toggle Visibility")
        visibility_button.setCheckable(True)
        visibility_button.setChecked(True)
        visibility_button.toggled.connect(
            lambda checked: self.update_visibility(checked, idx)
        )
        # self.visibility_buttons.append(visibility_button)
        group_layout.addRow("Visibility:", visibility_button)

        color_button = QPushButton("Select Tint Color")
        color_button.clicked.connect(lambda: self.show_color_dialog(idx))
        # self.color_tints.append(color_button)
        color_label = QLabel("None")

        color_layout = QHBoxLayout()
        color_layout.addWidget(color_button)
        color_layout.addWidget(color_label)
        group_layout.addRow("Tint Color:", color_layout)

        delete_button = QPushButton("Delete Layer")
        delete_button.clicked.connect(lambda: self.delete_layer(len(self.controls) - 1))
        # self.visibility_buttons.append(delete_button)
        group_layout.addRow("", delete_button)

        # self.opacity_sliders.append(opacity_slider)
        self.controls[idx].opacity_slider = opacity_slider
        self.controls[idx].tint_label = color_label

        group_box.setLayout(group_layout)
        self.controls[idx].layout = group_box
        self.scroll_layout.addWidget(group_box)

    def set_contrast_from_slider(self, idx):
        min_val, max_val = self.contrast_sliders[idx].value()
        self.update_contrast((min_val, max_val), idx)

    def eventFilter(self, source, event):
        if isinstance(source, qtrangeslider.QLabeledDoubleRangeSlider):
            if event.type() == QEvent.Type.MouseButtonRelease:
                for idx, slider in enumerate(self.contrast_sliders):
                    if slider is source:
                        self.set_contrast_from_slider(idx)
                        break
        return super().eventFilter(source, event)

    def update_opacity(self, value, idx):
        self.controls[idx].current_opacity = value / 100.0

        self.process_images()

    def update_layer_display(self, idx):
        # t0 = time.perf_counter()
        print(f"Updating contrast for: {idx}")
        c = self.controls[idx]
        contrast_key = tuple(c.current_contrast)
        min,max = contrast_key
        min /= 255.0
        max /= 255.0
        self.update_contrast_sig.emit(idx,contrast_key)

    def update_contrast(self, value, idx):
        value = [int(value[0]), int(value[1])]
        self.controls[idx].current_contrast = value
        self.update_layer_display(idx)
        self.process_images()

    def auto_contrast(self, idx, lower=0.1, upper=0.9):
        img = self.controls[idx].image
        assert isinstance(img, np.ndarray)
        new_min, new_max = auto_contrast_helper(img, lower, upper)
        self.contrast_sliders[idx].setValue((int(new_min), int(new_max)))
        self.update_contrast([new_min, new_max], idx)

    def update_visibility(self, checked, idx):

        self.controls[idx].current_visibility = checked
        self.process_images()

    import numpy as np

    def get_lut(self, color: QColor):
        color_name = color.name()
        if color_name in self.tint_lut_cache:
            return self.tint_lut_cache[color_name]

        # Ramp from 0..255 normalized to [0,1]
        ramp = np.linspace(0, 1, 256)[:, None]  # shape (256, 1)

        # Extract RGB values
        r, g, b, _ = color.getRgb()  # returns (r,g,b,a)
        rgb = np.array([r, g, b], dtype=np.float32)  # shape (3,)

        # Multiply ramp with RGB to create LUT
        lut = (ramp * rgb).astype(np.uint8)  # shape (256, 3)

        # Cache and return
        self.tint_lut_cache[color_name] = lut
        return lut


    def adjust_contrast(self, img, min=5, max=100):
        # pixvals = np.array(img)
        image = scale_adjust(img)
        lut = create_lut(min,max)
        res = np.clip(cv2.LUT(image, lut), 0, 254, dtype=np.uint8)
        return res

    def contrasted_image(self, img, contrast):
        min_val, max_val = contrast

        return adjust_contrast(img, min_val, max_val)

    def process_images(self, display=True):
        return
        if not self.controls:
            return

        # Find the first visible layer to determine the shape of the combined image
        first_visible_img = None
        for c in self.controls:
            if c.current_visibility and c.display_image.size > 0:
                first_visible_img = c.display_image
                break

        if first_visible_img is None:
            return

        combined_image = np.zeros_like(first_visible_img, dtype=np.float32)
        for c in self.controls:
            if c.current_visibility:
                display_img = c.display_image
                opacity = c.current_opacity

                # Ensure display_img is float for multiplication
                if display_img.dtype != np.float32:
                    display_img = display_img.astype(np.float32)

                combined_image += display_img * opacity

        combined_image = np.clip(combined_image, 0, 255).astype(np.uint8)

        q_image = numpy_to_qimage(combined_image)
        q_pixmap = QPixmap(q_image)
        if display:
            self.change_pix.emit(combined_image, 0)
        return combined_image

    def export_to_png(self):
        combined_image = self.process_images(False)
        if combined_image is None:
            return
        file_name, _ = QFileDialog.getSaveFileName(
            None, "Save PNG File", "protein_layers.png", "*.png;;All Files (*)"
        )
        if not file_name:
            return
        img = Image.fromarray(combined_image)
        img.save(file_name)

    def export_to_tif(self):
        if len(self.controls) == 0:
            QMessageBox.warning(None, "Warning", "No layers to export")
            return

        file_name, _ = QFileDialog.getSaveFileName(
            None, "Save TIF File", "protein_layers.tif", "*.tif;;All Files (*)"
        )

        if not file_name:
            return

        # Create an array to hold all the protein layer images as grayscale
        layers_data = []
        layer_names = []

        for i, c in enumerate(self.controls):
            if c.current_visibility:  # Only export visible layers
                img = c.image.copy()

                # Get original protein data in grayscale
                # If the image has 3 channels (RGB), convert to grayscale
                if len(img.shape) == 3 and img.shape[2] == 3:
                    img_gray = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_RGB2GRAY)
                else:
                    img_gray = img
                assert isinstance(img_gray, np.ndarray)
                # Apply contrast adjustment if needed
                if (
                    isinstance(c.current_contrast, list)
                    and len(c.current_contrast) == 2
                ):
                    # Apply contrast stretching
                    img_gray = self.contrasted_image(img_gray, c.current_contrast)

                final_img = img_gray.astype(np.float64) * c.current_opacity

                # Ensure we have valid data range
                final_img = np.clip(final_img, 0, 255)
                # if c.tint_yn:
                #     final_img = self.apply_tint(final_img, c.current_tint)
                # Convert to uint8
                final_img = final_img.astype(np.uint8)

                # Add to our stack
                layers_data.append(final_img)
                layer_names.append(c.name)

        if not layers_data:
            QMessageBox.warning(None, "Warning", "No visible layers to export")
            return

        # Stack all layers into a single 3D array (Z,Y,X) where Z is the protein layer
        tif_data = np.stack(layers_data)

        # Save as multi-layer TIF file
        try:
            # Use tifffile to save with ImageJ compatibility
            tiff.imwrite(file_name, tif_data.astype(np.uint8), imagej=True)

            # Save layer names to a text file
            txt_file = os.path.splitext(file_name)[0] + "_protein_order.txt"
            with open(txt_file, "w") as f:
                for i, name in enumerate(layer_names):
                    f.write(f"Layer {i+1}: {name}\n")

            QMessageBox.information(
                None,
                "Success",
                f"Multi-layered TIF file saved to {file_name}\n"
                f"Each layer contains a separate protein in grayscale\n"
                f"Protein order saved to {txt_file}",
            )
        except Exception as e:
            QMessageBox.critical(None, "Error", f"Failed to save TIF file: {str(e)}")


def numpy_to_qimage(array):
    if len(array.shape) == 2:  # Grayscale image
        height, width = array.shape
        qimage = QImage(
            array.data, width, height, width, QImage.Format.Format_Grayscale8
        )
    elif len(array.shape) == 3 and array.shape[2] == 3:  # RGB image
        height, width, channels = array.shape
        bytes_per_line = channels * width
        qimage = QImage(
            array.data, width, height, bytes_per_line, QImage.Format.Format_RGB888
        )
    elif len(array.shape) == 3 and array.shape[2] == 4:  # RGBA image
        height, width, channels = array.shape
        bytes_per_line = channels * width
        qimage = QImage(
            array.data, width, height, bytes_per_line, QImage.Format.Format_RGBA8888
        )
    else:
        raise ValueError("Unsupported array shape: {}".format(array.shape))
    return qimage
