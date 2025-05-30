import pandas as pd
import numpy as np
import os
import threading
import tifffile as tiff

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
from PIL import Image
from PyQt6.QtWidgets import QFileDialog, QMessageBox
import cv2
import os
from numba import njit
from core.Worker import Worker

Image.MAX_IMAGE_PIXELS = None

color_dict = {
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


class ControlsBox:
    def __init__(self):
        self.name = None
        self.image = None
        self.q_image = None

        self.current_opacity = 1.0
        self.current_contrast = 1.0
        self.current_visibility = True
        self.current_tint = QColor(255, 255, 255)

        # actual components that we just want to keep track of
        self.tint_label = None
        self.opacity_slider = None
        self.contrast_slider = None

        # entire component layout
        self.layout = None

        # self.tint_yn
        self.tint_yn = True


import time


def write_protein(protein_data, reduced_cell_img):
    t = time.time()
    print(protein_data)
    cnv = write_protein_sub(protein_data, reduced_cell_img)
    print(time.time() - t)

    return cnv


@njit(cache=True)
def write_protein_sub(protein_data=np.array([]), reduced_cell_img=np.array([[]])):
    # Copy the image
    cnv = reduced_cell_img.copy()

    # Extract protein data, winsorize, and rescale
    protein_1 = protein_data
    lower, upper = np.percentile(protein_1, [0, 100])
    protein_1 = np.clip(protein_1, lower, upper)
    # protein_1 = 60 + (protein_1 - lower) * (255 - 60) / (upper - lower)

    
    for i in range(cnv.shape[0]):
        for j in range(cnv.shape[1]):
            id = reduced_cell_img[i, j]
            if id > 0 and id <= len(protein_1):
                cnv[i, j] = protein_1[id - 1]

    return cnv

def precompile_jit():
    """Precompile the function in the background."""
    
    # Dummy protein data: Random values between 0 and 255
    dummy_protein_data = np.random.randint(0, 256, size=100, dtype=np.uint8)

    # Dummy reduced cell image: A 10x10 grid with random cell IDs
    dummy_reduced_cell_img = np.random.randint(0, 101, size=(10, 10), dtype=np.uint8)

    write_protein(dummy_protein_data, dummy_reduced_cell_img)




def tint_grayscale_image(grayscale_image, color):
    """
    Tint a grayscale image with an arbitrary color.

    Parameters:
    grayscale_image (numpy array): The input grayscale image array with shape (height, width).
    color (tuple): The RGB color to use for tinting (R, G, B).

    Returns:
    numpy array: The tinted image with shape (height, width, 3).
    """
    # Ensure the grayscale image is a 2D array
    if len(grayscale_image.shape) != 2:
        raise ValueError(
            "Input image must be a 2D array representing a grayscale image."
        )

    # Normalize the grayscale image to the range [0, 1]
    grayscale_image_normalized = grayscale_image / 255.0

    # Create an empty array with shape (height, width, 3) for the colored image
    tinted_image = np.zeros(
        (grayscale_image.shape[0], grayscale_image.shape[1], 3), dtype=np.uint8
    )

    # Apply the color tint
    for i in range(3):
        tinted_image[:, :, i] = grayscale_image_normalized * color[i]

    return tinted_image


import os
import numpy as np

import qtrangeslider
from PyQt6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QLabel,
    QSlider,
    QHBoxLayout,
    QGroupBox,
    QFormLayout,
    QScrollArea,
    QSizePolicy,
    QPushButton,
    QListWidget,
    QListWidgetItem,
    QDialog,
    QDialogButtonBox,
)
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QImage, QPixmap, QColor


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

        self.layout = QHBoxLayout()

        self.layer_list = QListWidget()
        for i, layer in enumerate(layers):
            item = QListWidgetItem(layer["name"])
            self.layer_list.addItem(item)
        self.layout.addWidget(self.layer_list)

        self.buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        self.buttons.accepted.connect(self.accept)
        self.buttons.rejected.connect(self.reject)
        self.layout.addWidget(self.buttons)

        self.setLayout(self.layout)

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

        self.layout = QVBoxLayout()

        self.color_list = QListWidget()
        for color_name in colors.keys():
            item = QListWidgetItem(color_name)
            self.color_list.addItem(item)
        self.layout.addWidget(self.color_list)

        self.buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        self.buttons.accepted.connect(self.accept)
        self.buttons.rejected.connect(self.reject)
        self.layout.addWidget(self.buttons)

        self.setLayout(self.layout)

    def get_selected_color_name(self):
        selected_items = self.color_list.selectedItems()
        if selected_items:
            return selected_items[0].text()
        return None


# changes made
def adjust_contrast(img, min=5, max=100):
    # pixvals = np.array(img)
    minval = np.percentile(img, min)  # room for experimentation
    maxval = np.percentile(img, max)  # room for experimentation
    img = np.clip(img, minval, maxval)
    img = ((img - minval) / (maxval - minval)) * 255
    return img.astype(np.uint8)


from PyQt6.QtWidgets import (
    QGraphicsView,
    QRubberBand,
    QGraphicsScene,
    QGraphicsPixmapItem,
    QGraphicsItem,
)


class ImageOverlay(QWidget):
    changePix = pyqtSignal(QGraphicsPixmapItem)
    progress = pyqtSignal(int, str)

    def __init__(self, pixmap_label, enc):
        super().__init__()

        self.enc = enc

        self.pixmap_label = pixmap_label

        # self.df_path =  "/Users/clark/Downloads/cell_data_8_8_Full_Dataset_Biopsy.xlsx"
        # self.im_path = "/Users/clark/Downloads/new_sd.png"

        self.df_path = "assets/sample_data/demo/df1.csv"
        self.im_path = "assets/sample_data/demo/sd1.png"
        self.overlay_path = "assets/sample_data/demo/sd1.png"

        self.controls = []

        self.loaded_df = None

        self.initUI()

    def load_stardist_image(self):
        if self.im_path == None:
            return None

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
        if self.df_path == None:
            return None

        if self.loaded_df is not None:
            return self.loaded_df

        # print("df path", self.df_path)
        if self.df_path.endswith("csv"):
            df = pd.read_csv(self.df_path)
        if self.df_path.endswith("xlsx"):
            df = pd.read_excel(self.df_path)

        # print("df raw", df)
        df = df[df.columns.drop(list(df.filter(regex="N/A")))]

        self.loaded_df = df
        return df
    
    def generate_image(self, index):

        
        # protein_name = self.df.columns[3 + index]
        # im = write_protein(np.array(self.df[protein_name]), np.array(self.reduced_cell_img))
        # im = adjust_contrast(im)
        # return tint_grayscale_image(im, [255, 255, 255])


        start = time.perf_counter()
        
        protein_name = self.df.columns[3 + index]
        print(f"Time after fetching protein name: {time.perf_counter() - start:.6f} sec")

        im = write_protein(np.array(self.df[protein_name]), np.array(self.reduced_cell_img))
        print(f"Time after writing protein: {time.perf_counter() - start:.6f} sec")

        im = adjust_contrast(im)
        print(f"Time after adjusting contrast: {time.perf_counter() - start:.6f} sec")

        result = tint_grayscale_image(im, [255, 255, 255])
        print(f"Time after tinting grayscale image: {time.perf_counter() - start:.6f} sec")

        return result
    
    
        
        

    def build_all(self):

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
        
        
        # Start compilation in a background thread

        start = time.time()
        self.progress.emit(15, "Compiling `write_protein`function...")
        

        threading.Thread(target=precompile_jit, daemon=True).start()

        reduced_cell_img = (self.load_stardist_image()).astype(np.uint16)
        self.reduced_cell_img = reduced_cell_img

        df = self.load_df()
        self.df = df

        end = time.time()

        print("TOTAL TIME ", end - start)
        # self.progress.emit(35, "Loaded Stardist image")
        
        
        print(" df loaded")
        # self.progress.emit(75, "Loaded Dataframe")

        

        ims = [None for i in range(len(df.columns[3:]))]

        # self.progress.emit(70, "Images generated")
        layer_names = list(df.columns[3:])

        self.progress.emit(96, "Almost complete...")

        print("df", df)

        self.ims = ims  # List of images as np.arrays
        self.color_dict = color_dict  # Dictionary of color names to RGB values

        self.layers = [
            {"name": layer_names[i], "image": ims[i]} for i in range(len(ims))
        ]

        self.progress.emit(100, "Done")

        self.scroll_area.setVisible(True)

        self.add_layer_button.setVisible(True)
        self.add_other_image_button.setVisible(True)
        self.open_image.setVisible(False)
        self.open_image_label.setVisible(False)
        self.open_df.setVisible(False)
        self.open_df_label.setVisible(False)

        self.scale_down_label.setVisible(False)
        self.scale_down.setVisible(False)

        # self.todo_label.setVisible(False)

        self.apply_button.setVisible(False)
        self.cancel_reset.setVisible(True)
        self.export_tif_button.setVisible(True)

        return (ims, layer_names)

    def get_layer_values_at(self, x, y):
        if len(self.controls) == 0:
            return None

        layer_values = []
        for c in self.controls:
            value = c.image[y, x]
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

        while self.enc.analysis_tab.delete_view():
            pass

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

    def initUI(self):
        main_layout = QVBoxLayout()

        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
        )
        self.scroll_area.setMinimumHeight(450)  # Set a reasonable minimum height
        self.scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)  # Disable horizontal scroll

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
        self.open_image.clicked.connect(self.load_example)
        main_layout.addWidget(self.open_image)

        self.open_image_label = QLabel("Path: ")
        self.open_image_label.setVisible(False)
        main_layout.addWidget(self.open_image_label)

        self.open_df = QPushButton("Open Cell Data")
        self.open_df.clicked.connect(self.get_df_path)
        main_layout.addWidget(self.open_df)

        self.open_df_label = QLabel("Path: ")
        self.open_df_label.setVisible(False)
        main_layout.addWidget(self.open_df_label)

        ### scale slider
        self.scale_down_label = QLabel("Scale Down Factor: ")
        main_layout.addWidget(self.scale_down_label)

        self.scale_down = QSlider(Qt.Orientation.Horizontal)
        self.scale_down.setTickPosition(QSlider.TicksAbove)
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
        
        # Add a spacer to ensure content can scroll all the way down
        main_layout.addStretch(1)  # Add stretch at the end to push content up
        
        self.setLayout(main_layout)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.update_image()

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

    def load_example(self):
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

    def get_df_path(self):

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
            selected_index = dialog.get_selected_layer_index()
            print("selected indexxxx is ", selected_index)
            if selected_index is not None:
                c = ControlsBox()

                print("potential error", selected_index)

                try:
                    if self.layers[selected_index]["image"] == None:
                        self.layers[selected_index]["image"] = self.generate_image(selected_index)
                except:
                    pass
                
                c.image = self.layers[selected_index]["image"]
                c.name = self.layers[selected_index]["name"]
                self.add_layer(c)

    def show_color_dialog(self, idx):
        print(idx)
        dialog = ColorDialog(self.color_dict, self)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            selected_color_name = dialog.get_selected_color_name()
            if selected_color_name:
                selected_color = self.color_dict[selected_color_name]
                print(f"Selected color: {selected_color_name} - {selected_color}")
                self.controls[idx].current_tint = QColor(*selected_color)
                self.controls[idx].tint_label.setText(selected_color_name)
                self.update_image()

    def add_layer(self, c):
        self.controls.append(c)
        self.add_layer_controls(c)
        self.update_image()

    def delete_layer(self, index):
        c = self.controls.pop(index)

        layer = c.layout
        self.scroll_layout.removeWidget(layer)
        layer.deleteLater()
        layer = None

        self.update_image()

        if len(self.controls) == 0:
            combined_image = np.zeros((10, 10, 10))

            height, width, _ = combined_image.shape
            bytes_per_line = 3

            q_image = QImage(
                combined_image.tobytes(), width, height, QImage.Format.Format_RGB888
            )  # interesting image.tobytes() works well, maybe you don't need to do

            self.changePix.emit(QGraphicsPixmapItem(QPixmap.fromImage(q_image)))

    def add_layer_controls(self, c):
        idx = len(self.controls) - 1

        group_box = QGroupBox(f"Layer {idx + 1}: {c.name}")
        group_box.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)

        group_layout = QFormLayout()
        group_layout.setSpacing(8)  # Add spacing between form rows

        opacity_slider = QSlider(Qt.Orientation.Horizontal)
        opacity_slider.setMinimum(0)
        opacity_slider.setMaximum(100)
        opacity_slider.setValue(100)
        opacity_slider.valueChanged.connect(
            lambda value: self.update_opacity(value, idx)
        )

        group_layout.addRow("Opacity:", opacity_slider)

        contrast_slider = qtrangeslider.QLabeledDoubleRangeSlider(
            Qt.Orientation.Horizontal
        )
        contrast_slider.valueChanged.connect(
            lambda value: self.update_contrast(value, idx)
        )
        contrast_slider.setMaximum(255)
        contrast_slider.setValue((0, 255))
        contrast_slider.setDecimals(0)

        # self.contrast_sliders.append(contrast_slider)
        group_layout.addRow("Contrast:", contrast_slider)

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

    def update_opacity(self, value, idx):
        self.controls[idx].current_opacity = value / 100.0

        # self.current_opacities[idx] = value / 100.0

        self.update_image()

    def update_contrast(self, value, idx):
        value = [int(value[0]), int(value[1])]
        self.controls[idx].current_contrast = value
        self.update_image()

    def update_visibility(self, checked, idx):

        self.controls[idx].current_visibility = checked
        self.update_image()

    def apply_tint(self, img, color):

        tint_img = np.zeros_like(img)
        for c in range(3):
            tint_img[:, :, c] = img[:, :, c] * (color.getRgb()[c] / 255.0)
        return tint_img

    def adjust_contrast(self, img, min=5, max=100):
        # pixvals = np.array(img)
        minval = np.percentile(img, min)  # room for experimentation
        maxval = np.percentile(img, max)  # room for experimentation
        img = np.clip(img, minval, maxval)
        img = ((img - minval) / (maxval - minval)) * 255
        return img.astype(np.uint8)

    def update_image(self):
        if len(self.controls) == 0:
            return

        combined_image = np.zeros_like(self.controls[0].image, dtype=np.float32)

        for c in self.controls:
            visible = c.current_visibility
            if visible:
                img = c.image

                opacity = c.current_opacity
                contrast = c.current_contrast
                tint = c.current_tint

                adjusted_img = img * 1.0
                # adjusted_img = winsorize_array(adjusted_img, 0, 255)
                if c.tint_yn:
                    if type(contrast) == type([]):
                        adjusted_img = np.clip(adjusted_img, contrast[0], contrast[1])
                        adjusted_img = scale_image_to_255(adjusted_img)
                        adjusted_img = self.apply_tint(adjusted_img, tint)
                        adjusted_img = np.clip(
                            adjusted_img, 0, 255
                        )  # Clip values to stay in the valid range

                if not c.tint_yn:
                    adjusted_img = scale_image_to_255(adjusted_img)

                combined_image += adjusted_img * opacity

        combined_image = np.clip(combined_image, 0, 255).astype(np.uint8)

        height, width, _ = combined_image.shape
        bytes_per_line = 3

        q_image = numpy_to_qimage(combined_image)


        # commented out
        self.changePix.emit(QGraphicsPixmapItem(QPixmap.fromImage(q_image)))

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
            if True:
                img = c.image.copy()
                
                # Get original protein data in grayscale
                # If the image has 3 channels (RGB), convert to grayscale
                if len(img.shape) == 3 and img.shape[2] == 3:
                    img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
                else:
                    img_gray = img
                
                # Apply contrast adjustment if needed
                if type(c.current_contrast) == type([]):
                    img_gray = np.clip(img_gray, c.current_contrast[0], c.current_contrast[1])
                
                # Scale to 0-255 range
                img_gray = scale_image_to_255(img_gray)
                
                # Add to our stack
                layers_data.append(img_gray)
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
            with open(txt_file, 'w') as f:
                for i, name in enumerate(layer_names):
                    f.write(f"Layer {i+1}: {name}\n")
                    
            QMessageBox.information(None, "Success", 
                f"Multi-layered TIF file saved to {file_name}\n"
                f"Each layer contains a separate protein in grayscale\n"
                f"Protein order saved to {txt_file}")
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
