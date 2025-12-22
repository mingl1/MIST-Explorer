import itertools

import cv2 as cv
import numpy as np
import pandas as pd
from PyQt6.QtCore import QThread, pyqtSignal
from PyQt6.QtWidgets import QFileDialog
from scipy.spatial import KDTree
from skimage.measure import regionprops

from core import ImageWrapper
from core.canvas import ImageStorage


class CellIntensity(QThread):
    error_signal = pyqtSignal(str)
    progress = pyqtSignal(int, str)

    def __init__(self):
        super().__init__()
        self.params = {
            "max_size": 23000,
            "num_decoding_cycles": 3,
            "num_decoding_colors": 3,
            "radius_fg": 2,
            "radius_bg": 6,
        }

        self.channel_to_color_code = {}
        self.stardist_labels = np.array([], dtype=np.uint8)
        self.df_cell_data = None
        self.storage = ImageStorage()
        self.protein_signal_array = None

    def load_protein_signal_array_from_storage(self, uuid, channel):
        if uuid is None:
            uuid = self.storage.get_data("canvas_uuid")
            if uuid is None:
                raise ValueError("Protein Image not found in storage.")
            uuid = uuid["value"]
        c = "Channel " + str(channel + 1)
        item = self.storage.get_data(uuid)
        assert item is not None, "item not found in storage"
        data = item.get("data", None)
        assert data is not None, "data not found in storage item"

        self.load_protein_signal_array(data[c].data)
        print("loaded protein signal array from storage")

    def load_stardist_labels_from_storage(self, uuid, channel):
        item = self.storage.get_data(uuid)
        assert item is not None, "item not found in storage"
        data = item.get("data", None)
        assert data is not None, "data not found in storage item"
        c = "Channel " + str(channel + 1)
        stardist_labels = data[c]
        self.load_stardist_labels(stardist_labels)
        print("loaded stardist labels from storage")

    def load_protein_signal_array(self, arr):
        print("loaded protein signal array")
        self.protein_signal_array = arr
        self.blur_and_set_protein_layer()

    def generate_cell_intensity_table(self):
        self.progress.emit(0, "Starting Cell Intensity...")
        if self.isRunning():
            self.critical_error(
                "Cell Intensity Calculation is already running")
            return
        self.start()

    def critical_error(self, msg):
        self.error_signal.emit(msg)
        self.progress.emit(100, "Error encountered, see message")
        return

    def compute_all_centroids(self):
        """
        Compute centroids for all unique labels in the mask (excluding 0).
        Returns a dict: {label: (cx, cy)}
        """
        centroids = {}
        m = np.max(self.stardist_labels)
        for region in regionprops(self.stardist_labels):
            progress = int((region.label / m) * 100)
            self.progress.emit(
                progress,
                f"Finding centroid for cell {region.label}/{m}",
            )
            cy, cx = region.centroid
            centroids[region.label] = (int(cx), int(cy))
        return centroids

    def infer_params(self):
        # Infer the number of decoding colors and cycles from the bead data and color code if not explicitly set
        # based on any of the color code files
        self.params["num_decoding_cycles"] = (
            self.color_code.columns.size - 1
        )  # minus 1 for protein name column

        color_code_np = self.color_code.iloc[:, 1:].to_numpy()
        max_color_value = np.max(color_code_np)
        self.params["num_decoding_colors"] = (
            max_color_value + 1
        )  # assuming colors are 0-indexed

    def run(self):

        # need the aligned and segmented cell image, bead_data, and color_code

        for channel, color_code in self.channel_to_color_code.items():
            self.color_code = color_code
            channel = int(channel.split(" ")[-1]) - 1
            print(f"generating channel {channel}")
            # this func expects channel to be zero-indexed integer.
            self.load_protein_signal_array_from_storage(None, channel)
            if (
                self.stardist_labels is None
                or self.bead_data is None
                or self.channel_to_color_code is None
                or self.protein_signal_array is None
            ):
                err_msg = "Missing: "
                if self.stardist_labels is None:
                    err_msg += "stardist labels, "
                if self.bead_data is None:
                    err_msg += "bead data, "
                if self.channel_to_color_code is None:
                    err_msg += "color code, "
                if self.protein_signal_array is None:
                    err_msg += "protein signal array, "
                err_msg = err_msg.rstrip(", ")  # remove trailing comma
                print(err_msg)
                self.critical_error(err_msg)
                return
            else:
                # calculate all possible combinations to id the protein, for 4 colors and 2 cycles, it would be 4^2 = 16 combinations
                # then, for each combination, convert it to index. So for
                # example, 16 combinations would be indexed 0 to 15.
                self.infer_params()
                print("Inferred params:", self.params)
                possible_values = list(
                    range(self.params["num_decoding_colors"]))
                all_perms = [
                    "".join(map(str, p))
                    for p in itertools.product(
                        possible_values, repeat=self.params["num_decoding_cycles"]
                    )
                ]
                color_code_to_index = {
                    int(k): i for i, k in enumerate(all_perms)}
                index_to_color_code = {
                    v: k for k, v in color_code_to_index.items()}

                # This is the structure created. For example, cell image with three unique labels + 5 proteins would look like this:
                # {
                # 1: [[], [], [], [], []],
                # 2: [[], [], [], [], []],
                # 3: [[], [], [], [], []]
                #     }
                num_proteins = len(color_code_to_index)
                max_cell_id = np.max(self.stardist_labels)
                cell_data_dict = {
                    cell_id: [[] for _ in range(num_proteins)]
                    for cell_id in range(1, max_cell_id + 1)
                }
                cycle_cols = self.bead_data[
                    :, 2: 2 + self.params["num_decoding_cycles"]
                ]
                data_modified = np.zeros((len(self.bead_data), 3))
                data_modified[:, 0:2] = self.bead_data[:, 0:2].astype("uint16")
                data_modified[:, 2] = np.array(
                    [int("".join(map(str, map(int, bead)))) for bead in cycle_cols]
                )

                radius_bg = self.params["radius_bg"]
                max_size = self.params["max_size"]
                # filter out beads that are not within bounds of
                # stardist_labels
                x_limit, y_limit = (
                    self.stardist_labels.shape[1],
                    self.stardist_labels.shape[0],
                )
                print("before filtering", data_modified.shape)
                data_modified = [
                    bead
                    for bead in data_modified
                    if bead[0] < x_limit and bead[1] < y_limit
                ]
                data_modified = np.array(data_modified)
                print("after filtering", data_modified.shape)
                # --- Replace your entire 'for' loop with this block ---

                self.progress.emit(25, "Finding beads within cells...")

                # --- 1. Vectorized Lookup and Filtering ---

                # Get all bead coordinates (assuming x is column 0, y is column 1)
                # Ensure they are integers for indexing
                bead_xs = data_modified[:, 0].astype(int)
                bead_ys = data_modified[:, 1].astype(int)

                # Get the cell ID for every bead in a single, fast operation
                # This is the core of the vectorization!
                cell_ids_for_beads = self.stardist_labels[bead_ys, bead_xs]

                # --- 2. Create Boolean Masks for All Conditions ---

                # Mask 1: Beads that are inside any cell (ID > 0)
                in_cell_mask = cell_ids_for_beads > 0

                # Mask 2: Beads that are within the processing boundaries
                # (This prevents errors in get_adjusted_median_intensity)
                in_bounds_mask = (
                    (bead_xs > radius_bg)
                    & (bead_ys > radius_bg)
                    & (bead_xs < (max_size - radius_bg))
                    & (bead_ys < (max_size - radius_bg))
                )

                # --- 3. Combine Masks ---
                # The final mask identifies beads that satisfy ALL conditions
                valid_bead_mask = in_cell_mask & in_bounds_mask

                # remove in_cell_mask and edit self.stardist_labels to include psuedo-cells,
                # note the labels should start with max(self.stardist_labels)+1;
                # after editing self.stardist_labels, rest of code shouldn't
                # need to be changed

                # --- 4. Filter the Data ---
                # Create a much smaller array containing only the beads we need
                # to process
                valid_beads = data_modified[valid_bead_mask]
                valid_cell_ids = cell_ids_for_beads[valid_bead_mask]

                # --- 5. Loop Over the SMALLER Filtered Dataset ---
                self.progress.emit(
                    50, f"Processing {len(valid_beads)} valid beads...")

                # This loop is much faster because it runs only on the subset
                # of relevant beads
                for i, bead in enumerate(valid_beads):
                    # Only update progress occasionally
                    if i % 1000 == 0:
                        self.progress.emit(
                            50
                            + int(
                                (i / len(valid_beads)) * 25
                            ),  # Progress from 50% to 75%
                            f"Adjusting bead intensity {i+1}/{len(valid_beads)}",
                        )

                    bead_x, bead_y, color_code = int(
                        bead[0]), int(bead[1]), bead[2]

                    # We already know this bead is in a cell, so we get its ID
                    cell_associated_id = valid_cell_ids[i]

                    # The expensive calculation is only called for valid beads
                    adjusted_median_intensity = self.get_adjusted_median_intensity(
                        bead_x, bead_y)

                    protein_idx = color_code_to_index.get(color_code)
                    if (
                        protein_idx is not None
                        and adjusted_median_intensity is not None
                    ):
                        cell_data_dict[cell_associated_id][protein_idx].append(
                            adjusted_median_intensity
                        )

                # group every bead location (x,y) by the color code index
                # (protein), store in KDTree for efficient nearest neighbor
                # search
                protein_kdtree_map = {}
                for i in range(num_proteins):
                    protein_code = index_to_color_code.get(i)
                    if protein_code is not None:
                        protein_beads = data_modified[
                            data_modified[:, 2] == protein_code
                        ][:, 0:2].astype(int)
                        if len(protein_beads) > 0:
                            protein_kdtree_map[i] = KDTree(protein_beads)
                # find centerpoint of every cell

                cell_centroids = self.compute_all_centroids()

                # find how many are different
                print("Finding values for cells with incomplete protein profiles")
                for i, cell_id in enumerate(cell_data_dict.keys()):
                    cell_center = cell_centroids[cell_id]
                    progress_update = int(
                        ((i + 1) / len(cell_data_dict)) * 100)
                    self.progress.emit(
                        progress_update,
                        f"Finding values for cells with incomplete protein profiles {i+1}/{len(cell_data_dict)}",
                    )
                    for protein_idx, intensities in enumerate(
                            cell_data_dict[cell_id]):
                        if (
                            not intensities
                        ):  # If no beads were found for this protein of cell_id
                            kdtree = protein_kdtree_map.get(protein_idx)
                            if kdtree:  # Check if a tree was successfully built
                                _, index = kdtree.query(cell_center)
                                nn_x, nn_y = kdtree.data[index]
                                if (
                                    nn_x > radius_bg
                                    and nn_y > radius_bg
                                    and nn_x < (max_size - radius_bg)
                                    and nn_y < (max_size - radius_bg)
                                ):
                                    adjusted_intensity = (
                                        self.get_adjusted_median_intensity(
                                            int(nn_x), int(nn_y)
                                        )
                                    )
                                    if adjusted_intensity is not None:
                                        cell_data_dict[cell_id][protein_idx].append(
                                            adjusted_intensity)
                    # use the median value for intensity for each protein in
                    # each cell.
                self.progress.emit(
                    0,
                    f"Finishing Up",
                )
                median_values_for_cell_data_dict = {}

                # set up as before for `cell_data_dict`
                for cell_id in cell_data_dict:
                    median_values_for_cell_data_dict[cell_id] = []
                    for i in range(num_proteins):
                        median_values_for_cell_data_dict[cell_id].append([])

                # populate with medians
                for cell_id in cell_data_dict:
                    array_of_subarrays = cell_data_dict[cell_id]
                    array_of_subarrays_medians = [
                        np.median(subarr) for subarr in array_of_subarrays
                    ]

                    median_values_for_cell_data_dict[cell_id] = (
                        array_of_subarrays_medians
                    )
                self.progress.emit(
                    25,
                    f"Finishing Up",
                )
                # drop rows with NaN that pandas includes for some reason lol
                # assert isinstance(self.color_code, pd.DataFrame)
                try:
                    self.color_code = self.color_code.dropna(
                        how="all", axis=1).dropna(how="all", axis=0)
                except Exception as e:
                    self.color_code = pd.DataFrame(self.color_code)
                    self.color_code = self.color_code.dropna(
                        how="all", axis=1).dropna(how="all", axis=0)
                color_code = self.color_code.to_numpy()
                self.progress.emit(
                    50,
                    f"Finishing Up",
                )
                # lets us go from code -> protein i.e. 112 -> Fox3 or whatever
                color_code_translation_dict = {}
                for row in color_code:
                    try:
                        protein_name = row[0]
                        code = int("".join([str(int(x)) for x in row[1:]]))
                        color_code_translation_dict[code] = protein_name
                    except ValueError:
                        code = None

                # then we use this to build the header string
                header = ["Global X", "Global Y"]
                for subarray_index in index_to_color_code:
                    coresponding_protein_code = index_to_color_code[subarray_index]
                    # print(coresponding_protein_code)
                    if coresponding_protein_code in color_code_translation_dict:
                        readable_protein_name = color_code_translation_dict[
                            coresponding_protein_code
                        ]
                        header.append(readable_protein_name)
                    else:
                        header.append("N/A")
                self.progress.emit(
                    75,
                    f"Finishing Up",
                )
                # Now get all the data out of the subarrays
                save_this = np.array(
                    [v for k, v in median_values_for_cell_data_dict.items()]
                )
                # and all the centroid data
                save_this = np.hstack(
                    ([v for k, v in cell_centroids.items()], save_this)
                )
                # and finally save everything
                if self.df_cell_data is None:

                    self.df_cell_data = pd.DataFrame(
                        save_this, columns=header
                    )  # --> use this to visualize
                else:
                    curr_cell_data = pd.DataFrame(save_this, columns=header)
                    self.df_cell_data = self.df_cell_data.merge(
                        curr_cell_data, on=["Global X", "Global Y"]
                    )
        self.progress.emit(100, "Cell Data is Generated")

    # !TODO: need to implement checking if self.isInterruptionRequested() inside run()
    def cancel(self):
        self.requestInterruption()

    def set_color_codes(self, channel_to_code):
        assert isinstance(channel_to_code, dict)
        self.channel_to_color_code = channel_to_code.copy()

    def save_cell_data(self):
        print("saving cell data")
        file_name, _ = QFileDialog.getSaveFileName(
            None, "Save Cell Data File", "cell_data.csv", "*.csv;;*.xlsx;; All Files(*)")
        if self.df_cell_data is not None:
            self.df_cell_data.to_csv(file_name, index=False)
        else:
            self.critical_error("Cannot save. No cell data available")

    def get_adjusted_median_intensity(
            self, bead_x, bead_y, bead_median_threshold=5000):
        """
        Calculate the adjusted median intensity given the bead coordinates

        :param bead_x: The x-coordinate of the bead
        :param bead_y: The y-coordinate of the bead
        :param bead_median_threshold: the threshold needed to apply median intensity correction
        :type bead_x: int
        :type bead_y: int
        :type bead_median_threshold: int

        :returns: The adjusted median intensity value of the bead
        :rtype: float
        """

        if self.protein_signal_array is None:
            return

        radius_bg = self.params["radius_bg"]
        radius_fg = self.params["radius_fg"]

        # Extract the 5x5 region around the bead
        bead_region = self.protein_signal_array[
            bead_y - radius_fg: bead_y + radius_fg + 1,
            bead_x - radius_fg: bead_x + radius_fg + 1,
        ]

        # Calculate the mean and median intensity of the 5x5 bead region
        mean_5x5 = np.mean(bead_region)
        bead_median_org = np.median(bead_region)
        bead_median = bead_median_org.copy()

        # Extract the 15x15 surrounding region
        surrounding_region = self.protein_signal_array[
            bead_y - radius_bg: bead_y + radius_bg + 1,
            bead_x - radius_bg: bead_x + radius_bg + 1,
        ]  # Convert to float to handle NaN values

        # Ensure the 15x15 region is valid
        if surrounding_region.shape != (15, 15):
            return bead_median_org  # Return unadjusted median if the 15x15 region is invalid

        # Mask out the 5x5 region from the 15x15 region
        surrounding_region[
            bead_y - radius_fg: bead_y + radius_fg + 1,
            bead_x - radius_fg: bead_x + radius_fg + 1,
        ] = 0

        # Calculate the mean intensity of the surrounding 15x15 area, excluding
        # the 5x5 region
        surrounding_mean_15x15 = np.nanmean(surrounding_region)

        # Apply correction only if 15x15 mean is 1.5x greater than 5x5 mean,
        # and bead median > threshold
        if (
            surrounding_mean_15x15 > 1.5 * mean_5x5
            and bead_median > bead_median_threshold
        ):
            # Calculate the correction factor and apply linear correction
            correction_factor = mean_5x5 * (mean_5x5 / surrounding_mean_15x15)
            y = self.linear_correction(correction_factor)

            # Apply the correction to the bead median
            bead_median = bead_median - y + 2000

        # Ensure no negative values
        if bead_median < 1:
            bead_median = 1

        # Return the final adjusted bead median
        return bead_median

    def linear_correction(self, x):
        """Define the linear function for the correction equation"""
        return 0.8266 * x + 3970.1

    def load_stardist_labels(self, stardist: ImageWrapper) -> None:
        print("stardist label dtype:", stardist.data.dtype)
        print(
            "stardist label max and min", np.max(
                stardist.data), np.min(
                stardist.data))
        self.stardist_labels = stardist.data

    def set_bead_data(self, bead_data):
        if isinstance(bead_data, np.ndarray):
            self.bead_data = bead_data

    def set_radius_fg(self, value):
        self.params["radius_fg"] = value

    def set_radius_bg(self, value):
        self.params["radius_bg"] = value

    def blur_and_set_protein_layer(self, blur_percentage=1):
        """
        Applies Gaussian blur to the 4th layer (index 3) of the image stack and subtracts
        the specified percentage of the blurred image from the original.
        """
        layer4 = self.protein_signal_array
        blurred_mask = cv.GaussianBlur(layer4, (101, 101), 0)
        blurred_mask_adjusted = (
            blurred_mask *
            blur_percentage).astype(
            np.uint16)
        corrected_layer4 = cv.subtract(layer4, blurred_mask_adjusted)
        corrected_layer4 = np.clip(
            corrected_layer4, 0, 65535).astype(
            np.uint16)
        self.protein_signal_array = corrected_layer4
        return True
