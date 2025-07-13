import concurrent.futures
import trace
import traceback
from calendar import c
from typing import Tuple, final

import cv2
import numpy as np
import SimpleITK as sitk
import tifffile
from itk import elxParameterObjectPython, itkElastixRegistrationMethodPython
from numpy.typing import NDArray
from PyQt6.QtCore import QThread, pyqtSignal
from pystackreg import StackReg
from pystackreg.util import to_uint16
from pytools import P
from scipy.ndimage import affine_transform, binary_fill_holes, rotate, zoom

from utils import (
    adjust_contrast,
    make_same_shape,
    match_histograms,
    remove_padding,
    to_uint8,
    warp_image,
)


class CellLayerAligner(QThread):
    """Two Staged Cell Layer Alignment Using ITK Intensity Based Alignment & pystackreg's Algorithm"""

    progress = pyqtSignal(int, str)
    error = pyqtSignal(str)
    aligned_image_signal = pyqtSignal(dict, np.ndarray, np.ndarray)

    def __init__(self, rotate_by=90, coarse_scale=1 / 16.0, fine_scale=0.5):
        super().__init__()
        self.target_image = np.zeros(0).astype(np.uint16)
        self.unaligned_image = np.zeros(0).astype(np.uint16)
        self.original_target_shape = None
        self.replace = False
        self.target_channel = "Channel 1"
        self.target_uuid = ""
        self.unaligned_channel = "Channel 1"
        self.unaligned_uuid = ""
        self.debug = False  # Set to True for debugging
        # itk parameters, it takes a while to initialize but it is done lazily, if we can
        # intialize it here, it will shave 10s (?); i believe it is not initializing currently
        parameter_object = elxParameterObjectPython.elastixParameterObject_New()
        default = parameter_object.GetDefaultParameterMap("affine")
        parameter_object.AddParameterMap(default)
        self.parameter_object = parameter_object

        # Coarse and fine alignment parameters
        self.coarse_scale = coarse_scale
        self.rotation_angles = list(range(0, 360, rotate_by))
        # self.rotation_angles = [0]
        self.fine_scale = fine_scale

    def set_target_image(self, target_image, channel_name, uuid):
        """Set target image and clear pyramid cache"""
        self.target_pyramid = []
        self.target_image = target_image
        self.original_target_shape = target_image.shape
        self.target_channel = channel_name
        self.target_uuid = uuid

    def set_unaligned_image(self, unaligned_image, channel_name, uuid):
        """Set unaligned image and clear pyramid cache"""
        self.unaligned_pyramid = []
        self.unaligned_image = unaligned_image
        self.unaligned_channel = channel_name
        self.unaligned_uuid = uuid

    def run(self):
        """Main processing function that runs in the thread"""
        if self.target_image is None or self.unaligned_image is None:
            self._fatal_error_message(
                "Both target and unaligned images must be provided"
            )
            return

        try:
            # Ensure images have the same shape for alignment
            self.progress.emit(5, "Preparing images")
            self.coarse_target, self.coarse_moving = self._prepare_images(
                self.target_image, self.unaligned_image, self.coarse_scale
            )
            self.progress.emit(30, "Intensity-based coarse alignment")
            coarse_transform = self._coarse_alignment()
            if self.debug:
                tifffile.imwrite("coarse_target.tif", self.coarse_target)
                tifffile.imwrite("coarse_moving.tif", self.coarse_moving)
                print("Coarse transform matrix:\n", coarse_transform)
                coarse_aligned = warp_image(self.coarse_moving, coarse_transform)
                tifffile.imwrite("coarse_aligned.tif", coarse_aligned)
            if coarse_transform is None:
                return

            fine_transform = self._scale_transform_matrix(
                coarse_transform[:2, :3].copy(), self.coarse_scale, self.fine_scale
            )
            fine_target, fine_unaligned = self._prepare_images(
                self.target_image, self.unaligned_image, self.fine_scale
            )
            if self.debug:
                tifffile.imwrite("fine_target.tif", fine_target)
                tifffile.imwrite("fine_unaligned.tif", fine_unaligned)
            fine_target_shape = (
                np.array(self.target_image.shape) * self.fine_scale
            ).astype(int)

            fine_moving = warp_image(fine_unaligned, fine_transform)
            fine_moving = to_uint8(fine_moving)
            fine_moving = remove_padding(fine_moving, fine_target_shape)
            fine_target_image = remove_padding(fine_target, fine_target_shape)
            if self.debug:
                tifffile.imwrite("fine_movinge.tif", fine_moving)
                tifffile.imwrite("fine_target_image.tif", fine_target_image)

            self.progress.emit(60, "StackReg fine alignment")
            refinement_transform, aligned_preview = self._alignment_stackreg(
                fine_target_image, fine_moving
            )

            if refinement_transform is None:
                print("StackReg refinement failed, skipping fine alignment")
                aligned_preview = fine_moving
                refinement_transform = np.eye(3)  # Identity transform
                return

            # Combine transformations and apply to full resolution
            self.progress.emit(80, "Applying final transformation")
            full_coarse_transform = self._scale_transform_matrix(
                coarse_transform[:2, :3], self.coarse_scale, 1
            )
            full_refinement_transform = self._scale_transform_matrix(
                refinement_transform[:2, :3], self.fine_scale, 1
            )
            print("Full refinement transform", full_refinement_transform)

            # Apply transformations sequentially for better accuracy
            _, padded_moving = make_same_shape(self.target_image, self.unaligned_image)
            intermediate_aligned = warp_image(padded_moving, full_coarse_transform)
            if self.debug:
                tifffile.imwrite("intermediate_aligned.tif", intermediate_aligned)
            intermediate_aligned = remove_padding(
                intermediate_aligned, self.target_image.shape
            )
            if self.debug:
                tifffile.imwrite(
                    "intermediate_aligned_after_padding.tif", intermediate_aligned
                )
            final_aligned_image = warp_image(
                intermediate_aligned, full_refinement_transform
            )

            target_preview = fine_target_image
            if self.debug:
                print(final_aligned_image.shape, final_aligned_image.dtype)
                tifffile.imwrite(
                    "final_aligned_image_raw.tif",
                    final_aligned_image,
                    photometric="minisblack",
                    imagej=True,
                )

            # Convert result back to original dtype
            result = self._convert_to_original_dtype(
                final_aligned_image, self.unaligned_image.dtype
            )
            if self.debug:
                print("Final aligned image dtype:", result.dtype)
                tifffile.imwrite(
                    "final_aligned_image_converted.tif",
                    result,
                    photometric="minisblack",
                    imagej=True,
                )
            result = {
                "uuid": self.target_uuid,
                "layer": self.target_channel,
                "replace": self.replace,
                "data": result,
            }
            self.progress.emit(100, "Two-stage alignment complete")
            print(
                "Preview image correlation:",
                calculate_alignment_metrics(target_preview, aligned_preview),
            )
            self.aligned_image_signal.emit(result, target_preview, aligned_preview)

        except Exception as e:
            self._fatal_error_message(f"Error during alignment: {str(e)}")

    def _coarse_alignment(self):
        moving, _ = self.coarse_moving, self.coarse_target
        angle, flip, params = self._itk_align(
            self.parameter_object,
            self.rotation_angles,
            self.coarse_target,
            self.coarse_moving,
        )
        # if self.debug:
        # tifffile.imwrite("best_coarse_result.tif", best_result)
        params = np.array(params)
        transform_info = extract_complete_transformation(
            params,  # ITK parameters from registration
            angle,  # Angle used in preprocessing
            flip,  # Flip used in preprocessing
            moving.shape,  # Original moving image shape
        )
        t = transform_info["combined_matrix"]
        if t.shape == (2, 3):
            t = np.vstack([t, [0, 0, 1]])
        inverted = np.linalg.inv(t)

        return inverted

    def _prepare_images(
        self, target_image, unaligned_image, coarse_scale
    ) -> Tuple[NDArray[np.uint8], NDArray[np.uint8]]:
        """
        Preprocess two images for coarse alignment.

        Args:
            target_image: The reference/target image
            unaligned_image: The image to be aligned to the target
            coarse_scale: Scale factor for resizing (e.g., 0.25 for 25% of original size)

        Returns:
            tuple: (coarse_target, coarse_moving) - preprocessed images
        """
        # Resize images
        coarse_target = cv2.resize(
            target_image,
            (0, 0),
            fx=coarse_scale,
            fy=coarse_scale,
            interpolation=cv2.INTER_AREA,
        )
        coarse_moving = cv2.resize(
            unaligned_image,
            (0, 0),
            fx=coarse_scale,
            fy=coarse_scale,
            interpolation=cv2.INTER_AREA,
        )

        # Convert to uint8
        coarse_target = to_uint8(coarse_target)
        coarse_moving = to_uint8(coarse_moving)

        # Histogram matching
        target_histogram = np.histogram(coarse_target.flatten(), bins=256)[0]
        coarse_moving = match_histograms(coarse_moving, target_histogram)

        # Clip and adjust moving image
        coarse_moving = np.clip(coarse_moving, 32, 255) - 32

        # Adjust contrast for both images
        coarse_target = adjust_contrast(coarse_target.astype(np.float64), 50, 99)
        coarse_moving = adjust_contrast(coarse_moving.astype(np.float64), 50, 99)

        # Normalize both images
        coarse_moving = cv2.normalize(
            coarse_moving, coarse_moving, 255, 0, cv2.NORM_MINMAX, cv2.CV_8U
        )
        coarse_target = cv2.normalize(
            coarse_target, coarse_target, 255, 0, cv2.NORM_MINMAX, cv2.CV_8U
        )

        # Apply morphological opening
        coarse_moving = morph_open(coarse_moving)
        # coarse_target = morph_open(coarse_target)

        # Ensure both images have the same shape
        coarse_target, coarse_moving = make_same_shape(coarse_target, coarse_moving)
        assert isinstance(coarse_target, np.ndarray)
        assert isinstance(coarse_moving, np.ndarray)
        assert coarse_target.dtype == np.uint8
        assert coarse_moving.dtype == np.uint8
        # cleaned_fixed = binary_fill_holes(coarse_target > 0) * 255
        # cleaned_fixed = cleaned_fixed.astype(np.uint8)
        # cleaned_moving = binary_fill_holes(coarse_moving > 0) * 255
        # cleaned_moving = cleaned_moving.astype(np.uint8)
        return coarse_target, coarse_moving

    def _itk_align(self, parameter_object, rotation_angles, fixed, moving):
        print("Stage 1: Finding best angles...")
        angle_results = []

        with concurrent.futures.ThreadPoolExecutor(max_workers=6) as executor:
            futures = []
            for angle in rotation_angles:
                futures.append(
                    executor.submit(
                        register_combination,
                        fixed,
                        moving,
                        angle,
                        False,
                        parameter_object,
                    )
                )

            for future in concurrent.futures.as_completed(futures):
                try:
                    result = future.result()
                    angle_results.append(result)
                except Exception as e:
                    print(f"Error during registration: {e}")
                    traceback.print_exc()

        # Sort by score and get top 3 valid results
        valid_angle_results = [
            r for r in angle_results if r[3] is not None
        ]  # Filter out None results
        sorted_angle_results = sorted(
            valid_angle_results, key=lambda x: x[0], reverse=True
        )
        top3_angles = sorted_angle_results[:3]

        print("Top 3 angles:")
        for i, (score, angle, _, params) in enumerate(top3_angles):
            print(f"  {i+1}. Angle {angle}: score = {score}")

        # Stage 2: Test flip options for top 3 angles
        print("Stage 2: Testing flip options for top 3 angles...")
        flip_results = []

        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
            futures = []
            for score, angle, _, params in top3_angles:
                futures.append(
                    executor.submit(
                        register_combination,
                        fixed,
                        moving,
                        angle,
                        True,
                        parameter_object,
                    )
                )

            for future in concurrent.futures.as_completed(futures):
                result = future.result()
                flip_results.append(result)

        # Combine all results and find the best overall
        all_results = (
            sorted_angle_results + flip_results
        )  # Use all angle results, not just top 3
        valid_all_results = [
            r for r in all_results if r[3] is not None
        ]  # Filter out None results
        final_sorted_results = sorted(
            valid_all_results, key=lambda x: x[0], reverse=True
        )

        if not final_sorted_results:
            raise RuntimeError("No valid alignment found.")

        best_score, best_angle, best_flip, params = final_sorted_results[0]

        print(f"Final best: angle={best_angle}, flip={best_flip}, score={best_score}")

        # Extract transform matrix

        return best_angle, best_flip, params

    def _alignment_stackreg(self, target_fine, unaligned_fine):
        """Enhanced StackReg alignment with better preprocessing"""
        try:
            # assert target_fine.dtype == np.float64
            # assert unaligned_fine.dtype == np.float64

            # Enhanced contrast adjustment
            # unaligned_matched = self._match_target_histogram(
            #     target_fine, unaligned_fine, clipped=True
            # )
            if self.debug:
                print(
                    "Before enhancement: target min",
                    target_fine.min(),
                    "max",
                    target_fine.max(),
                )
                print(
                    "Before enhancement: unaligned min",
                    unaligned_fine.min(),
                    "max",
                    unaligned_fine.max(),
                )
                print("Target dtype:", target_fine.dtype)
                print("Unaligned dtype:", unaligned_fine.dtype)
                # tifffile.imwrite("input_stackreg_target.tif", target_fine)
                # tifffile.imwrite("input_stackreg_unaligned.tif", unaligned_fine)
            target_enhanced, unaligned_enhanced = self._image_enhancement(
                target_fine.copy(), unaligned_fine.copy()
            )
            if self.debug:
                tifffile.imwrite("target_enhanced.tif", target_enhanced)
                tifffile.imwrite("unaligned_enhanced.tif", unaligned_enhanced)
            # Convert to uint16 for StackReg
            target_uint16 = to_uint16(target_enhanced)
            unaligned_uint16 = to_uint16(unaligned_enhanced)

            # Use StackReg with affine transformation
            sr = StackReg(StackReg.AFFINE)
            aligned_result = sr.register_transform(target_uint16, unaligned_uint16)
            refinement_matrix = sr.get_matrix()

            # Validate transformation matrix
            # if not self._is_valid_transform(refinement_matrix):
            #     print("Invalid transformation matrix detected")
            #     return None, None
            aligned_result = warp_image(unaligned_fine, refinement_matrix)
            return refinement_matrix[:2, :3], aligned_result

        except Exception as e:
            print(f"StackReg refinement error: {e}")
            return None, None

    def _image_enhancement(self, target, moving):
        """Original SIFT enhancement method"""

        target = to_uint8(target)
        moving = to_uint8(moving)
        cleaned_fixed = binary_fill_holes(target > 0) * 255
        cleaned_fixed = cleaned_fixed.astype(np.uint8)
        cleaned_moving = binary_fill_holes(moving > 0) * 255
        cleaned_moving = cleaned_moving.astype(np.uint8)
        # Step 3: Histogram matching
        # target_histogram = np.histogram(target.flatten(), bins=256)[0]
        # moving = match_histograms(moving, target_histogram)
        # # Step 4: Clip and adjust moving image (no change to target here)
        # target = np.clip(target, 32, 255) - 32
        # moving = np.clip(moving, 32, 255) - 32
        # # ts = int(32*coarse_scale)
        # # clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(ts, ts))
        # # Step 5: Adjust contrast
        # # coarse_target = clahe.apply(coarse_target)
        # target = adjust_contrast(target.astype(np.float64), 50, 99)
        # moving = adjust_contrast(moving.astype(np.float64), 50, 99)

        # # Step 6: Normalize
        # moving = cv2.normalize(moving, moving, 255, 0, cv2.NORM_MINMAX, cv2.CV_8U)
        # target = cv2.normalize(target, target, 255, 0, cv2.NORM_MINMAX, cv2.CV_8U)
        # print(f"After normalization: target min {target.min()}, max {target.max()}")

        # # Step 7: Morphological opening
        # moving = morph_open(moving)
        # # coarse_target = morph_open(coarse_target)
        # # Step 8: Make same shape
        # target, moving = make_same_shape(target, moving)

        return cleaned_fixed, cleaned_moving

    def _match_target_histogram(
        self, target, unaligned: NDArray[np.uint8], clipped=True
    ):
        """Match histogram of unaligned image to target image"""
        target_histogram = np.histogram(target.flatten(), bins=256)[0]
        res = match_histograms(unaligned, target_histogram)
        if clipped:
            res = np.clip(res, 32, 255)
        return match_histograms(unaligned, target_histogram)

    def _scale_transform_matrix(self, matrix, from_scale, to_scale):
        print("matrix shape:", matrix.shape)
        if matrix is None:
            return None
        # Calculate scale factor between pyramid levels
        scale_factor = to_scale / from_scale

        scaled_matrix = matrix.copy()

        if matrix.shape == (2, 3):  # Affine matrix
            scaled_matrix[0, 2] *= scale_factor  # x translation
            scaled_matrix[1, 2] *= scale_factor  # y translation
        elif matrix.shape == (3, 3):  # Homography matrix
            # Scale coordinates: H_scaled = S_to^-1 @ H @ S_from
            s_from = np.diag([from_scale, from_scale, 1.0])
            s_to_inv = np.diag([1 / to_scale, 1 / to_scale, 1.0])
            scaled_matrix = s_to_inv @ matrix @ s_from

        return scaled_matrix

    def _convert_to_original_dtype(self, img_float, original_dtype):
        """Convert result back to original dtype with proper range handling"""
        if original_dtype == np.uint16:
            if img_float.dtype == np.uint8:
                img_float = img_float.astype(np.uint16) * 256
            return to_uint16(img_float)
        elif original_dtype == np.uint8:
            return np.clip(img_float, 0, 255).astype(np.uint8)
        elif np.issubdtype(original_dtype, np.integer):
            max_val = np.iinfo(original_dtype).max
            return np.clip(img_float, 0, max_val).astype(original_dtype)
        else:
            # Float types
            if np.issubdtype(original_dtype, np.floating):
                return img_float.astype(original_dtype)
            else:
                return np.clip(img_float, 0, 1).astype(original_dtype)

    def _process_aligned_image(
        self,
        final_aligned_image,
        target_preview,
        aligned_preview,
        original_target_shape,
    ):
        """Process and remove padding from aligned images"""

        # Remove padding from main image
        final_height, final_width = original_target_shape
        final_aligned_image_cropped = remove_padding(
            final_aligned_image, original_target_shape
        )

        # Calculate scaling factor for preview images
        scaling = target_preview.shape[0] / final_aligned_image.shape[0]

        # Calculate preview dimensions after scaling
        preview_height = int(final_height * scaling)
        preview_width = int(final_width * scaling)

        # Remove padding from preview images
        target_preview_cropped = remove_padding(
            target_preview, (preview_height, preview_width)
        )
        aligned_preview_cropped = remove_padding(
            aligned_preview, (preview_height, preview_width)
        )

        return (
            final_aligned_image_cropped,
            target_preview_cropped,
            aligned_preview_cropped,
        )

    def _fatal_error_message(self, msg):
        self.error.emit(msg)
        self.progress.emit(100, "Retry Maybe")


def calculate_alignment_metrics(fixed_array, aligned_array):
    # Ensure same shape
    if fixed_array.shape != aligned_array.shape:
        # Resize if needed

        zoom_factors = [f / a for f, a in zip(fixed_array.shape, aligned_array.shape)]
        aligned_array = zoom(aligned_array, zoom_factors, order=1)

    # Flatten arrays for easier computation
    fixed_flat = fixed_array.flatten()
    aligned_flat = aligned_array.flatten()

    correlation = abs(np.corrcoef(fixed_flat, aligned_flat)[0, 1])
    if np.isnan(correlation):
        correlation = 0.0

    return correlation


def composite_to_matrix(composite_transform, reference_image):
    """
    Convert a CompositeTransform to a single 2x3 affine transformation matrix.

    Returns matrix in format: [[M00, M01, Tx], [M10, M11, Ty]]
    where the transformation is: x' = M00*x + M01*y + Tx
                                y' = M10*x + M11*y + Ty
    """
    try:
        # Method 1: Try to get matrix directly from individual transforms
        if composite_transform.GetNumberOfTransforms() == 1:
            single_transform = composite_transform.GetNthTransform(0)
            if hasattr(single_transform, "GetMatrix"):
                # Get 2x2 matrix and translation
                matrix_2x2 = single_transform.GetMatrix()
                translation = single_transform.GetTranslation()

                # Convert to 2x3 affine matrix
                return [
                    [matrix_2x2[0], matrix_2x2[1], translation[0]],  # [M00, M01, Tx]
                    [matrix_2x2[2], matrix_2x2[3], translation[1]],  # [M10, M11, Ty]
                ]

        # Method 2: For multiple transforms, compute equivalent matrix by sampling
        # Get image dimensions for reasonable test points
        size = reference_image.GetSize()
        spacing = reference_image.GetSpacing()
        origin = reference_image.GetOrigin()

        # Define test points in physical space
        center_x = origin[0] + (size[0] * spacing[0]) / 2
        center_y = origin[1] + (size[1] * spacing[1]) / 2

        # Test points: origin, and points offset by 1 unit in each direction
        test_points = [
            [0.0, 0.0],  # Origin
            [1.0, 0.0],  # Unit X
            [0.0, 1.0],  # Unit Y
        ]

        # Transform the test points
        transformed_points = []
        for point in test_points:
            transformed = composite_transform.TransformPoint(point)
            transformed_points.append(transformed)

        # Extract affine matrix from transformed points
        p_origin, p_unit_x, p_unit_y = transformed_points

        # The transformation of (1,0) gives us the first column + translation
        # The transformation of (0,1) gives us the second column + translation
        # The transformation of (0,0) gives us the translation

        # Extract matrix elements
        m00 = p_unit_x[0] - p_origin[0]  # How (1,0) transforms in X
        m10 = p_unit_x[1] - p_origin[1]  # How (1,0) transforms in Y
        m01 = p_unit_y[0] - p_origin[0]  # How (0,1) transforms in X
        m11 = p_unit_y[1] - p_origin[1]  # How (0,1) transforms in Y
        tx = p_origin[0]  # Translation in X
        ty = p_origin[1]  # Translation in Y

        # Return as 2x3 affine matrix
        return np.array([[m00, m01, tx], [m10, m11, ty]])

    except Exception as e:
        print(f"Warning: Could not convert CompositeTransform to matrix: {e}")
        # Fallback: return identity 2x3 matrix
        return np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])


def register_combination(fixed, moving, angle, flip, parameter_object):
    print(f"Registering with angle={angle}, flip={flip}")

    # Apply flip if specified
    t = np.flip(moving, axis=-2) if flip else moving

    # Apply rotation
    rotated = rotate(t, angle, reshape=False, order=1)
    print(f"Rotated shape: {rotated.shape}, Fixed shape: {fixed.shape}")

    # Convert to ITK images
    source_itk = sitk.GetImageFromArray(rotated)
    target_itk = sitk.GetImageFromArray(fixed)

    # Cast to float32 for registration
    source_itk = sitk.Cast(source_itk, sitk.sitkFloat32)
    target_itk = sitk.Cast(target_itk, sitk.sitkFloat32)

    # Create registration method
    registration_method = sitk.ImageRegistrationMethod()
    registration_method.SetMetricAsCorrelation()

    # Create initial transform (Similarity2D for 2D registration)
    initial_transform = sitk.CenteredTransformInitializer(
        source_itk,
        target_itk,
        sitk.Euler2DTransform(),
        sitk.CenteredTransformInitializerFilter.GEOMETRY,
    )

    try:
        # Set up registration parameters
        registration_method.SetInitialTransform(initial_transform, inPlace=False)
        registration_method.SetInterpolator(sitk.sitkLinear)

        # Set optimizer
        registration_method.SetOptimizerAsPowell()

        # Set optimizer scales (important for different parameter types)
        registration_method.SetOptimizerScalesFromPhysicalShift()

        # Execute registration
        final_transform = registration_method.Execute(source_itk, target_itk)

        # Get registration score (note: correlation metric should be maximized)
        score = registration_method.GetMetricValue()

        # Apply final transform to get registered image
        # registered_image = sitk.Resample(
        #     source_itk,
        #     target_itk,
        #     final_transform,
        #     sitk.sitkLinear,
        #     0.0,
        #     source_itk.GetPixelID(),
        # )
        # print(f"Registration score: {score}, angle: {angle}, flip: {flip}")

        # Get transformation matrix
        if hasattr(final_transform, "GetMatrix"):
            transform_matrix = np.array(final_transform.GetMatrix())
        else:
            # For CompositeTransform, get the matrix from the last transform
            transform_matrix = composite_to_matrix(final_transform, target_itk)
        score = -1 * score  # Invert score for consistency (higher is better)
        print(f"Score: {score} for angle={angle}, flip={flip}")
        return score, angle, flip, transform_matrix

    except Exception as e:
        print(f"Failed at angle={angle}, flip={flip}: {e}")
        return -1, angle, flip, None


def morph_open(img):
    blur = cv2.GaussianBlur(img, (7, 7), 0)
    # blur = img
    _, t = cv2.threshold(blur, 32, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernel = np.ones((5, 5), np.uint8)
    closed = cv2.morphologyEx(t, cv2.MORPH_OPEN, kernel)
    output = closed.copy()
    # Then extract contours
    # contours, _ = cv2.findContours(output, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return output


def extract_itk_transform_matrix(params):
    """
    Extract the transformation matrix from ITK elastix parameters.

    Args:
        params: ITK elastix parameter object

    Returns:
        numpy.ndarray: Transformation matrix from ITK registration
    """
    try:
        # Get the transformation parameters
        transform_params = params.GetParameterMap(0)

        # Get transformation parameters
        transform_parameters = [
            float(x) for x in transform_params.get("TransformParameters", [])
        ]

        # ITK 2D Affine: [m00, m01, m10, m11, tx, ty]
        matrix = np.array(
            [
                [
                    transform_parameters[0],
                    transform_parameters[1],
                    transform_parameters[5],
                ],
                [
                    transform_parameters[2],
                    transform_parameters[3],
                    transform_parameters[4],
                ],
                [0, 0, 1],
            ]
        )

        return matrix

    except Exception as e:
        print(f"Warning: Could not extract ITK transform matrix: {str(e)}")
        # Return identity matrix as fallback
        return np.eye(3)


def extract_itk_transform_matrix_verbose(params):
    """
    Extract the transformation matrix from ITK elastix parameters with detailed output.
    Useful for debugging and understanding the transformation components.
    """
    try:
        transform_params = params.GetParameterMap(0)

        # Get all relevant parameters
        transform_parameters = [
            float(x) for x in transform_params.get("TransformParameters", [])
        ]
        center_of_rotation = transform_params.get("CenterOfRotationPoint", ["0", "0"])
        cx, cy = float(center_of_rotation[0]), float(center_of_rotation[1])

        print(f"Transform parameters: {transform_parameters}")
        print(f"Center of rotation: ({cx}, {cy})")

        # Extract matrix components
        m00, m01, m10, m11, tx, ty = transform_parameters[:6]

        # Show the decomposition
        rotation_matrix = np.array([[m00, m01], [m10, m11]])
        center = np.array([cx, cy])
        translation = np.array([tx, ty])

        print(f"Rotation/scaling matrix R:\n{rotation_matrix}")
        print(f"Original translation: {translation}")
        print(f"Center of rotation: {center}")

        # Calculate effective translation
        rotation_times_center = rotation_matrix @ center
        effective_translation = translation + center - rotation_times_center

        print(f"R * center: {rotation_times_center}")
        print(f"Effective translation: {effective_translation}")

        # Final matrix
        matrix = np.array(
            [
                [m00, m01, effective_translation[0]],
                [m10, m11, effective_translation[1]],
                [0, 0, 1],
            ]
        )

        print(f"Final transformation matrix:\n{matrix}")
        return matrix

    except Exception as e:
        print(f"Error: {str(e)}")
        return np.eye(3)


def create_preprocessing_matrix(angle, flip, image_shape):
    """
    Create transformation matrix for preprocessing steps (flip + rotation).

    Args:
        angle: Rotation angle in degrees
        flip: Whether Y-axis was flipped
        image_shape: Shape of the image (height, width) or (depth, height, width)

    Returns:
        numpy.ndarray: Preprocessing transformation matrix
    """
    is_2d = len(image_shape) == 2
    assert is_2d
    height, width = image_shape
    center_x, center_y = width / 2.0, height / 2.0

    # Step 1: Translate to origin
    translate_to_origin = np.array([[1, 0, -center_x], [0, 1, -center_y], [0, 0, 1]])

    # Step 2: Y-flip (if applied)
    if flip:
        flip_matrix = np.array([[-1, 0, 0], [0, 1, 0], [0, 0, 1]])
    else:
        flip_matrix = np.eye(3)

    # Step 3: Rotation (if applied)
    if angle != 0:
        angle_rad = np.radians(angle)
        cos_a, sin_a = np.cos(angle_rad), np.sin(angle_rad)
        rotation_matrix = np.array([[cos_a, -sin_a, 0], [sin_a, cos_a, 0], [0, 0, 1]])
    else:
        rotation_matrix = np.eye(3)

    # Step 4: Translate back to center
    translate_back = np.array([[1, 0, center_x], [0, 1, center_y], [0, 0, 1]])

    # Combine: T2 * R * F * T1
    preprocessing_matrix = (
        translate_back @ rotation_matrix @ flip_matrix @ translate_to_origin
    )

    return preprocessing_matrix


def combine_transforms(itk_params, angle, flip, image_shape):
    """
    Combine ITK registration transform with preprocessing transforms.

    Args:
        itk_params: ITK elastix parameter object
        angle: Rotation angle used in preprocessing
        flip: Whether flip was applied in preprocessing
        image_shape: Shape of the original image

    Returns:
        numpy.ndarray: Combined transformation matrix
    """
    # Extract ITK transformation matrix
    itk_matrix = itk_params

    # Create preprocessing transformation matrix
    preprocessing_matrix = create_preprocessing_matrix(angle, flip, image_shape)

    # Make matrices compatible (both 3x3 or both 4x4)
    if itk_matrix.shape == (3, 3) and preprocessing_matrix.shape == (4, 4):
        print("Convert ITK 3x3 to 4x4")
        itk_4d = np.eye(4)
        itk_4d[:2, :2] = itk_matrix[:2, :2]
        itk_4d[:2, 3] = itk_matrix[:2, 2]
        itk_matrix = itk_4d
    elif itk_matrix.shape == (2, 3) and preprocessing_matrix.shape == (3, 3):
        print("Convert ITK 2x3 to 3x3")
        itk_3d = np.eye(3)
        itk_3d[:2, :2] = itk_matrix[:2, :2]
        itk_3d[:2, 2] = itk_matrix[:2, 2]
        itk_matrix = itk_3d
    elif itk_matrix.shape == (2, 3) and preprocessing_matrix.shape == (4, 4):
        print("Convert ITK 2x3 to 4x4")
        itk_4d = np.eye(4)
        itk_4d[:2, :2] = itk_matrix[:2, :2]
        itk_4d[:2, 3] = itk_matrix[:2, 2]
        itk_matrix = itk_4d

    # if itk_matrix.shape == (4, 4):
    #     itk_2d = np.eye(3)
    #     itk_2d[:2, :2] = itk_matrix[:2, :2]
    #     itk_2d[:2, 2] = itk_matrix[:2, 3]
    #     itk_matrix = itk_2d

    # if preprocessing_matrix.shape == (4, 4):
    #     prep_2d = np.eye(3)
    #     prep_2d[:2, :2] = preprocessing_matrix[:2, :2]
    #     prep_2d[:2, 2] = preprocessing_matrix[:2, 3]
    #     preprocessing_matrix = prep_2d

    # The complete transformation is: ITK_transform * preprocessing_transform
    # This applies preprocessing first, then ITK registration
    combined_matrix = itk_matrix @ preprocessing_matrix

    return combined_matrix


def extract_complete_transformation(
    itk_params, angle, flip, image_shape, verbose=False
):
    """
    Extract the complete transformation matrix that reproduces ITK registration results.

    Args:
        itk_params: ITK elastix parameter object from registration
        angle: Rotation angle used in preprocessing
        flip: Whether flip was applied in preprocessing
        image_shape: Shape of the original moving image

    Returns:
        dict: Contains combined matrix, ITK matrix, preprocessing matrix, and metadata
    """
    try:
        # Extract individual components
        # if verbose:
        #     itk_matrix = extract_itk_transform_matrix_verbose(itk_params)
        # else:
        #     itk_matrix = extract_itk_transform_matrix(itk_params)
        itk_matrix = itk_params
        preprocessing_matrix = create_preprocessing_matrix(angle, flip, image_shape)
        combined_matrix = combine_transforms(itk_matrix, angle, flip, image_shape)

        results = {
            "combined_matrix": combined_matrix,
            "itk_matrix": itk_matrix,
            "preprocessing_matrix": preprocessing_matrix,
            "angle": angle,
            "flip": flip,
            "image_shape": image_shape,
            "is_2d": len(image_shape) == 2,
        }

        # Add some diagnostics
        results["itk_determinant"] = np.linalg.det(
            itk_matrix[:2, :2] if len(image_shape) == 2 else itk_matrix[:3, :3]
        )
        results["preprocessing_determinant"] = np.linalg.det(
            preprocessing_matrix[:2, :2]
            if len(image_shape) == 2
            else preprocessing_matrix[:3, :3]
        )
        results["combined_determinant"] = np.linalg.det(
            combined_matrix[:2, :2]
            if len(image_shape) == 2
            else combined_matrix[:3, :3]
        )

        return results

    except Exception as e:
        print(f"Error extracting complete transformation: {e}")
        return {}


def apply_combined_transform(image, combined_matrix):
    """
    Apply the combined transformation matrix to an image.

    Args:
        image: Input image array
        combined_matrix: Combined transformation matrix

    Returns:
        numpy.ndarray: Transformed image
    """

    is_2d = len(image.shape) == 2

    if is_2d:
        if combined_matrix.shape == (3, 3):
            # Extract affine components
            linear_part = combined_matrix[:2, :2]
            offset_part = combined_matrix[:2, 2]
        else:
            # 4x4 matrix for 2D image
            linear_part = combined_matrix[:2, :2]
            offset_part = combined_matrix[:2, 3]
    else:
        if combined_matrix.shape == (4, 4):
            # Extract affine components
            linear_part = combined_matrix[:3, :3]
            offset_part = combined_matrix[:3, 3]
        else:
            # 3x3 matrix for 3D image (shouldn't happen)
            linear_part = combined_matrix[:2, :2]
            offset_part = combined_matrix[:2, 2]

    try:
        # Apply transformation (scipy needs inverse)
        inv_linear = np.linalg.inv(linear_part)
        inv_offset = -inv_linear @ offset_part

        transformed = affine_transform(
            image, inv_linear, offset=inv_offset, order=1, mode="constant", cval=0
        )
        return transformed

    except np.linalg.LinAlgError:
        print("Warning: Transformation matrix is singular")
        return image.copy()
