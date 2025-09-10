import concurrent.futures
import traceback
from typing import Tuple, final

import cv2
import numpy as np
import SimpleITK as sitk
from numpy.typing import NDArray
from PyQt6.QtCore import QThread, pyqtSignal
from pystackreg import StackReg
from pystackreg.util import to_uint16
from scipy.ndimage import binary_fill_holes, rotate, zoom

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
    # Changed signal to emit a single dictionary for flexible snapshot data
    aligned_image_signal = pyqtSignal(dict, np.ndarray, np.ndarray)
    snapshot = pyqtSignal(dict)

    def __init__(
        self,
        rotate_by=90,
        coarse_scale=1 / 16.0,
        fine_scale=0.5,
        unaligned_space=(1, 1),
        target_space=(1, 1),
    ):
        super().__init__()
        self.target_image = np.zeros(0).astype(np.uint16)
        self.unaligned_image = np.zeros(0).astype(np.uint16)
        self.original_target_shape = None
        self.replace = False
        self.target_channel = "Channel 1"
        self.target_uuid = ""
        self.unaligned_channel = "Channel 1"
        self.unaligned_uuid = ""
        self.debug = False  # Set to True for debugging, downloads debug images
        self.unaligned_spacing = unaligned_space
        self.target_spacing = target_space
        self.need_gradient_descent = True  # Use ITK for coarse alignment
        self.stackreg_type = "affine"
        # ITK metadata to be captured
        self.itk_angle = 0
        self.itk_flip = False
        self.rotate_by = rotate_by
        # Coarse and fine alignment parameters
        self.coarse_scale = coarse_scale
        self.rotation_angles = [0]  # Default to no rotation
        self.fine_scale = fine_scale

    def set_target_image(self, target_image, channel_name, uuid):
        self.target_pyramid = []
        self.target_image = target_image
        self.original_target_shape = target_image.shape
        self.target_channel = channel_name
        self.target_uuid = uuid

    def set_unaligned_image(self, unaligned_image, channel_name, uuid):
        self.unaligned_pyramid = []
        self.unaligned_image = unaligned_image
        self.unaligned_channel = channel_name
        self.unaligned_uuid = uuid

    def set_unaligned_spacing(self, spacing):
        self.unaligned_spacing = spacing

    def set_target_spacing(self, spacing):
        self.target_spacing = spacing

    def skip_coarse_alignment(self, skip: bool):
        """
        Skip the coarse alignment step, only 1 angle will be used, only use itk to try to center
        """
        if skip:
            self.rotation_angles = [0]
        else:
            self.rotation_angles = list(range(0, 360, self.rotate_by))

    def skip_gradient_descent(self, skip: bool):
        """
        Skip the gradient descent step, only use ITK for coarse alignment
        """
        self.need_gradient_descent = not skip

    def manually_align(self, aligned_image: np.ndarray):
        """
        Emit manually aligned image
        """
        self.progress.emit(100, "Manual alignment set")
        self.aligned_image_signal.emit(
            {
                "uuid": self.unaligned_uuid,
                "layer": self.unaligned_channel,
                "replace": self.replace,
                "data": aligned_image,
            },
            np.array(0),
            np.array(0),
        )

    def _scale_for_snapshot(
        self, image_array: np.ndarray, size=(1024, 1024)
    ) -> np.ndarray:
        """
        Resizes an image to the specified snapshot size and converts it to uint8 for display.
        """
        # Convert to uint8 for display, preserving contrast
        img_uint8 = to_uint8(image_array)
        # Resize using INTER_AREA for robust downscaling
        resized = cv2.resize(
            img_uint8, (size[1], size[0]), interpolation=cv2.INTER_AREA
        )
        return resized

    def _emit_snapshot(
        self,
        stage_name: str,
        metadata: dict,
        aligned_image: np.ndarray,
        target_snapshot_img=None,
    ):
        """
        Creates and emits a snapshot dictionary containing scaled images and metadata.
        """
        # Create scaled versions for preview
        if target_snapshot_img is None:
            target_snapshot_img = self._scale_for_snapshot(self.target_image)
        else:
            target_snapshot_img = self._scale_for_snapshot(target_snapshot_img)
        aligned_snapshot_img = self._scale_for_snapshot(aligned_image)
        metadata["stage"] = stage_name
        snapshot_data = {
            "metadata": metadata,
            "target_image": target_snapshot_img,
            "aligned_image": aligned_snapshot_img,
        }
        self.snapshot.emit(snapshot_data)

    def run(self):
        """Main processing function that runs in the thread"""
        if self.target_image is None or self.unaligned_image is None:
            self._fatal_error_message(
                "Both target and unaligned images must be provided"
            )
            return
        initial_metadata = {
            "unaligned_shape": self.unaligned_image.shape,
            "target_shape": self.target_image.shape,
            "unaligned_spacing": self.unaligned_spacing,
            "target_spacing": self.target_spacing,
        }
        try:
            if self.target_spacing != self.unaligned_spacing:
                print("Resampling unaligned image to match target spacing")
                moving = sitk.GetImageFromArray(self.unaligned_image)
                moving.SetSpacing(self.unaligned_spacing)
                fixed = sitk.GetImageFromArray(self.target_image)
                fixed.SetSpacing(self.target_spacing)
                resample = sitk.ResampleImageFilter()
                resample.SetReferenceImage(fixed)
                resample.SetInterpolator(sitk.sitkLinear)
                resample.SetTransform(sitk.Transform())
                resampled = resample.Execute(moving)
                self.unaligned_image = sitk.GetArrayFromImage(resampled)
                initial_metadata["unaligned_shape"] = self.unaligned_image.shape

            # Prepare images for coarse alignment
            self.progress.emit(5, "Preparing images for coarse alignment")
            self.coarse_target, self.coarse_moving = self._prepare_images(
                self.target_image, self.unaligned_image, self.coarse_scale
            )
            # self._emit_snapshot(
            #     "After coarse preprocessing",
            #     initial_metadata,
            #     self.coarse_moving,
            #     self.coarse_target,
            # )
            coarse_transform = np.eye(3)

            # Perform coarse alignment
            self.progress.emit(30, "Intensity-based coarse alignment (ITK)")
            coarse_transform = self._coarse_alignment()
            if coarse_transform is None:
                self._fatal_error_message("Coarse alignment (ITK) failed.")
                return

            itk_metadata = {
                "chosen_angle": self.itk_angle,
                "flip": self.itk_flip,
                "matrix": coarse_transform,
            }
            print(coarse_transform)
            # Prepare images for fine alignment
            self.progress.emit(50, "Preparing images for fine alignment")
            fine_transform = self._scale_transform_matrix(
                coarse_transform[:2, :3].copy(), self.coarse_scale, self.fine_scale
            )
            fine_target, fine_unaligned = self._prepare_images(
                self.target_image, self.unaligned_image, self.fine_scale
            )
            fine_target_shape = (
                np.array(self.target_image.shape) * self.fine_scale
            ).astype(int)
            print(fine_target_shape, fine_target.shape, fine_unaligned.shape)
            fine_moving = warp_image(fine_unaligned, fine_transform)
            fine_moving = to_uint8(fine_moving)
            fine_moving = remove_padding(fine_moving, fine_target_shape)
            fine_target_image = remove_padding(fine_target, fine_target_shape)
            print(
                f"Fine target shape: {fine_target_image.shape}, Fine moving shape: {fine_moving.shape}"
            )
            # self._emit_snapshot("itk", itk_metadata, fine_moving, fine_target_image)

            # Perform fine alignment
            self.progress.emit(60, "Fine alignment (pystackreg)")
            refinement_transform, aligned_preview = self._alignment_stackreg(
                fine_target_image, fine_moving
            )

            if refinement_transform is None:
                print("StackReg refinement failed, using only coarse alignment.")
                aligned_preview = fine_moving
                refinement_transform = np.eye(3)  # Identity transform
                refinement_transform = self._scale_transform_matrix(
                    refinement_transform[:2, :3], 1, self.fine_scale
                )
            assert (
                refinement_transform is not None
            ), "Refinement transform should not be None"

            # --- 3. PYSTACKREG (FINAL) SNAPSHOT ---
            assert aligned_preview is not None, "Aligned preview should not be None"
            pystackreg_metadata = {"transform_type": self.stackreg_type}
            # self._emit_snapshot(
            #     "pystackreg", pystackreg_metadata, aligned_preview, fine_target_image
            # )

            gradient_transform = None
            if self.need_gradient_descent:
                # refine alignment using gradient descent
                gradient_transform, gradient_aligned = gradient_descent_alignment(
                    aligned_preview, fine_target_image, 200
                )
                gradient_metadata = {
                    "matrix": gradient_transform,
                }
                # self._emit_snapshot(
                #     "gradient descent",
                #     gradient_metadata,
                #     gradient_aligned,
                #     fine_target_image,
                # )
            # Apply final transformation
            self.progress.emit(80, "Applying final transformation")
            full_coarse_matrix = self._scale_transform_matrix(
                coarse_transform[:2, :3], self.coarse_scale, 1
            )
            full_refinement_transform = self._scale_transform_matrix(
                refinement_transform[:2, :3], self.fine_scale, 1
            )
            _, padded_moving = make_same_shape(self.target_image, self.unaligned_image)
            intermediate_aligned = warp_image(padded_moving, full_coarse_matrix)
            intermediate_aligned = remove_padding(
                intermediate_aligned, self.target_image.shape
            )
            final_aligned_image = warp_image(
                intermediate_aligned, full_refinement_transform
            )
            if gradient_transform is not None:
                # Apply gradient descent refinement if available
                full_gradient_transform = self._scale_transform_matrix(
                    gradient_transform[:2, :3], self.fine_scale, 1
                )
                final_aligned_image = warp_image(
                    final_aligned_image, full_gradient_transform
                )
            # Convert result back to original dtype for storage
            result_data = self._convert_to_original_dtype(
                final_aligned_image, self.unaligned_image.dtype
            )

            result_payload = {
                "uuid": self.target_uuid,
                "layer": self.target_channel,
                "replace": self.replace,
                "data": result_data,
            }

            self.progress.emit(100, "Two-stage alignment complete")
            # target_preview = fine_target_image
            # aligned_preview = final_aligned_image
            # adjust contrast for display:
            # target_preview = adjust_contrast(
            #     self.target_image.astype(np.float64), 30, 99
            # )
            # aligned_preview = adjust_contrast(
            #     final_aligned_image.astype(np.float64), 30, 99
            # )
            self.aligned_image_signal.emit(
                result_payload, self.target_image, final_aligned_image
            )

        except Exception as e:
            self._fatal_error_message(
                f"Error during alignment: {traceback.format_exc()}"
            )

    def _coarse_alignment(self):
        moving, _ = self.coarse_moving, self.coarse_target
        angle, flip, params = self._itk_align(
            self.rotation_angles,
            self.coarse_target,
            self.coarse_moving,
        )
        # Store the metadata for the snapshot
        self.itk_angle = angle
        self.itk_flip = flip

        params = np.array(params)
        transform_info = extract_complete_transformation(
            params, angle, flip, moving.shape
        )
        t = transform_info["combined_matrix"]
        print(t.shape)
        if t.shape == (2, 3):
            t = np.vstack([t, [0, 0, 1]])
        inverted = np.linalg.inv(t)

        return inverted

    def _prepare_images(
        self, target_image, unaligned_image, coarse_scale
    ) -> Tuple[NDArray[np.uint8], NDArray[np.uint8]]:
        """
        Preprocess two images for coarse alignment.
        """
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
        coarse_target = to_uint8(coarse_target)
        coarse_moving = to_uint8(coarse_moving)
        target_histogram = np.histogram(coarse_target.flatten(), bins=256)[0]
        coarse_moving = match_histograms(coarse_moving, target_histogram)
        coarse_moving = np.clip(coarse_moving, 32, 255) - 32
        coarse_target = adjust_contrast(coarse_target.astype(np.float64), 50, 99)
        coarse_moving = adjust_contrast(coarse_moving.astype(np.float64), 50, 99)
        coarse_moving = cv2.normalize(
            coarse_moving, coarse_moving, 255, 0, cv2.NORM_MINMAX, cv2.CV_8U
        )
        coarse_target = cv2.normalize(
            coarse_target, coarse_target, 255, 0, cv2.NORM_MINMAX, cv2.CV_8U
        )
        coarse_moving = morph_open(coarse_moving)
        coarse_target, coarse_moving = make_same_shape(coarse_target, coarse_moving)
        return coarse_target, coarse_moving

    def _itk_align(self, rotation_angles, fixed, moving):
        print("Stage 1: Finding best angles...")
        angle_results = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=6) as executor:
            futures = [
                executor.submit(register_combination, fixed, moving, angle, False)
                for angle in rotation_angles
            ]
            for future in concurrent.futures.as_completed(futures):
                try:
                    angle_results.append(future.result())
                except Exception as e:
                    print(f"Error during registration: {e}")
                    traceback.print_exc()
        if len(angle_results) == 1:
            # If only one angle was tested, return it directly
            best_angle, best_flip, params = angle_results[0][1:]
            print(f"Only one angle tested: {best_angle}, flip: {best_flip}")
            return best_angle, best_flip, params
        valid_angle_results = [r for r in angle_results if r[3] is not None]
        sorted_angle_results = sorted(
            valid_angle_results, key=lambda x: x[0], reverse=True
        )
        top3_angles = sorted_angle_results[:3]

        print("Top 3 angles:")
        for i, (score, angle, _, params) in enumerate(top3_angles):
            print(f"  {i+1}. Angle {angle}: score = {score}")

        print("Stage 2: Testing flip options for top 3 angles...")
        flip_results = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
            futures = [
                executor.submit(register_combination, fixed, moving, angle, True)
                for score, angle, _, params in top3_angles
            ]
            for future in concurrent.futures.as_completed(futures):
                flip_results.append(future.result())

        all_results = sorted_angle_results + flip_results
        valid_all_results = [r for r in all_results if r[3] is not None]
        if not valid_all_results:
            raise RuntimeError("No valid alignment found after all ITK attempts.")

        final_sorted_results = sorted(
            valid_all_results, key=lambda x: x[0], reverse=True
        )
        best_score, best_angle, best_flip, params = final_sorted_results[0]
        print(f"Final best: angle={best_angle}, flip={best_flip}, score={best_score}")
        return best_angle, best_flip, params

    def _alignment_stackreg(self, target_fine, unaligned_fine):
        """Enhanced StackReg alignment with better preprocessing"""
        try:
            target_enhanced, unaligned_enhanced = self._image_enhancement(
                target_fine.copy(), unaligned_fine.copy()
            )
            target_uint16 = to_uint16(target_enhanced)
            unaligned_uint16 = to_uint16(unaligned_enhanced)
            sr = StackReg(StackReg.AFFINE)
            aligned_result = sr.register_transform(target_uint16, unaligned_uint16)
            refinement_matrix = sr.get_matrix()
            aligned_result = warp_image(unaligned_fine, refinement_matrix)
            return refinement_matrix[:2, :3], aligned_result
        except Exception as e:
            print(f"StackReg refinement error: {e}")
            return None, None

    def _image_enhancement(self, target, moving):
        target = to_uint8(target)
        moving = to_uint8(moving)
        cleaned_fixed = binary_fill_holes(target > 0.1)
        assert cleaned_fixed is not None, "Fixed image should not be None"
        cleaned_fixed = cleaned_fixed.astype(np.uint8) * 255
        cleaned_moving = binary_fill_holes(moving > 0.1)
        assert cleaned_moving is not None, "Moving image should not be None"
        cleaned_moving = cleaned_moving.astype(np.uint8) * 255
        return cleaned_fixed, cleaned_moving

    def _scale_transform_matrix(
        self, matrix, from_scale, to_scale
    ) -> np.ndarray | None:
        if matrix is None:
            return None
        scale_factor = to_scale / from_scale
        scaled_matrix = matrix.copy()
        if matrix.shape == (2, 3):
            scaled_matrix[0, 2] *= scale_factor
            scaled_matrix[1, 2] *= scale_factor
        elif matrix.shape == (3, 3):
            s_from = np.diag([from_scale, from_scale, 1.0])
            s_to_inv = np.diag([1 / to_scale, 1 / to_scale, 1.0])
            scaled_matrix = s_to_inv @ matrix @ s_from
        return scaled_matrix

    def _convert_to_original_dtype(self, img_float, original_dtype):
        if np.issubdtype(original_dtype, np.uint16):
            return to_uint16(img_float)
        elif np.issubdtype(original_dtype, np.uint8):
            return np.clip(img_float, 0, 255).astype(np.uint8)
        elif np.issubdtype(original_dtype, np.integer):
            max_val = np.iinfo(original_dtype).max
            return np.clip(img_float, 0, max_val).astype(original_dtype)
        elif np.issubdtype(original_dtype, np.floating):
            return img_float.astype(original_dtype)
        else:
            return img_float

    def _fatal_error_message(self, msg):
        self.error.emit(msg)
        self.progress.emit(100, "Failed")


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


def register_combination(fixed, moving, angle, flip):
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
        # results["itk_determinant"] = np.linalg.det(
        #     itk_matrix[:2, :2] if len(image_shape) == 2 else itk_matrix[:3, :3]
        # )
        # results["preprocessing_determinant"] = np.linalg.det(
        #     preprocessing_matrix[:2, :2]
        #     if len(image_shape) == 2
        #     else preprocessing_matrix[:3, :3]
        # )
        # results["combined_determinant"] = np.linalg.det(
        #     combined_matrix[:2, :2]
        #     if len(image_shape) == 2
        #     else combined_matrix[:3, :3]
        # )

        return results

    except Exception as e:
        print(f"Error extracting complete transformation: {e}")
        return {}


def gradient_descent_alignment(moving_image, fixed_image, num_histogram_bins=300):
    """
    Perform gradient descent alignment between two images.

    Args:
        moving_image: The image to be aligned (moving image).
        fixed_image: The reference image (fixed image).

    Returns:
        The optimized transformation parameters.
    """
    if isinstance(moving_image, np.ndarray):
        moving_image = sitk.GetImageFromArray(moving_image)
    if isinstance(fixed_image, np.ndarray):
        fixed_image = sitk.GetImageFromArray(fixed_image)
    initial_transform = sitk.CenteredTransformInitializer(
        fixed_image,
        moving_image,
        sitk.Similarity2DTransform(),
        sitk.CenteredTransformInitializerFilter.GEOMETRY,
    )
    registration_method = sitk.ImageRegistrationMethod()

    # Similarity metric settings.
    registration_method.SetMetricAsMattesMutualInformation(
        numberOfHistogramBins=num_histogram_bins
    )
    registration_method.SetMetricSamplingStrategy(registration_method.RANDOM)
    registration_method.SetMetricSamplingPercentage(0.1)

    registration_method.SetInterpolator(sitk.sitkLinear)

    # Optimizer settings.
    registration_method.SetOptimizerAsGradientDescent(
        learningRate=0.05,
        numberOfIterations=50,
        convergenceMinimumValue=1e-7,
        convergenceWindowSize=10,
    )
    registration_method.SetOptimizerScalesFromPhysicalShift()

    # Setup for the multi-resolution framework.
    registration_method.SetShrinkFactorsPerLevel(shrinkFactors=[4, 2, 1])
    registration_method.SetSmoothingSigmasPerLevel(smoothingSigmas=[2, 1, 0])
    registration_method.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()

    registration_method.SetInitialTransform(initial_transform, inPlace=False)

    final_transform = registration_method.Execute(
        sitk.Cast(fixed_image, sitk.sitkFloat32),
        sitk.Cast(moving_image, sitk.sitkFloat32),
    )
    moving_resampled = sitk.Resample(
        moving_image,
        fixed_image,
        final_transform,
        sitk.sitkLinear,
        0.0,
        fixed_image.GetPixelID(),
    )
    final_transform_matrix = composite_to_matrix(final_transform, fixed_image)
    final_transform = np.array(final_transform_matrix)
    if final_transform.shape == (3, 3):
        final_transform = final_transform[:2, :3]  # Convert to 2x3 for 2D images
    moving_resampled = sitk.GetArrayFromImage(moving_resampled)
    return final_transform, moving_resampled
