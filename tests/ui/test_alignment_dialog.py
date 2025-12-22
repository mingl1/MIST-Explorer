import math

import cv2
import numpy as np
import pytest
from PyQt6.QtWidgets import QApplication
from scipy.ndimage import center_of_mass

#
from ui.alignment.alignment_preview_dialog import (
    AlignmentPreviewDialog,
)  # Replace with your actual import

# --- Helper Functions for Testing ---


def create_test_image(shape=(2048, 2048), rect_size=(200, 400)):
    """Creates a black image with a distinct white rectangle for easy feature tracking."""
    img = np.zeros(shape, dtype=np.float32)
    h, w = shape
    rh, rw = rect_size
    # Place rectangle off-center to make rotation obvious
    y_start, x_start = h // 4, w // 4
    img[y_start : y_start + rh, x_start : x_start + rw] = 1.0
    return img


def get_feature_center(image: np.ndarray) -> tuple[float, float]:
    """
    Calculates the center of mass of the white feature in the image.
    This is robust to interpolation artifacts. Returns (row, col) or (y, x).
    """
    # Ensure there's something to find to avoid division by zero
    if np.sum(image) == 0:
        return (float("nan"), float("nan"))
    return center_of_mass(image)


# --- Test Fixture ---


@pytest.fixture
def dialog(qtbot):
    """Pytest fixture to create the dialog for each test."""
    # Ensure a QApplication instance exists
    QApplication.instance() or QApplication([])

    snapshot_data = {
        "target_image": create_test_image(),
        "aligned_image": create_test_image(),
        "metadata": {"stage": "test"},
    }
    # Create the dialog with editing enabled
    dialog_instance = AlignmentPreviewDialog(snapshot_data, can_edit=True)
    qtbot.addWidget(dialog_instance)
    yield dialog_instance
    # Teardown is handled automatically by qtbot


# --- The Test Suite ---


class TestAlignmentPreviewDialog:

    def test_initialization(self, dialog):
        """Test that the dialog initializes correctly."""
        assert dialog.can_edit is True
        assert dialog.offset_x == 0
        assert dialog.offset_y == 0
        assert np.array_equal(dialog.target_image, dialog.original_aligned_image)
        # Initial scale factor should be 2048 / 1024 = 2.0
        assert math.isclose(dialog.scale_factor, 2.0)

    def test_accept_no_changes(self, dialog):
        """Test accepting without any edits. Output should match input."""
        initial_image = dialog.original_aligned_image.copy()

        # ACT
        dialog.accept_alignment()
        final_image = dialog.get_current_aligned_image()

        # ASSERT
        assert np.array_equal(initial_image, final_image)

    def test_simple_translation_with_scaling(self, dialog):
        """Test if display translation is correctly scaled to the actual image."""
        initial_center = get_feature_center(dialog.original_aligned_image)

        # ARRANGE: Set scale and apply a display translation
        dialog._update_for_new_scale(4.0)  # Downscale by 4x
        assert math.isclose(dialog.scale_factor, 4.0)

        display_dx, display_dy = 10, -20
        dialog.move_aligned_image(display_dx, display_dy)

        # ACT
        dialog.accept_alignment()
        final_image = dialog.get_current_aligned_image()

        # ASSERT
        final_center = get_feature_center(final_image)
        actual_dx = display_dx * dialog.scale_factor  # 10 * 4.0 = 40
        actual_dy = display_dy * dialog.scale_factor  # -20 * 4.0 = -80

        # Center moves (y, x) which corresponds to (dy, dx)
        expected_center = (initial_center[0] + actual_dy, initial_center[1] + actual_dx)

        assert final_center == pytest.approx(expected_center, abs=1.0)

    def test_simple_rotation(self, dialog):
        """Test if rotation is applied correctly, independent of scale."""
        dialog.original_aligned_image = create_test_image(
            shape=(2000, 1000), rect_size=(400, 100)
        )
        dialog._update_for_new_scale(2.0)

        dialog.rotation_input.setText("90")
        dialog.apply_rotation()
        dialog.accept_alignment()
        final_image = dialog.get_current_aligned_image()

        # --- ASSERT (Corrected) ---
        # To create the expected image, we must apply the exact same logic
        # as the dialog: cv2.warpAffine with a fixed dsize, which causes clipping.
        original_image = dialog.original_aligned_image
        h, w = original_image.shape[:2]
        center = (w / 2, h / 2)
        rot_mat = cv2.getRotationMatrix2D(center, 90.0, 1.0)
        expected_image = cv2.warpAffine(
            original_image, rot_mat, (w, h)  # Use original (w, h) to match clipping
        )

        # Now, calculate the center from the correctly simulated (clipped) image
        expected_center = get_feature_center(expected_image)
        final_center = get_feature_center(final_image)

        assert final_center == pytest.approx(expected_center, abs=1.5)

    def test_rotate_then_translate(self, dialog):
        """Verify that rotation is applied before translation."""
        dialog._update_for_new_scale(2.0)

        dialog.rotation_input.setText("90")
        dialog.apply_rotation()
        dialog.move_aligned_image(dx=50, dy=25)

        dialog.accept_alignment()
        final_image = dialog.get_current_aligned_image()

        # --- ASSERT (Corrected) ---
        # 1. Manually perform the clipping rotation first.
        original_image = dialog.original_aligned_image
        h, w = original_image.shape[:2]
        center = (w / 2, h / 2)
        rot_mat = cv2.getRotationMatrix2D(center, 90.0, 1.0)
        rotated_image = cv2.warpAffine(original_image, rot_mat, (w, h))

        # 2. Manually translate the clipped, rotated image.
        # Note: the translation destination size must match the rotated_image size.
        actual_dx, actual_dy = 50 * 2.0, 25 * 2.0
        trans_mat = np.float32([[1, 0, actual_dx], [0, 1, actual_dy]])
        expected_image = cv2.warpAffine(
            rotated_image, trans_mat, (rotated_image.shape[1], rotated_image.shape[0])
        )

        final_center = get_feature_center(final_image)
        expected_center = get_feature_center(expected_image)

        assert final_center == pytest.approx(expected_center, abs=1.5)

    def test_translate_then_rotate(self, dialog):
        """Verify that the transformation list is processed in order."""
        dialog._update_for_new_scale(2.0)

        dialog.move_aligned_image(dx=50, dy=25)
        dialog.rotation_input.setText("90")
        dialog.apply_rotation()

        dialog.accept_alignment()
        final_image = dialog.get_current_aligned_image()

        # --- ASSERT (Corrected) ---
        # 1. Manually translate the original image first.
        original_image = dialog.original_aligned_image
        h, w = original_image.shape[:2]
        actual_dx, actual_dy = 50 * 2.0, 25 * 2.0
        trans_mat = np.float32([[1, 0, actual_dx], [0, 1, actual_dy]])
        translated_image = cv2.warpAffine(original_image, trans_mat, (w, h))

        # 2. Then perform the clipping rotation on the translated image.
        center = (w / 2, h / 2)
        rot_mat = cv2.getRotationMatrix2D(center, 90.0, 1.0)
        expected_image = cv2.warpAffine(translated_image, rot_mat, (w, h))

        final_center = get_feature_center(final_image)
        expected_center = get_feature_center(expected_image)

        assert final_center == pytest.approx(expected_center, abs=1.5)

    # In your test_alignment_dialog.py file, within the TestAlignmentPreviewDialog class...

    def test_multiple_rotations_with_interspersed_translation(self, dialog):
        """
        Tests a sequence of R -> T -> R. This ensures the transformation list
        is processed in the correct order.
        """
        # ARRANGE:
        # 1. Rotate by 30 degrees.
        # 2. Translate by (10, 15) display pixels.
        # 3. Rotate again by 60 degrees.
        dialog._update_for_new_scale(4.0)
        dialog.rotation_input.setText("30")
        dialog.apply_rotation()
        dialog.move_aligned_image(dx=10, dy=15)
        dialog.rotation_input.setText("60")
        dialog.apply_rotation()

        # ACT:
        dialog.accept_alignment()
        final_image = dialog.get_current_aligned_image()

        # ASSERT: Manually replicate the exact sequence on the original image.
        original_image = dialog.original_aligned_image
        h, w = original_image.shape[:2]
        center = (w / 2, h / 2)

        # Step 1: First rotation
        rot_mat1 = cv2.getRotationMatrix2D(center, 30.0, 1.0)
        step1_img = cv2.warpAffine(original_image, rot_mat1, (w, h))

        # Step 2: Translation (using the final scale factor)
        actual_dx, actual_dy = 10 * 4.0, 15 * 4.0
        trans_mat = np.float32([[1, 0, actual_dx], [0, 1, actual_dy]])
        step2_img = cv2.warpAffine(step1_img, trans_mat, (w, h))

        # Step 3: Second rotation
        rot_mat2 = cv2.getRotationMatrix2D(center, 60.0, 1.0)
        expected_image = cv2.warpAffine(step2_img, rot_mat2, (w, h))

        final_center = get_feature_center(final_image)
        expected_center = get_feature_center(expected_image)

        assert final_center == pytest.approx(expected_center, abs=1.5)

    def test_complex_sequence_with_scale_change_and_reset(self, dialog):
        """
        Tests a realistic user workflow:
        1. Make some edits.
        2. Hit "Reset".
        3. Change display scale.
        4. Make new edits.
        The final result should only reflect the edits made AFTER the reset.
        """
        # ARRANGE:
        # -- Part 1: Initial edits that will be discarded --
        dialog._update_for_new_scale(2.0)
        dialog.move_aligned_image(dx=100, dy=100)
        dialog.rotation_input.setText("45")
        dialog.apply_rotation()

        # -- Part 2: Reset the state --
        dialog.reset_position()
        assert dialog.offset_x == 0
        assert dialog.transformations == [[0.0, []]]

        # -- Part 3: New edits with a different scale factor --
        dialog._update_for_new_scale(8.0)  # Change the scale
        dialog.rotation_input.setText("-90")  # Negative rotation
        dialog.apply_rotation()
        dialog.move_aligned_image(dx=-20, dy=30)  # Negative translation

        # ACT:
        dialog.accept_alignment()
        final_image = dialog.get_current_aligned_image()

        # ASSERT: Manually replicate ONLY the transformations from Part 3.
        original_image = dialog.original_aligned_image
        h, w = original_image.shape[:2]
        center = (w / 2, h / 2)

        # Step 1 (after reset): Rotation
        rot_mat = cv2.getRotationMatrix2D(center, -90.0, 1.0)
        step1_img = cv2.warpAffine(original_image, rot_mat, (w, h))

        # Step 2 (after reset): Translation, scaled by the FINAL factor (8.0)
        actual_dx, actual_dy = -20 * 8.0, 30 * 8.0
        trans_mat = np.float32([[1, 0, actual_dx], [0, 1, actual_dy]])
        expected_image = cv2.warpAffine(step1_img, trans_mat, (w, h))

        final_center = get_feature_center(final_image)
        expected_center = get_feature_center(expected_image)

        assert final_center == pytest.approx(expected_center, abs=1.5)

    def test_translation_is_always_scaled_by_final_factor(self, dialog):
        """
        Verifies a key design choice: all display translations are summed and
        then scaled by the scale_factor that is active at the moment of acceptance,
        regardless of what the scale was when each translation was made.
        """
        # ARRANGE:
        # Step 1: Translate at a low scale
        dialog._update_for_new_scale(2.0)
        dialog.move_aligned_image(dx=50, dy=0)  # Total display dx=50

        # Step 2: Change scale, then translate more
        dialog._update_for_new_scale(10.0)  # HIGHLY different scale
        dialog.move_aligned_image(dx=10, dy=0)  # Total display dx=60

        # Final scale factor is 10.0. Total display offset is (60, 0).

        # ACT:
        dialog.accept_alignment()
        final_image = dialog.get_current_aligned_image()

        # ASSERT:
        # The total actual translation should be based on the total DISPLAY
        # offset scaled by the FINAL scale factor.
        initial_center = get_feature_center(dialog.original_aligned_image)
        total_display_dx = 50 + 10
        final_scale_factor = 10.0

        actual_dx = total_display_dx * final_scale_factor  # 60 * 10.0 = 600
        actual_dy = 0 * final_scale_factor  # 0

        expected_center = (initial_center[0] + actual_dy, initial_center[1] + actual_dx)
        final_center = get_feature_center(final_image)

        assert final_center == pytest.approx(expected_center, abs=1.0)

    def test_reapplying_transformations_on_scale_change(self, dialog):
        """
        This test verifies that the display image is correctly regenerated
        when the scale changes mid-edit.
        """
        # ARRANGE
        dialog._update_for_new_scale(2.0)
        dialog.rotation_input.setText("30")
        dialog.apply_rotation()
        dialog.move_aligned_image(dx=25, dy=25)

        # Capture the state of the display image
        display_before_scale_change = dialog.aligned_display.copy()

        # ACT: Change the scale. This should trigger _reapply_display_transformations
        dialog._update_for_new_scale(4.0)

        # ASSERT
        display_after_scale_change = dialog.aligned_display.copy()

        # The image content should be different due to different downscaling.
        assert display_before_scale_change.shape != display_after_scale_change.shape

        # To verify the transformation was reapplied correctly, we accept and check
        # the final high-res image. This indirectly proves the display was correct.
        dialog.accept_alignment()
        final_image = dialog.get_current_aligned_image()

        # Manually build the expected image
        original_image = dialog.original_aligned_image
        h, w = original_image.shape[:2]
        center = (w / 2, h / 2)
        rot_mat = cv2.getRotationMatrix2D(center, 30.0, 1.0)
        step1_img = cv2.warpAffine(original_image, rot_mat, (w, h))

        # Use the final scale factor of 4.0
        actual_dx, actual_dy = 25 * 4.0, 25 * 4.0
        trans_mat = np.float32([[1, 0, actual_dx], [0, 1, actual_dy]])
        expected_image = cv2.warpAffine(step1_img, trans_mat, (w, h))

        final_center = get_feature_center(final_image)
        expected_center = get_feature_center(expected_image)

        assert final_center == pytest.approx(expected_center, abs=1.5)
