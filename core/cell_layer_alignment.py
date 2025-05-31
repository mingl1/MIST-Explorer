import numpy as np
from PyQt6.QtCore import pyqtSignal, QThread
from pystackreg import StackReg
from pystackreg.util import to_uint16
from skimage import transform
from skimage.feature import match_descriptors, SIFT
from skimage.measure import ransac
from utils import build_optical_flow_pyramid_pure_numpy, adjust_contrast, make_same_shape,warp_image,remove_padding
class CellLayerAligner(QThread):
    """Two Staged Cell Layer Alignment Using SIFT & pystackreg's Algorithm"""
    progress = pyqtSignal(int, str)
    error = pyqtSignal(str)
    aligned_image_signal = pyqtSignal(np.ndarray, np.ndarray, np.ndarray)  
    
    def __init__(self, pyramid_level=6):
        super().__init__()
        self.target_image = np.zeros(0).astype(np.uint16)
        self.unaligned_image = np.zeros(0).astype(np.uint16)
        self.pyramid_level = pyramid_level # Should be at least 6? Enough to blur the cells together into a shape
        self.coarse_level = -1  # Smallest pyramid level for SIFT
        self.fine_level = 2    
        self.target_pyramid = []
        self.unaligned_pyramid = []
        self.original_target_shape = None
        self.sift_detector = SIFT(n_octaves=8, n_scales=3, sigma_min=1.6)
    
    def set_target_image(self, target_image):
        """Set target image and clear pyramid cache"""
        self.target_pyramid = []
        self.target_image = target_image
        self.original_target_shape = target_image.shape
    
    def set_unaligned_image(self, unaligned_image):
        """Set unaligned image and clear pyramid cache"""
        self.unaligned_pyramid = []
        self.unaligned_image = unaligned_image
    
    def run(self):
        """Main processing function that runs in the thread"""
        if self.target_image is None or self.unaligned_image is None:
            self._fatal_error_message("Both target and unaligned images must be provided")
            return
        
        try:
            # Ensure images have the same shape for alignment
            self.progress.emit(5, "Preparing images")
            target_aligned, unaligned_aligned = make_same_shape(self.target_image, self.unaligned_image)
            self.target_image = target_aligned
            self.unaligned_image = unaligned_aligned
            
            self.progress.emit(10, "Building image pyramids")
            if not self.target_pyramid or self.target_pyramid[0].shape != self.target_image.shape:
                self.target_pyramid = build_optical_flow_pyramid_pure_numpy(self.target_image, self.pyramid_level)
            
            if not self.unaligned_pyramid or self.unaligned_pyramid[0].shape != self.unaligned_image.shape:
                self.unaligned_pyramid = build_optical_flow_pyramid_pure_numpy(self.unaligned_image, self.pyramid_level)
            
            # Stage 1: SIFT-based coarse alignment
            self.progress.emit(30, "SIFT-based coarse alignment")
            coarse_transform = self._coarse_alignment_sift()
            
            if coarse_transform is None:
                return
            
            # Apply coarse transform and prepare for fine alignment
            fine_transform = self._scale_transform_matrix(coarse_transform, self.coarse_level, self.fine_level)
            fine_target_image = self.target_pyramid[self.fine_level]
            fine_unaligned_image = self.unaligned_pyramid[self.fine_level]
            
            # Apply coarse transformation to fine level image
            coarse_aligned_fine = warp_image(fine_unaligned_image, fine_transform)
            
            # Stage 2: StackReg fine alignment
            self.progress.emit(60, "StackReg fine alignment")
            refinement_transform, aligned_preview = self._alignment_stackreg(fine_target_image, coarse_aligned_fine)
            
            if refinement_transform is None:
                return
            
            # Combine transformations and apply to full resolution
            self.progress.emit(80, "Applying final transformation")
            full_coarse_transform = self._scale_transform_matrix(coarse_transform, self.coarse_level, 0)
            full_refinement_transform = self._scale_transform_matrix(refinement_transform, self.fine_level, 0)
            
            # Apply transformations sequentially for better accuracy
            intermediate_aligned = warp_image(self.unaligned_image, full_coarse_transform)
            final_aligned_image = warp_image(intermediate_aligned, full_refinement_transform)
            
            # Remove padding
            final_aligned_image,target_preview,aligned_preview = self._process_aligned_image(
                final_aligned_image, fine_target_image, aligned_preview, self.original_target_shape)
            
            # Convert result back to original dtype
            result = self._convert_to_original_dtype(final_aligned_image, self.unaligned_image.dtype)
            
            self.progress.emit(100, "Two-stage alignment complete")
            self.aligned_image_signal.emit(result, target_preview, aligned_preview)
            
        except Exception as e:
            self._fatal_error_message(f"Error during alignment: {str(e)}")
    
    def _coarse_alignment_sift(self):
        """Enhanced SIFT-based coarse alignment with fallback strategies"""
        try:
            # Get images at coarsest level
            target_coarse = self.target_pyramid[self.coarse_level]
            unaligned_coarse = self.unaligned_pyramid[self.coarse_level]
            
            enhance_method = self._image_enhancement
            detector = self.sift_detector
            
            target_enhanced = enhance_method(target_coarse)
            unaligned_enhanced = enhance_method(unaligned_coarse)
            
            # Detect features
            detector.detect_and_extract(target_enhanced)
            keypoints1 = detector.keypoints
            descriptors1 = detector.descriptors
            
            detector.detect_and_extract(unaligned_enhanced)
            keypoints2 = detector.keypoints
            descriptors2 = detector.descriptors
            
            feature_count_1 = len(descriptors1) if descriptors1 is not None else 0
            feature_count_2 = len(descriptors2) if descriptors2 is not None else 0
            
            print(f"  SIFT config: target={feature_count_1}, unaligned={feature_count_2}")
            if descriptors1 is None or descriptors2 is None or len(descriptors1) < 4 or len(descriptors2) < 4:
                self._fatal_error_message("SIFT Alignment Failed: Not Enough Descriptors")
                return None
            matches = match_descriptors(descriptors2, descriptors1, 
                        cross_check=True, max_ratio=0.8)
            if len(matches) < 4:
                self._fatal_error_message("SIFT Alignment Failed: Not Enough Matches")
                return None
            if keypoints1 is None or keypoints2 is None:
                self._fatal_error_message("SIFT Alignment Failed: Key Points Empty")
                return None
            src_pts = keypoints2[matches[:, 0]][:, ::-1]  
            dst_pts = keypoints1[matches[:, 1]][:, ::-1] 
            
            # Find homography using RANSAC
            model_robust, inliers = ransac(
                (src_pts, dst_pts),
                transform.AffineTransform,
                min_samples=4,
                residual_threshold=2,
                max_trials=1000
            )
            if model_robust is None:
                self._fatal_error_message("SIFT Alignment Failed: RANSAC Failed")
                return None
            H_coarse = model_robust.params
            if model_robust is None or np.sum(np.array(inliers)) < 4:
                self._fatal_error_message("SIFT Alignment Failed: Inliers < 4")
                return None
            return H_coarse
        
        except Exception as e:
            self._fatal_error_message(f"SIFT alignment error: {e}")
            return None

    def _alignment_stackreg(self, target_fine, unaligned_fine):
        """Enhanced StackReg alignment with better preprocessing"""
        try:
            assert target_fine.dtype == np.float64
            assert unaligned_fine.dtype == np.float64
            
            # Enhanced contrast adjustment
            target_enhanced = self._image_enhancement(target_fine)
            unaligned_enhanced = self._image_enhancement(unaligned_fine)
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
            aligned_result = to_uint16(aligned_result)
            return refinement_matrix, aligned_result
            
        except Exception as e:
            print(f"StackReg refinement error: {e}")
            return None, None
    
    def _image_enhancement(self, image):
        """Original SIFT enhancement method"""
        return adjust_contrast(image,50,99)

    def _scale_transform_matrix(self, matrix, from_level, to_level):
        """Enhanced transformation matrix scaling with validation"""
        if matrix is None:
            return None
        
        # Calculate scale factor between pyramid levels
        from_shape = self.target_pyramid[from_level].shape[0]
        to_shape = self.target_pyramid[to_level].shape[0]
        scale_factor = to_shape / from_shape
        
        scaled_matrix = matrix.copy()
        
        if matrix.shape == (2, 3):  # Affine matrix
            scaled_matrix[0, 2] *= scale_factor  # x translation
            scaled_matrix[1, 2] *= scale_factor  # y translation
        elif matrix.shape == (3, 3):  # Homography matrix
            # Scale coordinates: H_scaled = S_to^-1 @ H @ S_from
            S_from = np.diag([1/scale_factor, 1/scale_factor, 1.0])
            S_to_inv = np.diag([scale_factor, scale_factor, 1.0])
            scaled_matrix = S_to_inv @ matrix @ S_from
        
        return scaled_matrix

    def _convert_to_original_dtype(self, img_float, original_dtype):
        """Convert result back to original dtype with proper range handling"""
        if original_dtype == np.uint16:
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
    
    def _process_aligned_image(self,final_aligned_image, target_preview, aligned_preview, original_target_shape):
        """Process and remove padding from aligned images"""
        
        # Remove padding from main image
        finalH, finalW = original_target_shape
        final_aligned_image_cropped = remove_padding(final_aligned_image, original_target_shape)
        
        # Calculate scaling factor for preview images
        scaling = target_preview.shape[0] / final_aligned_image.shape[0]
        
        # Calculate preview dimensions after scaling
        preview_height = int(finalH * scaling)
        preview_width = int(finalW * scaling)
        
        # Remove padding from preview images
        target_preview_cropped = remove_padding(target_preview, (preview_height, preview_width))
        aligned_preview_cropped = remove_padding(aligned_preview, (preview_height, preview_width))
        
        return final_aligned_image_cropped, target_preview_cropped, aligned_preview_cropped
    
    def _fatal_error_message(self, msg):
        self.error.emit(msg)
        self.progress.emit(100,"Retry Maybe")