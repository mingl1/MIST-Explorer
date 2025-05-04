import numpy as np
import cv2
from PyQt6.QtCore import pyqtSignal, QThread, QObject

class CellLayerAligner(QThread):
    """Worker thread for aligning cell layers"""
    progress = pyqtSignal(int, str)
    error = pyqtSignal(str)
    aligned_image_signal = pyqtSignal(np.ndarray)
    
    def __init__(self, target_image=None, unaligned_image=None):
        super().__init__()
        self.target_image = target_image
        self.unaligned_image = unaligned_image
        self.result = None
        self.downscale_factor = 4  # For initial alignment
        
    def set_images(self, target_image, unaligned_image):
        """Set the target and unaligned images for processing"""
        self.target_image = target_image
        self.unaligned_image = unaligned_image
        
    def run(self):
        """Main processing function that runs in the thread"""
        if self.target_image is None or self.unaligned_image is None:
            self.error.emit("Both target and unaligned images must be provided")
            return
            
        try:
            self.progress.emit(10, "Preparing images for alignment")
            
            # Convert images to grayscale if they're not already
            if len(self.target_image.shape) > 2 and self.target_image.shape[2] > 1:
                target_gray = cv2.cvtColor(self.target_image, cv2.COLOR_BGR2GRAY)
            else:
                target_gray = self.target_image
                
            if len(self.unaligned_image.shape) > 2 and self.unaligned_image.shape[2] > 1:
                unaligned_gray = cv2.cvtColor(self.unaligned_image, cv2.COLOR_BGR2GRAY)
            else:
                unaligned_gray = self.unaligned_image
            
            # Ensure both images are uint8
            self.progress.emit(20, "Converting images")
            target_gray_8bit = self.to_uint8(target_gray)
            unaligned_gray_8bit = self.to_uint8(unaligned_gray)
            
            # Downscale images for faster processing
            self.progress.emit(25, "Downscaling images")
            target_small = target_gray_8bit[::self.downscale_factor, ::self.downscale_factor]
            unaligned_small = unaligned_gray_8bit[::self.downscale_factor, ::self.downscale_factor]
            
            # Enhance contrast using histogram equalization
            self.progress.emit(30, "Enhancing contrast")
            target_small = self.adjust_contrast(target_small)
            unaligned_small = self.adjust_contrast(unaligned_small)
            
            # Perform SIFT-based alignment on small images
            self.progress.emit(40, "Detecting features")
            H = self.align_images_sift(unaligned_small, target_small)
            
            if H is None:
                self.error.emit("Could not find a valid transformation - not enough matches")
                return
            
            # Apply homography to full-size image
            self.progress.emit(80, "Applying transformation to full image")
            height, width = target_gray.shape
            aligned_image = cv2.warpPerspective(self.unaligned_image, H, (width, height))
                
            self.result = aligned_image
            self.progress.emit(100, "Alignment complete")
            self.aligned_image_signal.emit(aligned_image)
            
        except Exception as e:
            self.error.emit(f"Error during alignment: {str(e)}")
    
    def to_uint8(self, image):
        """Convert image to uint8 with proper scaling"""
        # Check if image is already uint8
        if image.dtype == np.uint8:
            return image
            
        # Convert to float and scale to 0-255
        img_float = image.astype(np.float32)
        if img_float.max() > img_float.min():  # Check to avoid division by zero
            img_norm = ((img_float - img_float.min()) * (255.0 / (img_float.max() - img_float.min())))
            return img_norm.astype(np.uint8)
        else:
            return np.zeros_like(image, dtype=np.uint8)
    
    def adjust_contrast(self, img, min_percentile=2, max_percentile=98):
        """Adjust image contrast using percentile-based clipping"""
        # Calculate percentiles
        minval = np.percentile(img, min_percentile)
        maxval = np.percentile(img, max_percentile)
        
        # Clip and rescale
        img_adjusted = np.clip(img, minval, maxval)
        img_adjusted = ((img_adjusted - minval) / (maxval - minval)) * 255
        return img_adjusted.astype(np.uint8)
    
    def align_images_sift(self, floating_img, target_img):
        """Align images using SIFT feature detection and FLANN-based matching"""
        try:
            # Try to create SIFT detector
            sift = cv2.SIFT_create()
            
            # Detect keypoints and compute descriptors
            kp1, des1 = sift.detectAndCompute(floating_img, None)
            kp2, des2 = sift.detectAndCompute(target_img, None)
            
            using_sift = True
        except cv2.error:
            # Fall back to ORB if SIFT is not available
            self.progress.emit(45, "SIFT unavailable, using ORB")
            orb = cv2.ORB_create(nfeatures=2000)
            
            # Detect keypoints and compute descriptors with ORB
            kp1, des1 = orb.detectAndCompute(floating_img, None)
            kp2, des2 = orb.detectAndCompute(target_img, None)
            
            using_sift = False
        
        # Check if descriptors were found
        if des1 is None or des2 is None or len(des1) < 2 or len(des2) < 2:
            return None
        
        if using_sift:
            # Set up FLANN-based matcher for SIFT
            FLANN_INDEX_KDTREE = 1
            index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
            search_params = dict(checks=50)
            
            try:
                flann = cv2.FlannBasedMatcher(index_params, search_params)
                matches = flann.knnMatch(des1, des2, k=2)
                
                # Apply ratio test to get good matches
                self.progress.emit(50, "Filtering matches")
                good_matches = []
                for m, n in matches:
                    if m.distance < 0.7 * n.distance:
                        good_matches.append(m)
            except Exception:
                # Fall back to brute force matcher if FLANN fails
                self.progress.emit(50, "Using brute force matcher")
                bf = cv2.BFMatcher()
                matches = bf.knnMatch(des1, des2, k=2)
                
                # Apply ratio test
                good_matches = []
                for m, n in matches:
                    if m.distance < 0.7 * n.distance:
                        good_matches.append(m)
        else:
            # Use BFMatcher with Hamming distance for ORB
            bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
            matches = bf.match(des1, des2)
            
            # Sort matches by distance
            matches = sorted(matches, key=lambda x: x.distance)
            
            # Use only good matches
            good_matches = matches[:min(100, len(matches))]
        
        # Check if we have enough good matches
        if len(good_matches) < 10:
            return None
        
        # Get matched keypoints
        self.progress.emit(60, "Computing homography")
        if using_sift:
            src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
            dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        else:
            src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
            dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        
        # Find homography matrix
        H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        
        # Scale homography matrix for the full-size image
        if H is not None:
            # Adjust for downscaling
            H[0, 2] *= self.downscale_factor
            H[1, 2] *= self.downscale_factor
            
        return H 