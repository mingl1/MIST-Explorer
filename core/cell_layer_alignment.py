import cv2
import numpy as np
from PyQt6.QtCore import pyqtSignal, QThread, QObject
from pystackreg import StackReg
from skimage import transform
from utils import build_optical_flow_pyramid_pure_numpy, adjust_contrast, to_uint8
class CellLayerAligner(QThread):
    """Worker thread for aligning cell layers"""
    progress = pyqtSignal(int, str)
    error = pyqtSignal(str)
    aligned_image_signal = pyqtSignal(np.ndarray, np.ndarray, np.ndarray)  # Full aligned image, downscaled target, downscaled aligned
    
    def __init__(self, pyramid_level=3):
        super().__init__()
        self.target_image = np.zeros(0)
        self.unaligned_image = np.zeros(0)
        self.result = None
        self.pyramid_level = pyramid_level # For initial alignment
        self.downscale_level = -1 # Use most course level for now
        self.target_pyramid = []
        self.unaligned_pyramid =[]     
    
    def set_target_image(self,target_image):
        self.target_image = target_image
        # Ensure both images are uint8
        target_gray_8bit = to_uint8(self.target_image)
        # Downscale images for faster processing
        self.target_pyramid = build_optical_flow_pyramid_pure_numpy(target_gray_8bit,self.pyramid_level)
    
    def set_unaligned_image(self,unaligned_image):
        
        self.unaligned_image = unaligned_image
        unaligned_gray_8bit =to_uint8(self.unaligned_image)
        self.unaligned_pyramid = build_optical_flow_pyramid_pure_numpy(unaligned_gray_8bit,self.pyramid_level)
    
    def run(self):
        """Main processing function that runs in the thread"""
        if self.target_image is None or self.unaligned_image is None:
            self.error.emit("Both target and unaligned images must be provided")
            return
            
        try:
            target_small = self.target_pyramid[self.downscale_level] 
            unaligned_small = self.unaligned_pyramid[self.downscale_level]
            target_small = target_small[:min(target_small.shape[0], unaligned_small.shape[0]), :min(target_small.shape[1], unaligned_small.shape[1])]
            unaligned_small = unaligned_small[:min(target_small.shape[0], unaligned_small.shape[0]), :min(target_small.shape[1], unaligned_small.shape[1])]
            # Enhance contrast using histogram equalization
            self.progress.emit(30, "Enhancing contrast")
            target_small = adjust_contrast(target_small,50,99)
            unaligned_small = adjust_contrast(unaligned_small,50,99)
            
            # Perform SIFT-based alignment on small images
            self.progress.emit(40, "Detecting features")
            aligned_small,H = self.align_images(unaligned_small, target_small)
            if H is None:
                self.error.emit("Could not find a valid transformation - not enough matches")
                return
            
            # Apply homography to full-size image
            self.progress.emit(80, "Applying transformation to full image")
            aligned_image = self.warp_image(self.unaligned_image,H)
            self.result = to_uint8(aligned_image)
            self.progress.emit(100, "Alignment complete")
            self.aligned_image_signal.emit(self.result, target_small, aligned_small)
            
        except Exception as e:
            self.error.emit(f"Error during alignment: {str(e)}")
    

    
    def align_images(self, unaligned, target):
        sr = StackReg(StackReg.AFFINE)
        out_tra = sr.register_transform(target, unaligned)
        mat = sr.get_matrix()
        scale = unaligned.shape[0]/self.unaligned_image.shape[0]
        S = np.diag([scale,scale,1.0])
        S_inv = np.linalg.inv(S)
        H_full = S_inv @ mat @ S
        return out_tra, H_full
        
    def warp_image(self,img,H):
        # w,h = shape
        tform = transform.AffineTransform(matrix=H)
        tf_img = transform.warp(img, tform)
        return tf_img
