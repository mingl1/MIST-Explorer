import cv2
import numpy as np
from core.register import TileMap
import os
from pathlib import Path

def load_image(image_path):
    """Load an image and convert to grayscale if needed."""
    img = cv2.imread(str(image_path))
    if img is None:
        raise ValueError(f"Could not load image at {image_path}")
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img

def align_tiles_with_sift(fixed_img, moving_img, tile_size=512, overlap=100):
    """
    Align two images using SIFT feature matching on tiles.
    
    Args:
        fixed_img: Reference image
        moving_img: Image to be aligned
        tile_size: Size of each tile
        overlap: Overlap between tiles in pixels
    
    Returns:
        Aligned image
    """
    # Create tile maps
    fixed_map = TileMap("fixed", fixed_img, overlap, tile_size)
    moving_map = TileMap("moving", moving_img, overlap, tile_size)
    
    # Initialize SIFT detector
    sift = cv2.SIFT_create()
    
    # Create output image
    output = np.zeros_like(fixed_img)
    
    # Process each tile pair
    for (fixed_tile, fixed_bounds), (moving_tile, moving_bounds) in zip(fixed_map, moving_map):
        # Find keypoints and descriptors
        kp1, des1 = sift.detectAndCompute(fixed_tile, None)
        kp2, des2 = sift.detectAndCompute(moving_tile, None)
        
        y0, x0 = fixed_bounds["ymin"], fixed_bounds["xmin"]
        h, w = moving_tile.shape
        h_out = min(h, output.shape[0] - y0)
        w_out = min(w, output.shape[1] - x0)

        if des1 is None or des2 is None or len(kp1) < 2 or len(kp2) < 2:
            # If not enough features found, use the moving tile as is
            output[y0:y0+h_out, x0:x0+w_out] = moving_tile[:h_out, :w_out]
            continue
        
        # Match features
        bf = cv2.BFMatcher()
        matches = bf.knnMatch(des1, des2, k=2)
        
        # Apply ratio test
        good_matches = []
        for m, n in matches:
            if m.distance < 0.75 * n.distance:
                good_matches.append(m)
        
        if len(good_matches) < 4:
            # If not enough good matches, use the moving tile as is
            output[y0:y0+h_out, x0:x0+w_out] = moving_tile[:h_out, :w_out]
            continue
        
        # Get matched keypoints
        src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        
        # Find homography
        H, mask = cv2.findHomography(dst_pts, src_pts, cv2.RANSAC, 5.0)
        
        # Warp moving tile
        warped = cv2.warpPerspective(moving_tile, H, (fixed_tile.shape[1], fixed_tile.shape[0]))
        wh, ww = warped.shape
        wh_out = min(wh, output.shape[0] - y0)
        ww_out = min(ww, output.shape[1] - x0)
        # Place warped tile in output
        output[y0:y0+wh_out, x0:x0+ww_out] = warped[:wh_out, :ww_out]
    
    return output

from PIL import Image

def main():
    Image.MAX_IMAGE_PIXELS = 99999999999  
    # Example usage
    fixed_path = Path("path/to/fixed_image.png")
    moving_path = Path("path/to/moving_image.png")
    
    # Load images
    fixed_img = load_image(fixed_path)
    moving_img = load_image(moving_path)
    
    # Align images
    # aligned_img = align_tiles_with_sift(fixed_img, moving_img)
    
    # Save result
    output_path = Path("aligned_output.png")
    cv2.imwrite(str(output_path), aligned_img)
    print(f"Aligned image saved to {output_path}")

if __name__ == "__main__":
    main() 