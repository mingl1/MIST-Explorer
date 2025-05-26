from PyQt6.QtWidgets import QFileDialog
from PyQt6.QtGui import  QImage, QPixmap
from PyQt6.QtCore import QTimer
import tifffile as tiff, numpy as np
from PIL import Image, ImageSequence
import cv2
import time


def numpy_to_qimage(array:np.ndarray) -> QImage:

    if not array.data.contiguous:
        array = np.ascontiguousarray(array)

    if len(array.shape) == 2:
        # Grayscale image
        height, width = array.shape
        format = QImage.Format.Format_Grayscale16 if array.dtype == np.uint16 else QImage.Format.Format_Grayscale8
        bytes_per_pixel = 2 if array.dtype == np.uint16 else 1
        bytes_per_line = width * bytes_per_pixel # uint8
        qimage =  QImage(array.data, width, height, bytes_per_line, format)
    elif len(array.shape) == 3:
        height, width, channels = array.shape
        if channels == 3:
            # RGB image
            qimage = QImage(array.data, width, height, width*channels, QImage.Format.Format_RGB888)
        elif channels == 4:
            # RGBA image
            qimage = QImage(array.data, width, height, width * channels, QImage.Format.Format_RGBA8888)
    else:
        raise ValueError("Unsupported array shape: {}".format(array.shape))
    return qimage.copy()


def qimage_to_numpy(qimage: QImage):
    # Ensure the QImage format is suitable for conversion
    valid_formats = [QImage.Format.Format_Grayscale8, QImage.Format.Format_Grayscale16]
    ptr = qimage.bits()
    width = qimage.width()
    height = qimage.height()

    # convert to grayscale
    if qimage.format() not in valid_formats:
        if qimage.format() == QImage.Format.Format_RGB32:
    
            ptr.setsize(width * height *4)
            arr = np.array(ptr, dtype=np.uint8).reshape(height, width, 4)

            return arr
        else:
            raise ValueError("Unsupported dtype")
        
    elif qimage.format() == QImage.Format.Format_Grayscale16:
        qimage = qimage.convertToFormat(QImage.Format.Format_Grayscale8)

    elif qimage.format() == QImage.Format.Format_Grayscale8:
        qimage = qimage
    else:
        raise ValueError("Unsupported dtype")


    # Set buffer size based on dtype
    ptr.setsize(width * height)
    
    print(qimage.format())
    arr = np.array(ptr).reshape(height, width)

    return arr
 
    

# this and qimage to numpy seems repetitive, delete one of them
def pixmap_to_image(pixmap: QPixmap):
    
    if pixmap == None:
        return None
    # Convert QPixmap to QImage
    qimage = pixmap.toImage()

    # Convert QImage to numpy array
    width = qimage.width()
    height = qimage.height()
    ptr = qimage.bits()
    ptr.setsize(height * width * 4)
    arr = np.array(ptr).reshape(height, width, 4)  # 4 for RGBA

    # Save numpy array as an image file using OpenCV
    return arr


def is_grayscale(image: np.ndarray) -> bool:

    if len(image.shape) == 3 and image.shape[2] == 3:
        return False
    elif len(image.shape) == 2 or (len(image.shape) == 3 and image.shape[2] == 1):
        return True
    else:
        raise ValueError("Image format not recognized")


def normalize_to_uint8(data: np.ndarray) -> np.ndarray:
    normalized_data = 255 * (data - np.min(data)) / (np.max(data) - np.min(data))
    normalized_data = normalized_data.astype(np.uint8)
    return normalized_data

def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.2989, 0.5870, 0.1140])

def convert_image_to_gray(img: np.ndarray) -> np.ndarray:
    if img.dtype==np.uint16:
        img = img.astype(np.float32) / 65535.0
    if is_grayscale(img):
        return img
    else:
        return rgb2gray(img)

def adjustContrast(img, alpha=5, beta=15):  
    
    alpha = 5 # Contrast control
    beta = 15 # Brightness control
    return cv2.convertScaleAbs(img, alpha=alpha, beta=beta)


# uint16 to uint8
def scale_adjust(arr:np.ndarray):
    if arr.dtype == np.uint16:
        return cv2.convertScaleAbs(arr, alpha=(255.0/65535.0))
    elif arr.dtype == np.uint8:
        return arr
    elif arr.dtype == np.uint32:
        array_uint8 = ((arr / arr.max()) * 255).astype(np.uint8)
        return array_uint8
    else:
        print("unsupported array type: ",arr.dtype)

def auto_contrast(img):
    return adjustContrast(scale_adjust(img))


def gaussian_kernel_1d(sigma, radius=None):
    """Generate 1D Gaussian kernel."""
    if radius is None:
        radius = int(np.ceil(3 * sigma))
    size = 2 * radius + 1
    x = np.arange(-radius, radius + 1, dtype=np.float32)
    kernel = np.exp(-(x**2) / (2 * sigma**2))
    kernel /= kernel.sum()
    return kernel


def gaussian_blur_separable(image, sigma=1.0):
    """Apply separable Gaussian blur manually using numpy."""
    kernel = gaussian_kernel_1d(sigma)
    
    # Manual separable convolution - more efficient than apply_along_axis
    # Horizontal pass
    pad_width = len(kernel) // 2
    padded = np.pad(image, ((0, 0), (pad_width, pad_width)), mode='reflect')
    blurred_h = np.zeros_like(image)
    
    for i in range(image.shape[0]):
        blurred_h[i] = np.convolve(padded[i], kernel, mode='valid')
    
    # Vertical pass
    padded = np.pad(blurred_h, ((pad_width, pad_width), (0, 0)), mode='reflect')
    blurred = np.zeros_like(image)
    
    for j in range(image.shape[1]):
        blurred[:, j] = np.convolve(padded[:, j], kernel, mode='valid')
    
    return blurred

def downsample(image, scale=0.5):
    """Downsample image by given scale factor."""
    step = int(round(1 / scale))
    return image[::step, ::step]


def build_optical_flow_pyramid_pure_numpy(image, max_level=3, scale=0.5, sigma=1.0, min_size=16):
    """Version using only numpy with manual separable convolution."""
    assert image.ndim == 2, "Only grayscale images supported"
    image = image.astype(np.float32)
    
    pyramid = [image]
    current = image
    
    for level in range(max_level):
        if min(current.shape) < min_size:
            break
            
        blurred = gaussian_blur_separable(current, sigma=sigma)
        current = downsample(blurred, scale=scale)
        pyramid.append(current)
    
    return pyramid

def adjust_contrast(img, min_percentile=2, max_percentile=98):
    """Adjust image contrast using percentile-based clipping"""
    # Calculate percentiles
    minval = np.percentile(img, min_percentile)
    maxval = np.percentile(img, max_percentile)
    
    # Clip and rescale
    img_adjusted = np.clip(img, minval, maxval)
    img_adjusted = ((img_adjusted - minval) / (maxval - minval)) * 255
    return img_adjusted.astype(np.uint8)