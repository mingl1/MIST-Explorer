import gc
from typing import Dict, OrderedDict
import weakref
from PyQt6.QtWidgets import QGraphicsView, QGraphicsScene, QGraphicsPixmapItem, QWidget, QTableWidget, QTableWidgetItem, QVBoxLayout
from PyQt6.QtGui import QDragEnterEvent, QDropEvent, QPixmap,  QCursor, QImage
from PyQt6.QtCore import Qt, QSize, pyqtSignal, pyqtSlot, QThreadPool, QRunnable, QTimer, QObject
import tifffile as tiff, numpy as np, matplotlib as mpl, time, cv2, xml.etree.ElementTree as ET, os, copy
from core.Worker import Worker
from utils import *
from ui.Dialogs import ImageDialog
import numpy as np
from typing import Dict, List, Optional
import weakref

class PyramidImageWrapper:
    """Wrapper that stores Gaussian pyramid and provides memory-efficient access."""
    
    def __init__(self, image: np.ndarray, max_levels: int = 4, scale: float = 0.5, 
                 sigma: float = 1.0, min_size: int = 32, convert_to_uint8: bool = True):
        self.original_shape = image.shape
        self.original_dtype = image.dtype
        self.max_levels = max_levels
        self.scale = scale
        self.convert_to_uint8 = convert_to_uint8
        
        # Build pyramid first with original precision
        self.pyramid = build_optical_flow_pyramid_pure_numpy(
            image, max_levels, scale, sigma, min_size
        )
        
        # Convert each pyramid level to uint8 after creation if requested
        if convert_to_uint8:
            self.pyramid = [to_uint8(level) for level in self.pyramid]
            self.dtype = np.uint8
        else:
            self.dtype = image.dtype
        
        # Cache for frequently accessed levels
        self._level_cache = weakref.WeakValueDictionary()
    
    @property
    def data(self) -> np.ndarray:
        """Returns the original full-resolution image."""
        return self.pyramid[0]
    
    @property 
    def coarsest(self) -> np.ndarray:
        """Returns the coarsest (smallest) level of the pyramid."""
        return self.pyramid[-1]
    
    def get_level(self, level: int) -> np.ndarray:
        """Get specific pyramid level with caching."""
        if level >= len(self.pyramid):
            raise IndexError(f"Level {level} not available. Max level: {len(self.pyramid)-1}")
        
        # Use weak reference cache to avoid keeping multiple copies in memory
        cache_key = f"level_{level}"
        if cache_key in self._level_cache:
            return self._level_cache[cache_key]
        
        level_data = self.pyramid[level]
        self._level_cache[cache_key] = level_data
        return level_data
    
    def get_memory_usage(self) -> Dict[str, int]:
        """Calculate memory usage for each pyramid level."""
        usage = {}
        total = 0
        for i, level in enumerate(self.pyramid):
            size_bytes = level.nbytes
            usage[f"level_{i}"] = size_bytes
            total += size_bytes
        usage["total"] = total
        return usage
    

class MemoryEfficientImageCache:
    """Memory-efficient cache with size limits and automatic cleanup."""
    
    def __init__(self, max_cache_size_mb=500, max_entries_per_channel=5):
        self.max_cache_size_bytes = max_cache_size_mb * 1024 * 1024
        self.max_entries_per_channel = max_entries_per_channel
        self.cache = {}  # {channel: OrderedDict of {cache_key: image_data}}
        self.current_size_bytes = 0
    
    def get(self, channel, cache_key):
        """Get cached image if available."""
        if channel not in self.cache:
            return None
        
        channel_cache = self.cache[channel]
        if cache_key in channel_cache:
            # Move to end (most recently used)
            image_data = channel_cache[cache_key]
            del channel_cache[cache_key]
            channel_cache[cache_key] = image_data
            return image_data
        return None
    
    def put(self, channel, cache_key, image_data):
        """Cache image data with memory management."""
        if channel not in self.cache:
            self.cache[channel] = OrderedDict()
        
        channel_cache = self.cache[channel]
        image_size_bytes = image_data.nbytes
        
        # Remove if already exists to update size tracking
        if cache_key in channel_cache:
            old_data = channel_cache[cache_key]
            self.current_size_bytes -= old_data.nbytes
            del channel_cache[cache_key]
        
        # Check if we need to free memory
        while (self.current_size_bytes + image_size_bytes > self.max_cache_size_bytes or
               len(channel_cache) >= self.max_entries_per_channel):
            if not channel_cache:
                break
            
            # Remove least recently used item
            old_key, old_data = channel_cache.popitem(last=False)
            self.current_size_bytes -= old_data.nbytes
            del old_data  # Explicit deletion
        
        # Add new data
        channel_cache[cache_key] = image_data
        self.current_size_bytes += image_size_bytes
        
        # Force garbage collection if cache is getting large
        if self.current_size_bytes > self.max_cache_size_bytes * 0.8:
            gc.collect()
    
    def clear_channel(self, channel):
        """Clear cache for specific channel."""
        if channel in self.cache:
            channel_cache = self.cache[channel]
            for image_data in channel_cache.values():
                self.current_size_bytes -= image_data.nbytes
            channel_cache.clear()
            gc.collect()
    
    def clear_all(self):
        """Clear entire cache."""
        for channel_cache in self.cache.values():
            channel_cache.clear()
        self.cache.clear()
        self.current_size_bytes = 0
        gc.collect()
    
    def get_memory_info(self):
        """Get current memory usage info."""
        return {
            'current_size_mb': self.current_size_bytes / (1024*1024),
            'max_size_mb': self.max_cache_size_bytes / (1024*1024),
            'channels': len(self.cache),
            'total_entries': sum(len(ch_cache) for ch_cache in self.cache.values())
        }



def clear_contrast_cache(self):
    """Clear contrast cache when switching channels or loading new images."""
    if hasattr(self, 'memory_cache'):
        self.memory_cache.clear_all()
        print("Cleared contrast cache")

def get_cache_info(self):
    """Get current cache information for debugging."""
    if hasattr(self, 'memory_cache'):
        return self.memory_cache.get_memory_info()
    return {"status": "No cache initialized"}

# Memory monitoring utility
def monitor_memory_usage():
    """Simple memory monitoring function."""
    import psutil
    import os
    
    process = psutil.Process(os.getpid())
    memory_info = process.memory_info()
    
    return {
        'rss_mb': memory_info.rss / (1024*1024),  # Resident Set Size
        'vms_mb': memory_info.vms / (1024*1024),  # Virtual Memory Size
        'percent': process.memory_percent()
    }

class ImageStorage:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance.image_list = {}
        return cls._instance
    
    def add_data(self, image_id, data):
        self._instance.image_list[image_id] = data

    def get_data(self, image_id):
        return self._instance.image_list[image_id]
    
    def clear_data(self):
        self._instance.image_list.clear()

class ImageWrapper:
    def __init__(self, data, name="", cmap="gray"):
        if not isinstance(data, np.ndarray):
            raise TypeError("Data must be a numpy array.")
        
        self.name = name
        self.cmap = cmap
        self.data = data
        self.contrast_min = 0
        self.contrast_max = 255

class __BaseGraphicsView(QWidget):
    '''base class for graphics view'''
    multi_layer = pyqtSignal(dict, bool)
    single_layer = pyqtSignal(np.ndarray)
    updateProgress = pyqtSignal(int, str)
    errorSignal = pyqtSignal(str)
    fill_metadata = pyqtSignal(dict)
    update_manager = pyqtSignal(dict, str)
    add_image_to_storage = pyqtSignal(str, object)
    def __init__(self, parent=None):
        super().__init__(parent)

        
        self.setMinimumSize(QSize(300, 300))
        self.scene = QGraphicsView()
        self.scene.setScene(QGraphicsScene(self))
        self.reset_pixmap =  None
        self.reset_pixmapItem = None
        self.pixmap = None
        self.pixmapItem=None
        self.np_channels = {}
        self.reset_np_channels = {}
        self.currentChannelNum = 0
        self.is_layered = False
        self.image_cache = {}
        self.lut_cache = {}
        self.image_count = 0
        self.storage = ImageStorage()

    def dragEnterEvent(self, event: QDragEnterEvent):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dragMoveEvent(self, event: QDragEnterEvent):
        event.acceptProposedAction()

    def read_tiff_pages(self, file_path):
        """Generator that yields pages one at a time to avoid memory overload."""
        with tiff.TiffFile(file_path) as tif:
            total_pages = len(tif.pages)
            valid_page_count = 0
            
            for i, page in enumerate(tif.pages):
                try:
                    image = page.asarray()
                    # Check if the image is blank
                    if np.all(image == image.flat[0]):
                        continue
                    
                    valid_page_count += 1
                    self.updateProgress.emit(int((i+1)/total_pages*50), "Loading image")  # Use 50% for reading
                    yield image, valid_page_count
                    
                except Exception as e:
                    print(f"Error reading page {i}: {e}")
                    continue
            
    def build_pyramid(self, image, levels=4):
        '''
        Generates an image pyramid with progressively lower resolutions.

        :param image: The original image to pyramid.
        :type image: np.ndarray
        :param levels: Number of pyramid levels to generate.
        :type levels: int
        :returns: List of images from highest to lowest resolution.
        :rtype: list of np.ndarray
        '''
        pyramid = [image]
        for _ in range(1, levels):
            image = cv2.pyrDown(image)
            pyramid.append(image)
        return pyramid

    def subsample_for_display(self, image: np.ndarray, max_dimension: int = 1024) -> np.ndarray:
        """Simple subsampling for display purposes."""
        h, w = image.shape[:2]
        
        # Calculate subsample factor
        scale_factor = min(max_dimension / h, max_dimension / w, 1.0)
        
        if scale_factor >= 1.0:
            return image  # No need to subsample
        
        # Calculate new dimensions
        new_h = int(h * scale_factor)
        new_w = int(w * scale_factor)
        
        # Simple subsampling by taking every nth pixel
        step_h = max(1, h // new_h)
        step_w = max(1, w // new_w)
        return image[::step_h, ::step_w]

    def filename_to_image(self, file_name: str, adjust_contrast=False, 
                                subsample_for_emit=True, max_display_size=1024) -> np.ndarray:
        """Memory-efficient version that processes TIFF pages one at a time with simple subsampling."""
        
        if file_name.endswith((".tiff", ".tif")):
            # Handle metadata
            metadata_widget = MetaData()
            # metadata = metadata_widget.parse_metadata(file_name)
            # self.fill_metadata.emit(metadata)
            
            print("Starting streaming page processing")
            emit_data = {}  # Data to emit (subsampled versions)
            channel_one_image = None
            
            # Process pages one at a time using generator
            for image, channel_num in self.read_tiff_pages(file_name):
                channel_name = f'Channel {channel_num}'
                image_adjusted = image

                if adjust_contrast:
                    __scaled = scale_adjust(image) 
                    image_adjusted = adjustContrast(__scaled)

                print(f"Processing {channel_name}, shape: {image.shape}, dtype: {image.dtype}")
                
                # Convert to uint8 for storage (memory savings)
                image_uint8 = to_uint8(image_adjusted)
                
                # Store full resolution version
                self.np_channels[channel_name] = ImageWrapper(image_uint8)
                self.reset_np_channels[channel_name] = ImageWrapper(image_uint8.copy())
                
                # Create subsampled version for emission/display
                if subsample_for_emit:
                    subsampled = self.subsample_for_display(image_uint8, max_display_size)
                    emit_data[channel_name] = subsampled
                    print(f"  Original: {image_uint8.shape}, Subsampled: {subsampled.shape}")
                else:
                    emit_data[channel_name] = image_uint8
                
                # Store first channel for return value
                if channel_num == 1:
                    channel_one_image = emit_data[channel_name]
                
                # Print memory info
                full_size_mb = image_uint8.nbytes / (1024*1024)
                if subsample_for_emit:
                    display_size_mb = emit_data[channel_name].nbytes / (1024*1024)
                    print(f'{channel_name} - Full: {full_size_mb:.2f} MB, Display: {display_size_mb:.2f} MB')
                else:
                    print(f'{channel_name} - Size: {full_size_mb:.2f} MB')
                
                # Update progress
                progress = 50 + int(channel_num * 40 / max(1, channel_num))
                self.updateProgress.emit(progress, f"Processing {channel_name}")
                
                # Force garbage collection after each channel
                gc.collect()

            self.is_layered = True
            
            print("Done processing, preparing emission")
            
            # Emit subsampled data for display
            if emit_data:
                if subsample_for_emit:
                    # Create wrappers for subsampled data
                    display_wrappers = {
                        name: ImageWrapper(data) for name, data in emit_data.items()
                    }
                    self.multi_layer.emit(display_wrappers, True)
                else:
                    # Emit full resolution
                    self.multi_layer.emit(self.np_channels, True)
                
            if self.np_channels:
                first_channel = next(iter(self.np_channels.values()))
                print("Final stored dtype:", first_channel.data.dtype)

        else:  # not a .tif image
            self.np_channels.clear()
            self.reset_np_channels.clear()
            self.is_layered = False
            from PIL import Image
            print("Processing single image")
            
            channel_one_image = np.array(Image.open(file_name))
            channel_name = f'Channel 1'
            
            # Convert to uint8 for consistency
            channel_one_image = to_uint8(channel_one_image)
            
            # Store full resolution
            self.image_wrapper = ImageWrapper(channel_one_image)
            self.np_channels[channel_name] = self.image_wrapper
            
            # Emit subsampled version for display if image is large
            if subsample_for_emit and channel_one_image.size > max_display_size * max_display_size:
                subsampled = self.subsample_for_display(channel_one_image, max_display_size)
                self.single_layer.emit(subsampled)
                channel_one_image = subsampled
                print(f"Single image subsampled from {self.image_wrapper.data.shape} to {subsampled.shape}")
            else:
                self.single_layer.emit(channel_one_image)
                
        self.image_cache.clear()
        self.lut_cache.clear()
        self.updateProgress.emit(100, "Image Loaded")
        
        print("Emitting to update manager")
        self.update_manager.emit(self.np_channels, file_name)
        self.image_count += 1

        return channel_one_image
    
    def deleteImage(self):
        self.scene.scene().clear()


class ReferenceGraphicsView(__BaseGraphicsView):
    

    update_reference = pyqtSignal(QPixmap, bool)
    # referenceLoaded = pyqtSignal(dict)
    def __init__(self, parent=None):
        super().__init__(parent)
        self.is_layered = False

    def dropEvent(self, event: QDropEvent):
        if event.mimeData().hasUrls():
            for url in event.mimeData().urls():
                file_path = url.toLocalFile()
                if file_path:
                    self.addImage(file_path)
            event.acceptProposedAction()
    

    def addImage(self, file_path:str):

        self.reference_worker = Worker(self.filename_to_image, file_path, False)
        self.reference_worker.start()
        self.reference_worker.signal.connect(self.filename_to_image_complete)
        self.reference_worker.finished.connect(self.reference_worker.quit)
        self.reference_worker.finished.connect(self.reference_worker.deleteLater)

    def filename_to_image_complete(self, image):
        if not self.np_channels.get("Channel 1") == None:
            self.is_layered = True

        else:
            self.is_layered = False

        qimage = numpy_to_qimage(image)
        self.pixmap = QPixmap(qimage)

        self.update_reference.emit(self.pixmap, self.is_layered)

##########################################################
class ImageGraphicsView(__BaseGraphicsView):
    
    canvasUpdated = pyqtSignal(QPixmap)
    newImageAdded = pyqtSignal(QGraphicsPixmapItem)
    saveImage = pyqtSignal(QGraphicsPixmapItem)  
    changeSlider = pyqtSignal(tuple)

    def __init__(self, parent=None):

        super().__init__(parent)
        self.reset_pixmap=None
        self.reset_pixmapItem=None
        self.pixmap=None
        self.pixmapItem=None
        self.begin_crop = False
        self.crop_cursor =  QCursor(Qt.CursorShape.CrossCursor)
        self.stardist_image_count = 0
        self.memory_cache = MemoryEfficientImageCache(max_cache_size_mb=300)

    def toPixmapItem(self, data:QPixmap|np.ndarray|QImage):
        '''Sends a pixmap to the canvas for display'''
        #convert pixmap to pixmapItem
        if type(data) == QPixmap:

            self.pixmap = data
        elif type(data) == QImage:
            self.pixmap = QPixmap(data)
        else:
            self.pixmap = QPixmap(numpy_to_qimage(data))
    
        self.canvasUpdated.emit(self.pixmap)
    
    update_cmap = pyqtSignal(str)

    def swap_channel(self, index):
        '''swaps between channels of a multi-layered tiff image'''
        
        if self.is_layered:

            self.currentChannelNum = index
            channel_num = f'Channel {index+1}'
            if hasattr(self, 'memory_cache'):
                self.memory_cache.clear_channel(f"Channel {self.currentChannelNum + 1}")
            
            self.image_wrapper = self.np_channels.get(channel_num)# self.image always needs to be updated. this is the current image that is being operated on

            if channel_num in self.np_channels.keys():
                self.update_image(cmap_text = self.image_wrapper.cmap) # this also updates the contrast

        
    
    # def _apply_contrast_and_cache(self, channel_num, cache_key, contrast_min, contrast_max):
    #     """
    #     This helper function applies the contrast and caches the processed image.
    #     If the contrast has already been applied, it uses the cached version.
    #     """

    #     if (channel_num is None and self.image_cache.get(cache_key) is None) or \
    #     (channel_num is not None and self.image_cache[channel_num].get(cache_key) is None):

    #         image_to_display = self.apply_contrast(contrast_min, contrast_max)
            
    #         if channel_num:

    #             self.image_cache[channel_num][cache_key] = image_to_display

    #         else:

    #             self.image_cache[cache_key] = image_to_display

    #         contrast_pixmap = QPixmap(numpy_to_qimage(image_to_display))  # Convert to pixmap for display

    #         self.canvasUpdated.emit(contrast_pixmap)

    #     else:
    #         # use cached image if available

    #         image_to_display = self.image_cache[channel_num][cache_key] if channel_num else self.image_cache[cache_key]
            
    #         contrast_pixmap = QPixmap(numpy_to_qimage(image_to_display))
    #         self.canvasUpdated.emit(contrast_pixmap)
    # Updated contrast update methods for your ImageGraphicsView class
    def update_contrast_memory_efficient(self, values):
        """Memory-efficient version of update_contrast method."""
        if self.image_wrapper is None:
            self.errorSignal.emit("Canvas is empty")
            return

        contrast_min, contrast_max = int(values[0]), int(values[1])
        self.image_wrapper.contrast_min = contrast_min
        self.image_wrapper.contrast_max = contrast_max

        # Initialize memory-efficient cache if not exists
        if not hasattr(self, 'memory_cache'):
            self.memory_cache = MemoryEfficientImageCache()

        contrast_key = (contrast_min, contrast_max)
        cmap_key = self.image_wrapper.cmap
        cache_key = (cmap_key, contrast_key)

        if self.is_layered:
            print("Processing layered image with memory management")
            channel_num = f"Channel {self.currentChannelNum + 1}"
            self.image_wrapper = self.np_channels[channel_num]
            
            # Check cache first
            cached_image = self.memory_cache.get(channel_num, cache_key)
            if cached_image is not None:
                print(f"Using cached image for {channel_num}")
                contrast_pixmap = QPixmap(numpy_to_qimage(cached_image))
                self.canvasUpdated.emit(contrast_pixmap)
            else:
                print(f"Processing new contrast for {channel_num}")
                self._apply_contrast_memory_efficient(channel_num, cache_key, contrast_min, contrast_max)
        else:
            # Single layer processing
            cached_image = self.memory_cache.get("single", cache_key)
            if cached_image is not None:
                contrast_pixmap = QPixmap(numpy_to_qimage(cached_image))
                self.canvasUpdated.emit(contrast_pixmap)
            else:
                self._apply_contrast_memory_efficient("single", cache_key, contrast_min, contrast_max)

        # Update slider
        self.changeSlider.emit((self.image_wrapper.contrast_min, self.image_wrapper.contrast_max))
        
    def _apply_contrast_memory_efficient(self, channel_key, cache_key, contrast_min, contrast_max):
        """Memory-efficient contrast application with caching."""
        try:
            # Apply contrast
            image_to_display = self.apply_contrast(contrast_min, contrast_max)
            
            # Cache the result
            self.memory_cache.put(channel_key, cache_key, image_to_display.copy())
            
            # Display
            contrast_pixmap = QPixmap(numpy_to_qimage(image_to_display))
            self.canvasUpdated.emit(contrast_pixmap)
            
            # Clean up local reference
            del image_to_display
            
        except MemoryError:
            print("Memory error - clearing cache and retrying")
            self.memory_cache.clear_all()
            gc.collect()
            
            # Retry with no caching
            image_to_display = self.apply_contrast(contrast_min, contrast_max)
            contrast_pixmap = QPixmap(numpy_to_qimage(image_to_display))
            self.canvasUpdated.emit(contrast_pixmap)
            del image_to_display
    def change_cmap(self, cmap_text="default"):
        '''changes the colormap given a colormap str valid in matplotlib'''

        if self.image_wrapper is None:
            self.errorSignal.emit("Canvas is empty")
            return

        if self.is_layered:
            channel_num = f"Channel {self.currentChannelNum + 1}" 
            self.image_wrapper = self.np_channels[channel_num]  # wrapper


        if cmap_text == "default":
            cmap_text = self.image_wrapper.cmap
        
        if cmap_text not in self.lut_cache:
            lut = self.generate_lut(cmap_text)
            self.lut_cache[cmap_text] = lut # cache to avoid recalculating LUT
        else:
            lut = self.lut_cache[cmap_text]  # Reuse the cached LUT
        
        self.image_wrapper.cmap = cmap_text


        __copy = (self.image_wrapper.data).copy()

        return self.label2rgb(scale_adjust(__copy), lut).astype(np.uint8)

    def update_image(self, cmap_text="default"):
        '''Updates the current image using the current colormap and contrast settings.
        This only changes the display and does not change the underlying data.'''
        
        #update the color map
        image_to_display = self.change_cmap(cmap_text)
        self.toPixmapItem(image_to_display)  
        self.update_cmap.emit(cmap_text)
        # update the contrast
        min, max = self.image_wrapper.contrast_min, self.image_wrapper.contrast_max # read contrast settings
        self.update_contrast((min,max))

    def generate_lut(self, cmap:str):
        '''generate a 8 bit look-up table and converts to rgb space'''
        label_range = np.linspace(0, 1, 256)
        return np.uint8(mpl.colormaps[cmap](label_range)[:,2::-1]*256).reshape(256, 1, 3)
    def update_contrast(self, values):
        return self.update_contrast_memory_efficient(values)
    def _apply_contrast_and_cache(self, channel_num, cache_key, contrast_min, contrast_max):
        return self._apply_contrast_memory_efficient(channel_num, cache_key, contrast_min, contrast_max)
    def label2rgb(self, labels, lut):
        '''applys the look-up table and merges r, g, b channels to form colored image '''
        print(type(labels))
        if len(labels.shape) == 3 and labels.shape[2] == 3:
            r, g, b = cv2.split(labels)
            return cv2.LUT(cv2.merge((r, g, b)), lut)
        else:
            # Ensure labels is 2D before merging
            if len(labels.shape) > 2:
                labels = labels[:,:,0]  # Take first channel if multi-channel
            return cv2.LUT(cv2.merge((labels, labels, labels)), lut) # gray to color
    

    def loadStardistLabels(self, stardist: ImageWrapper):

        self.is_layered = False
        self.stardist_labels = stardist.data
        cmap = self.np_channels[f"Channel {self.currentChannelNum+1}"].cmap
        self.image_wrapper = ImageWrapper(self.stardist_labels.copy(), name="stardist_label", cmap=cmap)
        self.update_image(cmap_text=cmap)
        # !TODO: need to fix; think is broken now after changing update_manger behavior
        self.update_manager.emit(cmap, self.image_wrapper.name + " " + str(self.stardist_image_count))
        self.stardist_image_count+=1
    
    def addImage(self, file:str):
        '''add a new image'''
        self.np_channels.clear()
        if hasattr(self, 'memory_cache'):
            self.memory_cache.clear_all()
        self.scene.resetTransform()

        if isinstance(file, str):

           self.image_worker = Worker(self.filename_to_image, file)
           self.image_worker.signal.connect(self.onFileNameToPixmapCompleted)
           self.image_worker.error.connect(self.onError)
           self.image_worker.finished.connect(self.image_worker.quit)
           self.image_worker.start()


    @pyqtSlot(object)
    def onFileNameToPixmapCompleted(self, image):
        '''handles operation after the file is loaded into the canvas'''
        if image.dtype != np.uint8:
            image = scale_adjust(image)

        qimage = numpy_to_qimage(image)
        pixmap = QPixmap(qimage)
        self.reset_pixmap=pixmap
        self.reset_pixmapItem = QGraphicsPixmapItem(pixmap)
        self.pixmap = pixmap
        self.pixmapItem = QGraphicsPixmapItem(pixmap)
        self.newImageAdded.emit(self.pixmapItem) # emit uint16, change to uint8 in canvas_ui

    
    def pixmap_to_numpy_array(self):
        # Convert the QPixmap to a QImage
        image = self.pixmap.toImage()

        # Get image dimensions
        width = image.width()
        height = image.height()

        # Convert QImage to byte array
        ptr = image.bits()
        ptr.setsize(height * width * 4)

        # Convert byte array to numpy array
        array = np.array(ptr).reshape((height, width, 4))

        return array

    def reset_image(self):
        '''resets the image to original state'''
        if self.pixmapItem: 

            self.np_channels = copy.deepcopy(self.reset_np_channels)
            self.multi_layer.emit(self.np_channels, False)

            channel_num = f"Channel {self.currentChannelNum + 1}"
            self.image_wrapper = self.np_channels.get(channel_num)

            self.image_cache.clear()
            self.update_image("gray")

    def rotate_image_task(self, channels:dict, angle):
        t = time.time()
        rotated_arrays = []

        if self.is_layered:
            for channel_num, wrapper in channels.items():
                try:
                    arr = wrapper.data
                    cmap = wrapper.cmap
                    if not arr.data.contiguous:
                        arr = np.ascontiguousarray(arr, dtype='uint16')
                except Exception as e:
                    print("error: ", str(e)) # should be a QMessageBox

                # rotate image
                h,w = arr.shape
                center = (w/2, h/2)
                rotation_matrix = cv2.getRotationMatrix2D(center, -angle, 1)
                cos = np.abs(rotation_matrix[0,0])
                sin = np.abs(rotation_matrix[0,1])
                updated_w = int((h*sin) + (w*cos))
                updated_h = int((h*cos) + (w*sin))
                rotation_matrix[0,2] += (updated_w/2) - w/2
                rotation_matrix[1,2] += (updated_h/2) - h/2
                rotated_arr = cv2.warpAffine(arr, rotation_matrix, (updated_h, updated_h))
                
                self.np_channels[channel_num].data = rotated_arr
                # self.rotated_wrapper = ImageWrapper(rotated_arr, cmap = cmap)
                # rotated_arrays.append(self.rotated_wrapper)
            return self.np_channels
        else:
            h,w = self.image_wrapper.data.shape
            center = (w/2, h/2)
            rotation_matrix = cv2.getRotationMatrix2D(center, -angle, 1)
            cos = np.abs(rotation_matrix[0,0])
            sin = np.abs(rotation_matrix[0,1])
            updated_w = int((h*sin) + (w*cos))
            updated_h = int((h*cos) + (w*sin))
            rotation_matrix[0,2] += (updated_w/2) - w/2
            rotation_matrix[1,2] += (updated_h/2) - h/2
            rotated_arr = cv2.warpAffine(self.image_wrapper.data, rotation_matrix, (updated_h, updated_h))

            return rotated_arr

    def rotateImage(self, angle_text: str):
        try:
            angle = float(angle_text)
        except ValueError:
            print("Error: Please enter a valid number.") # this should be a QMessageBox
            return

        if self.pixmap and angle is not None:
            self.rotation_worker = Worker(self.rotate_image_task, self.np_channels, angle)
            self.rotation_worker.signal.connect(self.onRotationCompleted) # result is rotated_channels
            self.rotation_worker.error.connect(self.onError)
            self.rotation_worker.finished.connect(self.rotation_worker.quit)
            self.rotation_worker.finished.connect(self.rotation_worker.deleteLater)
            self.rotation_worker.start()

    @pyqtSlot(object)
    def onRotationCompleted(self, result):

        if type(result) == dict:
            print("completing rotation")
            self.np_channels = result

            # channel_image = list(self.np_channels.values())[self.currentChannelNum].data
            channel_cmap = list(self.np_channels.values())[self.currentChannelNum].cmap
            # channel_image = scale_adjust(channel_image)
            # self.image = channel_image

            self.image_cache.clear()
            self.update_image(cmap_text=channel_cmap) # this also updates the contrast
            self.multi_layer.emit(self.np_channels, False)

        else:
            # channel_image = scale_adjust(result)
            # self.image = channel_image
            self.update_image(cmap_text="gray") # this also updates the contrast


    @pyqtSlot(str)
    def onError(self, error_message):
        print(f"Error: {error_message}")

    def updateChannels(self, channels:dict, clear:bool) -> None: #cropsignal will update this
        self.np_channels = channels # replace channels with new, cropped/rotated, etc
        self.multi_layer.emit(self.np_channels, clear)


    def updateCurrentImage(self, data_dict):
        self.image = data_dict[f"Channel {self.currentChannelNum + 1}"].data


    def auto_contrast(self, lower = 0.1, upper=.9):
        if self.is_layered:
            channel_num = f"Channel {self.currentChannelNum + 1}"
            channel = scale_adjust(self.np_channels[channel_num].data)
        else:
            channel = scale_adjust(self.image_wrapper.data)

        flat_channel = channel.flatten()
    
        hist, _ = np.histogram(flat_channel, bins=256, range=(0, 255))
        total_pixels = flat_channel.size
        cumulative_hist = np.cumsum(hist) / total_pixels
        new_min= np.argmax(cumulative_hist > lower)  
        new_max = np.argmax(cumulative_hist > upper)  

        self.update_contrast((new_min, new_max))

    def apply_contrast(self, new_min, new_max):

        qimage = self.pixmap.toImage() # get current image
        image = qimage_to_numpy(qimage) # returns uint8
        lut = self.create_lut(new_min, new_max)
        return cv2.LUT(image, lut)

    def create_lut(self, new_min, new_max):

        lut = np.zeros(256, dtype=np.uint8) # uint8 for display
        lut[new_min:new_max+1] = np.linspace(start=0, stop=255, num=(new_max - new_min + 1), endpoint=True, dtype=np.uint8)
        lut[:new_min] = 0 # clip between 0 and 255
        lut[new_max+1:] = 255

        return lut
    

    def blur_layer(self, blur_percentage:float, confirm=False):
        '''start gaussian blur in a separate thread'''
        self.blur_worker = Worker(self.blur_layer_task, blur_percentage, confirm)
        # self.blur_worker.signal.connect() # result is rotated_channels
        self.blur_worker.error.connect(self.onError)
        self.blur_worker.finished.connect(self.blur_worker.quit)
        self.blur_worker.finished.connect(self.blur_worker.deleteLater)
        self.blur_worker.start()

    def blur_layer_task(self, blur_percentage:float, confirm=False):
        """
        Applies Gaussian blur chosen of the image stack and subtracts
        the specified percentage of the blurred image from the original.
        """

        self.image_cache.clear()
        self._blur_layer = f'Channel {self.currentChannelNum+ 1}'
        if not confirm:
            
            # blur_percentage = self._blur_percentage
            layer_to_blur = (self.np_channels[self._blur_layer].data).copy()
            blurred_mask = cv2.GaussianBlur(layer_to_blur, (101, 101), 0)
            blurred_mask_adjusted = (blurred_mask * blur_percentage).astype(np.uint16)
            self.corrected_layer = cv2.subtract(layer_to_blur, blurred_mask_adjusted)
            self.corrected_layer = np.clip(self.corrected_layer, 0, 65535).astype(np.uint16)
            # cmap = self.np_channels[self._blur_layer].cmap
            self.toPixmapItem(self.corrected_layer)
            print("blurring")
            # self.update_image(cmap)

        else:
            print("Error from gaussian blur")


        if confirm and hasattr(self, "corrected_layer") and (not self.np_channels.get(self._blur_layer) == None):
            self.np_channels[self._blur_layer].data = self.corrected_layer # Replace with the corrected version
            cmap = self.np_channels[self._blur_layer].cmap
            self.update_image(cmap)
            self.multi_layer.emit(self.np_channels, False)
            self.updateProgress.emit(100, f"Replaced {self._blur_layer}")


    def crop(self, image_rect):
        """start crop thread"""

        print("starting crop")
        left = image_rect.x()
        top = image_rect.y()
        right = image_rect.right()
        bottom = image_rect.bottom()

        cropped_array = self.image_wrapper.data[top:bottom+1, left:right+1] # this is the current image. if layered then its the current channel
        cropped_array_copy = cropped_array.copy()

        if not cropped_array.data.contiguous:
            cropped_array = np.ascontiguousarray(cropped_array, dtype=cropped_array.dtype)

        cropped_array_copy = cropped_array.copy()

        contrast = (self.image_wrapper.contrast_min, self.image_wrapper.contrast_max)
            
        self.crop_dialog = ImageDialog(self, cropped_array_copy, contrast, self.image_wrapper.cmap)
        self.crop_dialog.exec()

        if not self.is_layered:
            self.image_wrapper.data = cropped_array_copy
            self.cropSignal.emit(False) #set crop status
            return # return if there are no other layers or is not a tiff

        if self.crop_dialog.confirm_crop:
            self.crop_worker = Worker(self.cropImageTask, image_rect)
            self.crop_worker.signal.connect(self.onCropCompleted) 
            self.crop_worker.finished.connect(self.crop_worker.quit)
            self.crop_worker.finished.connect(self.crop_worker.deleteLater)
            self.crop_worker.start()
        else:
            self.cropSignal.emit(False)

    cropSignal = pyqtSignal(bool)
    @pyqtSlot(dict)
    def onCropCompleted(self, cropped_wrappers: dict):
        """Handle completed crop operation"""

        if not cropped_wrappers == {}:
            self.np_channels = cropped_wrappers
            channel_num = f"Channel {self.currentChannelNum + 1}"
            self.image_wrapper = self.np_channels.get(channel_num)
            self.image_cache.clear()

        self.cropSignal.emit(False)

    def cropImageTask(self, image_rect) -> dict:
        """Process crop in background thread"""
        left = image_rect.x()
        top = image_rect.y()
        right = image_rect.right()
        bottom = image_rect.bottom()

        if self.is_layered:
            for channel_name, image_arr in self.np_channels.items(): # iterate over wrappers
                arr = image_arr.data 
                cropped_array = arr[top:bottom+1, left:right+1]
                if not cropped_array.data.contiguous:
                    cropped_array = np.ascontiguousarray(cropped_array, dtype=arr.dtype)

                if not hasattr(self, "cropped_wrappers"):
                    self.cropped_wrappers = {}

                self.np_channels[channel_name].data = cropped_array

            return self.np_channels
            
        else: return {}
    
    def flip_horizontal(self):
        """Flip the image horizontally"""
        if self.is_layered:
            for channel_name, wrapper in self.np_channels.items():
                wrapper.data = cv2.flip(wrapper.data, 1)  # 1 for horizontal flip
            self.image_cache.clear()  # Clear cache to force redraw
            self.update_image()
        else:
            self.image_wrapper.data = cv2.flip(self.image_wrapper.data, 1)
            self.image_cache.clear()  # Clear cache to force redraw
            self.update_image()
        print("Image flipped horizontally")

    def flip_vertical(self):
        """Flip the image vertically"""
        if self.is_layered:
            for channel_name, wrapper in self.np_channels.items():
                wrapper.data = cv2.flip(wrapper.data, 0)  # 0 for vertical flip
            self.image_cache.clear()  # Clear cache to force redraw
            self.update_image()
        else:
            self.image_wrapper.data = cv2.flip(self.image_wrapper.data, 0)
            self.image_cache.clear()  # Clear cache to force redraw
            self.update_image()
        print("Image flipped vertically")

class MetaData(QWidget):
    '''Class to handle metadata of images'''
    def __init__(self, parent=None):
        super().__init__(parent)
        self.row_count = 8
        self.column_count = 1
        self.table = QTableWidget()
        self.table.setRowCount(8)
        self.table.setColumnCount(1)

        self.headers = ['Name', 'URI', 'Pixel Type', 'Width', 
                                 'Height', 'Dimension (CZT)', 'PhysicalSizeX', 'PhysicalSizeY']

        self.table.setVerticalHeaderLabels(self.headers)

        self.table.setHorizontalHeaderLabels(['Value'])
        self.table.setColumnWidth(0, 300)

        editable_fields = [3, 7, 8]
        # for row in range(self.row_count):
            
            # if row in editable_fields:
            #     value_item.setFlags(Qt.ItemIsSelectable | Qt.ItemIsEnabled)  # Make it read-only

        layout = QVBoxLayout(self)
        layout.addWidget(self.table)
        self.setLayout(layout)

    def set_item(self, row, value):
        item = QTableWidgetItem(value)
        self.table.setItem(row, 0, item)

    def populate_table(self, metadata: dict):
        self.table.setRowCount(len(self.headers))

        for row, key in enumerate(self.headers):
            value = metadata.get(key, "Unknown")  
            value_item = QTableWidgetItem(str(value))

            # key_item.setFlags(key_item.flags() ^ Qt.ItemFlag.ItemIsEditable)
            self.table.setItem(row, 0, value_item)

    def parse_metadata(self, filename):
        file_name = os.path.basename(filename)  
        name = os.path.splitext(file_name)[0]    

        with tiff.TiffFile(filename) as tif:
            raw_meta_data = {}
            page = tif.pages[0]
            for tag in page.tags.values():
                raw_meta_data[tag.name] = tag.value
                
        try:
            desc = raw_meta_data["ImageDescription"]
            metadata = {}
            root = ET.fromstring(desc)
            namespace_uri = root.tag[root.tag.find('{')+1 : root.tag.find('}')]
            ns = {'ome': namespace_uri}
            pixels = root.find('.//ome:Pixels', namespaces=ns)

            if pixels is not None:

                metadata = {
                    'Name': name,
                    'URI': filename,
                    'Width': pixels.attrib.get('SizeX'),
                    'Height': pixels.attrib.get('SizeY'),
                    'Dimension (CZT)': f"{pixels.attrib.get('SizeC')} x {pixels.attrib.get('SizeZ')} x {pixels.attrib.get('SizeT')}",
                    'Pixel Type': pixels.attrib.get('Type'),
                    'PhysicalSizeX': f"{pixels.attrib.get('PhysicalSizeX')} {pixels.attrib.get('PhysicalSizeXUnit')}",
                    'PhysicalSizeY': f"{pixels.attrib.get('PhysicalSizeY')} {pixels.attrib.get('PhysicalSizeYUnit')}",
                    'DimensionOrder': pixels.attrib.get('DimensionOrder')
                }

                for k, v in metadata.items():
                    print(f"{k}: {v}")
            else:
                print("Pixels element not found.")
                
        except ET.ParseError as e:
            print("Parse error has occurred:", e)

        finally:

            if not metadata:
                metadata["Name"] = name
                metadata["URI"] = filename
                metadata["Width"] = raw_meta_data["ImageWidth"]
                metadata["Height"] = raw_meta_data["ImageLength"]
                metadata['Pixel Type'] = f'uint{raw_meta_data["BitsPerSample"]}'
                metadata['Dimension (CZT)'] = "Unknown"
                metadata['PhysicalSizeX'] = "Unknown"
                metadata['PhysicalSizeY'] = "Unknown"
                metadata['DimensionOrder'] = "Unknown"
    
        return metadata
    


# class ContrastWorkerSignals(QObject):
#     finished = pyqtSignal(np.ndarray)

# class ContrastWorker(QRunnable):

#     def __init__(self, image, contrast_min, contrast_max):
#         super().__init__()
#         self.image = image
#         self.contrast_min = contrast_min
#         self.contrast_max = contrast_max
#         self.signals = ContrastWorkerSignals()

#     def run(self):
#         pass
    

#     def update(self):
#         '''displays the image between a lower and upper limit'''
#         if self.image_wrapper is None:
#             self.errorSignal.emit("Canvas is empty")
#             return

#         contrast_min, contrast_max = int(values[0]), int(values[1])
#         self.image_wrapper.contrast_min = contrast_min
#         self.image_wrapper.contrast_max = contrast_max  # Save contrast settings
        
#         contrast_key = (contrast_min, contrast_max)
#         cmap_key = self.image_wrapper.cmap
#         cache_key = (cmap_key, contrast_key)

#         # Check if the current image (single-layer or multi-layer) is in the cache
#         if self.is_layered: 
#             print("Checking layered")
#             channel_num = f"Channel {self.currentChannelNum + 1}"

#             # Initialize cache for this channel if it's not already initialized
#             if channel_num not in self.image_cache:
#                 self.image_cache[channel_num] = {}


#             self.image_wrapper = self.np_channels[channel_num]  # Set the current channel wrapper


#             self._apply_contrast_and_cache(channel_num, cache_key, contrast_min, contrast_max)

#         else:  # Single-layer logic
#             self._apply_contrast_and_cache(None, cache_key, contrast_min, contrast_max)

#         # Update the contrast slider with the new settings
#         self.changeSlider.emit((self.image_wrapper.contrast_min, self.image_wrapper.contrast_max))  # Update the slider

#     def apply_contrast(self, new_min, new_max):

#         qimage = self.pixmap.toImage() # get current image
#         image = qimage_to_numpy(qimage) # returns uint8
#         lut = self.create_lut(new_min, new_max)
#         return cv2.LUT(image, lut)

#     def create_lut(self, new_min, new_max):

#         lut = np.zeros(256, dtype=np.uint8) # uint8 for display
#         lut[new_min:new_max+1] = np.linspace(start=0, stop=255, num=(new_max - new_min + 1), endpoint=True, dtype=np.uint8)
#         lut[:new_min] = 0 # clip between 0 and 255
#         lut[new_max+1:] = 255

#         return lut