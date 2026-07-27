import numpy as np
import cv2

class Pipeline:
    def __init__(self):
        pass

    @staticmethod
    def get_perimeter_histogram(img, band_width):
        """
        Use only the outer N pixels of an image for histogram analysis.

        img: input image (grayscale)
        band_width: width of the outer band to consider (in pixels)
        """
        h, w = img.shape
        # Create a mask for the perimeter
        perimeter_mask = np.ones((h, w), dtype=np.uint8)*255
        perimeter_mask[band_width:-band_width, band_width:-band_width] = 0
        
        # Calculate histogram only for the perimeter pixels
        hist = cv2.calcHist([img], [0], perimeter_mask, [256], [0, 256])
        
        return hist

    @staticmethod
    def border_classification(hist, threshold, ratio):
        """
        Classify if the histogram indicates a border being present in an image.

        hist: histogram of the imageborder
        threshold: the gray value (0-255) considerd 'black'
        ratio: the ratio of black pixels to total pixels in the perimeter to consider it a border
        """

        total_pixels = np.sum(hist)
        if total_pixels > 0 and (np.sum(hist[:int(threshold)]) / total_pixels) > ratio:
            return True
        else:
            return False
        