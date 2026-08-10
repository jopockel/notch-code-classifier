import os
import cv2
import numpy as np

class LegacyPipeline:
    def __init__(self):
        # The ML pipeline initializes models here. 
        pass

    @staticmethod
    def _get_perimeter_histogram(img, band_width):
        """Use only the outer N pixels of an image for histogram analysis."""
        h, w = img.shape
        perimeter_mask = np.ones((h, w), dtype=np.uint8) * 255
        perimeter_mask[band_width:-band_width, band_width:-band_width] = 0
        hist = cv2.calcHist([img], [0], perimeter_mask, [256], [0, 256])
        return hist

    @staticmethod
    def _analyze_perimeter_properties(hist, dark_threshold, ratio_cutoff):
        """Classifies if a histogram represents a digital black border."""
        total_pixels = np.sum(hist)
        if total_pixels == 0: 
            return False, 0.0
        
        dark_count = np.sum(hist[:dark_threshold])
        dark_ratio = dark_count / total_pixels
        has_border = dark_ratio > ratio_cutoff
        return has_border

    @staticmethod
    def _find_peak_decay_threshold(hist, peak_val, safety_margin=10):
        """
        Finds the gray value where the histogram count drops below a fraction of the peak.
        """
        peak_height = hist[peak_val]
        target = peak_height * 0.6

        sigma_idx = peak_val
        for i in range(peak_val + 1, len(hist)):
            if hist[i] < target:
                sigma_idx = i
                break

        sigma = sigma_idx - peak_val
        final_threshold = peak_val + 3 * sigma + safety_margin
        
        return int(final_threshold)

    @staticmethod
    def _thresholding(img, threshold):
        """
        Make a binary mask of the film area using a strict global threshold.
        """
        _, mask = cv2.threshold(img, threshold, 255, cv2.THRESH_BINARY)
        return mask
    
    @staticmethod
    def _adaptive_thresholding(img, block_size=21, C=10, use_gaussian=False):
        """Make a binary mask of the film area using adaptive thresholding."""
        blurred = cv2.GaussianBlur(img, (7, 7), 0)
        
        if use_gaussian:
            adaptive_method = cv2.ADAPTIVE_THRESH_GAUSSIAN_C
        else:
            adaptive_method = cv2.ADAPTIVE_THRESH_MEAN_C

        mask = cv2.adaptiveThreshold(
            blurred, 255, adaptive_method, cv2.THRESH_BINARY_INV, block_size, C
        )
        
        return mask

    @staticmethod
    def _hull_concavity(binary_mask, band_width=150):
        """
        Identifies notches by finding concavities (missing film) using the global convex hull.
        """
        contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return np.zeros_like(binary_mask), np.zeros_like(binary_mask)
        
        all_points = np.vstack(contours)
        global_hull = cv2.convexHull(all_points)

        hull_mask = np.zeros_like(binary_mask)
        cv2.drawContours(hull_mask, [global_hull], -1, 255, thickness=cv2.FILLED)

        all_concavities = cv2.subtract(hull_mask, binary_mask)
        
        h, w = binary_mask.shape
        perimeter_mask = np.ones((h, w), dtype=np.uint8)*255
        perimeter_mask[band_width:-band_width, band_width:-band_width] = 0
        notches = cv2.bitwise_and(all_concavities, perimeter_mask)

        return notches, hull_mask

    @staticmethod
    def _filter_notches_by_geometry(concavity_mask, min_area=300, max_area=1500):
        """
        Filters noise and dust out of the concavity mask using aspect ratios, solidity, and hierarchy.
        """
        contours, hierarchy = cv2.findContours(concavity_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        final_notch_mask = np.zeros_like(concavity_mask)
        
        if hierarchy is None:
            return final_notch_mask

        for i, c in enumerate(contours):
            child_idx = hierarchy[0][i][2]
            
            if child_idx != -1:
                child_area = cv2.contourArea(contours[child_idx])
                if child_area > 20: 
                    continue
            
            area = cv2.contourArea(c)
            if area < min_area or area > max_area:
                continue
             
            x, y, w, h = cv2.boundingRect(c)
            aspect_ratio = float(w) / h
            
            if aspect_ratio > 5.0 or aspect_ratio < 0.2:
                continue
                
            hull = cv2.convexHull(c)
            hull_area = cv2.contourArea(hull)
            if hull_area == 0: continue
            solidity = float(area) / hull_area
            
            if solidity < 0.5:
                continue

            cv2.drawContours(final_notch_mask, [c], -1, 255, thickness=cv2.FILLED)
            
        return final_notch_mask

    @staticmethod
    def _classify_notch_shapes(notch_mask, min_area=50, epsilon_mult=0.04, box_padding=0):
        """Classify shape using OpenCV polygon approximation and format for the ML test notebook."""
        labels = []
        contours, _ = cv2.findContours(notch_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        img_h, img_w = notch_mask.shape[:2]

        for c in contours:
            area = cv2.contourArea(c)
            if area < min_area:
                continue

            perim = cv2.arcLength(c, True)
            if perim == 0:
                continue

            epsilon = epsilon_mult * perim
            approx = cv2.approxPolyDP(c, epsilon, True)
            n_vertices = len(approx)

            x, y, w, h = cv2.boundingRect(c)
            aspect = w / float(h) if h > 0 else 1.0
            circularity = 4 * np.pi * area / (perim ** 2)

            if n_vertices >= 8 and circularity > 0.6:
                shape = "circle"
            elif n_vertices == 4:
                shape = "rectangle"
            elif n_vertices == 3:
                shape = "triangle"
            else:
                shape = "semicircle"

            x1 = max(0, x - box_padding)
            y1 = max(0, y - box_padding)
            x2 = min(img_w, x + w + box_padding)
            y2 = min(img_h, y + h + box_padding)

            labels.append({
                'coords': [x1, y1, x2, y2],           # Bounding box [x1, y1, x2, y2]
                'shape': shape,
                'shape_confidence': 1.0,              # Legacy rules are binary/absolute
                'needs_human_review': False,          # Legacy doesn't calculate uncertainty
                'svm_notch_prob': 1.0                 # Dummy value so grouping logic doesn't break
            })

        return labels

    def process_image(self, 
        img_path, 
        perimeter_band_width=15, 
        notch_band_width=125, 
        dark_threshold=40, 
        border_ratio=0.84, 
        use_adaptive_threshold=False,
        adaptive_block_size=37,
        adaptive_c=8,
        adaptive_use_gaussian=True,
        peak_safety_margin=1,
        filter_min_area=50,
        filter_max_area=4000,
        shape_min_area=50,
        shape_epsilon_mult=0.04,
        box_padding=16):
        """
        The main pipeline flow mimicking the ML process_image method:
        1. Check for border.
        2. Threshold the image (using Adaptive or Peak Decay).
        3. Extract concavities using Global Convex Hull.
        4. Filter by geometry and classify shapes.
        """
        img = cv2.imread(img_path)
        if img is None:
            return None
        
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
        
        # 1. Check for border
        hist = self._get_perimeter_histogram(gray, perimeter_band_width)
        has_border = self._analyze_perimeter_properties(hist, dark_threshold, border_ratio)
        
        if not has_border:
            return {
                "border_present": False,
                "accepted_notches": [],
                "rejected_notches": []
            }
            
        # 2. Thresholding Method Selection
        if use_adaptive_threshold:
            mask = self._adaptive_thresholding(
                gray, 
                block_size=adaptive_block_size, 
                C=adaptive_c, 
                use_gaussian=adaptive_use_gaussian
            )
        else:
            peak_val = int(np.argmax(hist))
            thresh_val = self._find_peak_decay_threshold(hist, peak_val, safety_margin=peak_safety_margin)
            mask = self._thresholding(gray, thresh_val)

        # 3. Concavity Extraction
        raw_notches_mask, _ = self._hull_concavity(mask, band_width=notch_band_width)

        # 4. Filter and Classify
        filtered_notch_mask = self._filter_notches_by_geometry(
            raw_notches_mask, 
            min_area=filter_min_area, 
            max_area=filter_max_area
        )
        
        accepted_notches = self._classify_notch_shapes(
            filtered_notch_mask, 
            min_area=shape_min_area, 
            epsilon_mult=shape_epsilon_mult,
            box_padding=box_padding
        )
        
        return {
            "border_present": True,
            "accepted_notches": accepted_notches,
            "rejected_notches": []  
        }