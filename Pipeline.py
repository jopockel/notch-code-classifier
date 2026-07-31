import numpy as np
import cv2
import torch
import os
import joblib
import requests
from PIL import Image
from torchvision import transforms
from ultralytics import YOLO

class Pipeline:
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

    @staticmethod
    def _find_largest_true_segment(bool_array):
        """
        Internal helper: Finds the start and end indices of the longest 
        contiguous sequence of True values in a 1D boolean array.
        """
        # Pad with False at both ends so we always detect transitions
        padded = np.concatenate(([False], bool_array, [False]))
        
        # np.diff finds where the array changes. 1 means False->True, -1 means True->False
        diff = np.diff(padded.astype(int))
        
        starts = np.where(diff == 1)[0]
        ends = np.where(diff == -1)[0]
        
        if len(starts) == 0:
            return 0, 0
            
        # Calculate lengths of all True segments and find the longest one
        lengths = ends - starts
        max_idx = np.argmax(lengths)
        
        # Return inclusive start and end indices
        return starts[max_idx], ends[max_idx] - 1

    @staticmethod
    def _extract_and_align_notch(img, coords, film_bbox, padding):
        """Internal helper: Extracts the notch crop and rotates it based on film edge."""
        x1, y1, x2, y2 = coords
        fx1, fy1, fx2, fy2 = film_bbox
        img_h, img_w = img.shape[:2]
        
        cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
        
        distances = {
            'top': abs(cy - fy1),
            'bottom': abs(fy2 - cy),
            'left': abs(cx - fx1),
            'right': abs(fx2 - cx)
        }
        closest_edge = min(distances, key=distances.get)
        
        crop_x1, crop_y1 = max(0, int(x1) - padding), max(0, int(y1) - padding)
        crop_x2, crop_y2 = min(img_w, int(x2) + padding), min(img_h, int(y2) + padding)
        crop = img[crop_y1:crop_y2, crop_x1:crop_x2]
        
        if crop.size == 0:
            return crop, closest_edge
            
        if closest_edge == 'top':
            aligned_crop = cv2.rotate(crop, cv2.ROTATE_180)
        elif closest_edge == 'left':
            aligned_crop = cv2.rotate(crop, cv2.ROTATE_90_COUNTERCLOCKWISE)
        elif closest_edge == 'right':
            aligned_crop = cv2.rotate(crop, cv2.ROTATE_90_CLOCKWISE)
        else:
            aligned_crop = crop
            
        return aligned_crop, closest_edge

    def __init__(self, tag="v1.0.0", device_name=None):
        
        # Setup Device
        if device_name is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device_name)

        print(f"Initializing Pipeline on: {self.device}")

        script_dir = os.path.dirname(os.path.abspath(__file__))
        assets_dir = os.path.join(script_dir, "assets")
        os.makedirs(assets_dir, exist_ok=True)

        # Define all required assets (YOLO weights + 3 SVMs)
        assets = [
            "yolo_weights.pt",
            "svm_notch_noise.pkl",
            "svm_shape_grouped.pkl",
            # "svm_direction.pkl",
            # "dinov2_features.npz"
        ]

        # Check and Download Assets
        for filename in assets:
            file_path = os.path.join(assets_dir, filename)
            if not os.path.exists(file_path):
                url = f"https://github.com/jopockel/notch-code-classifier/releases/download/{tag}/{filename}"
                print(f"Downloading {filename} from GitHub Release ({tag})...")
                try:
                    response = requests.get(url)
                    response.raise_for_status()
                    with open(file_path, "wb") as f:
                        f.write(response.content)
                except Exception as e:
                    print(f"Error: Could not download {filename} from {url}. Details: {e}")

        # Load models
        try:
            self.yolo_model = YOLO(os.path.join(assets_dir, "yolo_weights.pt"))
            self.svm_notch_noise = joblib.load(os.path.join(assets_dir,"svm_notch_noise.pkl"))
            self.svm_shape_grouped = joblib.load(os.path.join(assets_dir,"svm_shape_grouped.pkl"))
            # self.svm_direction.pkl = joblib.load(os.path.join(assets_dir,"svm_direction.pkl"))
        except FileNotFoundError as e:
            print(f"Error loading model files: {e}")

        print("Loading DINOv2...")
        self.dinov2 = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')
        self.dinov2 = self.dinov2.to(self.device)
        self.dinov2.eval()

        # Setup transformer
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def get_yolo_candidates(self, img, yolo_conf=0.15, iou_threshold=0.7):
        """Ask YOLO for all possible notch candidates."""
        results = self.yolo_model.predict(
            source=img, 
            conf=yolo_conf, 
            iou=iou_threshold,
            agnostic_nms=True, 
            imgsz=1024,
            verbose=False,
            end2end=False
        )
        
        candidates = []
        for box in results[0].boxes:
            candidates.append({
                'coords': [int(c) for c in box.xyxy[0].tolist()], 
                'prob': float(box.conf[0])                        
            })
        return candidates

    def _get_film_bounding_box(self, img, strip_tolerance=0.02, patch_size=15):
            """
            Finds the main bounding box of the film material, adapting
            to the dark background of the scanner bed.
            """
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
            h, w = gray.shape
            
            # Safely handle images that might be smaller than our patch size
            p_h = min(patch_size, h // 2)
            p_w = min(patch_size, w // 2)
            
            # Extract the four corners of the image
            top_left = gray[0:p_h, 0:p_w]
            top_right = gray[0:p_h, w-p_w:w]
            bottom_left = gray[h-p_h:h, 0:p_w]
            bottom_right = gray[h-p_h:h, w-p_w:w]
            
            corners_combined = np.concatenate([
                top_left.flatten(), top_right.flatten(), 
                bottom_left.flatten(), bottom_right.flatten()
            ])
            
            bg_median = np.median(corners_combined)
            bg_threshold = bg_median + round(256 * 0.01)
            
            # Create mask of bright "film" pixels
            film_mask = (gray > bg_threshold).astype(np.uint8)
            
            # Calculate fractions
            row_fractions = np.sum(film_mask, axis=1) / w
            col_fractions = np.sum(film_mask, axis=0) / h
            
            # Create boolean arrays where the strip has enough film pixels
            valid_rows = row_fractions > strip_tolerance
            valid_cols = col_fractions > strip_tolerance
            
            # Find the largest continuous block of valid rows and columns
            y1, y2 = self._find_largest_true_segment(valid_rows)
            x1, x2 = self._find_largest_true_segment(valid_cols)
            
            # Fallback if no valid segments are found
            if y1 == y2 == 0 or x1 == x2 == 0:
                return [0, 0, w, h]
                
            return [int(x1), int(y1), int(x2), int(y2)]

    def process_image(self, img, band_width=15, border_thresh=40, border_ratio=0.84, yolo_conf=0.3, svm_notch_noise_threshold=0.2, padding=10):
        """
        The main pipeline flow:
        1. Check for border.
        2. If border exists, run YOLO to get candidates.
        3. Rotate candidates and run DINOv2 + SVMs to filter noise and classify shape.
        """
        # Convert to grayscale for perimeter check
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
        
        # Step 1: Perimeter and Border Classification
        hist = self.get_perimeter_histogram(gray, band_width)
        is_border_present = self.border_classification(hist, border_thresh, border_ratio)
        
        if not is_border_present:
            return {
                "border_present": False,
                "accepted_notches": [],
                "rejected_notches": []
            }
            
        # Step 2: Get YOLO Candidates (Tunable Thresholds)
        boxes = self.get_yolo_candidates(img, yolo_conf=yolo_conf)
        
        # Step 3: Extract, Align, and Classify
        svm_accepted = []
        svm_rejected = []
        film_bbox = self._get_film_bounding_box(img)
        
        for box in boxes:
            crop, edge_location = self._extract_and_align_notch(img, box['coords'], film_bbox, padding)
            if crop.size == 0: 
                continue
                
            # Convert to PIL for DINOv2
            pil_crop = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
            img_tensor = self.transform(pil_crop).unsqueeze(0).to(self.device)
            
            # Extract features
            with torch.no_grad():
                feature_vector = self.dinov2(img_tensor).cpu().numpy().flatten()
                
            # SVM 1: Is it a Notch or Noise?
            notch_idx = list(self.svm_notch_noise.classes_).index('notch')
            prob_notch = self.svm_notch_noise.predict_proba([feature_vector])[0][notch_idx]
            
            candidate_data = {
                'coords': box['coords'], 
                'yolo_prob': box['prob'],
                'svm_notch_prob': prob_notch,
                'edge_location': edge_location
            }
            
            # SVM 2: Shape Classification (if it is a notch)
            if prob_notch >= svm_notch_noise_threshold:
                shape_probs = self.svm_shape_grouped.predict_proba([feature_vector])[0]
                best_shape_idx = np.argmax(shape_probs)
                shape_confidence = shape_probs[best_shape_idx]
                shape_name = self.svm_shape_grouped.classes_[best_shape_idx]

                candidate_data['shape'] = shape_name
                candidate_data['shape_confidence'] = shape_confidence
                candidate_data['needs_human_review'] = bool(shape_confidence < 0.60)
                
                svm_accepted.append(candidate_data)
            else:
                svm_rejected.append(candidate_data)
                
        return {
            "border_present": True,
            "accepted_notches": svm_accepted,
            "rejected_notches": svm_rejected
        }