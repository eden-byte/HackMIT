import cv2
import numpy as np
import imagehash
from PIL import Image

# --- Configuration for Image Quality ---
BLUR_THRESHOLD = 100.0
CONTRAST_THRESHOLD = 20

# --- Configuration for CLAHE enhancement ---
CLAHE_CLIP_LIMIT = 2.0
CLAHE_TILE_GRID_SIZE = (8, 8)

def is_blurry(gray_image, threshold=BLUR_THRESHOLD):
    """Determines if a grayscale image is blurry by measuring the variance of the Laplacian."""
    laplacian_var = cv2.Laplacian(gray_image, cv2.CV_64F).var()
    return laplacian_var < threshold

def is_low_contrast(gray_image, threshold=CONTRAST_THRESHOLD):
    """Checks if a grayscale image has low contrast by measuring the standard deviation of pixel intensity."""
    return gray_image.std() < threshold

def enhance_image_details(gray_image):
    """
    Enhances local contrast in a grayscale image using CLAHE.
    """
    clahe = cv2.createCLAHE(clipLimit=CLAHE_CLIP_LIMIT, tileGridSize=CLAHE_TILE_GRID_SIZE)
    enhanced_gray = clahe.apply(gray_image)
    return enhanced_gray

def process_image_frame_from_memory(image_frame):
    """
    Takes a single frame, performs quality checks, and enhances its details.
    """
    if image_frame is None:
        return None

    try:
        # --- Optimization: Convert to grayscale once for all checks and processing ---
        gray_frame = cv2.cvtColor(image_frame, cv2.COLOR_BGR2GRAY)

        # Pass the pre-converted grayscale image to the quality checks
        if is_blurry(gray_frame):
            print("Discarding frame: Blurry")
            return None
        if is_low_contrast(gray_frame):
            print("Discarding frame: Low contrast")
            return None

        # Enhance the grayscale image directly for output
        processed_image_to_save = enhance_image_details(gray_frame)
        
        # Hashing should still use the original color frame for consistency
        image_rgb = cv2.cvtColor(image_frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(image_rgb)
        frame_hash = str(imagehash.phash(pil_image))
        
        (h, w) = image_frame.shape[:2]
        features = {
            'dimensions': {'width': w, 'height': h},
            'perceptual_hash': frame_hash,
            'enhancement_method': 'CLAHE_Grayscale'
        }
        
        return {
            "features": features,
            "processed_image": processed_image_to_save
        }
        
    except Exception as e:
        print(f"Error processing frame in-memory: {e}")
        return None
