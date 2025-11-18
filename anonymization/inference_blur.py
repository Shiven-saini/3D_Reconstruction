import os
from pathlib import Path
from ultralytics import YOLO
import cv2

MODEL_PATH = "../assets/yolov11m_face_lp.pt" 
TESTING_DIR = "testing"
OUTPUT_DIR = "output"
BLUR_STRENGTH = 99

# Create output directory if it doesn't exist
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load your fine-tuned YOLOv11m model
model = YOLO(MODEL_PATH)

def apply_gaussian_blur(image, x1, y1, x2, y2, kernel_size=99):
    """
    Apply strong Gaussian blur to a region of interest (irreversible)
    
    Args:
        image: Input image
        x1, y1: Top-left coordinates
        x2, y2: Bottom-right coordinates
        kernel_size: Blur kernel size (must be odd, higher = stronger blur)
    """
    # Ensure kernel size is odd
    if kernel_size % 2 == 0:
        kernel_size += 1
    
    # Extract ROI
    roi = image[y1:y2, x1:x2]
    
    # Apply Gaussian blur
    blurred_roi = cv2.GaussianBlur(roi, (kernel_size, kernel_size), 0)
    
    # Replace original ROI with blurred version
    image[y1:y2, x1:x2] = blurred_roi
    
    return image

def apply_pixelation(image, x1, y1, x2, y2, blocks=10):
    """
    Apply pixelation blur to a region of interest (irreversible)
    
    Args:
        image: Input image
        x1, y1: Top-left coordinates
        x2, y2: Bottom-right coordinates
        blocks: Number of blocks (lower = stronger pixelation)
    """
    # Extract ROI
    roi = image[y1:y2, x1:x2]
    h, w = roi.shape[:2]
    
    # Calculate block size
    block_w = max(1, w // blocks)
    block_h = max(1, h // blocks)
    
    # Resize down and then up to create pixelation effect
    temp = cv2.resize(roi, (block_w, block_h), interpolation=cv2.INTER_LINEAR)
    pixelated_roi = cv2.resize(temp, (w, h), interpolation=cv2.INTER_NEAREST)
    
    # Replace original ROI with pixelated version
    image[y1:y2, x1:x2] = pixelated_roi
    
    return image

# Get all image files from testing directory
image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.webp']
image_files = []

for ext in image_extensions:
    image_files.extend(Path(TESTING_DIR).glob(f'*{ext}'))
    image_files.extend(Path(TESTING_DIR).glob(f'*{ext.upper()}'))

print(f"Found {len(image_files)} images in {TESTING_DIR} directory")

# Run inference on each image
for img_path in image_files:
    print(f"Processing: {img_path.name}")
    
    # Load image
    image = cv2.imread(str(img_path))
    
    if image is None:
        print(f"  Error loading image, skipping...")
        continue
    
    # Run inference
    results = model(str(img_path), conf=0.25)
    
    # Process results
    for result in results:
        boxes = result.boxes
        
        if len(boxes) > 0:
            print(f"  Detected {len(boxes)} objects:")
            
            # Process each detection
            for box in boxes:
                # Get bounding box coordinates
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                cls_name = model.names[cls_id]
                
                # Ensure coordinates are within image bounds
                x1 = max(0, x1)
                y1 = max(0, y1)
                x2 = min(image.shape[1], x2)
                y2 = min(image.shape[0], y2)
                
                print(f"    - Blurring {cls_name} (confidence: {conf:.2f})")
                
                # Apply Gaussian blur (choose one method)
                image = apply_gaussian_blur(image, x1, y1, x2, y2, kernel_size=BLUR_STRENGTH)
                
                # OR use pixelation instead (uncomment to use)
                # image = apply_pixelation(image, x1, y1, x2, y2, blocks=8)
        else:
            print(f"  No objects detected")
    
    # Save the anonymized image
    output_path = os.path.join(OUTPUT_DIR, img_path.name)
    cv2.imwrite(output_path, image)
    print(f"  Saved to: {output_path}")

print(f"\nAnonymization complete! Blurred images saved to '{OUTPUT_DIR}' directory")
