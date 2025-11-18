import os
from pathlib import Path
from ultralytics import YOLO
import cv2

MODEL_PATH = "../assets/yolov11m_face_lp.pt"  # Path to your fine-tuned YOLOv11m model
TESTING_DIR = "testing"  # Directory containing test images
OUTPUT_DIR = "output"  # Directory to save annotated images

# Create output directory if it doesn't exist
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load your fine-tuned YOLOv11m model
model = YOLO(MODEL_PATH)

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
    
    # Run inference
    results = model(str(img_path), conf=0.25)  # conf threshold can be adjusted
    
    # Process results
    for i, result in enumerate(results):
        # Plot bounding boxes on the image
        annotated_img = result.plot(
            conf=True,        # Show confidence scores
            labels=True,      # Show class labels
            boxes=True,       # Show bounding boxes
            line_width=2      # Adjust box line width
        )
        
        # Save the annotated image
        output_path = os.path.join(OUTPUT_DIR, img_path.name)
        cv2.imwrite(output_path, annotated_img)
        
        # Print detection summary
        detections = result.boxes
        if len(detections) > 0:
            print(f"  Detected {len(detections)} objects:")
            for box in detections:
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                cls_name = model.names[cls_id]
                print(f"    - {cls_name} (confidence: {conf:.2f})")
        else:
            print(f"  No objects detected")

print(f"\nInference complete! Annotated images saved to '{OUTPUT_DIR}' directory")
