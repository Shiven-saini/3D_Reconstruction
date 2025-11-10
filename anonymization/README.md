# Anonymization of User Data on Edge Device

This project focuses on privacy-preserving visual data processing by detecting and anonymizing sensitive information such as faces and vehicle license plates directly on the edge device. It uses fine-tuned **YOLOv11** models for real-time detection and **OpenCV Gaussian Blur** for anonymization.

---

## Overview

In many real-world applications such as surveillance, autonomous driving, and smart cities, video data can unintentionally expose personal information. This project ensures privacy protection at the edge by automatically detecting and blurring sensitive regions before the data leaves the device.

---

## Features

* Detection of faces and license plates using fine-tuned YOLOv11 models
* Multiple YOLOv11 variants trained and evaluated to balance model size, inference speed, and accuracy
* Gaussian blur anonymization using OpenCV for detected regions
* Command-line interface for flexible usage:

  * Visualize detection bounding boxes for accuracy testing
  * Apply anonymization to single images or video files
  * Option to toggle between face, license-plate, or both

---

## Datasets Used

### 1. License Plate Recognition

**Dataset:** [License Plate Recognition (Roboflow)](https://universe.roboflow.com/roboflow-universe-projects/license-plate-recognition-rxg4e)
**Training Images:** ~7,000
**Purpose:** Detect vehicle license plates for anonymization

### 2. Face Detection

**Dataset:** [Face Dataset (Roboflow)](https://universe.roboflow.com/thomas-febr7/face-d7xbs)
**Purpose:** Detect human faces for anonymization

*Note:* Although the Roboflow face dataset is used initially, **WIDER-Face** is a strong candidate for future improvement due to its higher variability and robustness.

---

## Model Variants

Several **YOLOv11** variants (n, s, m, l, x) were fine-tuned to assess trade-offs between:

* Model size (for deployment feasibility on edge devices)
* Inference speed (real-time performance)
* Detection accuracy

Each variant can be selected based on the edge device’s computational resources and target performance.

## Usage

The script supports anonymizing both images and videos, as well as visualizing detection results.

### Example Commands

**1. Show Bounding Boxes (Test Accuracy)**

```bash
python anonymize.py --input path/to/image_or_video --show-box
```

**2. Blur Faces**

```bash
python anonymize.py --input path/to/image_or_video --blur face
```

**3. Blur License Plates**

```bash
python anonymize.py --input path/to/image_or_video --blur plate
```

**4. Blur Both Faces and Plates**

```bash
python anonymize.py --input path/to/image_or_video --blur all
```

---

## Implementation Details

1. **Inference**
   The YOLOv11 model detects bounding boxes for faces and license plates.

2. **Post-Processing**
   OpenCV applies Gaussian blur to each detected region:

   ```python
   blurred_region = cv2.GaussianBlur(region, (51, 51), 30)
   ```

3. **Output**
   The script saves or displays an anonymized image/video based on command-line arguments.

---

## Future Improvements

* Replace or try the face dataset with **WIDER-Face** for improved generalization
* Add object tracking for consistent anonymization across video frames
* Experimental : Integrate **ONNX** or **TensorRT** export for faster deployment on edge devices
