# -------------------------------
# 🔹 Import Required Libraries
# -------------------------------
import torch
import cv2
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# -------------------------------
# 🔹 Load Pretrained YOLOv5 Model
# -------------------------------
# 'yolov5s' = small model (fast and accurate)
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

# -------------------------------
# 🔹 Load Image (Update path for your image)
# -------------------------------
img_path = r'C:\Users\admin\OneDrive\YOLO\YOLO Image 3.jpg'

# Load the image
img = cv2.imread(img_path)

# Check if image is loaded properly
if img is None:
    raise FileNotFoundError(f"⚠️ Image not found at {img_path}")

# -------------------------------
# 🔹 Convert and Display the Image
# -------------------------------
# OpenCV loads in BGR, convert to RGB for correct colors
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Display original image
plt.imshow(img_rgb)
plt.axis('off')
plt.title("Original Image")
plt.show()

# -------------------------------
# 🔹 Perform Object Detection
# -------------------------------
results = model(img_rgb)

# Print detection results in console
results.print()

# Show detection results (opens a window with bounding boxes)
results.show()

# -------------------------------
# 🔹 Extract Results as a DataFrame
# -------------------------------
# Convert detections to Pandas DataFrame
detections = results.pandas().xyxy[0]

# Print detected objects (label, confidence, coordinates)
print("🧠 Detected Objects:")
print(detections)

# -------------------------------
# 🔹 Optional: Save Results Image
# -------------------------------
# Saves the image with boxes automatically to 'runs/detect/exp'
results.save()
print("✅ Detection image saved in 'runs/detect/exp' folder.")
