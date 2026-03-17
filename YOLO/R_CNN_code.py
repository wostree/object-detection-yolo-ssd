import torch
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.transforms import functional as F
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# ✅ Correct image path (use raw string to avoid errors)
image_path = r"C:\Users\admin\OneDrive\YOLO\R-CNN Image 1.jpeg"

# Load and prepare image
image = Image.open(image_path).convert("RGB")
image_tensor = F.to_tensor(image)

# Load pretrained Faster R-CNN model
from torchvision.models.detection import FasterRCNN_ResNet50_FPN_Weights
model = fasterrcnn_resnet50_fpn(weights=FasterRCNN_ResNet50_FPN_Weights.DEFAULT)
model.eval()

# Run inference (disable gradient calculations for speed)
with torch.no_grad():
    outputs = model([image_tensor])

# Extract predicted boxes, labels, and scores
boxes = outputs[0]['boxes']
labels = outputs[0]['labels']
scores = outputs[0]['scores']

# Visualize results
fig, ax = plt.subplots(1, figsize=(12, 8))
ax.imshow(image)

for box, label, score in zip(boxes, labels, scores):
    if score > 0.8:  # Show only confident detections
        x1, y1, x2, y2 = box
        rect = patches.Rectangle(
            (x1, y1), x2 - x1, y2 - y1,
            linewidth=2, edgecolor='r', facecolor='none'
        )
        ax.add_patch(rect)
        # Add label text (optional)
        ax.text(x1, y1, f"{label.item()} ({score:.2f})", 
                color='white', fontsize=10,
                bbox=dict(facecolor='red', alpha=0.5))

plt.axis("off")
plt.show()
