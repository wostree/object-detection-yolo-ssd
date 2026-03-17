import torch
from torchvision.models.detection import ssd300_vgg16
from torchvision.transforms import functional as F
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# === 1. Load the image ===
image_path = r"C:\Users\admin\OneDrive\YOLO\R-CNN Image 1.jpeg"
image = Image.open(image_path).convert("RGB")

# === 2. Convert to tensor ===
image_tensor = F.to_tensor(image)

# === 3. Load pre-trained SSD model ===
model = ssd300_vgg16(pretrained=True)
model.eval()

# === 4. Make predictions ===
with torch.no_grad():
    outputs = model([image_tensor])

boxes = outputs[0]['boxes']
labels = outputs[0]['labels']
scores = outputs[0]['scores']

# === 5. Display results ===
fig, ax = plt.subplots(1, figsize=(12, 8))
ax.imshow(image)

# Draw boxes where confidence > 0.5
for box, score in zip(boxes, scores):
    if score > 0.5:
        x1, y1, x2, y2 = box
        rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1,
                                 linewidth=2, edgecolor='lime', facecolor='none')
        ax.add_patch(rect)
        ax.text(x1, y1, f"{score:.2f}", color='yellow', fontsize=8, weight='bold')

plt.axis("off")
plt.show()
