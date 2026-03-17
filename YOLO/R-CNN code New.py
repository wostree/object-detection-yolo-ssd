import torch
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.transforms import functional as F
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# 1️⃣ Load Pretrained Faster R-CNN model
model = fasterrcnn_resnet50_fpn(pretrained=True)
model.eval()  # Set to evaluation mode

# 2️⃣ Load your image
image_path = r"C:\Users\admin\OneDrive\YOLO\R-CNN Image 1.jpeg"  # 🔹 Note the 'r' before string
image = Image.open(image_path).convert("RGB")

# 3️⃣ Transform image to tensor
image_tensor = F.to_tensor(image)

# 4️⃣ Perform detection
with torch.no_grad():
    predictions = model([image_tensor])

# 5️⃣ Get results
boxes = predictions[0]['boxes']
labels = predictions[0]['labels']
scores = predictions[0]['scores']

# 6️⃣ Filter detections with score threshold
threshold = 0.8
selected_indices = [i for i, score in enumerate(scores) if score > threshold]

# 7️⃣ Visualize the detections
fig, ax = plt.subplots(1, figsize=(12, 8))
ax.imshow(image)

for i in selected_indices:
    box = boxes[i].cpu().numpy()
    label = labels[i].item()
    score = scores[i].item()

    # Draw bounding box
    rect = patches.Rectangle(
        (box[0], box[1]), box[2] - box[0], box[3] - box[1],
        linewidth=2, edgecolor='red', facecolor='none'
    )
    ax.add_patch(rect)
    ax.text(box[0], box[1] - 5, f"Label: {label}, Score: {score:.2f}",
            color='red', fontsize=10, backgroundcolor='white')

plt.axis('off')
plt.show(block=True)

