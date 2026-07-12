import torch
import torch.nn as nn
from torchvision import models, transforms
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# Import Grad-CAM from the jacobgil library
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image

# 1. Load the Trained Model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

num_classes = 5 # Adjust if your SIPaKMeD subset has a different number
model = models.resnet50(weights=None)
num_features = model.fc.in_features
model.fc = nn.Linear(num_features, num_classes)

# Load the weights you saved previously
model.load_state_dict(torch.load("sipakmed_resnet50.pth", map_location=device))
model = model.to(device)
model.eval()

# 2. Setup Grad-CAM
# For ResNet, the best layer for Grad-CAM is usually the last convolutional block
target_layers = [model.layer4[-1]]

cam = GradCAM(model=model, target_layers=target_layers)

# 3. Load and Prepare Image and Masks
# REPLACE these with actual paths from your dataset/segmentation model
image_path = "path/to/sipakmed_image.bmp"
cell_mask_path = "path/to/cell_mask.bmp"
nucleus_mask_path = "path/to/nucleus_mask.bmp"

# Load image for the model (requires exact normalization used in training)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

pil_image = Image.open(image_path).convert('RGB')
input_tensor = transform(pil_image).unsqueeze(0).to(device)

# Load image for visualization (RGB, float, scaled between 0 and 1)
rgb_img = cv2.imread(image_path, 1)[:, :, ::-1] # BGR to RGB
rgb_img = cv2.resize(rgb_img, (224, 224))
rgb_img = np.float32(rgb_img) / 255

# Load masks (Grayscale, resize to match model input, threshold to binary 0 or 1)
def load_and_prep_mask(path):
    mask = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    mask = cv2.resize(mask, (224, 224))
    _, mask = cv2.threshold(mask, 127, 1, cv2.THRESH_BINARY)
    return mask

cell_mask = load_and_prep_mask(cell_mask_path)
nucleus_mask = load_and_prep_mask(nucleus_mask_path)

# 4. Generate Grad-CAM Heatmap
# We want to see what parts of the image contribute to the highest scoring class
# If you want to force it to look at a specific class index, pass e.g., [ClassifierOutputTarget(2)]
targets = None 

# Generate the CAM (returns a numpy array)
grayscale_cam = cam(input_tensor=input_tensor, targets=targets)
grayscale_cam = grayscale_cam[0, :] # Take the first image in the batch

# Create the visual overlay
visualization = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)

# 5. Analyze Correspondence (Overlap Metric)
# Threshold the CAM to find the "highest attention" areas (e.g., top 30% of activations)
cam_threshold = 0.7 
attention_mask = (grayscale_cam >= cam_threshold).astype(np.uint8)

# Calculate what percentage of the model's top attention falls inside the masks
attention_area = np.sum(attention_mask)

if attention_area > 0:
    cell_overlap = np.sum(attention_mask & cell_mask) / attention_area
    nucleus_overlap = np.sum(attention_mask & nucleus_mask) / attention_area
else:
    cell_overlap, nucleus_overlap = 0.0, 0.0

print(f"Percentage of high-attention area inside Cell Mask: {cell_overlap * 100:.2f}%")
print(f"Percentage of high-attention area inside Nucleus Mask: {nucleus_overlap * 100:.2f}%")

# 6. Visualize Results
fig, axes = plt.subplots(1, 4, figsize=(20, 5))

axes[0].imshow(rgb_img)
axes[0].set_title('Original Image')
axes[0].axis('off')

# Display cell mask (blue) and nucleus (red) combined for visual clarity
combined_mask = np.zeros_like(rgb_img)
combined_mask[cell_mask == 1] = [0, 0, 1]  # Blue for cell
combined_mask[nucleus_mask == 1] = [1, 0, 0]  # Red for nucleus
axes[1].imshow(combined_mask)
axes[1].set_title('Ground Truth Masks\n(Blue=Cell, Red=Nucleus)')
axes[1].axis('off')

axes[2].imshow(grayscale_cam, cmap='jet')
axes[2].set_title('Raw Grad-CAM Heatmap')
axes[2].axis('off')

axes[3].imshow(visualization)
axes[3].set_title('Grad-CAM Overlay')
axes[3].axis('off')

plt.tight_layout()
plt.show()
