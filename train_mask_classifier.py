import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import models, transforms
from torchvision.models import ResNet50_Weights
from PIL import Image

# ==========================================
# 1. Custom Dataset for 5-Channel Input
# ==========================================
class SIPaKMeDMaskDataset(Dataset):
    def __init__(self, images_dir, cell_masks_dir, nucleus_masks_dir, transforms_dict):
        """
        Assumes directories are structured as:
        dir/
          class_1/
            img_001.bmp
          class_2/
            img_002.bmp
        """
        self.images_dir = images_dir
        self.cell_masks_dir = cell_masks_dir
        self.nucleus_masks_dir = nucleus_masks_dir
        self.transforms_dict = transforms_dict
        
        self.classes = sorted(os.listdir(images_dir))
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}
        
        self.image_paths = []
        self.labels = []
        
        # Map all valid files
        for cls_name in self.classes:
            cls_dir = os.path.join(images_dir, cls_name)
            if not os.path.isdir(cls_dir): continue
            for img_name in os.listdir(cls_dir):
                self.image_paths.append(os.path.join(cls_name, img_name))
                self.labels.append(self.class_to_idx[cls_name])

    def __len__(self):
        return len(self.image_paths)
        
    def __getitem__(self, idx):
        rel_path = self.image_paths[idx]
        label = self.labels[idx]
        
        # Load Image (RGB) and Masks (Grayscale)
        image = Image.open(os.path.join(self.images_dir, rel_path)).convert("RGB")
        cell_mask = Image.open(os.path.join(self.cell_masks_dir, rel_path)).convert("L")
        nucleus_mask = Image.open(os.path.join(self.nucleus_masks_dir, rel_path)).convert("L")
        
        # Apply transforms separately (masks don't get RGB normalization)
        image = self.transforms_dict['image'](image)
        cell_mask = self.transforms_dict['mask'](cell_mask)
        nucleus_mask = self.transforms_dict['mask'](nucleus_mask)
            
        # Ensure masks are strictly binary (0.0 or 1.0)
        cell_mask = (cell_mask > 0.5).float()
        nucleus_mask = (nucleus_mask > 0.5).float()
        
        # Concatenate along the channel dimension (Dim 0 in PyTorch)
        # image (3xHxW) + cell_mask (1xHxW) + nucleus_mask (1xHxW) = 5xHxW
        combined_input = torch.cat((image, cell_mask, nucleus_mask), dim=0)
        
        return combined_input, label

# ==========================================
# 2. Setup Data Loading
# ==========================================
# REPLACE these with your actual directory paths
IMAGES_DIR = "path/to/images"
CELL_MASKS_DIR = "path/to/cell_masks"
NUCLEUS_MASKS_DIR = "path/to/nucleus_masks"

# Dictionaries for transforms
transforms_dict = {
    'image': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ]),
    'mask': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor() # Converts to [0, 1] range automatically
    ])
}

print("Loading dataset...")
dataset = SIPaKMeDMaskDataset(IMAGES_DIR, CELL_MASKS_DIR, NUCLEUS_MASKS_DIR, transforms_dict)
num_classes = len(dataset.classes)

train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=2)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=2)

# ==========================================
# 3. Modify ResNet50 for 5 Channels
# ==========================================
print("Initializing modified ResNet50...")
model = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)

# 3A. Replace conv1
old_conv1 = model.conv1
# Create a new layer with 5 input channels instead of 3
new_conv1 = nn.Conv2d(in_channels=5, 
                      out_channels=old_conv1.out_channels, 
                      kernel_size=old_conv1.kernel_size, 
                      stride=old_conv1.stride, 
                      padding=old_conv1.padding, 
                      bias=False)

with torch.no_grad():
    # Copy pre-trained weights to the first 3 channels (RGB)
    new_conv1.weight[:, :3, :, :] = old_conv1.weight
    # Initialize the new 2 channels (Masks) using Kaiming Normalization
    nn.init.kaiming_normal_(new_conv1.weight[:, 3:, :, :], mode='fan_out', nonlinearity='relu')

model.conv1 = new_conv1

# 3B. Replace the classification head
num_features = model.fc.in_features
model.fc = nn.Linear(num_features, num_classes)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# ==========================================
# 4. Training Setup (Modified Warmup)
# ==========================================
criterion = nn.CrossEntropyLoss()

# We need to train BOTH the new conv1 layer AND the new fc layer during warmup
# because they both contain randomly initialized weights.
for name, param in model.named_parameters():
    if 'conv1' in name or 'fc' in name:
        param.requires_grad = True
    else:
        param.requires_grad = False

# Pass only the unfreezed parameters to the optimizer
optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.001)

# ==========================================
# 5. Training Loop
# ==========================================
epochs = 5 # Set to desired warmup epochs (Stage 1)
print(f"Starting warmup training on {device}...")

for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * inputs.size(0)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
    epoch_loss = running_loss / len(train_dataset)
    epoch_acc = correct / total
    print(f"Epoch {epoch+1}/{epochs} | Loss: {epoch_loss:.4f} | Accuracy: {epoch_acc:.4f}")

print("Warmup complete! You can now unfreeze the rest of the model for Stage 2 fine-tuning.")
