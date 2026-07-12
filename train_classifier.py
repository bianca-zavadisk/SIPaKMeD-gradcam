import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torchvision.models import ResNet50_Weights
from torch.utils.data import DataLoader, random_split
import kagglehub

# 1. Dataset Setup
print("Downloading SIPaKMeD dataset...")
dataset_path = kagglehub.dataset_download("prahladmehandiratta/cervical-cancer-largest-dataset-sipakmed")
data_dir = dataset_path 

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

dataset = datasets.ImageFolder(root=data_dir, transform=transform)
num_classes = len(dataset.classes)

train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=2)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=2)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 2. Model Initialization
print("Loading pre-trained ResNet50...")
model = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)

def evaluate_model(model, dataloader, criterion):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return running_loss / len(dataloader.dataset), correct / total

criterion = nn.CrossEntropyLoss()

# STAGE 1: Warmup the Classification Head
print("\n--- STAGE 1: Training the classification head only ---")

# 1. Freeze all layers
for param in model.parameters():
    param.requires_grad = False

# 2. Replace the head (implicitly sets requires_grad=True for the new layer)
num_features = model.fc.in_features
model.fc = nn.Linear(num_features, num_classes)
model = model.to(device)

# 3. Optimize ONLY the head with a standard learning rate
optimizer_stage1 = optim.Adam(model.fc.parameters(), lr=0.001)

stage1_epochs = 3
for epoch in range(stage1_epochs):
    model.train()
    running_loss = 0.0
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer_stage1.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer_stage1.step()
        running_loss += loss.item() * inputs.size(0)
        
    train_loss = running_loss / len(train_dataset)
    val_loss, val_acc = evaluate_model(model, val_loader, criterion)
    print(f"Stage 1 - Epoch {epoch+1}/{stage1_epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")

# STAGE 2: Full Fine-Tuning (Stable)
print("\n--- STAGE 2: Fine-tuning the entire network with a low learning rate ---")

# 1. Unfreeze the entire network
for param in model.parameters():
    param.requires_grad = True

# 2. Re-initialize the optimizer with all parameters but a very small learning rate
# This prevents the pre-trained weights from altering drastically
optimizer_stage2 = optim.Adam(model.parameters(), lr=1e-5) 

stage2_epochs = 5
for epoch in range(stage2_epochs):
    model.train()
    running_loss = 0.0
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer_stage2.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer_stage2.step()
        running_loss += loss.item() * inputs.size(0)
        
    train_loss = running_loss / len(train_dataset)
    val_loss, val_acc = evaluate_model(model, val_loader, criterion)
    print(f"Stage 2 - Epoch {epoch+1}/{stage2_epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")

print("\nMulti-stage training complete!")

save_path = "sipakmed_resnet50.pth"
torch.save(model.state_dict(), save_path)

print(f"Model weights successfully saved to: {save_path}")
