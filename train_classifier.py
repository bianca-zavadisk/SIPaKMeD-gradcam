import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torchvision.models import ResNet50_Weights
from torch.utils.data import DataLoader, random_split
import kagglehub
import os
from tqdm.auto import tqdm # Import tqdm
from torch.utils.data import Dataset
from PIL import Image
import glob
import numpy as np
import logging
from datetime import datetime

# ==========================================
# Configure SEED for Reproducibility
# ==========================================
SEED = 42

torch.manual_seed(SEED)
np.random.seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# ==========================================
# Configure Logging
# ==========================================
log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
log_filename = os.path.join(log_dir, f"training_{timestamp}.log")

# Configure logging to both file and console
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_filename),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

logger.info(f"=== Training Session Started at {timestamp} ===")
logger.info(f"SEED: {SEED}")
logger.info(f"Log file: {log_filename}")

class SipakMedCroppedDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.transform = transform
        self.samples = []

        classes = [
            "im_Koilocytotic",
            "im_Superficial-Intermediate",
            "im_Dyskeratotic",
            "im_Parabasal",
            "im_Metaplastic"
        ]

        self.classes = classes

        self.class_to_idx = {
            cls: idx
            for idx, cls in enumerate(self.classes)
}

        for cls in classes:

            cropped_dir = glob.glob(
                os.path.join(root_dir, cls, "**", "CROPPED"),
                recursive=True
            )[0]

            for img_path in sorted(glob.glob(os.path.join(cropped_dir, "*.bmp"))):

                self.samples.append(
                    (
                        img_path,
                        self.class_to_idx[cls]
                    )
                )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):

        img_path, label = self.samples[idx]

        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, label

# ==========================================
# 1. Dataset Setup
# ==========================================
logger.info("Downloading SIPaKMeD dataset...")
print("Downloading SIPaKMeD dataset...")
dataset_path = kagglehub.dataset_download("prahladmehandiratta/cervical-cancer-largest-dataset-sipakmed")
data_dir = dataset_path
logger.info(f"Dataset path: {data_dir}")

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

dataset = SipakMedCroppedDataset(
    data_dir,
    transform=transform
)
num_classes = len(dataset.classes)
logger.info(f"Total samples: {len(dataset)}")
logger.info(f"Number of classes: {num_classes}")
logger.info(f"Classes: {dataset.classes}")

# Split dataset into train (70%), validation (15%), test (15%)
train_size = int(0.70 * len(dataset))
val_size = int(0.15 * len(dataset))
test_size = len(dataset) - train_size - val_size

train_dataset, val_dataset, test_dataset = random_split(
    dataset, 
    [train_size, val_size, test_size]
)

logger.info(f"Train set size: {len(train_dataset)}")
logger.info(f"Validation set size: {len(val_dataset)}")
logger.info(f"Test set size: {len(test_dataset)}")

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=2)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=2)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=2)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Device: {device}")

# ==========================================
# 2. Model Initialization
# ==========================================
logger.info("Loading pre-trained ResNet50...")
print("Loading pre-trained ResNet50...")
model = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)

def evaluate_model(model, dataloader, criterion, dataset_name="Validation"):
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
    avg_loss = running_loss / len(dataloader.dataset)
    accuracy = correct / total
    return avg_loss, accuracy

criterion = nn.CrossEntropyLoss()
logger.info(f"Loss function: CrossEntropyLoss")

# ==========================================
# STAGE 1: Warmup the Classification Head
# ==========================================
logger.info("\n--- STAGE 1: Training the classification head only ---")
print("\n--- STAGE 1: Training the classification head only ---")
for param in model.parameters():
    param.requires_grad = False

num_features = model.fc.in_features
model.fc = nn.Linear(num_features, num_classes)
model = model.to(device)

optimizer_stage1 = optim.Adam(model.fc.parameters(), lr=0.001)
logger.info(f"Optimizer Stage 1: Adam, lr=0.001")

stage1_epochs = 5
for epoch in range(stage1_epochs):
    model.train()
    running_loss = 0.0
    # Wrap train_loader with tqdm
    for inputs, labels in tqdm(train_loader, desc=f"Stage 1 Epoch {epoch+1}/{stage1_epochs}"):
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer_stage1.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer_stage1.step()
        running_loss += loss.item() * inputs.size(0)

    train_loss = running_loss / len(train_dataset)
    val_loss, val_acc = evaluate_model(model, val_loader, criterion, "Validation")
    log_msg = f"Stage 1 - Epoch {epoch+1}/{stage1_epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}"
    print(log_msg)
    logger.info(log_msg)

# ==========================================
# STAGE 2: Full Fine-Tuning (Stable)
# ==========================================
logger.info("\n--- STAGE 2: Fine-tuning the entire network with a low learning rate ---")
print("\n--- STAGE 2: Fine-tuning the entire network with a low learning rate ---")
for param in model.parameters():
    param.requires_grad = True

optimizer_stage2 = optim.Adam(model.parameters(), lr=1e-5)
logger.info(f"Optimizer Stage 2: Adam, lr=1e-5")

stage2_epochs = 10
for epoch in range(stage2_epochs):
    model.train()
    running_loss = 0.0
    # Wrap train_loader with tqdm
    for inputs, labels in tqdm(train_loader, desc=f"Stage 2 Epoch {epoch+1}/{stage2_epochs}"):
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer_stage2.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer_stage2.step()
        running_loss += loss.item() * inputs.size(0)

    train_loss = running_loss / len(train_dataset)
    val_loss, val_acc = evaluate_model(model, val_loader, criterion, "Validation")
    log_msg = f"Stage 2 - Epoch {epoch+1}/{stage2_epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}"
    print(log_msg)
    logger.info(log_msg)

logger.info("\nMulti-stage training complete!")
print("\nMulti-stage training complete!")

# Evaluate on Test Set
logger.info("\n--- Evaluating on Test Set ---")
print("\n--- Evaluating on Test Set ---")
test_loss, test_acc = evaluate_model(model, test_loader, criterion, "Test")
test_log_msg = f"Test Set Results | Loss: {test_loss:.4f} | Accuracy: {test_acc:.4f}"
print(test_log_msg)
logger.info(test_log_msg)

# Save Model
save_path = "sipakmed_resnet50.pth"
torch.save(model.state_dict(), save_path)
save_msg = f"Model weights successfully saved to: {save_path}"
print(save_msg)
logger.info(save_msg)
logger.info(f"=== Training Session Completed ===")