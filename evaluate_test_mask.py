import os
import torch
import torch.nn as nn
from torchvision import models, transforms
from torchvision.models import ResNet50_Weights
from torch.utils.data import Dataset, DataLoader, random_split
from PIL import Image
import logging
from datetime import datetime
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import json
import glob

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
log_filename = os.path.join(log_dir, f"test_evaluation_mask_{timestamp}.log")

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_filename),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

logger.info(f"=== Mask Classifier Test Evaluation Session Started at {timestamp} ===")
logger.info(f"SEED: {SEED}")
logger.info(f"Log file: {log_filename}")

# ==========================================
# Dataset Definition
# ==========================================
class SIPaKMeDMaskDataset(Dataset):
    def __init__(self, images_dir, cell_masks_dir, nucleus_masks_dir, transforms_dict):
        self.images_dir = images_dir
        self.cell_masks_dir = cell_masks_dir
        self.nucleus_masks_dir = nucleus_masks_dir
        self.transforms_dict = transforms_dict
        
        self.classes = ['im_Koilocytotic', 'im_Superficial-Intermediate', 'im_Dyskeratotic', 'im_Parabasal', 'im_Metaplastic']
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}
        
        self.image_paths = [] 
        self.targets = []
        
        for cls_name in self.classes:
            cropped_dirs = glob.glob(
                os.path.join(images_dir, cls_name, "**", "CROPPED"),
                recursive=True
            )

            if len(cropped_dirs) == 0:
                logger.warning(f"CROPPED folder not found for {cls_name}")
                continue

            cls_dir = cropped_dirs[0]

            for img_name in sorted(os.listdir(cls_dir)):
                if img_name.lower().endswith((".bmp", ".png", ".jpg", ".jpeg")):
                    self.image_paths.append((cls_name, cls_dir, img_name))
                    self.targets.append(self.class_to_idx[cls_name])

    def __len__(self):
        return len(self.image_paths)
        
    def __getitem__(self, idx):
        cls_name, cls_dir, img_name = self.image_paths[idx]
        label = self.targets[idx]
        
        img_path = os.path.join(cls_dir, img_name)
        image = Image.open(img_path).convert("RGB")
        
        cell_path = os.path.join(self.cell_masks_dir, cls_name, img_name)
        nucleus_path = os.path.join(self.nucleus_masks_dir, cls_name, img_name)
        
        cell_mask = Image.open(cell_path).convert("L")
        nucleus_mask = Image.open(nucleus_path).convert("L")
        
        image = self.transforms_dict['image'](image)
        cell_mask = self.transforms_dict['mask'](cell_mask)
        nucleus_mask = self.transforms_dict['mask'](nucleus_mask)
            
        cell_mask = (cell_mask > 0.5).float()
        nucleus_mask = (nucleus_mask > 0.5).float()
        
        combined_input = torch.cat((image, cell_mask, nucleus_mask), dim=0)
        
        return combined_input, label

# ==========================================
# Setup Device
# ==========================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Device: {device}")

# ==========================================
# Load Dataset
# ==========================================
logger.info("Loading SIPaKMeD dataset with masks...")
IMAGES_DIR = "/home/al.bianca.abreu/.cache/kagglehub/datasets/prahladmehandiratta/cervical-cancer-largest-dataset-sipakmed/versions/1"
BASE_OUTPUT_DIR = os.path.join(os.getcwd(), "sipakmed_generated_masks")
CELL_MASKS_DIR = os.path.join(BASE_OUTPUT_DIR, "cell_masks")
NUCLEUS_MASKS_DIR = os.path.join(BASE_OUTPUT_DIR, "nucleus_masks")

transforms_dict = {
    'image': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ]),
    'mask': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])
}

dataset = SIPaKMeDMaskDataset(IMAGES_DIR, CELL_MASKS_DIR, NUCLEUS_MASKS_DIR, transforms_dict)
num_classes = len(dataset.classes)
logger.info(f"Total samples: {len(dataset)}")
logger.info(f"Number of classes: {num_classes}")
logger.info(f"Classes: {dataset.classes}")

# Split dataset (same proportions as training)
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

test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=2)

# ==========================================
# Load Model
# ==========================================
logger.info("Loading 5-channel model...")
model = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)

old_conv1 = model.conv1
new_conv1 = nn.Conv2d(in_channels=5, 
                      out_channels=old_conv1.out_channels, 
                      kernel_size=old_conv1.kernel_size, 
                      stride=old_conv1.stride, 
                      padding=old_conv1.padding, 
                      bias=False)

with torch.no_grad():
    new_conv1.weight[:, :3, :, :] = old_conv1.weight
    nn.init.kaiming_normal_(new_conv1.weight[:, 3:, :, :], mode='fan_out', nonlinearity='relu')

model.conv1 = new_conv1
num_features = model.fc.in_features
model.fc = nn.Linear(num_features, num_classes)

model_path = "sipakmed_resnet50_5channels_best.pth"
if os.path.exists(model_path):
    model.load_state_dict(torch.load(model_path, map_location=device))
    logger.info(f"Model loaded from: {model_path}")
else:
    logger.error(f"Model file not found: {model_path}")
    raise FileNotFoundError(f"Model file not found: {model_path}")

model = model.to(device)
model.eval()

# ==========================================
# Evaluate on Test Set
# ==========================================
logger.info("\n=== Evaluating 5-Channel Model on Test Set ===")
print("\n=== Evaluating 5-Channel Model on Test Set ===")

criterion = nn.CrossEntropyLoss()

all_predictions = []
all_labels = []
running_loss = 0.0

with torch.no_grad():
    for inputs, labels in test_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        running_loss += loss.item() * inputs.size(0)
        
        _, predicted = torch.max(outputs, 1)
        all_predictions.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

all_predictions = np.array(all_predictions)
all_labels = np.array(all_labels)

# Calculate metrics
test_loss = running_loss / len(test_dataset)
test_accuracy = accuracy_score(all_labels, all_predictions)

logger.info(f"Test Loss: {test_loss:.4f}")
logger.info(f"Test Accuracy: {test_accuracy:.4f}")
print(f"Test Loss: {test_loss:.4f}")
print(f"Test Accuracy: {test_accuracy:.4f}")

# Classification Report
logger.info("\n=== Classification Report ===")
print("\n=== Classification Report ===")
class_report = classification_report(
    all_labels, 
    all_predictions, 
    target_names=dataset.classes,
    digits=4
)
logger.info(f"\n{class_report}")
print(class_report)

# Confusion Matrix
logger.info("\n=== Confusion Matrix ===")
print("\n=== Confusion Matrix ===")
conf_matrix = confusion_matrix(all_labels, all_predictions)
logger.info(f"\n{conf_matrix}")
print(f"\n{conf_matrix}")

# Save detailed report
report_path = os.path.join(log_dir, f"test_report_mask_{timestamp}.json")
report_dict = {
    "timestamp": timestamp,
    "seed": SEED,
    "model_type": "ResNet50 with 5 Channels (RGB + Cell Mask + Nucleus Mask)",
    "test_loss": float(test_loss),
    "test_accuracy": float(test_accuracy),
    "test_set_size": len(test_dataset),
    "classes": dataset.classes,
    "classification_report": class_report,
    "confusion_matrix": conf_matrix.tolist()
}

with open(report_path, 'w') as f:
    json.dump(report_dict, f, indent=4)

logger.info(f"\nDetailed report saved to: {report_path}")
print(f"\nDetailed report saved to: {report_path}")

logger.info(f"=== Test Evaluation Session Completed ===")
print("\n=== Test Evaluation Session Completed ===")
