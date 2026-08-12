import torch
import torch.nn as nn
from torchvision import models, transforms
from torchvision.models import ResNet50_Weights
from torch.utils.data import DataLoader, random_split
import kagglehub
import os
import logging
from datetime import datetime
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import json
from PIL import Image
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
log_filename = os.path.join(log_dir, f"test_evaluation_{timestamp}.log")

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_filename),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

logger.info(f"=== Test Evaluation Session Started at {timestamp} ===")
logger.info(f"SEED: {SEED}")
logger.info(f"Log file: {log_filename}")

# ==========================================
# Dataset Definition
# ==========================================
class SipakMedCroppedDataset(torch.utils.data.Dataset):
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
# Setup Device
# ==========================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Device: {device}")

# ==========================================
# Load Dataset
# ==========================================
logger.info("Loading SIPaKMeD dataset...")
dataset_path = kagglehub.dataset_download("prahladmehandiratta/cervical-cancer-largest-dataset-sipakmed")
data_dir = dataset_path
logger.info(f"Dataset path: {data_dir}")

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

dataset = SipakMedCroppedDataset(data_dir, transform=transform)
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
logger.info("Loading model...")
model = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
num_features = model.fc.in_features
model.fc = nn.Linear(num_features, num_classes)

model_path = "sipakmed_resnet50.pth"
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
logger.info("\n=== Evaluating on Test Set ===")
print("\n=== Evaluating on Test Set ===")

criterion = nn.CrossEntropyLoss()

all_predictions = []
all_labels = []
running_loss = 0.0

with torch.no_grad():
    for inputs, labels in DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=2):
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
report_path = os.path.join(log_dir, f"test_report_{timestamp}.json")
report_dict = {
    "timestamp": timestamp,
    "seed": SEED,
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
