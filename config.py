"""
Configuration module for SIPaKMeD training and evaluation.
Centralizes seed, dataset splits, and logging settings.
"""

import os
import logging
from datetime import datetime

# ==========================================
# SEED CONFIGURATION
# ==========================================
SEED = 42

# ==========================================
# DATASET CONFIGURATION
# ==========================================
# Dataset split proportions
TRAIN_RATIO = 0.70  # 70% for training
VAL_RATIO = 0.15    # 15% for validation
TEST_RATIO = 0.15   # 15% for testing

# Dataset parameters
DATASET_CLASSES = [
    "im_Koilocytotic",
    "im_Superficial-Intermediate",
    "im_Dyskeratotic",
    "im_Parabasal",
    "im_Metaplastic"
]
NUM_CLASSES = len(DATASET_CLASSES)

# Image preprocessing
IMAGE_SIZE = (224, 224)
NORMALIZE_MEAN = [0.485, 0.456, 0.406]
NORMALIZE_STD = [0.229, 0.224, 0.225]

# DataLoader settings
BATCH_SIZE = 32
NUM_WORKERS = 2

# ==========================================
# TRAINING CONFIGURATION
# ==========================================
# Stage 1: Classification Head Training
STAGE1_EPOCHS = 5
STAGE1_LR = 0.001

# Stage 2: Fine-tuning
STAGE2_EPOCHS = 10
STAGE2_LR = 1e-5

# ==========================================
# LOGGING CONFIGURATION
# ==========================================
LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)

def get_logger(name, log_file=None):
    """
    Create and return a configured logger.
    
    Args:
        name: Logger name (usually __name__)
        log_file: Optional specific log file name
        
    Returns:
        logging.Logger: Configured logger instance
    """
    if log_file is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = f"{name.replace('.', '_')}_{timestamp}.log"
    
    log_path = os.path.join(LOG_DIR, log_file)
    
    logger = logging.getLogger(name)
    
    # Remove existing handlers
    if logger.hasHandlers():
        logger.handlers.clear()
    
    logger.setLevel(logging.INFO)
    
    # File handler
    file_handler = logging.FileHandler(log_path)
    file_handler.setLevel(logging.INFO)
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    
    # Formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger, log_path

# ==========================================
# MODEL CONFIGURATION
# ==========================================
MODEL_NAME = "ResNet50"
PRETRAINED = True  # Use ImageNet pretrained weights
MODEL_SAVE_PATH = "sipakmed_resnet50.pth"

# ==========================================
# LOSS FUNCTION
# ==========================================
LOSS_FUNCTION = "CrossEntropyLoss"

# ==========================================
# DEVICE
# ==========================================
import torch
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==========================================
# Print Configuration Summary
# ==========================================
def print_config():
    """Print current configuration."""
    config_summary = f"""
    ============================================
    SIPaKMeD Configuration Summary
    ============================================
    SEED: {SEED}
    
    Dataset Split:
      - Training:   {TRAIN_RATIO*100:.0f}% ({int(TRAIN_RATIO*1000)}% of total)
      - Validation: {VAL_RATIO*100:.0f}% ({int(VAL_RATIO*1000)}% of total)
      - Testing:    {TEST_RATIO*100:.0f}% ({int(TEST_RATIO*1000)}% of total)
    
    Model: {MODEL_NAME} (Pretrained: {PRETRAINED})
    Device: {DEVICE}
    
    Training:
      - Stage 1: {STAGE1_EPOCHS} epochs, LR={STAGE1_LR}
      - Stage 2: {STAGE2_EPOCHS} epochs, LR={STAGE2_LR}
    
    Batch Size: {BATCH_SIZE}
    Image Size: {IMAGE_SIZE}
    Loss Function: {LOSS_FUNCTION}
    
    Classes ({NUM_CLASSES}): {', '.join(DATASET_CLASSES)}
    ============================================
    """
    print(config_summary)

if __name__ == "__main__":
    print_config()
