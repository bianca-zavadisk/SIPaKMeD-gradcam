import os
import kagglehub
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import models, transforms
from PIL import Image
import glob
from tqdm.auto import tqdm

# ==========================================
# 1. Custom Dataset (5 Canais)
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
                print(f"CROPPED folder not found for {cls_name}")
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
# 2. Configuração e DataLoader
# ==========================================
IMAGES_DIR = kagglehub.dataset_download("prahladmehandiratta/cervical-cancer-largest-dataset-sipakmed")
    
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

print("Carregando dataset de 5 canais...")
dataset = SIPaKMeDMaskDataset(IMAGES_DIR, CELL_MASKS_DIR, NUCLEUS_MASKS_DIR, transforms_dict)
num_classes = len(dataset.classes)

train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

# Configurando DataLoader de Validação com num_workers=0 para evitar loops/deadlocks
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=0)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==========================================
# 3. Recriação do Modelo (5 Canais) e Carregamento dos Pesos
# ==========================================
print("Recriando a arquitetura ResNet50 modificada...")
model = models.resnet50()

old_conv1 = model.conv1
new_conv1 = nn.Conv2d(
    in_channels=5, 
    out_channels=old_conv1.out_channels, 
    kernel_size=old_conv1.kernel_size, 
    stride=old_conv1.stride, 
    padding=old_conv1.padding, 
    bias=False
)
model.conv1 = new_conv1

num_features = model.fc.in_features
model.fc = nn.Linear(num_features, num_classes)

# Carregando os pesos do melhor modelo salvo
model_path = "sipakmed_resnet50_5channels_best.pth"
print(f"Carregando pesos salvos de '{model_path}'...")
model.load_state_dict(torch.load(model_path, map_location=device))
model = model.to(device)

criterion = nn.CrossEntropyLoss()

# ==========================================
# 4. Avaliação
# ==========================================
def evaluate_model(model, dataloader, criterion):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    print(f"Iniciando avaliação no conjunto de validação usando: {device}...")
    
    with torch.no_grad():
        progress_bar = tqdm(dataloader, desc="Avaliando Validação")
        
        for inputs, labels in progress_bar:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item() * inputs.size(0)
            
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            progress_bar.set_postfix(loss=f"{loss.item():.4f}", acc=f"{(correct/total):.4f}")
            
    return running_loss / len(dataloader.dataset), correct / total

if __name__ == "__main__":
    # Aqui passamos o val_loader em vez do train_loader
    val_loss, val_acc = evaluate_model(model, val_loader, criterion)
    
    print("\n" + "="*50)
    print("RESULTADOS DA AVALIAÇÃO COM MÁSCARAS (VALIDAÇÃO)")
    print("="*50)
    print(f"Loss (Custo):  {val_loss:.4f}")
    print(f"Accuracy:      {val_acc:.4f} ({(val_acc*100):.2f}%)")
    print("="*50)