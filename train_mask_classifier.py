import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import models, transforms
from torchvision.models import ResNet50_Weights
from PIL import Image
from tqdm import tqdm
import glob

# ==========================================
# 1. Custom Dataset Adaptado para a Estrutura do Seu Cache
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

                    self.image_paths.append(
                        (cls_name, cls_dir, img_name)
                    )

                    self.targets.append(
                        self.class_to_idx[cls_name]
                    )

    def __len__(self):
        return len(self.image_paths)
        
    def __getitem__(self, idx):
        cls_name, cls_dir, img_name = self.image_paths[idx]
        label = self.targets[idx]
        
        img_path = os.path.join(
            cls_dir,
            img_name
        )
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
# 2. Configuração dos Caminhos Exatos
# ==========================================
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

print("Carregando dataset de 5 canais...")
dataset = SIPaKMeDMaskDataset(IMAGES_DIR, CELL_MASKS_DIR, NUCLEUS_MASKS_DIR, transforms_dict)
num_classes = len(dataset.classes)

train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=2)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=2)

# ==========================================
# 3. Modificar ResNet50 para 5 Canais
# ==========================================
print("Inicializando ResNet50 modificada...")
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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

criterion = nn.CrossEntropyLoss()

# ==========================================
# 4. FASE 1: Loop de Treinamento (Warmup)
# ==========================================
# Congela o corpo da ResNet, treina apenas conv1 e fc
for name, param in model.named_parameters():
    if 'conv1' in name or 'fc' in name:
        param.requires_grad = True
    else:
        param.requires_grad = False

optimizer_warmup = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.001)

warmup_epochs = 3
print(f"\n[FASE 1] Iniciando treinamento de warmup (5 épocas) em: {device}...")

for epoch in range(warmup_epochs):
    model.train()
    running_loss, correct, total = 0.0, 0, 0
    
    progress_bar = tqdm(train_loader, desc=f"Warmup Epoch {epoch+1}/{warmup_epochs}", leave=True)
    for inputs, labels in progress_bar:
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer_warmup.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer_warmup.step()
        
        running_loss += loss.item() * inputs.size(0)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
        progress_bar.set_postfix(batch_loss=f"{loss.item():.4f}", running_acc=f"{(correct/total):.4f}")
        
    print(f"-> Resumo Warmup Ep {epoch+1} | Loss: {(running_loss/len(train_dataset)):.4f} | Acc: {(correct/total):.4f}\n")

print("Warmup concluído com sucesso!")

# ==========================================
# 5. FASE 2: Loop de Treinamento (Fine-Tuning Global)
# ==========================================
print("\n" + "="*50)
print("[FASE 2] Inicializando Fine-Tuning de Todo o Modelo...")
print("="*50)

# 1. DESCONGELAR TODAS AS CAMADAS DO MODELO
for param in model.parameters():
    param.requires_grad = True

# 2. DEFINIR UMA TAXA DE APRENDIZADO MUITO BAIXA (1e-5) para ajustes cirúrgicos
optimizer_finetune = optim.Adam(model.parameters(), lr=1e-5)

fine_tune_epochs = 5
best_val_acc = 0.0

for epoch in range(fine_tune_epochs):
    # --- Passo de Treino ---
    model.train()
    train_loss, train_correct, train_total = 0.0, 0, 0
    
    progress_bar = tqdm(train_loader, desc=f"Fine-Tune Epoch {epoch+1}/{fine_tune_epochs}", leave=True)
    for inputs, labels in progress_bar:
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer_finetune.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer_finetune.step()
        
        train_loss += loss.item() * inputs.size(0)
        _, predicted = torch.max(outputs, 1)
        train_total += labels.size(0)
        train_correct += (predicted == labels).sum().item()
        
        progress_bar.set_postfix(loss=f"{loss.item():.4f}", acc=f"{(train_correct/train_total):.4f}")
        
    epoch_train_loss = train_loss / len(train_dataset)
    epoch_train_acc = train_correct / train_total
    
    # --- Passo de Validação (Obrigatório no Fine-Tuning para monitorar Overfitting) ---
    model.eval()
    val_loss, val_correct, val_total = 0.0, 0, 0
    
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            val_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            val_total += labels.size(0)
            val_correct += (predicted == labels).sum().item()
            
    epoch_val_loss = val_loss / len(val_dataset)
    epoch_val_acc = val_correct / val_total
    
    print(f"-> FIM DA ÉPOCA {epoch+1}/{fine_tune_epochs}")
    print(f"   [TREINO] Loss: {epoch_train_loss:.4f} | Acurácia: {epoch_train_acc:.4f}")
    print(f"   [VALIDAÇÃO] Loss: {epoch_val_loss:.4f} | Acurácia: {epoch_val_acc:.4f}")
    
    # Salva os pesos se a acurácia de validação bater o recorde anterior
    if epoch_val_acc > best_val_acc:
        best_val_acc = epoch_val_acc
        torch.save(model.state_dict(), "sipakmed_resnet50_5channels_best.pth")
        print("   ⭐ Melhoria detectada! Pesos salvos em 'sipakmed_resnet50_5channels_best.pth'")
    print("-" * 50 + "\n")

print(f"Treinamento Completo! Melhor Acurácia de Validação alcançada: {best_val_acc:.4f}")