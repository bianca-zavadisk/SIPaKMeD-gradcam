import torch
import torch.nn as nn
from torchvision import transforms, models
from torch.utils.data import DataLoader, random_split
import kagglehub
import os
import glob
from torch.utils.data import Dataset
from PIL import Image

# ==========================================
# 1. Definição do Dataset
# ==========================================
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
                self.samples.append((img_path, self.class_to_idx[cls]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, label

# ==========================================
# 2. Configuração do Dataset e DataLoader
# ==========================================
print("Baixando/Verificando dataset SIPaKMeD...")
data_dir = kagglehub.dataset_download("prahladmehandiratta/cervical-cancer-largest-dataset-sipakmed")

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

dataset = SipakMedCroppedDataset(data_dir, transform=transform)
num_classes = len(dataset.classes)

# Refazendo a divisão para obter o conjunto de treino (80%)
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size

# DICA: Para garantir consistência em testes futuros, recomenda-se usar uma seed fixa no script de treino e avaliação.
# ex: torch.manual_seed(42)
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

# shuffle=False é suficiente pois estamos apenas avaliando
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=0)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==========================================
# 3. Inicialização e Carregamento do Modelo
# ==========================================
print("Carregando o modelo salvo...")
model = models.resnet50() # Não precisamos dos pesos do ImageNet aqui, pois vamos carregar do arquivo

# Ajustar a última camada para o número de classes do nosso dataset
num_features = model.fc.in_features
model.fc = nn.Linear(num_features, num_classes)

# Carregar os pesos salvos localmente
save_path = "sipakmed_resnet50.pth"
model.load_state_dict(torch.load(save_path, map_location=device))
model = model.to(device)

criterion = nn.CrossEntropyLoss()

# ==========================================
# 4. Função de Avaliação
# ==========================================
from tqdm.auto import tqdm

def evaluate_model(model, dataloader, criterion):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    print(f"Iniciando avaliação no conjunto de validação usando: {device}...")
    
    with torch.no_grad():
        # Envolvemos o dataloader no tqdm para visualizar o progresso
        progress_bar = tqdm(dataloader, desc="Avaliando Validação")
        
        for inputs, labels in progress_bar:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item() * inputs.size(0)
            
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            # Atualiza a barra de progresso com a acurácia e erro em tempo real
            progress_bar.set_postfix(loss=f"{loss.item():.4f}", acc=f"{(correct/total):.4f}")
            
    return running_loss / len(dataloader.dataset), correct / total
# ==========================================
# 5. Execução e Exibição de Resultados
# ==========================================
if __name__ == "__main__":
    val_loss, val_acc = evaluate_model(model, val_loader, criterion)
    
    print("\n" + "="*40)
    print("RESULTADOS DA AVALIAÇÃO (CONJUNTO DE VALIDAÇÃO)")
    print("="*40)
    print(f"Loss (Custo):  {val_loss:.4f}")
    print(f"Accuracy:      {val_acc:.4f} ({(val_acc*100):.2f}%)")
    print("="*40)