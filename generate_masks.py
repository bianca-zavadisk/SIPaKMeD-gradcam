import os
import torch
import cv2
import numpy as np
from torchvision import transforms
from PIL import Image

# Importa sua UNet do arquivo UNet_model.py
from UNet_model import UNet 

# ==========================================
# 1. CAMINHOS DEFINIDOS MANUALMENTE (SEU AMBIENTE)
# ==========================================
# Caminho exato do seu cache que contém as imagens (.bmp)
BASE_DATASET_DIR = "/home/al.bianca.abreu/.cache/kagglehub/datasets/prahladmehandiratta/cervical-cancer-largest-dataset-sipakmed/versions/1"

# Onde as máscaras geradas pela UNet serão salvas
BASE_OUTPUT_DIR = os.path.join(os.getcwd(), "sipakmed_generated_masks")
OUTPUT_CELL_DIR = os.path.join(BASE_OUTPUT_DIR, "cell_masks")
OUTPUT_NUCLEUS_DIR = os.path.join(BASE_OUTPUT_DIR, "nucleus_masks")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==========================================
# 2. Carregar o Modelo U-Net
# ==========================================
print("Carregando U-Net...")
# Como vimos no seu UNet_model.py, o padding=1 mantém o tamanho de saída idêntico ao de entrada.
unet = UNet(n_class=2).to(device)
unet.load_state_dict(torch.load("unet_model.pth", map_location=device))
unet.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)), 
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# ==========================================
# 3. Processar Imagens e Gerar Máscaras
# ==========================================
classes = ['im_Koilocytotic', 'im_Superficial-Intermediate', 'im_Dyskeratotic', 'im_Parabasal', 'im_Metaplastic']

for cls in classes:
    # Resolve a estrutura duplicada do Kaggle (ex: im_Koilocytotic/im_Koilocytotic)
    class_images_dir = os.path.join(BASE_DATASET_DIR, cls, cls)
    
    if not os.path.exists(class_images_dir):
        print(f"Aviso: Caminho não encontrado: {class_images_dir}")
        continue
        
    # Cria as pastas de saída estruturadas por classe
    os.makedirs(os.path.join(OUTPUT_CELL_DIR, cls), exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_NUCLEUS_DIR, cls), exist_ok=True)
    
    # Lista apenas os arquivos de imagem válidos
    images = [f for f in os.listdir(class_images_dir) if f.lower().endswith(('.bmp', '.png', '.jpg', '.jpeg'))]
    print(f"Processando {len(images)} imagens da classe '{cls}'...")
    
    for img_name in images:
        img_path = os.path.join(class_images_dir, img_name)
        img = Image.open(img_path).convert("RGB")
        
        input_tensor = transform(img).unsqueeze(0).to(device)
        
        with torch.no_grad():
            output = unet(input_tensor)
            probs = torch.sigmoid(output).squeeze(0).cpu().numpy()
            
        # Separa os canais preditos (0: Célula, 1: Núcleo)
        cell_mask = (probs[0] > 0.5).astype(np.uint8) * 255
        nucleus_mask = (probs[1] > 0.5).astype(np.uint8) * 255
        
        # Redimensiona para o formato final que a ResNet50 usará (224x224)
        cell_mask_resized = cv2.resize(cell_mask, (224, 224), interpolation=cv2.INTER_NEAREST)
        nucleus_mask_resized = cv2.resize(nucleus_mask, (224, 224), interpolation=cv2.INTER_NEAREST)
        
        # Salva os arquivos finais
        cv2.imwrite(os.path.join(OUTPUT_CELL_DIR, cls, img_name), cell_mask_resized)
        cv2.imwrite(os.path.join(OUTPUT_NUCLEUS_DIR, cls, img_name), nucleus_mask_resized)

print("\nGeração de máscaras concluída! Verifique a pasta 'sipakmed_generated_masks'.")