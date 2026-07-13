import os
import torch
import torch.nn as nn
from torchvision import models, transforms
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import glob
import kagglehub

# Import Grad-CAM from the jacobgil library
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image

plt.rcParams.update({
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.labelsize": 10,
})

# ==========================================
# 1. Carregar o Modelo Treinado (5 Canais)
# ==========================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

num_classes = 5 
model = models.resnet50(weights=None)

# Modificar a conv1 para aceitar 5 canais
old_conv1 = model.conv1
model.conv1 = nn.Conv2d(in_channels=5, 
                        out_channels=old_conv1.out_channels, 
                        kernel_size=old_conv1.kernel_size, 
                        stride=old_conv1.stride, 
                        padding=old_conv1.padding, 
                        bias=False)

num_features = model.fc.in_features
model.fc = nn.Linear(num_features, num_classes)

# Carregue os pesos que você salvou no fine-tuning de 5 canais
model.load_state_dict(torch.load("sipakmed_resnet50_5channels_best.pth", map_location=device))
model = model.to(device)
model.eval()

# ==========================================
# 2. Configurar o Grad-CAM e Transformações
# ==========================================
target_layers = [model.layer4[-1]]
cam = GradCAM(model=model, target_layers=target_layers)

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

def load_and_prep_mask(path):
    mask = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        return np.zeros((224, 224), dtype=np.uint8) # Prevenção de erro
    mask = cv2.resize(mask, (224, 224))
    _, mask = cv2.threshold(mask, 127, 1, cv2.THRESH_BINARY)
    return mask

def read_dat_file(dat_path):
    coords = []
    with open(dat_path, "r") as f:
        for line in f:
            line = line.strip()

            if line == "" or line.startswith("#"):
                continue

            x, y = line.split(",")
            coords.append((int(float(x)), int(float(y))))

    return coords

def generate_mask_from_contours(height, width, nucleus_coords, cytoplasm_coords):

    nucleus_mask = np.zeros((height, width), dtype=np.uint8)
    cytoplasm_mask = np.zeros((height, width), dtype=np.uint8)

    cv2.fillPoly(
        nucleus_mask,
        [np.array(nucleus_coords, dtype=np.int32)],
        1,
    )

    cv2.fillPoly(
        cytoplasm_mask,
        [np.array(cytoplasm_coords, dtype=np.int32)],
        1,
    )

    cytoplasm_mask = cytoplasm_mask - nucleus_mask
    cytoplasm_mask[cytoplasm_mask < 0] = 0

    return nucleus_mask, cytoplasm_mask

# ==========================================
# 3. Mapear as Imagens (5ª de cada classe)
# ==========================================
BASE_DATASET_DIR = "/home/al.bianca.abreu/.cache/kagglehub/datasets/prahladmehandiratta/cervical-cancer-largest-dataset-sipakmed/versions/1"
BASE_OUTPUT_DIR = os.path.join(os.getcwd(), "sipakmed_generated_masks")

classes = ['im_Koilocytotic', 'im_Superficial-Intermediate', 'im_Dyskeratotic', 'im_Parabasal', 'im_Metaplastic']

# Configurando o Plot com 5 linhas (classes) e 5 colunas (visualizações)
fig, axes = plt.subplots(
    5,
    5,
    figsize=(17,15),
    constrained_layout=True
)

titles = [
    "Original",
    "Ground Truth",
    "Predicted\nSegmentation",
    "Grad-CAM",
    "Overlay",
]

for j, t in enumerate(titles):
    axes[0, j].set_title(t, fontsize=10)

for i, cls in enumerate(classes):
    print(f"\n--- Processando Classe: {cls} ---")
    
    possible = glob.glob(
        os.path.join(BASE_DATASET_DIR, cls, "**", "CROPPED"),
        recursive=True,
    )

    pasta_imagens = possible[0]
    
    # Pega automaticamente todos os nomes e seleciona a 5ª imagem (índice 4)
    todas_imagens = sorted([f for f in os.listdir(pasta_imagens) if f.lower().endswith(('.bmp', '.png', '.jpg', '.jpeg'))])
    nome_imagem = todas_imagens[4]
    print(f"Arquivo selecionado: {nome_imagem}")
    
    # Constrói os caminhos
    image_path = os.path.join(pasta_imagens, nome_imagem)

    base = os.path.splitext(image_path)[0]

    cyt_path = base + "_cyt.dat"
    nuc_path = base + "_nuc.dat"

    img = Image.open(image_path)

    width, height = img.size

    nucleus_coords = read_dat_file(nuc_path)
    cytoplasm_coords = read_dat_file(cyt_path)

    gt_nucleus, gt_cell = generate_mask_from_contours(
        height,
        width,
        nucleus_coords,
        cytoplasm_coords,
    )

    gt_nucleus = cv2.resize(
        gt_nucleus,
        (224,224),
        interpolation=cv2.INTER_NEAREST
    )

    gt_cell = cv2.resize(
        gt_cell,
        (224,224),
        interpolation=cv2.INTER_NEAREST
    )

    cell_mask_path = os.path.join(
        BASE_OUTPUT_DIR,
        "cell_masks",
        cls,
        nome_imagem,
    )

    nucleus_mask_path = os.path.join(
        BASE_OUTPUT_DIR,
        "nucleus_masks",
        cls,
        nome_imagem,
    )

    # Preparar a imagem original e máscaras preditas para o modelo de 5 canais
    pil_image = Image.open(image_path).convert('RGB')
    pil_cell = Image.open(cell_mask_path).convert('L')
    pil_nucleus = Image.open(nucleus_mask_path).convert('L')
    
    t_image = transforms_dict['image'](pil_image)
    t_cell = transforms_dict['mask'](pil_cell)
    t_nucleus = transforms_dict['mask'](pil_nucleus)
    
    t_cell_processed = (t_cell > 0.5).float()
    t_nucleus_processed = (t_nucleus > 0.5).float()
    
    # Concatena em um tensor de 5 canais
    combined_input = torch.cat((t_image, t_cell_processed, t_nucleus_processed), dim=0)
    input_tensor = combined_input.unsqueeze(0).to(device)

    # Preparar a imagem original para visualização
    rgb_img = cv2.imread(image_path, 1)[:, :, ::-1] # BGR to RGB
    rgb_img = cv2.resize(rgb_img, (224, 224))
    rgb_img = np.float32(rgb_img) / 255

    ground_truth = np.zeros_like(rgb_img)

    ground_truth[gt_cell == 1] = [0,0,1]
    ground_truth[gt_nucleus == 1] = [1,0,0]

    # Carregar as máscaras da célula e do núcleo
    cell_mask = load_and_prep_mask(cell_mask_path)
    nucleus_mask = load_and_prep_mask(nucleus_mask_path)

    # Gerar o Grad-CAM Heatmap
    grayscale_cam = cam(input_tensor=input_tensor, targets=None)
    grayscale_cam = grayscale_cam[0, :]

    # Criar a sobreposição do Grad-CAM na imagem
    visualization = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)

    # Calcular as métricas de Sobreposição (Overlap) usando a máscara predita (como foi passada pro modelo)
    cam_threshold = 0.7 
    attention_mask = (grayscale_cam >= cam_threshold).astype(np.uint8)
    attention_area = np.sum(attention_mask)

    if attention_area > 0:
        cell_overlap = np.sum(attention_mask & cell_mask) / attention_area
        nucleus_overlap = np.sum(attention_mask & nucleus_mask) / attention_area
    else:
        cell_overlap, nucleus_overlap = 0.0, 0.0

    print(f"  % de alta atenção na Célula Predita (Azul): {cell_overlap * 100:.2f}%")
    print(f"  % de alta atenção no Núcleo Predito (Vermelho): {nucleus_overlap * 100:.2f}%")

    # Juntar a máscara predita (célula e núcleo) em uma imagem RGB para visualizar
    combined_mask = np.zeros_like(rgb_img)
    combined_mask[cell_mask == 1] = [0, 0, 1]  # Azul = Célula Predita
    combined_mask[nucleus_mask == 1] = [1, 0, 0]  # Vermelho = Núcleo Predito

    # ==========================================
    # 4. Adicionar ao Subplot na linha 'i'
    # ==========================================
    nome_limpo_classe = cls.replace("im_", "") # Limpa o nome para o plot ficar mais bonito
    
    # Row label
    axes[i,0].imshow(rgb_img)
    axes[i,0].axis("off")
    axes[i,0].set_ylabel(
        nome_limpo_classe,
        fontsize=10,
        rotation=90,
        fontweight="bold",
        labelpad=15,
    )

    # Ground truth
    axes[i,1].imshow(ground_truth)
    axes[i,1].axis("off")

    # Predicted segmentation
    axes[i,2].imshow(combined_mask)
    axes[i,2].axis("off")

    # GradCAM
    axes[i,3].imshow(grayscale_cam, cmap="jet")
    axes[i,3].axis("off")

    # Overlay
    axes[i,4].imshow(visualization)
    axes[i,4].axis("off")
    
os.makedirs("visualizations", exist_ok=True)

save_path = os.path.join("visualizations", "gradcam_resnet50_5channels.png")

plt.savefig(save_path, dpi=300, bbox_inches="tight")
print(f"Figure saved to {save_path}")

plt.show()