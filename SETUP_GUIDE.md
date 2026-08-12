# SIPaKMeD Training and Evaluation - Configuração

## ✨ Novas Funcionalidades Adicionadas

### 1. **Configuração de SEED (Reproduzibilidade)**

A seed foi configurada para garantir **reproduzibilidade completa** dos resultados:

```python
SEED = 42

torch.manual_seed(SEED)
np.random.seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
```

**O que isso significa:**
- Os mesmos valores de seed produzem os mesmos resultados sempre
- Importante para pesquisa, debugging e compartilhamento de resultados
- Pode ser alterado em `config.py` se necessário

---

### 2. **Dataset de Teste (Train/Val/Test Split)**

O dataset agora é dividido em **três conjuntos**:

| Conjunto | Proporção | Uso |
|----------|-----------|-----|
| **Training** | 70% | Treinar o modelo |
| **Validation** | 15% | Validar durante o treinamento |
| **Testing** | 15% | Avaliar performance final |

**Código relevante:**
```python
train_size = int(0.70 * len(dataset))
val_size = int(0.15 * len(dataset))
test_size = len(dataset) - train_size - val_size

train_dataset, val_dataset, test_dataset = random_split(
    dataset, 
    [train_size, val_size, test_size]
)
```

---

### 3. **Sistema de Logging Completo**

Agora todos os eventos de treinamento são registrados em **arquivos de log**:

#### Estrutura de Logs
```
logs/
├── training_20240812_143025.log          # Log do treinamento
└── test_evaluation_20240812_150530.log   # Log da avaliação de teste
```

#### O que é registrado:
- ✅ Seed utilizado
- ✅ Tamanho do dataset
- ✅ Configuração do dispositivo (CPU/CUDA)
- ✅ Parâmetros do otimizador
- ✅ Loss e acurácia em cada época
- ✅ Resultados finais do teste
- ✅ Relatório detalhado de classificação

---

## 🚀 Como Usar

### 1. **Treinar o Modelo**

```bash
python train_classifier.py
```

**O que acontece:**
- Carrega o dataset SIPaKMeD
- Configura seed = 42 automaticamente
- Divide dados em Train/Val/Test
- Treina em 2 estágios
- Avalia no conjunto de teste
- Salva logs em `logs/training_TIMESTAMP.log`

---

### 2. **Avaliar Apenas o Conjunto de Teste**

```bash
python evaluate_test.py
```

**Gera:**
- Log detalhado: `logs/test_evaluation_TIMESTAMP.log`
- Relatório JSON: `logs/test_report_TIMESTAMP.json`
- Métricas: accuracy, precision, recall, F1-score por classe
- Matriz de confusão

---

### 3. **Personalizar Configurações**

Edite `config.py` para alterar:

```python
# Seed
SEED = 42  # Altere para outro valor se desejar

# Dataset split
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

# Treinamento
STAGE1_EPOCHS = 5
STAGE1_LR = 0.001
STAGE2_EPOCHS = 10
STAGE2_LR = 1e-5

# Batch size
BATCH_SIZE = 32
```

---

## 📊 Arquivos de Log

### Exemplo de Log de Treinamento

```
2024-08-12 14:30:25 - __main__ - INFO - === Training Session Started at 20240812_143025 ===
2024-08-12 14:30:25 - __main__ - INFO - SEED: 42
2024-08-12 14:30:26 - __main__ - INFO - Total samples: 966
2024-08-12 14:30:26 - __main__ - INFO - Number of classes: 5
2024-08-12 14:30:26 - __main__ - INFO - Train set size: 676
2024-08-12 14:30:26 - __main__ - INFO - Validation set size: 145
2024-08-12 14:30:26 - __main__ - INFO - Test set size: 145
2024-08-12 14:35:42 - __main__ - INFO - Stage 1 - Epoch 1/5 | Train Loss: 1.2345 | Val Loss: 1.0234 | Val Acc: 0.6345
...
2024-08-12 14:52:15 - __main__ - INFO - Test Set Results | Loss: 0.8567 | Accuracy: 0.7234
```

### Exemplo de Relatório de Teste (JSON)

```json
{
    "timestamp": "20240812_143025",
    "seed": 42,
    "test_loss": 0.8567,
    "test_accuracy": 0.7234,
    "test_set_size": 145,
    "classes": ["im_Koilocytotic", "im_Superficial-Intermediate", ...],
    "classification_report": "precision    recall  f1-score   support\n...",
    "confusion_matrix": [[...], [...], ...]
}
```

---

## 📁 Estrutura de Arquivos Após Treinamento

```
SIPaKMeD-gradcam/
├── train_classifier.py              ✏️ Modificado (seed, logging, test split)
├── evaluate_test.py                 ✨ Novo (avaliação detalhada)
├── config.py                        ✨ Novo (configurações centralizadas)
├── logs/                            ✨ Novo (diretório)
│   ├── training_20240812_143025.log
│   ├── test_evaluation_20240812_150530.log
│   └── test_report_20240812_150530.json
├── sipakmed_resnet50.pth            (modelo treinado)
└── ...
```

---

## 🔍 Verificando Reproduzibilidade

Para verificar se a seed está funcionando corretamente:

```bash
# Primeira execução
python train_classifier.py

# Segunda execução
python train_classifier.py
```

Com a mesma seed, ambas execuções devem ter:
- Mesmos valores de loss em cada época
- Mesma acurácia final
- Mesmos pesos do modelo (comparar com diff ou hash)

---

## 💡 Dicas

1. **Alterar seed para exploração:** Treine com diferentes seeds para ver variabilidade natural
2. **Analisar logs:** Use `grep`, `cat` ou editores de texto para revisar logs
3. **Comparar resultados:** Salve logs de diferentes execuções para comparar performance
4. **Backup:** Copie logs importantes antes de sobrescrever (usam timestamp automático)

---

## 📝 Resumo das Mudanças

| Arquivo | Mudança | Benefício |
|---------|---------|-----------|
| `train_classifier.py` | + Seed, logging, test split | Reproduzibilidade e rastreabilidade |
| `evaluate_test.py` | ✨ Novo | Avaliação detalhada em dataset separado |
| `config.py` | ✨ Novo | Configurações centralizadas e reutilizáveis |
| `logs/` | ✨ Novo | Registro permanente de treinamentos |

---

**Desenvolvido com ❤️ para reproducibilidade e rastreabilidade**
