# 08 — Scripts e Snippets de Referência
## Blocos de Código Prontos para Aceleração da Implementação

---

## 1. PyTorch: Dataset Customizado e DataLoader com Prevenção de Leakage

```python
import os
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import transforms

class CardioImageDataset(Dataset):
    """
    Dataset customizado para radiografias torácicas do CardioIA.
    Suporta metadados demográficos para auditoria de fairness.
    """
    def __init__(self, df, img_dir, transform=None):
        self.df = df.reset_index(drop=True)
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.img_dir, row["filename"])
        
        # Carregamento com Pillow e conversão para 3 canais RGB
        image = Image.open(img_path).convert("RGB")
        
        if self.transform:
            image = self.transform(image)
            
        label = torch.tensor(row["label"], dtype=torch.long)
        
        # Metadados opcionais para avaliação de fairness
        gender = row.get("patient_gender", "U")
        age = row.get("patient_age", 0)
        
        return image, label, {"gender": gender, "age": age, "patient_id": row.get("patient_id", "")}

# Transformações recomendadas
train_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomRotation(degrees=7),
    transforms.RandomAffine(degrees=0, translate=(0.04, 0.04), scale=(0.96, 1.04)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

val_test_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
```

---

## 2. Modelos Convolucionais: CNN do Zero vs Transfer Learning

```python
import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights

# A) CNN Construída do Zero
class CardioCNN_Scratch(nn.Module):
    def __init__(self, num_classes=2):
        super(CardioCNN_Scratch, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2), # 112x112

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2), # 56x56

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2), # 28x28

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((4, 4)) # 4x4
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 4 * 4, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        return self.classifier(self.features(x))

# B) Transfer Learning com ResNet18
def get_cardio_resnet(num_classes=2, freeze_backbone=True):
    model = resnet18(weights=ResNet18_Weights.DEFAULT)
    if freeze_backbone:
        for param in model.parameters():
            param.requires_grad = False
            
    num_ftrs = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Linear(num_ftrs, 256),
        nn.ReLU(inplace=True),
        nn.Dropout(0.35),
        nn.Linear(256, num_classes)
    )
    return model
```

---

## 3. Função de Avaliação de Métricas Clínicas

```python
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix, roc_curve
)

def evaluate_model(model, dataloader, device):
    model.eval()
    y_true, y_pred, y_probs = [], [], []
    
    with torch.no_grad():
        for images, labels, _ in dataloader:
            images = images.to(device)
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)[:, 1].cpu().numpy()
            preds = torch.argmax(outputs, dim=1).cpu().numpy()
            
            y_true.extend(labels.numpy())
            y_pred.extend(preds)
            y_probs.extend(probs)
            
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_probs = np.array(y_probs)
    
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    auc = roc_auc_score(y_true, y_probs)
    cm = confusion_matrix(y_true, y_pred)
    
    return {
        "acuracia": acc,
        "precisao": prec,
        "recall_sensibilidade": rec,
        "f1_score": f1,
        "roc_auc": auc,
        "matriz_confusao": cm,
        "y_true": y_true,
        "y_probs": y_probs
    }
```

---

## 4. Auditoria de Fairness e Métricas de Justiça

```python
def compute_fairness_audit(y_true, y_pred, sensitive_attributes):
    """
    sensitive_attributes: array com valores como 'M' ou 'F'.
    """
    subgroups = np.unique(sensitive_attributes)
    audit = {}
    
    for grp in subgroups:
        mask = (sensitive_attributes == grp)
        rec = recall_score(y_true[mask], y_pred[mask], zero_division=0)
        prec = precision_score(y_true[mask], y_pred[mask], zero_division=0)
        pos_rate = np.mean(y_pred[mask])
        
        audit[grp] = {
            "total_amostras": int(np.sum(mask)),
            "recall": rec,
            "precisao": prec,
            "taxa_predicao_positiva": pos_rate
        }
        
    # Disparate Impact Ratio (DIR)
    grp_a, grp_b = subgroups[0], subgroups[1]
    dir_ratio = audit[grp_a]["taxa_predicao_positiva"] / max(1e-5, audit[grp_b]["taxa_predicao_positiva"])
    equal_opportunity_diff = abs(audit[grp_a]["recall"] - audit[grp_b]["recall"])
    
    return {
        "subgrupos": audit,
        "disparate_impact_ratio": dir_ratio,
        "equal_opportunity_difference": equal_opportunity_diff
    }
```

---

## 5. Flask API: Manipulação de Imagem e Inferência

```python
import io
from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image
import torch
import torchvision.transforms as transforms

app = Flask(__name__)
CORS(app)

# Carrega modelo e define transformações
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = None # Inicializar no startup

infer_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

@app.route("/api/v1/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"erro": "Nenhum arquivo enviado no campo 'file'"}), 400
        
    file = request.files["file"]
    try:
        image = Image.open(io.BytesIO(file.read())).convert("RGB")
        tensor = infer_transform(image).unsqueeze(0).to(device)
        
        with torch.no_grad():
            outputs = model(tensor)
            probs = torch.softmax(outputs, dim=1)[0].cpu().numpy()
            
        prob_cardiomegalia = float(probs[1])
        diag = "Cardiomegalia" if prob_cardiomegalia >= 0.50 else "Silhueta Cardíaca Normal"
        
        return jsonify({
            "diagnostico": diag,
            "probabilidade": prob_cardiomegalia,
            "confianca_pct": round(prob_cardiomegalia * 100, 2),
            "status": "ALERTA_CRITICO" if prob_cardiomegalia >= 0.50 else "NORMAL"
        })
    except Exception as e:
        return jsonify({"erro": str(e)}), 500
```
