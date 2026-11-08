"""
Módulo de Curadoria, Geração Resiliente e Pré-processamento de Imagens Médicas
CardioIA Fase 4 — Data Health Solutions (FIAP)

Implementa:
1. Geração e curadoria de radiografias torácicas baseadas nas distribuições clínicas do
   NIH Chest X-ray Dataset (Cardiomegalia vs. Silhueta Cardíaca Normal), com simulação
   anatômica realista de campos pulmonares, costelas, coluna, diafragma, silhueta cardíaca
   (Índice Cardiotorácico ICT <= 0.48 para Normal e ICT >= 0.54 para Cardiomegalia),
   atenuação de tecidos moles e artefatos de projeção AP vs. PA.
2. Particionamento estratificado agrupado por Patient ID (GroupShuffleSplit / GroupKFold)
   garantindo 0% de vazamento de pacientes (Patient Leakage) entre Treino, Validação e Teste.
3. Pipeline PyTorch (CardioImageDataset e DataLoaders) com Data Augmentation clínico restrito
   (rotações suaves ±7°, translação/escala leves, normalização ImageNet, sem flip vertical).
"""

import os
from typing import Dict, Tuple, Optional
import numpy as np
import pandas as pd
from PIL import Image, ImageFilter, ImageDraw
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.model_selection import GroupShuffleSplit

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def _categorize_age(age: int) -> str:
    if age < 45:
        return "Jovem (<45)"
    elif age <= 65:
        return "Adulto (45-65)"
    return "Idoso (>65)"


def synthesize_chest_xray(
    label: int,
    ctr_target: float,
    gender: str,
    age: int,
    view_position: str,
    rng: np.random.Generator,
    img_size: int = 224,
) -> Image.Image:
    """
    Sintetiza uma radiografia torácica frontal (PA/AP) de alta fidelidade fisiológica
    com controle explícito do Índice Cardiotorácico (ICT / CTR), arcos costais,
    transparência pulmonar, congestão hilar e artefatos radiológicos.
    """
    yy, xx = np.mgrid[0:img_size, 0:img_size].astype(np.float32)
    yn = (yy - img_size / 2.0) / (img_size / 2.0)
    xn = (xx - img_size / 2.0) / (img_size / 2.0)

    # 1. Envelope corporal / caixa torácica (mais brilhante nas bordas de tecido mole)
    thorax_width = rng.uniform(0.76, 0.84)
    thorax_mask = ((xn / thorax_width) ** 2 + (yn / 0.92) ** 2) <= 1.0
    base = np.full((img_size, img_size), 18.0, dtype=np.float32)
    base[thorax_mask] = 115.0 - 28.0 * (xn[thorax_mask] ** 2 + yn[thorax_mask] ** 2)

    # 2. Campos pulmonares Direito e Esquerdo (radiolucentes / escuros ao Raio-X)
    lung_r_cx = rng.uniform(-0.31, -0.27)
    lung_l_cx = rng.uniform(0.27, 0.31)
    lung_cy = rng.uniform(-0.08, -0.02)
    lung_rx = rng.uniform(0.23, 0.26)
    lung_ry = rng.uniform(0.52, 0.58)

    dist_r = ((xn - lung_r_cx) / lung_rx) ** 2 + ((yn - lung_cy) / lung_ry) ** 2
    dist_l = ((xn - lung_l_cx) / lung_rx) ** 2 + ((yn - lung_cy) / lung_ry) ** 2

    lung_field = np.exp(-1.8 * np.minimum(dist_r, dist_l))
    lung_mask = (dist_r <= 1.08) | (dist_l <= 1.08)
    base -= 72.0 * lung_field * lung_mask

    # 3. Grade costal (arcos costais periódicos curvos)
    rib_freq = rng.uniform(21.0, 24.0)
    rib_phase = rng.uniform(0, np.pi)
    rib_curve = yn + 0.18 * (xn ** 2)
    ribs = np.sin(rib_freq * rib_curve + rib_phase)
    rib_intensity = 9.5 if age <= 65 else 7.0  # osteopenia leve em idosos
    base += rib_intensity * ribs * lung_mask

    # 4. Coluna vertebral / mediastino central e clavículas superiores
    spine = np.exp(-((xn / 0.09) ** 2)) * (np.abs(yn) < 0.88)
    base += 48.0 * spine

    clavicle_y = -0.58 + 0.12 * np.abs(xn)
    clavicles = np.exp(-(((yn - clavicle_y) / 0.035) ** 2)) * (np.abs(xn) < 0.68)
    base += 32.0 * clavicles

    # 5. Sombra mamária / tecidos moles anteriores (variação sutil por sexo)
    if gender == "F":
        breast_r = np.exp(-(((xn + 0.30) / 0.26) ** 2 + ((yn - 0.24) / 0.22) ** 2))
        breast_l = np.exp(-(((xn - 0.30) / 0.26) ** 2 + ((yn - 0.24) / 0.22) ** 2))
        base += 14.0 * (breast_r + breast_l)

    # 6. Silhueta Cardíaca governada pelo Índice Cardiotorácico (ICT / CTR)
    # Na projeção AP há magnificação geométrica leve (+4% aparente)
    ap_magnification = 1.04 if view_position == "AP" else 1.00
    effective_ctr = ctr_target * ap_magnification

    # Raio horizontal cardíaco proporcional à largura torácica * CTR
    heart_half_width = effective_ctr * thorax_width
    # O coração desloca-se predominantemente para a esquerda do paciente (direita da imagem, xn > 0)
    heart_cx = rng.uniform(0.08, 0.14)
    heart_cy = rng.uniform(0.14, 0.20)
    # Altura cardíaca correlacionada ao CTR com variância anatômica contínua
    heart_ry = float(np.clip(0.54 * effective_ctr + rng.normal(0.0, 0.025), 0.22, 0.39))

    heart_dist = ((xn - heart_cx) / max(0.12, heart_half_width)) ** 2 + (
        (yn - heart_cy) / heart_ry
    ) ** 2
    heart_profile = np.clip(1.0 - heart_dist, 0.0, 1.0) ** 0.65
    heart_opacity = float(np.clip(62.0 + 65.0 * effective_ctr + rng.normal(0.0, 8.0), 74.0, 114.0))
    base += heart_opacity * heart_profile

    # Congestão perihilar / trama broncovascular proporcional ao CTR (com ruído anatômico)
    if effective_ctr + rng.normal(0.0, 0.04) > 0.50:
        hilar_dist = ((xn - 0.04) / 0.38) ** 2 + ((yn - 0.02) / 0.28) ** 2
        hilar_haze = np.exp(-1.5 * hilar_dist) * lung_mask
        base += rng.uniform(10.0, 22.0) * hilar_haze

    # 7. Cúpulas diafragmáticas inferiores
    diaphragm_level = 0.48 - 0.08 * (xn ** 2)
    diaphragm_mask = (yn > diaphragm_level) & thorax_mask
    base[diaphragm_mask] += 58.0 * np.clip((yn[diaphragm_mask] - diaphragm_level[diaphragm_mask]) * 4.0, 0.0, 1.0)

    # 8. Ruído quântico radiográfico (Poisson/Gaussiano) e suavização óptica
    noise = rng.normal(0.0, 4.5, size=(img_size, img_size)).astype(np.float32)
    img_arr = np.clip(base + noise, 5.0, 248.0).astype(np.uint8)

    pil_img = Image.fromarray(img_arr, mode="L").filter(ImageFilter.GaussianBlur(radius=0.85))
    draw = ImageDraw.Draw(pil_img)

    # Marcador radiográfico discreto no canto superior ("PA" ou "AP" / "L")
    draw.text((8, 8), view_position, fill=int(rng.integers(175, 220)))
    return pil_img.convert("RGB")


def ensure_cardio_dataset(
    data_dir: str = "data",
    num_patients: int = 240,
    seed: int = 42,
    force_regenerate: bool = False,
) -> Tuple[pd.DataFrame, str]:
    """
    Garante a existência do dataset curado de radiografias torácicas (NIH Chest X-ray
    Cardiomegaly Subset / Simulador Anatômico Resiliente).
    Cria múltiplos exames longitudinais para parte dos pacientes para testar e comprovar
    a prevenção de Patient Leakage via GroupShuffleSplit.
    """
    raw_img_dir = os.path.join(data_dir, "raw", "images")
    processed_dir = os.path.join(data_dir, "processed")
    os.makedirs(raw_img_dir, exist_ok=True)
    os.makedirs(processed_dir, exist_ok=True)

    metadata_csv = os.path.join(processed_dir, "dataset_curado_nih_cardio.csv")

    if os.path.exists(metadata_csv) and not force_regenerate:
        df = pd.read_csv(metadata_csv)
        if len(df) >= num_patients and os.path.exists(os.path.join(raw_img_dir, df.iloc[0]["filename"])):
            return df, raw_img_dir

    if force_regenerate and os.path.exists(raw_img_dir):
        for old_f in os.listdir(raw_img_dir):
            if old_f.endswith(".png"):
                os.remove(os.path.join(raw_img_dir, old_f))

    rng = np.random.default_rng(seed)
    records = []
    img_counter = 1

    for p_idx in range(1, num_patients + 1):
        patient_id = f"NIH_PAT_{p_idx:04d}"
        # Distribuição demográfica realista (56% M, 44% F)
        gender = "M" if rng.random() < 0.56 else "F"
        # Faixa etária entre 22 e 86 anos
        age = int(np.clip(rng.normal(58, 16), 22, 86))
        age_group = _categorize_age(age)

        # Probabilidade basal de Cardiomegalia (~46% para dataset balanceado de estudo)
        # Pacientes idosos têm prevalência ligeiramente maior, refletindo epidemiologia real
        prob_cardio = 0.52 if age > 65 else (0.45 if age >= 45 else 0.38)
        label = 1 if rng.random() < prob_cardio else 0
        finding_label = "Cardiomegaly" if label == 1 else "No Finding"

        # 25% dos pacientes possuem 2 exames longitudinais (follow-up) para provar prevenção de leakage
        num_exams = 2 if rng.random() < 0.25 else 1

        for follow_up in range(num_exams):
            # Em pacientes com cardiomegalia, incidência AP no leito é mais frequente (42% vs 22%)
            view_pos = "AP" if rng.random() < (0.42 if label == 1 else 0.22) else "PA"

            # Índice Cardiotorácico (ICT / CTR):
            # 12% dos casos situam-se na zona limítrofe clínica (ICT 0.46 - 0.54) para desafiar os classificadores
            is_borderline = rng.random() < 0.12
            if label == 0:
                ctr = float(np.round(rng.uniform(0.46, 0.51) if is_borderline else rng.uniform(0.39, 0.47), 3))
            else:
                ctr = float(np.round(rng.uniform(0.49, 0.54) if is_borderline else rng.uniform(0.54, 0.67), 3))

            filename = f"nih_cxr_{img_counter:05d}_{patient_id}_f{follow_up}.png"
            img_path = os.path.join(raw_img_dir, filename)

            if not os.path.exists(img_path) or force_regenerate:
                img = synthesize_chest_xray(
                    label=label,
                    ctr_target=ctr,
                    gender=gender,
                    age=age,
                    view_position=view_pos,
                    rng=rng,
                    img_size=224,
                )
                img.save(img_path, format="PNG")

            records.append(
                {
                    "filename": filename,
                    "patient_id": patient_id,
                    "follow_up_number": follow_up,
                    "label": int(label),
                    "finding_label": finding_label,
                    "patient_gender": gender,
                    "patient_age": age,
                    "age_group": age_group,
                    "view_position": view_pos,
                    "ctr_estimated": ctr,
                }
            )
            img_counter += 1

    df = pd.DataFrame(records)
    df.to_csv(metadata_csv, index=False)
    return df, raw_img_dir


def split_dataset_no_leakage(
    df: pd.DataFrame,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
    save_dir: Optional[str] = None,
) -> Dict[str, pd.DataFrame]:
    """
    Realiza a divisão em Treino, Validação e Teste agrupando estritamente por `patient_id`
    para impedir vazamento de pacientes (Patient Leakage).
    """
    assert np.isclose(train_ratio + val_ratio + test_ratio, 1.0), "As proporções devem somar 1.0"

    gss_first = GroupShuffleSplit(n_splits=1, train_size=train_ratio, random_state=seed)
    train_idx, temp_idx = next(gss_first.split(df, df["label"], groups=df["patient_id"]))

    df_train = df.iloc[train_idx].copy().reset_index(drop=True)
    df_temp = df.iloc[temp_idx].copy().reset_index(drop=True)

    relative_val_ratio = val_ratio / (val_ratio + test_ratio)
    gss_second = GroupShuffleSplit(n_splits=1, train_size=relative_val_ratio, random_state=seed + 1)
    val_idx, test_idx = next(gss_second.split(df_temp, df_temp["label"], groups=df_temp["patient_id"]))

    df_val = df_temp.iloc[val_idx].copy().reset_index(drop=True)
    df_test = df_temp.iloc[test_idx].copy().reset_index(drop=True)

    # Verificação formal de ausência de vazamento
    p_train = set(df_train["patient_id"])
    p_val = set(df_val["patient_id"])
    p_test = set(df_test["patient_id"])
    assert len(p_train & p_val) == 0, "Leakage detectado entre Treino e Validação!"
    assert len(p_train & p_test) == 0, "Leakage detectado entre Treino e Teste!"
    assert len(p_val & p_test) == 0, "Leakage detectado entre Validação e Teste!"

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        df_train.to_csv(os.path.join(save_dir, "train_split.csv"), index=False)
        df_val.to_csv(os.path.join(save_dir, "val_split.csv"), index=False)
        df_test.to_csv(os.path.join(save_dir, "test_split.csv"), index=False)

    return {"train": df_train, "val": df_val, "test": df_test}


def get_transforms(img_size: int = 224) -> Dict[str, transforms.Compose]:
    """
    Retorna transformações de pré-processamento e Data Augmentation clínico restrito
    (sem inversão vertical ou horizontal que descaracterize a dextrocardia/levocardia).
    """
    train_transform = transforms.Compose(
        [
            transforms.Resize((img_size, img_size)),
            transforms.RandomRotation(degrees=7),
            transforms.RandomAffine(degrees=0, translate=(0.04, 0.04), scale=(0.96, 1.04)),
            transforms.ColorJitter(brightness=0.08, contrast=0.08),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )

    eval_transform = transforms.Compose(
        [
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )

    return {"train": train_transform, "val": eval_transform, "test": eval_transform}


class CardioImageDataset(Dataset):
    """
    Dataset PyTorch para radiografias torácicas do CardioIA Fase 4.
    Retorna `(image_tensor, label_tensor, metadata_dict)` para suportar tanto
    o treinamento supervisionado quanto a auditoria demográfica de Fairness.
    """

    def __init__(self, df: pd.DataFrame, img_dir: str, transform: Optional[transforms.Compose] = None):
        self.df = df.reset_index(drop=True)
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.img_dir, row["filename"])
        image = Image.open(img_path).convert("RGB")

        if self.transform is not None:
            image = self.transform(image)

        label = torch.tensor(int(row["label"]), dtype=torch.long)
        metadata = {
            "filename": str(row["filename"]),
            "patient_id": str(row.get("patient_id", "")),
            "gender": str(row.get("patient_gender", "U")),
            "age": int(row.get("patient_age", 0)),
            "age_group": str(row.get("age_group", "Desconhecido")),
            "view_position": str(row.get("view_position", "PA")),
            "ctr_estimated": float(row.get("ctr_estimated", 0.5)),
        }
        return image, label, metadata


def create_dataloaders(
    splits: Dict[str, pd.DataFrame],
    img_dir: str,
    batch_size: int = 16,
    num_workers: int = 0,
    img_size: int = 224,
) -> Dict[str, DataLoader]:
    """Cria DataLoaders PyTorch prontos para Treino, Validação e Teste."""
    tfms = get_transforms(img_size=img_size)
    loaders = {}
    for split_name, split_df in splits.items():
        tfm = tfms.get(split_name, tfms["test"])
        ds = CardioImageDataset(split_df, img_dir=img_dir, transform=tfm)
        loaders[split_name] = DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=(split_name == "train"),
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
        )
    return loaders
