import os
import torch
import numpy as np
import pytest

from src.models import CardioCNN_Scratch, get_cardio_resnet, count_parameters, export_model_onnx
from src.dataset import ensure_cardio_dataset, split_dataset_no_leakage, create_dataloaders
from src.train import train_model
from src.evaluate import evaluate_model


def test_cardio_cnn_scratch_forward_and_params():
    model = CardioCNN_Scratch(num_classes=2)
    dummy = torch.randn(4, 3, 224, 224)
    out = model(dummy)
    assert out.shape == (4, 2)
    params = count_parameters(model)
    assert params["total"] > 100_000
    assert params["trainable"] == params["total"]


def test_cardio_resnet_transfer_learning_forward_and_freezing():
    model = get_cardio_resnet(num_classes=2, freeze_backbone=True, unfreeze_layer4=True, pretrained=False)
    dummy = torch.randn(2, 3, 224, 224)
    out = model(dummy)
    assert out.shape == (2, 2)
    params = count_parameters(model)
    assert params["trainable"] < params["total"]
    assert params["trainable"] > 100_000


def test_train_and_evaluate_pipeline_plus_onnx_export(tmp_path):
    data_dir = str(tmp_path / "data")
    df, img_dir = ensure_cardio_dataset(data_dir=data_dir, num_patients=40, seed=42)
    splits = split_dataset_no_leakage(df, seed=42)
    loaders = create_dataloaders(splits, img_dir=img_dir, batch_size=8)

    device = torch.device("cpu")
    model = CardioCNN_Scratch(num_classes=2)
    ckpt_path = str(tmp_path / "best_scratch.pt")

    history = train_model(
        model=model,
        train_loader=loaders["train"],
        val_loader=loaders["val"],
        device=device,
        epochs=2,
        lr=1e-3,
        save_path=ckpt_path,
    )
    assert os.path.exists(ckpt_path)
    assert len(history["train_loss"]) == 2
    assert len(history["val_acc"]) == 2

    metrics = evaluate_model(model, loaders["test"], device=device)
    for key in ["acuracia", "precisao", "recall_sensibilidade", "f1_score", "roc_auc", "matriz_confusao"]:
        assert key in metrics
    assert metrics["matriz_confusao"].shape == (2, 2)

    onnx_path = str(tmp_path / "model.onnx")
    export_model_onnx(model, onnx_path, device=device)
    assert os.path.exists(onnx_path)
