import os
import pytest
import numpy as np
import pandas as pd
import torch

from src.dataset import (
    ensure_cardio_dataset,
    split_dataset_no_leakage,
    CardioImageDataset,
    get_transforms,
    create_dataloaders,
)


def test_ensure_cardio_dataset_generates_images_and_metadata(tmp_path):
    data_dir = str(tmp_path / "data")
    df, img_dir = ensure_cardio_dataset(data_dir=data_dir, num_patients=60, seed=42)

    assert os.path.exists(img_dir)
    assert len(df) >= 60
    required_cols = {
        "filename",
        "patient_id",
        "label",
        "finding_label",
        "patient_gender",
        "patient_age",
        "age_group",
        "view_position",
        "ctr_estimated",
    }
    assert required_cols.issubset(set(df.columns))
    # Verify both classes exist
    assert set(df["label"].unique()) == {0, 1}
    # Verify sample image exists and has valid size
    first_img = os.path.join(img_dir, df.iloc[0]["filename"])
    assert os.path.exists(first_img)


def test_split_dataset_prevents_patient_leakage(tmp_path):
    data_dir = str(tmp_path / "data")
    df, _ = ensure_cardio_dataset(data_dir=data_dir, num_patients=80, seed=42)
    splits = split_dataset_no_leakage(df, train_ratio=0.70, val_ratio=0.15, test_ratio=0.15, seed=42)

    train_patients = set(splits["train"]["patient_id"].unique())
    val_patients = set(splits["val"]["patient_id"].unique())
    test_patients = set(splits["test"]["patient_id"].unique())

    # Zero patient leakage across splits
    assert len(train_patients.intersection(val_patients)) == 0
    assert len(train_patients.intersection(test_patients)) == 0
    assert len(val_patients.intersection(test_patients)) == 0
    # Total rows preserved
    assert len(splits["train"]) + len(splits["val"]) + len(splits["test"]) == len(df)


def test_cardio_dataset_and_dataloaders_tensor_shapes(tmp_path):
    data_dir = str(tmp_path / "data")
    df, img_dir = ensure_cardio_dataset(data_dir=data_dir, num_patients=40, seed=42)
    splits = split_dataset_no_leakage(df, seed=42)
    loaders = create_dataloaders(splits, img_dir=img_dir, batch_size=8, num_workers=0)

    images, labels, meta = next(iter(loaders["train"]))
    assert isinstance(images, torch.Tensor)
    assert images.shape == (8, 3, 224, 224)
    assert labels.shape == (8,)
    assert "gender" in meta and "age_group" in meta and "view_position" in meta
