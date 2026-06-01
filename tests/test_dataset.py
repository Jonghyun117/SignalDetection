# tests/test_dataset.py
import torch
import numpy as np
import sys
sys.path.insert(0, 'training')
from dataset import AMCDataset, _preprocess
from simulate import MODULATIONS

def test_dataset_output_shape():
    ds = AMCDataset(n_per_class_per_snr=2, augment=False)
    x, label = ds[0]
    assert x.shape == (3, 1024), f"Expected (3, 1024), got {x.shape}"
    assert 0 <= label < len(MODULATIONS)

def test_preprocess_power_normalized():
    i_arr = np.random.randn(1024).astype(np.float32) * 5.0
    q_arr = np.random.randn(1024).astype(np.float32) * 5.0
    feat = _preprocess(i_arr, q_arr)
    # amplitude channel: mean(A²) should be ~1
    mean_power = np.mean(feat[0] ** 2)
    assert abs(mean_power - 1.0) < 0.05, f"mean power={mean_power:.4f}"

def test_augmentation_changes_signal():
    ds_clean = AMCDataset(n_per_class_per_snr=2, augment=False)
    ds_aug   = AMCDataset(n_per_class_per_snr=2, augment=True)
    x_clean, _ = ds_clean[0]
    x_aug,   _ = ds_aug[0]
    assert not torch.allclose(x_clean, x_aug), "Augmented signal should differ"
