# test_dataloader.py
import torch
from models.dataset import IndustrialADDataset
from torch.utils.data import DataLoader

# 1. 测试 VisA (candle)
print("=== Testing VisA (candle) ===")
dataset = IndustrialADDataset(
    config_path="configs/dataset.yaml",
    dataset_name="visa",
    category="candle",
    split="test"
)
loader = DataLoader(dataset, batch_size=4, shuffle=False)
batch = next(iter(loader))
print(f"Image shape: {batch['image'].shape}")
print(f"GT shape: {batch['gt'].shape}")
print(f"Labels: {batch['label']}")
print(f"Paths: {batch['img_path'][0]}")

# 2. 测试 MVTec (bottle)
print("\n=== Testing MVTec (bottle) ===")
dataset = IndustrialADDataset(
    config_path="configs/dataset.yaml",
    dataset_name="mvtec",
    category="bottle",
    split="test"
)
loader = DataLoader(dataset, batch_size=4, shuffle=False)
batch = next(iter(loader))
print(f"Image shape: {batch['image'].shape}")
print(f"GT shape: {batch['gt'].shape}")
print(f"Labels: {batch['label']}")

# 3. 测试 Train Split (仅正常)
print("\n=== Testing MVTec Train (bottle) ===")
dataset = IndustrialADDataset(
    config_path="configs/dataset.yaml",
    dataset_name="mvtec",
    category="bottle",
    split="train"
)
print(f"Train samples: {len(dataset)}")
print(f"All labels normal? {all(dataset.labels[i] == 0 for i in range(len(dataset)))}")