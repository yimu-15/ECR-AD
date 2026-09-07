# training/build_memory_bank.py
"""
构建正常特征记忆库
从 MVTec AD 所有类别的 train/good 样本中提取投影特征，构建视觉和语义两个记忆库。
"""
import os
import sys
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, ConcatDataset
from torchvision import transforms as T
from tqdm import tqdm
import yaml

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models.dataset import IndustrialADDataset
from models.dino_encoder import DINOv2Encoder
from models.clip_encoder import CLIPEncoder
from models.projection import ProjectionHead
from models.memory_bank import MemoryBank


def build_memory_bank(config_path='configs/dataset.yaml', projection_weights='weights/projection_aligned.pth'):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[INFO] Device: {device}")
    
    # 1. 加载模型
    print("[INFO] Loading models...")
    dino = DINOv2Encoder().to(device).eval()
    clip = CLIPEncoder().to(device).eval()
    
    proj = ProjectionHead().to(device).eval()
    proj.load_state_dict(torch.load(projection_weights, map_location=device))
    print(f"[INFO] Loaded projection weights from {projection_weights}")
    
    # 2. 加载 MVTec 所有正常训练样本
    with open(config_path, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)['mvtec']
    
    transform_dino = T.Compose([
        T.Resize((518, 518)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    all_datasets = []
    for cat in cfg['categories']:
        ds = IndustrialADDataset(config_path, 'mvtec', cat, split='train', transform=transform_dino)
        all_datasets.append(ds)
        print(f"  {cat}: {len(ds)} samples")
    
    joint_dataset = ConcatDataset(all_datasets)
    loader = DataLoader(joint_dataset, batch_size=16, shuffle=False, num_workers=4, pin_memory=True)
    print(f"\n[INFO] Total normal samples: {len(joint_dataset)}")
    
    # 3. 提取特征
    visual_features_list = []  # 存储 DINOv2 投影特征
    semantic_features_list = []  # 存储 CLIP 投影特征
    
    print("\n[INFO] Extracting features from normal samples...")
    with torch.no_grad():
        for batch in tqdm(loader, desc="Feature Extraction"):
            images_dino = batch['image'].to(device)  # [B, 3, 518, 518]
            
            # CLIP 需要 224x224
            images_clip = F.interpolate(images_dino, size=(224, 224), mode='bilinear', align_corners=False)
            # 重新归一化 for CLIP
            dino_mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(device)
            dino_std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(device)
            clip_mean = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(1, 3, 1, 1).to(device)
            clip_std = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(1, 3, 1, 1).to(device)
            images_clip = (images_clip * dino_std + dino_mean - clip_mean) / clip_std
            
            # DINOv2 特征: [B, 1370, 384]
            dino_out = dino(images_dino)
            dino_patches = dino_out[:, 1:, :]  # [B, 1369, 384] 去掉 CLS
            
            # CLIP 特征: [B, 50, 512]
            clip_out = clip(images_clip)
            clip_patches = clip_out[:, 1:, :]  # [B, 49, 512] 去掉 CLS
            
            B = images_dino.shape[0]
            
            # 投影视觉特征 (patch-level)
            N_v = dino_patches.shape[1]
            dino_flat = dino_patches.reshape(B * N_v, -1)
            z_v_flat = proj.forward_visual(dino_flat)  # 只走视觉分支 proj_v
            z_v = z_v_flat.reshape(B, N_v, -1)  # [B, 1369, 256]
            # 按 patch 展平后收集: [B*N_v, 256] (记忆库按 patch 行存储)
            visual_features_list.append(z_v.reshape(-1, z_v.shape[-1]).cpu())
            
            # 投影语义特征 (patch-level)
            N_t = clip_patches.shape[1]
            clip_flat = clip_patches.reshape(B * N_t, -1)
            z_t_flat = proj.forward_text(clip_flat)  # 只走语义分支 proj_t
            z_t = z_t_flat.reshape(B, N_t, -1)  # [B, 49, 256]
            semantic_features_list.append(z_t.reshape(-1, z_t.shape[-1]).cpu())
    
    # 4. 构建记忆库
    print("\n[INFO] Building visual memory bank...")
    visual_bank = MemoryBank(device='cpu')
    visual_bank.build(visual_features_list, coreset_ratio=0.1)
    
    print("\n[INFO] Building semantic memory bank...")
    semantic_bank = MemoryBank(device='cpu')
    semantic_bank.build(semantic_features_list, coreset_ratio=0.1)
    
    # 5. 保存
    save_dir = 'weights'
    os.makedirs(save_dir, exist_ok=True)
    
    visual_path = os.path.join(save_dir, 'memory_bank_visual.pth')
    semantic_path = os.path.join(save_dir, 'memory_bank_semantic.pth')
    
    torch.save(visual_bank.features, visual_path)
    torch.save(semantic_bank.features, semantic_path)
    
    print(f"\n✅ Visual memory bank saved to {visual_path}")
    print(f"   Shape: {visual_bank.features.shape}")
    print(f"✅ Semantic memory bank saved to {semantic_path}")
    print(f"   Shape: {semantic_bank.features.shape}")


if __name__ == "__main__":
    build_memory_bank()