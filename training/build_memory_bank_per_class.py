# training/build_memory_bank_per_class.py
"""
按类别构建记忆库
为每个目标类别单独构建正常特征记忆库，提升 Zero-Shot 异常检测的区分力。
"""
import os
import sys
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms as T
from tqdm import tqdm
import yaml

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models.dataset import IndustrialADDataset
from models.dino_encoder import DINOv2Encoder
from models.clip_encoder import CLIPEncoder
from models.projection import ProjectionHead
from models.memory_bank import MemoryBank


def build_class_memory_bank(dataset_name, category, config_path='configs/dataset.yaml',
                            projection_weights='weights/projection_aligned.pth',
                            use_projection=True):
    """
    为指定类别构建记忆库
    use_projection=False: A4 消融——跳过投影头，直接用原始 DINOv2(384D)/CLIP(512D) patch 特征建库
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[INFO] Device: {device}")
    print(f"[INFO] Building memory bank for: {dataset_name}/{category}")
    
    # 1. 加载模型
    dino = DINOv2Encoder().to(device).eval()
    clip = CLIPEncoder().to(device).eval()
    
    proj = None
    if use_projection:
        proj = ProjectionHead().to(device).eval()
        proj.load_state_dict(torch.load(projection_weights, map_location=device))
    
    # 2. 加载该类别的正常训练样本
    transform_dino = T.Compose([
        T.Resize((518, 518)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    train_dataset = IndustrialADDataset(config_path, dataset_name, category, split='train', transform=transform_dino)
    loader = DataLoader(train_dataset, batch_size=16, shuffle=False, num_workers=4, pin_memory=True)
    print(f"[INFO] Normal training samples: {len(train_dataset)}")
    
    # 3. 提取特征
    visual_features_list = []
    semantic_features_list = []
    
    with torch.no_grad():
        for batch in tqdm(loader, desc=f"Extracting {category}"):
            images_dino = batch['image'].to(device)
            
            # CLIP 输入
            images_clip = F.interpolate(images_dino, size=(224, 224), mode='bilinear', align_corners=False)
            dino_mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(device)
            dino_std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(device)
            clip_mean = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(1, 3, 1, 1).to(device)
            clip_std = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(1, 3, 1, 1).to(device)
            images_clip = (images_clip * dino_std + dino_mean - clip_mean) / clip_std
            
            # DINOv2 patch 特征
            dino_out = dino(images_dino)
            dino_patches = dino_out[:, 1:, :]  # [B, 1369, 384]
            
            # CLIP patch 特征
            clip_out = clip(images_clip)
            clip_patches = clip_out[:, 1:, :]  # [B, 49, 512]
            
            B = images_dino.shape[0]
            
            if use_projection:
                # 投影视觉 patch 特征 (只走视觉分支 proj_v, 384 -> 256)
                N_v = dino_patches.shape[1]
                dino_flat = dino_patches.reshape(B * N_v, -1)
                z_v_flat = proj.forward_visual(dino_flat)
                z_v = z_v_flat.reshape(B, N_v, -1)
                # 按 patch 行展平后收集: [B*N_v, 256] (记忆库按 patch 行存储)
                visual_features_list.append(z_v.reshape(-1, z_v.shape[-1]).cpu())

                # 投影语义 patch 特征 (只走语义分支 proj_t, 512 -> 256)
                N_t = clip_patches.shape[1]
                clip_flat = clip_patches.reshape(B * N_t, -1)
                z_t_flat = proj.forward_text(clip_flat)
                z_t = z_t_flat.reshape(B, N_t, -1)
                semantic_features_list.append(z_t.reshape(-1, z_t.shape[-1]).cpu())
            else:
                # A4 消融：无投影，直接存原始 patch 特征（维度保持 DINOv2 384 / CLIP 512）
                visual_features_list.append(dino_patches.reshape(-1, dino_patches.shape[-1]).cpu())
                semantic_features_list.append(clip_patches.reshape(-1, clip_patches.shape[-1]).cpu())
    
    # 4. 构建记忆库（类别特化的，不下采样，保留全部特征）
    print(f"\n[INFO] Building class-specific visual memory bank...")
    visual_bank = MemoryBank(device='cpu')
    visual_bank.build(visual_features_list, coreset_ratio=1.0)  # 保留全部
    
    print(f"[INFO] Building class-specific semantic memory bank...")
    semantic_bank = MemoryBank(device='cpu')
    semantic_bank.build(semantic_features_list, coreset_ratio=1.0)  # 保留全部
    
    # 5. 保存（无投影库存独立目录 memory_banks_noproj，避免与投影库混淆/覆盖）
    bank_root = 'weights/memory_banks' if use_projection else 'weights/memory_banks_noproj'
    save_dir = f'{bank_root}/{dataset_name}/{category}'
    os.makedirs(save_dir, exist_ok=True)
    
    torch.save(visual_bank.features, os.path.join(save_dir, 'visual.pth'))
    torch.save(semantic_bank.features, os.path.join(save_dir, 'semantic.pth'))
    
    print(f"\n✅ Saved to {save_dir}")
    print(f"   Visual: {visual_bank.features.shape}")
    print(f"   Semantic: {semantic_bank.features.shape}")


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='visa', choices=['mvtec', 'visa', 'btad', 'mpdd'])
    parser.add_argument('--category', type=str, default='candle')
    parser.add_argument('--config', type=str, default='configs/dataset.yaml')
    parser.add_argument('--weights', type=str, default='weights/projection_aligned.pth')
    parser.add_argument('--no_projection', action='store_true',
                        help='A4 消融：跳过投影头，用原始 DINOv2/CLIP 特征建库')
    args = parser.parse_args()
    
    build_class_memory_bank(args.dataset, args.category, args.config, args.weights,
                            use_projection=not args.no_projection)


if __name__ == "__main__":
    main()