# training/train_projection.py
import os
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms as T
from tqdm import tqdm
import yaml

from models.dataset import IndustrialADDataset
from models.dino_encoder import DINOv2Encoder
from models.clip_encoder import CLIPEncoder
from models.projection import ProjectionHead

def train_projection(config_path='configs/dataset.yaml'):
    # ==========================================
    # 1. 基础配置与模型加载
    # ==========================================
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # 冻结的骨干网络
    dino = DINOv2Encoder().to(device).eval()
    clip = CLIPEncoder().to(device).eval()
    
    # 可训练的 Projection Head (384->256, 512->256)
    proj = ProjectionHead().to(device).train()
    
    # 优化器 (只训练 Projection Head)
    optimizer = torch.optim.AdamW(proj.parameters(), lr=1e-4, weight_decay=1e-4)
    
    # ==========================================
    # 2. 数据加载 (MVTec AD 所有类别的正常样本)
    # ==========================================
    with open(config_path, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)['mvtec']
    
    # 构建一个包含所有类别的联合数据集
    all_datasets = []
    for cat in cfg['categories']:
        ds = IndustrialADDataset(
            config_path=config_path,
            dataset_name='mvtec',
            category=cat,
            split='train',  # 仅加载 train/good
            transform=T.Compose([
                T.Resize((518, 518)),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        )
        all_datasets.append(ds)
        print(f"Loaded {cat}: {len(ds)} normal samples")
        
    # 使用 ConcatDataset 合并
    from torch.utils.data import ConcatDataset
    joint_dataset = ConcatDataset(all_datasets)
    print(f"\nTotal joint normal samples: {len(joint_dataset)}")
    
    loader = DataLoader(
        joint_dataset, 
        batch_size=16, 
        shuffle=True, 
        num_workers=4, 
        pin_memory=True,
        drop_last=True
    )
    
    # ==========================================
    # 3. 训练循环
    # ==========================================
    epochs = 10
    save_path = "weights/projection_aligned.pth"
    os.makedirs("weights", exist_ok=True)
    
    for epoch in range(epochs):
        proj.train()
        total_loss = 0.0
        pbar = tqdm(loader, desc=f"Epoch {epoch+1}/{epochs}")
        
        for batch in pbar:
            images = batch['image'].to(device)
            
            # 前向传播 (Backbones 冻结)
            with torch.no_grad():
                # DINOv2 输出: [B, 1370, 384] -> 取 CLS token: [B, 384]
                dino_feat = dino(images)[:, 0, :]
                # CLIP 需要 224x224 输入 (ViT-B/32 patch=32 -> 7x7=49+1=50)
                clip_images = F.interpolate(images, size=(224, 224), mode='bilinear', align_corners=False)
                clip_feat = clip(clip_images)[:, 0, :]  # 取 CLS token: [B, 512]
            
            # Projection
            z_v, z_t = proj(dino_feat, clip_feat)  # 都是 [B, 256]
            
            # 1. InfoNCE Contrastive Loss
            logits = torch.matmul(z_v, z_t.T) / 0.07  # [B, B]
            labels = torch.arange(logits.size(0)).to(device)
            loss_nce = F.cross_entropy(logits, labels)
            
            # 2. Cosine Alignment Loss
            loss_align = (1 - F.cosine_similarity(z_v, z_t, dim=1)).mean()
            
            # 总 Loss
            loss = loss_nce + loss_align
            
            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})
            
        avg_loss = total_loss / len(loader)
        print(f"Epoch {epoch+1} | Avg Loss: {avg_loss:.4f}")
        
    # ==========================================
    # 4. 保存权重
    # ==========================================
    torch.save(proj.state_dict(), save_path)
    print(f"\n✅ Projection Head trained and saved to {save_path}")

if __name__ == "__main__":
    train_projection()