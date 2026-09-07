# inference/zero_shot.py
"""
ECR-AD 2.0 Zero-Shot Inference & Evaluation Pipeline
=====================================================
完整的 Zero-Shot 工业异常检测推理流程：
. 加载冻结的 Backbones (DINOv2 + CLIP) + 训练好的 Projection Head
. 多尺度 Patch 特征提取与投影
. 视觉证据 (Visual Evidence) 与语义证据 (Semantic Evidence) 计算
. Uncertainty 与 Disagreement 计算
. Reliability Router 动态权重仲裁
. 异常分数生成 + 多尺度聚合 + 双边滤波 (Bilateral Refinement)
. Image-level & Pixel-level AUROC 评估
"""

import os
import sys
import argparse
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import transforms as T
from models.memory_bank import MemoryBank
from tqdm import tqdm
import yaml
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score

# 确保能找到项目模块
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models.dataset import IndustrialADDataset
from models.dino_encoder import DINOv2Encoder
from models.clip_encoder import CLIPEncoder
from models.projection import ProjectionHead
from models.evidence import VisualEvidence, SemanticEvidence
from models.uncertainty import UncertaintyEstimator
from models.disagreement import CrossModalDisagreement
from models.reliability_router import ReliabilityRouter


class ECRADInferencePipeline:
    """
    ECR-AD 2.0 完整推理 Pipeline
    封装所有组件，提供端到端的 Zero-Shot 异常检测能力
    """
    
    def __init__(self, config_path, dataset_name, category, projection_weights_path, device='cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.dataset_name = dataset_name
        self.category = category
        
        print(f"[INFO] Initializing ECR-AD 2.0 Pipeline on {self.device}")
        
        # 1. 加载冻结的 Backbone 编码器
        print("[INFO] Loading frozen backbones...")
        self.dino_encoder = DINOv2Encoder().to(self.device).eval()
        self.clip_encoder = CLIPEncoder().to(self.device).eval()
        
        # 2. 加载训练好的 Projection Head
        print(f"[INFO] Loading Projection Head from {projection_weights_path}")
        self.projection = ProjectionHead().to(self.device).eval()
        state_dict = torch.load(projection_weights_path, map_location=self.device)
        self.projection.load_state_dict(state_dict)
        
        # 3. 初始化证据计算模块
        self.visual_evidence = VisualEvidence(feature_dim=256).to(self.device).eval()
        self.semantic_evidence = SemanticEvidence().to(self.device).eval()

        # 计算 CLIP 文本特征 (normal / anomalous) 并投影到 latent space
        with torch.no_grad():
            prompts = self.semantic_evidence.get_prompts()
            text_feats = self.clip_encoder.encode_text(prompts)  # [2, 512]
            text_feats = self.projection.forward_text(text_feats)  # [2, 256]
            text_normal = text_feats[0:1].to(self.device)     # [1, 256]
            text_anomalous = text_feats[1:2].to(self.device)  # [1, 256]
            self.semantic_evidence.set_text_features(text_normal, text_anomalous)
        
        # 4. 初始化 Uncertainty 和 Disagreement 模块
        self.uncertainty = UncertaintyEstimator().to(self.device).eval()
        self.disagreement = CrossModalDisagreement().to(self.device).eval()
        
        # 5. 初始化 Reliability Router
        self.router = ReliabilityRouter().to(self.device).eval()
        
        # 6. 数据变换 (DINOv2: 518x518, CLIP: 224x224)
        self.transform_dino = T.Compose([
            T.Resize((518, 518)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        self.transform_clip = T.Compose([
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize(mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711])
        ])

        # 7. 加载类别特化的记忆库
        class_bank_dir = os.path.join('weights', 'memory_banks', dataset_name, category)
        visual_bank_path = os.path.join(class_bank_dir, 'visual.pth')
        semantic_bank_path = os.path.join(class_bank_dir, 'semantic.pth')

        # 如果类别特化记忆库不存在，回退到通用记忆库
        if not os.path.exists(visual_bank_path):
            print(f"[WARN] Class-specific memory bank not found at {class_bank_dir}")
            print(f"[WARN] Falling back to global memory bank")
            visual_bank_path = os.path.join(os.path.dirname(projection_weights_path), 'memory_bank_visual.pth')
            semantic_bank_path = os.path.join(os.path.dirname(projection_weights_path), 'memory_bank_semantic.pth')

        print("[INFO] Loading memory banks...")
        self.visual_bank = MemoryBank(device=self.device)
        self.visual_bank.features = torch.load(visual_bank_path, map_location=self.device)
        self.visual_bank.is_built = True
        print(f"[INFO] Visual memory bank: {self.visual_bank.features.shape}")

        self.semantic_bank = MemoryBank(device=self.device)
        self.semantic_bank.features = torch.load(semantic_bank_path, map_location=self.device)
        self.semantic_bank.is_built = True
        print(f"[INFO] Semantic memory bank: {self.semantic_bank.features.shape}")

        print("[INFO] Pipeline initialized successfully.")
    
    @torch.no_grad()
    def extract_features(self, images_dino, images_clip):
        """
        提取多尺度 Patch 特征并投影到统一语义空间
        Args:
            images_dino: [B, 3, 518, 518] DINOv2 输入
            images_clip: [B, 3, 224, 224] CLIP 输入
        Returns:
            z_v_patches: [B, N_v, 256] 视觉投影特征
            z_t_patches: [B, N_t, 256] 语义投影特征
        """
        # DINOv2 特征提取: [B, 1370, 384] (37x37 patches + 1 CLS)
        dino_out = self.dino_encoder(images_dino)
        
        # CLIP 特征提取: [B, 50, 512] (7x7 patches + 1 CLS)
        clip_out = self.clip_encoder(images_clip)
        
        # 分离 CLS token 和 Patch tokens
        dino_cls = dino_out[:, 0, :]      # [B, 384]
        dino_patches = dino_out[:, 1:, :] # [B, 1369, 384]
        
        clip_cls = clip_out[:, 0, :]      # [B, 512]
        clip_patches = clip_out[:, 1:, :] # [B, 49, 512]
        
        # Projection: CLS level
        z_v_cls, z_t_cls = self.projection(dino_cls, clip_cls)  # [B, 256], [B, 256]
        
        # Projection: Patch level (逐 patch 投影)
        B, N_v, D_v = dino_patches.shape
        B, N_t, D_t = clip_patches.shape
        
        dino_patches_flat = dino_patches.reshape(B * N_v, D_v)
        clip_patches_flat = clip_patches.reshape(B * N_t, D_t)
        
        z_v_patches_flat, z_t_patches_flat = self.projection(dino_patches_flat, clip_patches_flat)
        
        z_v_patches = z_v_patches_flat.reshape(B, N_v, 256)  # [B, 1369, 256]
        z_t_patches = z_t_patches_flat.reshape(B, N_t, 256)  # [B, 49, 256]
        
        return z_v_cls, z_t_cls, z_v_patches, z_t_patches
    
    @torch.no_grad()
    def compute_anomaly_score(self, images_dino, images_clip):
        """
        完整的异常分数计算流程
        Returns:
            anomaly_map: [B, 1, H, W] 像素级异常热力图
            image_score: [B] 图像级异常分数
        """
        # Step 1: 特征提取与投影
        z_v_cls, z_t_cls, z_v_patches, z_t_patches = self.extract_features(images_dino, images_clip)
        
        # Step 2: 计算视觉证据与语义证据 (Patch level)
        ev_v = self.visual_evidence(z_v_patches)  # [B, N_v, 1]
        ev_t = self.semantic_evidence(z_t_patches) # [B, N_t, 1]
        
        # Step 3: 计算 Uncertainty
        unc_v = self.uncertainty(ev_v)  # [B, N_v, 1]
        unc_t = self.uncertainty(ev_t)  # [B, N_t, 1]
        
        # Step 4: 计算 Cross-Modal Disagreement
        # 需要将 z_t_patches 上采样到与 z_v_patches 相同的空间分辨率
        # 或者直接计算 CLS-level 的 disagreement 作为全局权重
        dis_score = self.disagreement(
            visual_features=z_v_cls,
            textual_features=z_t_cls,
            visual_weights=ev_v.mean(dim=1),
            textual_weights=ev_t.mean(dim=1)
        )  # [B, 1] 或标量
        
        # Step 5: Reliability Router 计算动态权重
        w_v, w_t = self.router(
            uncertainty_v=unc_v.mean(dim=1),  # [B, 1]
            uncertainty_t=unc_t.mean(dim=1),  # [B, 1]
            disagreement=dis_score             # [B, 1]
        )  # w_v: [B, 1], w_t: [B, 1]
        
       # Step 6: 生成 Patch-level 异常分数（基于记忆库 k-NN 距离）
        score_v_patches = self._compute_patch_anomaly_score(z_v_patches, self.visual_bank, k=5)   # [B, N_v]
        score_t_patches = self._compute_patch_anomaly_score(z_t_patches, self.semantic_bank, k=5)  # [B, N_t]
        
        # Step 7: 将 CLIP 的低分辨率分数上采样到 DINOv2 的高分辨率
        B = images_dino.shape[0]
        H_grid = W_grid = 37  # DINOv2 patch grid: 37x37 = 1369
        
        # 视觉分数直接 reshape
        score_v_map = score_v_patches.reshape(B, H_grid, W_grid)  # [B, 37, 37]
        
        # 语义分数需要上采样: 7x7 -> 37x37
        score_t_map = score_t_patches.reshape(B, 7, 7)  # [B, 7, 7]
        score_t_map = F.interpolate(
            score_t_map.unsqueeze(1),  # [B, 1, 7, 7]
            size=(H_grid, W_grid),
            mode='bilinear',
            align_corners=False
        ).squeeze(1)  # [B, 37, 37]
        
        # Step 8: Router 加权融合
        # w_v/w_t 形状为 [B, 1]，unsqueeze(-1) 一次得到 [B, 1, 1] 即可与 [B, 37, 37] 广播
        w_v_expanded = w_v.unsqueeze(-1)  # [B, 1, 1]
        w_t_expanded = w_t.unsqueeze(-1)  # [B, 1, 1]
        
        fused_score_map = w_v_expanded * score_v_map + w_t_expanded * score_t_map  # [B, 37, 37]
        
        # Step 9: 双边滤波精炼 (Bilateral Refinement)
        anomaly_map = self._bilateral_refinement(fused_score_map, images_dino)  # [B, 518, 518]
        
        # Step 10: 图像级分数 (Top-10% patch 最大异常分数聚合)
        score_flat = fused_score_map.reshape(B, -1)  # [B, 37*37]
        k = max(int(score_flat.shape[1] * 0.10), 1)  # Top 10% of patches
        topk_scores, _ = torch.topk(score_flat, k=k, dim=1)
        image_score = topk_scores.mean(dim=1)  # [B]
        
        return anomaly_map.unsqueeze(1), image_score
    
    def _compute_patch_anomaly_score(self, z_patches, memory_bank, k=5):
        """
        基于记忆库 k-NN 距离的 Patch-level 异常分数
        Args:
            z_patches: [B, N, D] 投影后的 patch 特征
            memory_bank: MemoryBank 实例
            k: 最近邻数量
        Returns:
            scores: [B, N] 每个 patch 的异常分数
        """
        scores = memory_bank.compute_anomaly_score(z_patches, k=k)  # [B, N]
        return scores
    
    def _bilateral_refinement(self, score_map, images):
        """
        双边滤波精炼：保留边缘的平滑异常图
        Args:
            score_map: [B, 37, 37] 低分辨率异常分数
            images: [B, 3, 518, 518] 原始图像 (用于引导滤波)
        """
        B = score_map.shape[0]
        
        # 上采样到原始分辨率
        score_map_up = F.interpolate(
            score_map.unsqueeze(1),  # [B, 1, 37, 37]
            size=(518, 518),
            mode='bilinear',
            align_corners=False
        )  # [B, 1, 518, 518]
        
        # 简化的双边滤波: 使用高斯模糊 + 原图边缘引导
        # 这里用一个简单的 Gaussian blur 近似
        kernel_size = 15
        sigma = 3.0
        pad = kernel_size // 2

        # 创建 1D 高斯核 (separable 可分离卷积)
        coords = torch.arange(kernel_size, dtype=torch.float32, device=score_map.device) - pad
        g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
        g = g / g.sum()
        kernel_h = g.reshape(1, 1, -1, 1)  # [1, 1, 15, 1] 垂直方向
        kernel_w = g.reshape(1, 1, 1, -1)  # [1, 1, 1, 15] 水平方向

        # 先沿 H 再沿 W 卷积，每步各自 reflect padding，保证输出分辨率仍为 518
        blurred = F.pad(score_map_up, (0, 0, pad, pad), mode='reflect')
        blurred = F.conv2d(blurred, kernel_h)  # [B, 1, 518, 518]
        blurred = F.pad(blurred, (pad, pad, 0, 0), mode='reflect')
        blurred = F.conv2d(blurred, kernel_w)  # [B, 1, 518, 518]

        return blurred.squeeze(1)  # [B, 518, 518]
    
    @torch.no_grad()
    def evaluate(self, dataloader, save_vis_dir=None):
        """
        完整评估流程
        Returns:
            image_auroc: 图像级 AUROC
            pixel_auroc: 像素级 AUROC
        """
        all_image_scores = []
        all_image_labels = []
        all_pixel_scores = []   # 每张图抽样后的分数数组 (np.float32)
        all_pixel_labels = []   # 每张图抽样后的标签数组 (np.uint8)

        # 像素级分层子采样参数 (控制内存，避免 Python list 累积导致 OOM)
        rng = np.random.default_rng(42)   # 固定种子，结果可复现
        MAX_POS_PER_IMG = 50_000   # 单图最多保留的异常像素
        MAX_NEG_PER_IMG = 10_000   # 单图最多保留的正常像素
        
        if save_vis_dir:
            os.makedirs(save_vis_dir, exist_ok=True)
        
        self.dino_encoder.eval()
        self.clip_encoder.eval()
        self.projection.eval()
        self.visual_evidence.eval()
        self.semantic_evidence.eval()
        self.uncertainty.eval()
        self.disagreement.eval()
        self.router.eval()
        
        pbar = tqdm(dataloader, desc="Zero-Shot Inference")
        for batch in pbar:
            images = batch['image'].to(self.device)  # [B, 3, 518, 518]
            gt_masks = batch['gt'].to(self.device)   # [B, 1, 518, 518]
            labels = batch['label']                   # [B]
            img_paths = batch['img_path']
            
            # 为 CLIP 准备 224x224 输入
            images_clip = F.interpolate(images, size=(224, 224), mode='bilinear', align_corners=False)
            # 重新归一化 (CLIP 使用不同的 mean/std)
            # 先反归一化到 [0,1]，再用 CLIP 的 mean/std 归一化
            images_clip = self._renormalize_for_clip(images_clip)
            
            # 计算异常分数
            anomaly_map, image_score = self.compute_anomaly_score(images, images_clip)
            
            # 收集结果
            # 图像级分数：用 anomaly_map 的 Top-1% 最大值（而非全局平均）
            B = anomaly_map.shape[0]
            amap_flat = anomaly_map.reshape(B, -1)  # [B, 518*518]
            k = max(int(amap_flat.shape[1] * 0.01), 1)  # Top 1%
            topk_scores, _ = torch.topk(amap_flat, k=k, dim=1)
            image_scores_topk = topk_scores.mean(dim=1)  # [B]

            all_image_scores.extend(image_scores_topk.cpu().numpy().tolist())
            all_image_labels.extend(labels.numpy().tolist())
            
            # 像素级评估：按 GT 分层子采样后再收集，避免全像素 Python list 累积导致 OOM
            # (异常像素通常占比小，尽量全保留；正常像素每张图随机抽样封顶)
            pixel_scores = anomaly_map.squeeze(1).cpu().numpy()  # [B, 518, 518]
            pixel_labels = gt_masks.squeeze(1).cpu().numpy()     # [B, 518, 518]

            for i in range(len(img_paths)):
                scores_f = pixel_scores[i].reshape(-1).astype(np.float32)  # [H*W]
                labels_f = pixel_labels[i].reshape(-1) > 0.5               # [H*W] bool

                pos_idx = np.flatnonzero(labels_f)
                neg_idx = np.flatnonzero(~labels_f)

                if len(pos_idx) > MAX_POS_PER_IMG:
                    pos_idx = rng.choice(pos_idx, size=MAX_POS_PER_IMG, replace=False)
                if len(neg_idx) > MAX_NEG_PER_IMG:
                    neg_idx = rng.choice(neg_idx, size=MAX_NEG_PER_IMG, replace=False)

                keep = np.sort(np.concatenate([pos_idx, neg_idx]))
                all_pixel_scores.append(scores_f[keep])
                all_pixel_labels.append(labels_f[keep].astype(np.uint8))
            
            # 保存可视化
            if save_vis_dir:
                self._save_visualization(img_paths, images, anomaly_map, gt_masks, save_vis_dir)
        
        # 计算 AUROC
        image_auroc = roc_auc_score(all_image_labels, all_image_scores)

        # 像素级 AUROC (分层子采样后的全部像素，numpy 拼接后统一计算)
        if len(all_pixel_scores) > 0:
            all_pixel_scores = np.concatenate(all_pixel_scores)
            all_pixel_labels = np.concatenate(all_pixel_labels)
            if all_pixel_labels.sum() > 0 and (all_pixel_labels == 0).sum() > 0:
                pixel_auroc = roc_auc_score(all_pixel_labels, all_pixel_scores)
            else:
                pixel_auroc = 0.0
        else:
            pixel_auroc = 0.0
        
        return image_auroc, pixel_auroc
    
    def _renormalize_for_clip(self, images):
        """将 DINOv2 归一化的图像转换为 CLIP 归一化"""
        # DINOv2 mean/std
        dino_mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(images.device)
        dino_std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(images.device)
        # CLIP mean/std
        clip_mean = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(1, 3, 1, 1).to(images.device)
        clip_std = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(1, 3, 1, 1).to(images.device)
        
        # 反归一化
        images = images * dino_std + dino_mean
        # 重新归一化
        images = (images - clip_mean) / clip_std
        return images
    
    def _save_visualization(self, img_paths, images, anomaly_maps, gt_masks, save_dir):
        """保存异常热力图可视化"""
        for i in range(len(img_paths)):
            # 获取文件名
            fname = os.path.basename(img_paths[i]).replace('.JPG', '.png').replace('.jpg', '.png')
            
            # 原图
            img = images[i].cpu()
            # 反归一化
            img = img * torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1) + torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
            img = img.clamp(0, 1).permute(1, 2, 0).numpy()
            
            # 异常图
            amap = anomaly_maps[i, 0].cpu().numpy()
            amap = (amap - amap.min()) / (amap.max() - amap.min() + 1e-8)
            
            # GT
            gt = gt_masks[i, 0].cpu().numpy()
            
            # 保存
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            axes[0].imshow(img)
            axes[0].set_title("Input")
            axes[0].axis('off')
            
            axes[1].imshow(amap, cmap='jet')
            axes[1].set_title("Anomaly Map")
            axes[1].axis('off')
            
            axes[2].imshow(gt, cmap='gray')
            axes[2].set_title("Ground Truth")
            axes[2].axis('off')
            
            plt.tight_layout()
            plt.savefig(os.path.join(save_dir, fname), dpi=150, bbox_inches='tight')
            plt.close()


def main():
    parser = argparse.ArgumentParser(description="ECR-AD 2.0 Zero-Shot Inference")
    parser.add_argument('--config', type=str, default='configs/dataset.yaml', help='Dataset config path')
    parser.add_argument('--dataset', type=str, default='visa', choices=['mvtec', 'visa', 'btad', 'mpdd'])
    parser.add_argument('--category', type=str, default='candle', help='Category name')
    parser.add_argument('--weights', type=str, default='weights/projection_aligned.pth', help='Projection weights')
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--save_vis', type=str, default='results/vis', help='Visualization save dir')
    parser.add_argument('--device', type=str, default='cuda')
    args = parser.parse_args()
    
    # 1. 初始化 Pipeline
    pipeline = ECRADInferencePipeline(
        config_path=args.config,
        dataset_name=args.dataset,
        category=args.category,
        projection_weights_path=args.weights,
        device=args.device
    )
    
    # 2. 加载测试集
    print(f"\n[INFO] Loading {args.dataset} / {args.category} test set...")
    test_dataset = IndustrialADDataset(
        config_path=args.config,
        dataset_name=args.dataset,
        category=args.category,
        split='test'
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    print(f"[INFO] Test samples: {len(test_dataset)}")
    
    # 3. 运行评估
    print("\n[INFO] Starting Zero-Shot Evaluation...")
    image_auroc, pixel_auroc = pipeline.evaluate(
        test_loader,
        save_vis_dir=args.save_vis
    )
    
    # 4. 输出结果
    print("\n" + "="*50)
    print(f"Dataset: {args.dataset} | Category: {args.category}")
    print(f"Image-level AUROC: {image_auroc:.4f}")
    print(f"Pixel-level AUROC: {pixel_auroc:.4f}")
    print("="*50)
    print(f"\nVisualizations saved to: {args.save_vis}")


if __name__ == "__main__":
    main()