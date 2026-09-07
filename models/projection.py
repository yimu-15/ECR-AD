
"""
Projection Head (投影层)
----------------------
- 将 DINOv2 (384维) 和 CLIP (512维) 的特征投影到统一的 latent space (256维)
- 包含 Cross-Modal Contrastive Alignment 训练目标
- 训练完成后 Freeze，后续推理阶段不参与训练
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class ProjectionHead(nn.Module):
    """跨模态投影头。

    将 DINOv2 (384维) 和 CLIP (512维) 的特征投影到统一的 latent space。

    架构:
        DINOv2 (384) -> Linear -> ReLU -> Dropout -> Linear -> L2 Norm -> (256)
        CLIP (512)   -> Linear -> ReLU -> Dropout -> Linear -> L2 Norm -> (256)
    """

    def __init__(
        self,
        dino_dim: int = 384,
        clip_dim: int = 512,
        latent_dim: int = 256,
        dropout: float = 0.1,
    ):
        """
        Args:
            dino_dim: DINOv2 特征维度 (384)
            clip_dim: CLIP 特征维度 (512)
            latent_dim: 投影后 latent 维度 (256)
            dropout: Dropout 比率
        """
        super().__init__()

        # DINOv2 投影路径: 384 -> 256 -> 256
        self.proj_v = nn.Sequential(
            nn.Linear(dino_dim, latent_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(latent_dim, latent_dim),
        )

        # CLIP 投影路径: 512 -> 256 -> 256
        self.proj_t = nn.Sequential(
            nn.Linear(clip_dim, latent_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(latent_dim, latent_dim),
        )

        self.latent_dim = latent_dim

    def forward_visual(self, x: torch.Tensor) -> torch.Tensor:
        """投影视觉特征。

        Args:
            x: [B, 384] DINOv2 特征

        Returns:
            [B, 256] 归一化后的投影特征
        """
        x = self.proj_v(x)
        x = F.normalize(x, dim=-1)
        return x

    def forward_text(self, x: torch.Tensor) -> torch.Tensor:
        """投影文本特征。

        Args:
            x: [B, 512] CLIP 文本特征

        Returns:
            [B, 256] 归一化后的投影特征
        """
        x = self.proj_t(x)
        x = F.normalize(x, dim=-1)
        return x

    def forward(self, v: torch.Tensor, t: torch.Tensor):
        """同时投影视觉和文本特征。

        Args:
            v: [B, 384] DINOv2 特征
            t: [B, 512] CLIP 文本特征

        Returns:
            z_v: [B, 256] 投影后的视觉特征
            z_t: [B, 256] 投影后的文本特征
        """
        z_v = self.forward_visual(v)
        z_t = self.forward_text(t)
        return z_v, z_t

    def count_parameters(self) -> int:
        """统计可训练参数数量。"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class CrossModalContrastiveLoss(nn.Module):
    """跨模态对比学习损失 (NT-Xent / InfoNCE)。

    用于在 Source Dataset (如 ImageNet-1K) 上无监督训练 Projection Head。

    损失函数:
        L = -log( exp(sim(z_v, z_t) / tau) / sum_j exp(sim(z_v, z_tj) / tau) )
    """

    def __init__(self, temperature: float = 0.07):
        """
        Args:
            temperature: 温度系数 tau
        """
        super().__init__()
        self.temperature = temperature
        self.cos_sim = nn.CosineSimilarity(dim=-1)

    def forward(
        self,
        z_v: torch.Tensor,
        z_t: torch.Tensor,
    ) -> torch.Tensor:
        """计算对比损失。

        Args:
            z_v: [B, D] 投影后的视觉特征 (已 L2 归一化)
            z_t: [B, D] 投影后的文本特征 (已 L2 归一化)

        Returns:
            loss: 标量损失值
        """
        logits = self.cos_sim(z_v.unsqueeze(1), z_t.unsqueeze(0)) / self.temperature
        labels = torch.arange(logits.size(0), device=logits.device)
        loss = nn.CrossEntropyLoss()(logits, labels)
        return loss


class AlignmentLoss(nn.Module):
    """余弦对齐损失。

    L_align = 1 - cos(z_v, z_t)
    """

    def __init__(self):
        super().__init__()
        self.cos_sim = nn.CosineSimilarity(dim=-1)

    def forward(
        self,
        z_v: torch.Tensor,
        z_t: torch.Tensor,
    ) -> torch.Tensor:
        """计算对齐损失。

        Args:
            z_v: [B, D] 投影后的视觉特征
            z_t: [B, D] 投影后的文本特征

        Returns:
            loss: 标量损失值 = 1 - mean(cosine_similarity)
        """
        sim = self.cos_sim(z_v, z_t)
        return torch.mean(1 - sim)


def create_projection_head(
    dino_dim: int = 384,
    clip_dim: int = 512,
    latent_dim: int = 256,
    dropout: float = 0.1,
) -> ProjectionHead:
    """工厂函数：创建投影头。"""
    return ProjectionHead(
        dino_dim=dino_dim,
        clip_dim=clip_dim,
        latent_dim=latent_dim,
        dropout=dropout,
    )


def create_loss_functions(
    temperature: float = 0.07,
    align_weight: float = 1.0,
) -> Tuple:
    """工厂函数：创建所有损失函数。"""
    return (
        CrossModalContrastiveLoss(temperature=temperature),
        AlignmentLoss(),
        align_weight,
    )