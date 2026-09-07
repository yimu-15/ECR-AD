
"""
跨模态分歧度计算模块
-----------------
- 计算 DINOv2 (视觉) 和 CLIP (语义) 特征之间的分歧度
- 使用投影后的 latent space 中的余弦距离
- 分歧度用于 Reliability Router 的不对称惩罚
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple
from typing import Optional

class CrossModalDisagreement(nn.Module):
    """跨模态分歧度计算器。

    计算视觉证据和语义证据之间的不一致程度。
    当两个模态对同一区域的判断存在显著分歧时，
    Router 应更加谨慎地分配权重。

    公式:
        D = 1 - cos(z_v, z_t)
    其中 z_v 和 z_t 是投影到统一 latent space 后的特征。
    """

    def __init__(self, epsilon: float = 1e-8):
        """
        Args:
            epsilon: 数值稳定性常数
        """
        super().__init__()
        self.epsilon = epsilon

    def forward(
        self,
        visual_features: torch.Tensor,
        textual_features: torch.Tensor,
        visual_weights: Optional[torch.Tensor] = None,
        textual_weights: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """计算跨模态分歧度 (图像级)。

        Args:
            visual_features: [B, D] 视觉特征
            textual_features: [B, D] 语义特征
            visual_weights: [B] 可选视觉权重
            textual_weights: [B] 可选语义权重 (未使用，保留接口一致性)

        Returns:
            disagreement: [B] 跨模态分歧度
        """
        # [B] 余弦距离
        cos_sim = F.cosine_similarity(visual_features, textual_features, dim=-1)
        disagreement = 1.0 - cos_sim  # [B]

        if visual_weights is not None:
            disagreement = disagreement * visual_weights.squeeze(-1)

        return disagreement.unsqueeze(-1)  # [B, 1]

    def compute_patch_level(
        self,
        z_v: torch.Tensor,
        z_t: torch.Tensor,
    ) -> torch.Tensor:
        """计算 patch-level 分歧度。

        Args:
            z_v: [B, N, D] 投影后的视觉 patch 特征
            z_t: [B, 1, D] 投影后的语义特征 (广播到所有 patch)

        Returns:
            disagreement: [B, N] 每个 patch 的分歧度 (范围 [0, 2])
        """
        # 确保 z_t 形状正确
        if z_t.dim() == 2:
            z_t = z_t.unsqueeze(1)  # [B, 1, D]

        # 余弦相似度
        cos_sim = F.cosine_similarity(z_v, z_t, dim=-1)  # [B, N]

        # 分歧度 = 1 - cosine_similarity
        disagreement = 1.0 - cos_sim  # [B, N]

        return disagreement

    def compute_image_level(
        self,
        disagreement_patch: torch.Tensor,
    ) -> torch.Tensor:
        """计算图像级分歧度 (取所有 patch 的均值)。

        Args:
            disagreement_patch: [B, N] patch-level 分歧度

        Returns:
            disagreement_img: [B] 图像级分歧度
        """
        return disagreement_patch.mean(dim=1)  # [B]

    def compute_weighted(
        self,
        z_v: torch.Tensor,
        z_t: torch.Tensor,
        visual_weights: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """计算加权分歧度。

        可以使用视觉权重对分歧度进行加权，突出重要区域。

        Args:
            z_v: [B, N, D] 投影后的视觉 patch 特征
            z_t: [B, 1, D] 投影后的语义特征
            visual_weights: [B, N] 可选的视觉权重

        Returns:
            disagreement: [B, N] 加权后的分歧度
        """
        base_disagree = self.compute_patch_level(z_v, z_t)

        if visual_weights is not None:
            # 加权: 高视觉可靠性的区域分歧度更可信
            weighted = base_disagree * visual_weights
            return weighted
        return base_disagree