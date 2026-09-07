
"""
不确定性计算模块
--------------
- Semantic Uncertainty: 基于 softmax 概率的熵
- Visual Uncertainty: 基于 patch 特征分散度的熵
- 支持 patch-level 和 image-level 的不确定性聚合
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


class SemanticUncertainty:
    """语义不确定性 (基于 CLIP softmax 概率的熵)。

    U_t = -sum_c p_c * log(p_c)
    其中 p_c 来自 CLIP 文本相似度的 softmax 概率。
    """

    def __init__(self, epsilon: float = 1e-8):
        """
        Args:
            epsilon: 数值稳定性常数
        """
        self.epsilon = epsilon

    def compute(
        self,
        p_normal: torch.Tensor,
        p_anomalous: torch.Tensor,
    ) -> torch.Tensor:
        """计算语义不确定性 (熵)。

        Args:
            p_normal: [B] 正常概率
            p_anomalous: [B] 异常概率

        Returns:
            uncertainty: [B] 熵值 (越高越不确定)
        """
        p_normal = torch.clamp(p_normal, min=self.epsilon, max=1.0)
        p_anomalous = torch.clamp(p_anomalous, min=self.epsilon, max=1.0)

        # 二元熵
        entropy = -(
            p_normal * torch.log(p_normal) +
            p_anomalous * torch.log(p_anomalous)
        )
        return entropy

    def normalize(
        self,
        entropy: torch.Tensor,
        eps: float = 1e-8,
    ) -> torch.Tensor:
        """将熵归一化到 [0, 1] 范围。

        Args:
            entropy: [B] 原始熵值
            eps: 数值稳定性常数

        Returns:
            normalized: [B] 归一化到 [0, 1] 的不确定性
        """
        # 二元熵的最大值为 log(2)
        max_entropy = torch.log(torch.tensor(2.0, device=entropy.device))
        normalized = entropy / (max_entropy + eps)
        return torch.clamp(normalized, 0.0, 1.0)


class VisualUncertainty:
    """视觉不确定性 (基于 patch 特征分散度)。

    通过计算 patch 特征之间的余弦相似度分散度来衡量视觉证据的可靠性。
    如果所有 patch 特征一致，则不确定性低；如果分散，则不确定性高。
    """

    def __init__(self, epsilon: float = 1e-8):
        self.epsilon = epsilon

    def compute_patch_uncertainty(
        self,
        patch_features: torch.Tensor,
    ) -> torch.Tensor:
        """基于 patch 特征分散度计算不确定性。

        方法: 计算每个 patch 与 patch 均值的余弦距离的均值。

        Args:
            patch_features: [B, N, D] patch 特征

        Returns:
            uncertainty: [B, N] 每个 patch 的不确定性
        """
        # 计算 patch 均值
        mean_feat = patch_features.mean(dim=1, keepdim=True)  # [B, 1, D]
        mean_feat = F.normalize(mean_feat, dim=-1)

        # 归一化所有 patch 特征
        patch_norm = F.normalize(patch_features, dim=-1)  # [B, N, D]

        # 余弦相似度 [B, N]
        cos_sim = torch.sum(patch_norm * mean_feat, dim=-1)  # [B, N]

        # 不确定性 = 1 - 平均余弦相似度 (越高越不确定)
        uncertainty = 1.0 - cos_sim  # [B, N]

        return uncertainty

    def compute_image_uncertainty(
        self,
        patch_features: torch.Tensor,
    ) -> torch.Tensor:
        """计算整张图像的不确定性 (取所有 patch 的均值)。

        Args:
            patch_features: [B, N, D] patch 特征

        Returns:
            uncertainty: [B] 图像级不确定性
        """
        patch_unc = self.compute_patch_uncertainty(patch_features)  # [B, N]
        return patch_unc.mean(dim=1)  # [B]

    def normalize(
        self,
        uncertainty: torch.Tensor,
        eps: float = 1e-8,
    ) -> torch.Tensor:
        """将不确定性归一化到 [0, 1] 范围。

        Args:
            uncertainty: [B] 原始不确定性值
            eps: 数值稳定性常数

        Returns:
            normalized: [B] 归一化到 [0, 1] 的不确定性
        """
        normalized = (uncertainty - uncertainty.min()) / (
            uncertainty.max() - uncertainty.min() + eps
        )
        return normalized


class UncertaintyEstimator(nn.Module):
    """统一不确定性估计器。

    接收视觉或语义证据 (evidence in [0,1]，越高越正常/可靠)，
    输出不确定性 in [0,1]。

    使用二分类熵作为不确定性度量：evidence=0.5 时不确定性最高，
    evidence→0 或 1 时不确定性最低（置信度高）。
    """

    def __init__(self, epsilon: float = 1e-8):
        super().__init__()
        self.epsilon = epsilon

    def forward(self, evidence: torch.Tensor) -> torch.Tensor:
        """计算不确定性。

        Args:
            evidence: [B, N, 1] 或 [B, N]，证据值 (越高越正常/可靠)

        Returns:
            uncertainty: 与输入同形状，归一化到 [0, 1] 的不确定性
        """
        p = torch.clamp(evidence, self.epsilon, 1.0 - self.epsilon)
        entropy = -(p * torch.log(p) + (1.0 - p) * torch.log(1.0 - p))
        max_entropy = torch.log(torch.tensor(2.0, device=evidence.device))
        uncertainty = entropy / (max_entropy + self.epsilon)
        return torch.clamp(uncertainty, 0.0, 1.0)