
"""
多模态证据提取模块
----------------
- Visual Evidence: 基于 Universal Feature Distribution 的 Visual Typicality
- Semantic Evidence: 基于 CLIP 文本相似度的 Semantic Normality
- 支持 Class-Agnostic 和 Class-Aware 两种 prompt 模式
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, List


class VisualEvidence(nn.Module):
    """视觉证据提取器。

    基于 Universal Feature Distribution 计算 Visual Typicality。
    不使用 target category 的任何样本，仅依赖预训练的视觉特征空间。

    方法:
        - 使用 DINOv2 提取 patch 特征
        - 计算 patch 特征到全局均值的 Mahalanobis distance
        - 转为 Visual Typicality: q_v = exp(-d_v / sigma)
        - 定义为 Visual Evidence Reliability: R_v = 1 / (1 + U_v)
    """

    def __init__(
        self,
        feature_dim: int = 384,
        sigma: float = 1.0,
        device: str = "cuda",
    ):
        """
        Args:
            feature_dim: 视觉特征维度 (384 for DINOv2 ViT-S/14)
            sigma: Typicality 计算的温度参数，控制分布的集中程度
            device: 设备
        """
        super().__init__()
        self.device = torch.device(device)
        self.feature_dim = feature_dim
        self.sigma = sigma

        # Universal Feature Distribution 的统计量 (从 source domain 预计算)
        self.register_buffer("global_mean", torch.zeros(feature_dim))
        self.register_buffer("global_inv_cov", torch.eye(feature_dim))
        self.is_initialized = False

    def initialize_from_stats(
        self,
        mean: torch.Tensor,
        cov_inv: torch.Tensor,
    ):
        """从预计算的统计量初始化 Universal Distribution。

        Args:
            mean: [D] 全局特征均值
            cov_inv: [D, D] 协方差的逆 (伪逆)
        """
        self.global_mean = mean.to(self.device)
        self.global_inv_cov = cov_inv.to(self.device)
        self.is_initialized = True

    def compute_mahalanobis_distance(
        self,
        patch_features: torch.Tensor,
    ) -> torch.Tensor:
        """计算 Mahalanobis distance。

        Args:
            patch_features: [B, N, D] patch 特征 (N patches)

        Returns:
            distances: [B, N] Mahalanobis distance per patch
        """
        if not self.is_initialized:
            # 如果未初始化，使用简单的 L2 distance 作为 fallback
            mean = self.global_mean.unsqueeze(0).unsqueeze(0)  # [1, 1, D]
            dist = torch.cdist(patch_features, mean, p=2)  # [B, N, 1]
            return dist.squeeze(-1)  # [B, N]

        # Mahalanobis distance: sqrt((x - mu)^T * Sigma^-1 * (x - mu))
        diff = patch_features - self.global_mean  # [B, N, D]
        left = torch.matmul(diff, self.global_inv_cov)  # [B, N, D]
        dist_sq = torch.sum(left * diff, dim=-1)  # [B, N]
        dist_sq = torch.clamp(dist_sq, min=1e-8)  # 防止数值不稳定
        return torch.sqrt(dist_sq)

    def compute_typicality(
        self,
        patch_features: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """计算 Visual Typicality 和 Reliability。

        Args:
            patch_features: [B, N, D] patch 特征

        Returns:
            typicality: [B, N] 典型性分数 (越高越 normal)
            reliability: [B, N] 可靠性分数
        """
        distances = self.compute_mahalanobis_distance(patch_features)  # [B, N]

        # Visual Typicality: q_v = exp(-d_v / sigma)
        typicality = torch.exp(-distances / self.sigma)  # [B, N]

        # Visual Evidence Reliability: R_v = 1 / (1 + U_v)
        # 其中 U_v 为 normalized feature dispersion
        U_v = distances / (distances.max(dim=1, keepdim=True).values + 1e-8)
        reliability = 1.0 / (1.0 + U_v)  # [B, N]

        return typicality, reliability

    def forward(self, patch_features: torch.Tensor) -> torch.Tensor:
        """计算视觉证据 (Visual Typicality)。

        Args:
            patch_features: [B, N, D] patch 特征

        Returns:
            evidence: [B, N, 1] 视觉典型性证据 (越高越正常)
        """
        typicality, _ = self.compute_typicality(patch_features)  # [B, N]
        return typicality.unsqueeze(-1)  # [B, N, 1]

    def get_device(self) -> torch.device:
        return self.device


class SemanticEvidence(nn.Module):
    """语义证据提取器。

    基于 CLIP 文本编码器计算 Semantic Normality。
    支持两种 prompt 模式:
        - Class-Agnostic: "a photo of a normal object" / "a photo of an anomalous object"
        - Class-Aware: "a normal bottle" / "an anomalous bottle" (仅使用类别名)

    概率计算:
        p_N = exp(s_N / tau) / (exp(s_N / tau) + exp(s_A / tau))
        p_A = exp(s_A / tau) / (exp(s_N / tau) + exp(s_A / tau))
    """

    def __init__(
        self,
        temperature: float = 0.07,
        mode: str = "class_agnostic",
        category_name: Optional[str] = None,
        device: str = "cuda",
    ):
        """
        Args:
            temperature: softmax 温度
            mode: 'class_agnostic' 或 'class_aware'
            category_name: 类别名称 (class_aware 模式下使用，如 "bottle")
            device: 设备
        """
        super().__init__()
        self.device = torch.device(device)
        self.temperature = temperature
        self.mode = mode
        self.category_name = category_name

        # 定义 prompt 模板
        if mode == "class_agnostic":
            self.normal_prompt = "a photo of a normal object"
            self.anomalous_prompt = "a photo of an anomalous object"
        else:  # class_aware
            if category_name is None:
                raise ValueError("category_name must be provided for class_aware mode")
            self.normal_prompt = f"a normal {category_name}"
            self.anomalous_prompt = f"an anomalous {category_name}"

        self.prompts = [self.normal_prompt, self.anomalous_prompt]

    def set_category(self, category_name: str):
        """动态设置类别名称 (用于 class_aware 模式)。"""
        self.category_name = category_name
        self.normal_prompt = f"a normal {category_name}"
        self.anomalous_prompt = f"an anomalous {category_name}"
        self.prompts = [self.normal_prompt, self.anomalous_prompt]

    def compute_semantic_normality(
        self,
        image_features: torch.Tensor,
        text_normal: torch.Tensor,
        text_anomalous: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """计算语义正常度。

        Args:
            image_features: [B, 512] 归一化图像特征
            text_normal: [1, 512] 归一化正常文本特征
            text_anomalous: [1, 512] 归一化异常文本特征

        Returns:
            p_normal: [B] 正常概率
            p_anomalous: [B] 异常概率
            semantic_score: [B] 异常分数 (p_anomalous - p_normal)
        """
        # 计算相似度 [B, 1]
        sim_normal = torch.matmul(image_features, text_normal.T).squeeze(-1)  # [B]
        sim_anomalous = torch.matmul(image_features, text_anomalous.T).squeeze(-1)  # [B]

        # 温度缩放的 softmax
        logits = torch.stack([sim_normal, sim_anomalous], dim=-1) / self.temperature  # [B, 2]
        probs = F.softmax(logits, dim=-1)  # [B, 2]

        p_normal = probs[:, 0]  # [B]
        p_anomalous = probs[:, 1]  # [B]

        # 异常分数
        semantic_score = p_anomalous - p_normal  # [B]

        return p_normal, p_anomalous, semantic_score

    def get_prompts(self) -> List[str]:
        """获取当前使用的 prompt 列表。"""
        return self.prompts

    def set_text_features(
        self,
        text_normal: torch.Tensor,
        text_anomalous: torch.Tensor,
    ):
        """设置预计算的文本特征 (需与 patch 特征处于同一 latent space)。

        Args:
            text_normal: [1, D] 归一化的正常文本特征
            text_anomalous: [1, D] 归一化的异常文本特征
        """
        self.register_buffer("text_normal", text_normal)
        self.register_buffer("text_anomalous", text_anomalous)

    def forward(self, patch_features: torch.Tensor) -> torch.Tensor:
        """计算语义证据 (Semantic Normality 概率)。

        Args:
            patch_features: [B, N, D] patch 特征 (需与文本特征同空间)

        Returns:
            evidence: [B, N, 1] 正常概率 (越高越正常)
        """
        if not hasattr(self, "text_normal"):
            # 未设置文本特征时，使用特征范数作为 fallback 证据
            norm = torch.norm(patch_features, dim=-1, keepdim=True)
            evidence = 1.0 / (1.0 + norm)
            return evidence

        B, N, D = patch_features.shape
        patches_flat = patch_features.reshape(B * N, D)  # [B*N, D]

        sim_normal = torch.matmul(patches_flat, self.text_normal.T).squeeze(-1)  # [B*N]
        sim_anomalous = torch.matmul(patches_flat, self.text_anomalous.T).squeeze(-1)  # [B*N]

        logits = torch.stack([sim_normal, sim_anomalous], dim=-1) / self.temperature
        probs = F.softmax(logits, dim=-1)  # [B*N, 2]
        p_normal = probs[:, 0].reshape(B, N, 1)  # [B, N, 1]
        return p_normal