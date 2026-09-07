
"""
Reliability Router (可靠性路由仲裁器)
----------------------------------
- 核心创新: Asymmetric Reliability Arbitration
- 根据模态不确定性 + 跨模态分歧度，动态分配视觉/语义权重

核心公式:
    A_m = -U_m * (1 + lambda * D)
    w_m = softmax(A_v, A_t)

物理意义:
    当模态冲突 D 增大时，不确定性 U_m 越高的模态会受到指数级放大的惩罚，
    权重自动向高可靠性的模态倾斜。
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict, Optional


class ReliabilityRouter(nn.Module):
    """跨模态可靠性路由仲裁器。

    核心功能: 根据每个 patch 的视觉/语义不确定性和模态分歧度，
    动态计算每个 patch 应信任视觉还是语义。

    不对称仲裁公式:
        A_v = -U_v * (1 + lambda * D)   # 视觉注意力分数
        A_t = -U_t * (1 + lambda * D)   # 语义注意力分数
        w_v = softmax(A_v, A_t)          # 视觉权重
        w_t = softmax(A_v, A_t)          # 语义权重

    最终异常分数:
        S_final = w_v * S_v + w_t * S_t
    """

    def __init__(
        self,
        lambda_weight: float = 1.0,
        temperature: float = 1.0,
        epsilon: float = 1e-8,
    ):
        """
        Args:
            lambda_weight: 分歧度放大系数 lambda，控制分歧对权重的影响强度
            temperature: softmax 温度，控制权重分配的平滑程度
            epsilon: 数值稳定性常数
        """
        super().__init__()
        self.lambda_weight = lambda_weight
        self.temperature = temperature
        self.epsilon = epsilon

    def forward(
        self,
        uncertainty_v: torch.Tensor,
        uncertainty_t: torch.Tensor,
        disagreement: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """计算路由权重。

        Args:
            uncertainty_v: [B, 1] 视觉不确定性
            uncertainty_t: [B, 1] 语义不确定性
            disagreement: [B, 1] 跨模态分歧度

        Returns:
            w_v: [B, 1] 视觉权重
            w_t: [B, 1] 语义权重
        """
        # 不对称惩罚: A_m = -U_m * (1 + lambda * D)
        score_v = -uncertainty_v * (1.0 + self.lambda_weight * disagreement)
        score_t = -uncertainty_t * (1.0 + self.lambda_weight * disagreement)

        # 温度缩放 softmax
        logits = torch.cat([score_v, score_t], dim=-1) / self.temperature
        weights = F.softmax(logits, dim=-1)

        w_v = weights[:, 0:1]  # [B, 1]
        w_t = weights[:, 1:2]  # [B, 1]
        return w_v, w_t

    def compute_asymmetric_scores(
        self,
        visual_uncertainty: torch.Tensor,
        semantic_uncertainty: torch.Tensor,
        disagreement: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """计算不对称注意力分数。

        A_m = -U_m * (1 + lambda * D)

        Args:
            visual_uncertainty: [B, N] 视觉不确定性 (已归一化到 [0,1])
            semantic_uncertainty: [B, N] 语义不确定性 (已归一化到 [0,1])
            disagreement: [B, N] 跨模态分歧度

        Returns:
            score_v: [B, N] 视觉注意力分数
            score_t: [B, N] 语义注意力分数
        """
        # 不对称惩罚: 分歧度 D 放大不确定性高的模态的负分数
        # 当 D 大时，U 大的模态 score 更负 -> softmax 后权重更小
        score_v = -visual_uncertainty * (1.0 + self.lambda_weight * disagreement)
        score_t = -semantic_uncertainty * (1.0 + self.lambda_weight * disagreement)

        return score_v, score_t

    def compute_weights(
        self,
        visual_uncertainty: torch.Tensor,
        semantic_uncertainty: torch.Tensor,
        disagreement: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """计算最终的路由权重。

        Args:
            visual_uncertainty: [B, N] 视觉不确定性 (已归一化)
            semantic_uncertainty: [B, N] 语义不确定性 (已归一化)
            disagreement: [B, N] 跨模态分歧度

        Returns:
            weight_v: [B, N] 视觉权重 (sum to 1)
            weight_t: [B, N] 语义权重 (sum to 1)
        """
        score_v, score_t = self.compute_asymmetric_scores(
            visual_uncertainty, semantic_uncertainty, disagreement
        )

        # 温度缩放的 softmax
        logits = torch.stack([score_v, score_t], dim=-1) / self.temperature
        weights = F.softmax(logits, dim=-1)

        weight_v = weights[:, :, 0]  # [B, N]
        weight_t = weights[:, :, 1]  # [B, N]

        return weight_v, weight_t

    def route(
        self,
        visual_score: torch.Tensor,
        semantic_score: torch.Tensor,
        visual_uncertainty: torch.Tensor,
        semantic_uncertainty: torch.Tensor,
        disagreement: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """完整的路由仲裁流程。

        Args:
            visual_score: [B, N] 视觉异常分数
            semantic_score: [B, N] 语义异常分数
            visual_uncertainty: [B, N] 视觉不确定性 (已归一化)
            semantic_uncertainty: [B, N] 语义不确定性 (已归一化)
            disagreement: [B, N] 跨模态分歧度

        Returns:
            results: dict 包含:
                - final_score: [B, N] 路由后的最终分数
                - weight_v: [B, N] 视觉权重
                - weight_t: [B, N] 语义权重
                - score_v: [B, N] 视觉注意力分数
                - score_t: [B, N] 语义注意力分数
        """
        # 确保所有输入形状一致 [B, N]
        B, N = visual_score.shape

        # 计算路由权重
        weight_v, weight_t = self.compute_weights(
            visual_uncertainty, semantic_uncertainty, disagreement
        )

        # 加权融合
        final_score = weight_v * visual_score + weight_t * semantic_score

        return {
            "final_score": final_score,
            "weight_v": weight_v,
            "weight_t": weight_t,
            "score_v": self.compute_asymmetric_scores(
                visual_uncertainty, semantic_uncertainty, disagreement
            )[0],
            "score_t": self.compute_asymmetric_scores(
                visual_uncertainty, semantic_uncertainty, disagreement
            )[1],
        }

    def modality_dropout_forward(
        self,
        visual_score: torch.Tensor,
        semantic_score: torch.Tensor,
        visual_uncertainty: torch.Tensor,
        semantic_uncertainty: torch.Tensor,
        disagreement: torch.Tensor,
        dropout_rate: float = 0.25,
        training: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """模态 Dropout 前向传播。

        训练时随机 mask 视觉或语义分支，让 Router 学会在模态缺失/不可靠时
        依赖另一模态。这是一种 self-supervised robustness objective。

        Args:
            visual_score: [B, N] 视觉异常分数
            semantic_score: [B, N] 语义异常分数
            visual_uncertainty: [B, N] 视觉不确定性
            semantic_uncertainty: [B, N] 语义不确定性
            disagreement: [B, N] 跨模态分歧度
            dropout_rate: mask 概率
            training: 是否在训练模式

        Returns:
            results: 同上
        """
        if training and dropout_rate > 0:
            # 随机选择每个样本使用哪个模态
            # Case 1: 只用视觉 (概率 0.25)
            # Case 2: 只用语义 (概率 0.25)
            # Case 3: 融合 (概率 0.5)
            batch_size = visual_score.shape[0]
            mode = torch.full((batch_size,), 0.5, device=visual_score.device)  # 0: vis, 1: sem, 2: both
            mode = torch.multinomial(mode, num_samples=1).squeeze(-1)

            # 根据 mode 选择策略
            weight_v = torch.ones_like(visual_uncertainty)
            weight_t = torch.ones_like(semantic_uncertainty)

            mask_vis = (mode == 0) | (mode == 2)  # 使用视觉
            mask_sem = (mode == 1) | (mode == 2)  # 使用语义

            if not mask_vis.all():
                weight_v[mask_vis.logical_not()] = 0.0
            if not mask_sem.all():
                weight_t[mask_sem.logical_not()] = 0.0

            # 重新归一化
            total = weight_v + weight_t
            weight_v = weight_v / (total + self.epsilon)
            weight_t = weight_t / (total + self.epsilon)

            final_score = weight_v * visual_score + weight_t * semantic_score

            return {
                "final_score": final_score,
                "weight_v": weight_v,
                "weight_t": weight_t,
                "score_v": torch.zeros_like(visual_score),
                "score_t": torch.zeros_like(semantic_score),
                "dropout_mode": mode,
            }
        else:
            # 推理模式: 正常路由
            return self.route(
                visual_score, semantic_score,
                visual_uncertainty, semantic_uncertainty, disagreement
            )