
"""
ECR-AD 2.0 完整 Pipeline
======================
Evidence-Calibrated Cross-Modal Routing for Zero-Shot Anomaly Detection

架构总览:
    Input Image (518x518)
         |
    +----+----+
    v         v
 DINOv2    CLIP (Image)
    |         |
 patch feat  img feat
    |         |
    v         v
 Projection Head (frozen after alignment)
    |         |
 z_v (256)  z_t (256)
    |         |
    v         v
Visual      Semantic
Evidence    Evidence
    |         |
    v         v
Uncertainty Uncertainty
    |         |
    +----+----+
         v
  Cross-Modal Disagreement
         v
  Reliability Router (Asymmetric Arbitration)
         v
  Weighted Fusion -> Anomaly Score
         v
  Multi-scale Aggregation + Bilateral Refinement
         v
  Pixel Anomaly Map
"""
import torch
import torch.nn as nn
from typing import Dict, Optional, List, Tuple
import numpy as np

from models.dino_encoder import DINOv2Encoder
from models.clip_encoder import CLIPEncoder
from models.projection import ProjectionHead
from models.evidence import VisualEvidence, SemanticEvidence
from models.uncertainty import SemanticUncertainty, VisualUncertainty
from models.disagreement import CrossModalDisagreement
from models.reliability_router import ReliabilityRouter


class ECRAD(nn.Module):
    """ECR-AD 2.0 完整模型。

    包含:
        1. DINOv2 ViT-S/14 视觉编码器 (frozen)
        2. CLIP ViT-B/32 图文编码器 (frozen)
        3. Projection Head (trainable, frozen after alignment)
        4. Visual Evidence (Universal Feature Distribution)
        5. Semantic Evidence (CLIP text prompts)
        6. Uncertainty Estimation
        7. Cross-Modal Disagreement
        8. Reliability Router (Asymmetric Arbitration)
        9. Multi-scale Aggregation
        10. Bilateral Refinement
    """

    def __init__(
        self,
        device: str = "cuda",
        latent_dim: int = 256,
        lambda_weight: float = 1.0,
        temperature: float = 1.0,
        sigma: float = 1.0,
        prompt_mode: str = "class_agnostic",
        category_name: Optional[str] = None,
        multi_scale: bool = True,
    ):
        """
        Args:
            device: 'cuda' or 'cpu'
            latent_dim: Projection latent space dimension
            lambda_weight: Disagreement amplification coefficient
            temperature: Softmax temperature for router
            sigma: Visual typicality temperature
            prompt_mode: 'class_agnostic' or 'class_aware'
            category_name: Category name for class_aware mode
            multi_scale: Whether to use multi-scale DINOv2 features
        """
        super().__init__()

        self.device = torch.device(device)
        self.prompt_mode = prompt_mode
        self.category_name = category_name
        self.multi_scale = multi_scale

        # === Stage 1: Frozen Foundation Models ===
        self.dino = DINOv2Encoder(device=device, multi_scale=multi_scale)
        self.clip = CLIPEncoder(device=device)

        # === Projection Head (trainable, then frozen) ===
        self.projection = ProjectionHead(
            dino_dim=384,
            clip_dim=512,
            latent_dim=latent_dim,
            dropout=0.1,
        )

        # === Stage 2: Evidence Extraction ===
        self.visual_evidence = VisualEvidence(
            feature_dim=384,
            sigma=sigma,
            device=device,
        )
        self.semantic_evidence = SemanticEvidence(
            temperature=0.07,
            mode=prompt_mode,
            category_name=category_name,
            device=device,
        )

        # === Uncertainty Estimation ===
        self.semantic_uncertainty = SemanticUncertainty()
        self.visual_uncertainty = VisualUncertainty()

        # === Cross-Modal Disagreement ===
        self.disagreement = CrossModalDisagreement()

        # === Reliability Router ===
        self.router = ReliabilityRouter(
            lambda_weight=lambda_weight,
            temperature=temperature,
        )

        # === Multi-scale aggregation weights ===
        if multi_scale:
            self.scale_weights = nn.Parameter(torch.tensor([0.3, 0.3, 0.4]))

        # 将子模块移到指定设备
        self.to(device)

    def set_category(self, category_name: str):
        """设置当前类别名称 (用于 class_aware 模式)。"""
        self.category_name = category_name
        self.semantic_evidence.set_category(category_name)

    def forward(
        self,
        images: torch.Tensor,
        training: bool = False,
        dropout_rate: float = 0.25,
    ) -> Dict[str, torch.Tensor]:
        """完整前向传播。

        Args:
            images: [B, 3, 518, 518] RGB 图像
            training: 是否在训练模式 (用于 modality dropout)
            dropout_rate: modality dropout 概率

        Returns:
            results: dict 包含所有中间结果和最终分数
        """
        results = {}

        # === Stage 1: Feature Extraction ===
        # DINOv2 visual features
        if self.multi_scale:
            dino_features_list = self.dino.forward_multi_scale(images)
        else:
            dino_features_list = [self.dino.forward(images)]

        # CLIP image features (from 224x224, need to resize)
        clip_images = torch.nn.functional.interpolate(
            images, size=224, mode='bilinear', align_corners=False
        )
        clip_img_features = self.clip.encode_image(clip_images)  # [B, 512]

        # CLIP text features
        text_normal = self.clip.encode_text(
            [self.semantic_evidence.normal_prompt]
        )  # [1, 512]
        text_anomalous = self.clip.encode_text(
            [self.semantic_evidence.anomalous_prompt]
        )  # [1, 512]

        # === Projection ===
        # Project visual features (use CLS token + patch mean)
        z_v_list = []
        for dino_feats in dino_features_list:
            cls_token = dino_feats[:, 0, :]  # [B, 384]
            patch_mean = dino_feats[:, 1:, :].mean(dim=1)  # [B, 384]
            combined = (cls_token + patch_mean) / 2  # [B, 384]
            z_v = self.projection.forward_visual(combined)  # [B, 256]
            z_v_list.append(z_v)

        # Project text features
        z_t_normal = self.projection.forward_text(text_normal)  # [1, 256]
        z_t_anomalous = self.projection.forward_text(text_anomalous)  # [1, 256]

        results["z_v_list"] = z_v_list
        results["z_t_normal"] = z_t_normal
        results["z_t_anomalous"] = z_t_anomalous

        # === Stage 2: Visual Evidence ===
        visual_typicality_list = []
        visual_reliability_list = []
        visual_uncertainty_list = []

        for dino_feats in dino_features_list:
            patch_tokens = dino_feats[:, 1:, :]  # [B, 1369, 384]
            typicality, reliability = self.visual_evidence.compute_typicality(
                patch_tokens
            )
            unc = self.visual_uncertainty.compute_patch_uncertainty(patch_tokens)
            visual_typicality_list.append(typicality)
            visual_reliability_list.append(reliability)
            visual_uncertainty_list.append(unc)

        # Multi-scale aggregation (weighted average)
        if self.multi_scale:
            visual_typicality = sum(
                w * t for w, t in zip(
                    torch.softmax(self.scale_weights, dim=0),
                    visual_typicality_list
                )
            )
            visual_reliability = sum(
                w * r for w, r in zip(
                    torch.softmax(self.scale_weights, dim=0),
                    visual_reliability_list
                )
            )
            visual_uncertainty = sum(
                w * u for w, u in zip(
                    torch.softmax(self.scale_weights, dim=0),
                    visual_uncertainty_list
                )
            )
        else:
            visual_typicality = visual_typicality_list[0]
            visual_reliability = visual_reliability_list[0]
            visual_uncertainty = visual_uncertainty_list[0]

        results["visual_typicality"] = visual_typicality
        results["visual_reliability"] = visual_reliability
        results["visual_uncertainty"] = visual_uncertainty

        # === Stage 3: Semantic Evidence ===
        # 为每个 patch 计算语义证据 (使用 CLS token 作为图像代表)
        B = images.shape[0]
        N = 1369  # number of patches

        # 广播语义特征到所有 patch
        z_t_n = z_t_normal.unsqueeze(1).expand(B, N, -1)  # [B, N, 256]
        z_t_a = z_t_anomalous.unsqueeze(1).expand(B, N, -1)  # [B, N, 256]

        # 计算语义概率
        # 使用投影后的特征计算相似度
        p_normal = torch.softmax(
            torch.sum(z_v_list[0] * z_t_n, dim=-1) / 0.07, dim=-1
        ) if False else None  # placeholder

        # 简化: 直接使用 CLIP 原始特征的相似度
        sim_normal = torch.matmul(clip_img_features, text_normal.T).squeeze(-1)  # [B]
        sim_anomalous = torch.matmul(clip_img_features, text_anomalous.T).squeeze(-1)  # [B]

        logits = torch.stack([sim_normal, sim_anomalous], dim=-1) / 0.07
        probs = torch.softmax(logits, dim=-1)

        p_normal = probs[:, 0]  # [B]
        p_anomalous = probs[:, 1]  # [B]

        # 广播到 patch level
        p_normal_patch = p_normal.unsqueeze(-1).expand(B, N)  # [B, N]
        p_anomalous_patch = p_anomalous.unsqueeze(-1).expand(B, N)  # [B, N]

        # Semantic score (anomaly score)
        semantic_score = p_anomalous_patch - p_normal_patch  # [B, N]

        # Semantic uncertainty
        semantic_unc = self.semantic_uncertainty.compute(p_normal, p_anomalous)
        semantic_unc_patch = semantic_unc.unsqueeze(-1).expand(B, N)  # [B, N]

        results["semantic_score"] = semantic_score
        results["p_normal"] = p_normal
        results["p_anomalous"] = p_anomalous
        results["semantic_uncertainty"] = semantic_unc_patch

        # === Stage 4: Cross-Modal Disagreement ===
        # 使用投影后的视觉特征和语义特征计算分歧
        z_v_patch = z_v_list[0].unsqueeze(1).expand(B, N, -1)  # [B, N, 256]
        disagreement = self.disagreement.compute_patch_level(
            z_v_patch, z_t_normal
        )  # [B, N]

        results["disagreement"] = disagreement

        # === Stage 5: Reliability Routing ===
        # 归一化不确定性到 [0, 1]
        vis_unc_norm = self.visual_uncertainty.normalize(
            self.visual_uncertainty.compute_image_uncertainty(
                dino_features_list[0][:, 1:, :]
            )
        ).unsqueeze(-1).expand(B, N)

        sem_unc_norm = self.semantic_uncertainty.normalize(semantic_unc)
        sem_unc_norm_patch = sem_unc_norm.unsqueeze(-1).expand(B, N)

        # 视觉分数: 基于 typicality (越高越 normal -> 异常分数 = 1 - typicality)
        visual_anomaly_score = 1.0 - visual_typicality  # [B, N]

        # 路由仲裁
        router_output = self.router.modality_dropout_forward(
            visual_score=visual_anomaly_score,
            semantic_score=semantic_score,
            visual_uncertainty=vis_unc_norm,
            semantic_uncertainty=sem_unc_norm_patch,
            disagreement=disagreement,
            dropout_rate=dropout_rate,
            training=training,
        )

        results["final_score"] = router_output["final_score"]
        results["weight_v"] = router_output["weight_v"]
        results["weight_t"] = router_output["weight_t"]

        return results

    @torch.no_grad()
    def infer(
        self,
        images: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """推理模式 (无 dropout)。

        Args:
            images: [B, 3, 518, 518]

        Returns:
            results: 包含 final_score, weight_v, weight_t 等
        """
        return self.forward(images, training=False, dropout_rate=0.0)

    def count_trainable_parameters(self) -> int:
        """统计可训练参数数量。"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def freeze_backbones(self):
        """冻结 backbone 和 projection (仅训练 router 等轻量模块)。"""
        self.dino.model.eval()
        for param in self.dino.model.parameters():
            param.requires_grad = False
        self.clip.model.eval()
        for param in self.clip.model.parameters():
            param.requires_grad = False
        self.projection.eval()
        for param in self.projection.parameters():
            param.requires_grad = False

    def get_device(self) -> torch.device:
        return self.device