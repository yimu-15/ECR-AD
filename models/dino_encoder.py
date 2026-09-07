
"""
DINOv2 ViT-S/14 视觉编码器
--------------------------
- 输入: RGB 图像 [B, 3, 518, 518]
- 输出: patch tokens [B, 1370, 384] (1 CLS + 1369 patches)
- 支持提取多层特征用于多尺度聚合
"""
import torch
import timm
from typing import Tuple, Optional, List


class DINOv2Encoder:
    """DINOv2 ViT-S/14 特征提取器。

    使用 timm 库加载 DINOv2 ViT-Small/14 预训练模型。
    输入尺寸: 518x518 (patch_size=14 -> 37x37=1369 patches + 1 CLS token)
    输出维度: 384
    """

    def __init__(
        self,
        device: str = "cuda",
        multi_scale: bool = False,
        layer_indices: Optional[List[int]] = None,
    ):
        """
        Args:
            device: 'cuda' or 'cpu'
            multi_scale: 是否提取多层特征（用于多尺度异常图聚合）
            layer_indices: 要提取的层索引列表。默认 [6, 9, 11]
        """
        self.device = torch.device(device)
        self.multi_scale = multi_scale

        if layer_indices is None:
            layer_indices = [6, 9, 11] if multi_scale else [11]

        self.layer_indices = layer_indices

        # 创建模型
        self.model = timm.create_model(
            "vit_small_patch14_dinov2.lvd142m",
            pretrained=True,
            num_classes=0,
            exportable=False,
        )
        self.model = self.model.to(self.device)
        self.model.eval()

        # 输入尺寸
        self.input_size = 518
        self.patch_size = 14
        self.num_patches = (self.input_size // self.patch_size) ** 2  # 1369
        self.feature_dim = 384

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward(x)

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播，返回最后一层特征。

        Args:
            x: 输入图像 [B, 3, 518, 518]

        Returns:
            features: [B, 1370, 384]
        """
        x = x.to(self.device)
        features = self.model.forward_features(x)
        return features  # [B, 1370, 384]

    @torch.no_grad()
    def forward_multi_scale(self, x: torch.Tensor) -> List[torch.Tensor]:
        """多尺度前向传播，返回多层特征。

        Args:
            x: 输入图像 [B, 3, 518, 518]

        Returns:
            features_list: List of [B, 1370, 384]
        """
        x = x.to(self.device)
        all_features = {}

        def make_hook(layer_idx):
            def hook(module, input, output):
                all_features[layer_idx] = output
            return hook

        handles = []
        for name, module in self.model.named_modules():
            if "blocks" in name and "block" in name:
                try:
                    layer_num = int(name.split(".")[-2])
                    if layer_num in self.layer_indices:
                        h = module.register_forward_hook(make_hook(layer_num))
                        handles.append(h)
                except (ValueError, IndexError):
                    pass

        with torch.no_grad():
            _ = self.model.forward_features(x)

        result = [all_features[i] for i in sorted(all_features.keys())]
        return result

    @torch.no_grad()
    def extract_patch_features(
        self, images: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """提取 patch 特征并分离 CLS token。

        Args:
            images: [B, 3, 518, 518]

        Returns:
            cls_token: [B, 384]
            patch_tokens: [B, 1369, 384]
        """
        features = self.forward(images)
        cls_token = features[:, 0, :]
        patch_tokens = features[:, 1:, :]
        return cls_token, patch_tokens

    @torch.no_grad()
    def extract_patch_grid(
        self, images: torch.Tensor
    ) -> torch.Tensor:
        """提取 patch 特征并 reshape 为网格形式。

        Returns:
            grid: [B, 37, 37, 384]
        """
        _, patch_tokens = self.extract_patch_features(images)
        B, _, C = patch_tokens.shape
        H = W = int(patch_tokens.shape[1] ** 0.5)  # 37
        grid = patch_tokens.reshape(B, H, W, C)
        return grid

    def get_device(self) -> torch.device:
        return self.device

    def to(self, device):
        self.device = device
        self.model = self.model.to(device)
        return self

    def eval(self):
        self.model.eval()
        return self