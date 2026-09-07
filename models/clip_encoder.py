
"""
CLIP ViT-B/32 图文编码器
------------------------
- 图像编码: PIL Image -> [B, 512]
- 文本编码: 文本列表 -> [B, 512]
- 使用 open_clip 库加载 Laion2B 预训练权重
"""
import torch
import open_clip
from PIL import Image
from typing import List, Optional, Union


class CLIPEncoder:
    """CLIP ViT-B/32 图文编码器。

    使用 open_clip 库加载 Laion2B-s/34B 预训练权重。
    图像输入: 224x224 (CLIP 标准尺寸)
    文本输入: 经过 tokenizer 处理的 token IDs
    输出维度: 512
    """

    def __init__(
        self,
        model_name: str = "ViT-B-32",
        pretrained: str = "laion2b_s34b_b79k",
        device: str = "cuda",
    ):
        """
        Args:
            model_name: CLIP 模型架构名
            pretrained: 预训练权重名
            device: 'cuda' or 'cpu'
        """
        self.device = torch.device(device)

        # 加载模型和预处理
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            model_name, pretrained=pretrained
        )
        self.model = self.model.to(self.device)
        self.model.eval()

        # 加载 tokenizer
        self.tokenizer = open_clip.get_tokenizer(model_name)

        # 模型配置
        self.image_dim = 512
        self.text_dim = 512
        self.input_size = 224

    def __call__(self, x) -> torch.Tensor:
        return self.encode_image(x)

    @torch.no_grad()
    def encode_image(
        self,
        images: Union[torch.Tensor, List[Image.Image], Image.Image],
        normalize: bool = True,
    ) -> torch.Tensor:
        """编码图像为 CLIP patch tokens。

        通过 forward hook 捕获视觉 transformer 在 ln_post 之后的完整 token
        序列 (含 CLS token)，并应用 proj 映射到 embedding 维度。

        Args:
            images: 可以是 torch.Tensor [B, 3, 224, 224] / List[PIL.Image] / PIL.Image
            normalize: 是否 L2 归一化特征

        Returns:
            features: [B, N+1, 512] patch tokens (含 CLS)，ViT-B/32 时 N=49
        """
        self.model.eval()

        if isinstance(images, Image.Image):
            images = [images]
        if isinstance(images, list):
            images = torch.stack([self.preprocess(img) for img in images])
        images = images.to(self.device)

        # 捕获 ln_post 输出的完整 token 序列 [B, N+1, D]
        captured = {}

        def _hook(module, inp, out):
            captured["tokens"] = out

        handle = self.model.visual.ln_post.register_forward_hook(_hook)
        _ = self.model.encode_image(images)  # 触发前向
        handle.remove()

        tokens = captured["tokens"]  # [B, N+1, D]

        # 应用 proj (D -> embed_dim)，与 encode_image 的 CLS 输出保持同空间
        proj = getattr(self.model.visual, "proj", None)
        if proj is not None:
            tokens = tokens @ proj  # [B, N+1, embed_dim]

        if normalize:
            tokens = torch.nn.functional.normalize(tokens, dim=-1)
        return tokens

    @torch.no_grad()
    def encode_text(
        self,
        texts: List[str],
        normalize: bool = True,
    ) -> torch.Tensor:
        """编码文本为 CLIP 特征。

        Args:
            texts: 文本列表
            normalize: 是否 L2 归一化特征

        Returns:
            features: [B, 512] 归一化文本特征
        """
        self.model.eval()
        tokens = self.tokenizer(texts).to(self.device)
        features = self.model.encode_text(tokens)
        if normalize:
            features = torch.nn.functional.normalize(features, dim=-1)
        return features

    @torch.no_grad()
    def get_text_prompts(
        self,
        prompt_templates: Optional[List[str]] = None,
        normalize: bool = True,
    ) -> torch.Tensor:
        """获取文本 prompt 特征。

        Args:
            prompt_templates: prompt 模板列表
                           默认: ["a photo of a normal object", "a photo of an anomalous object"]
            normalize: 是否归一化

        Returns:
            text_features: [num_prompts, 512]
        """
        if prompt_templates is None:
            prompt_templates = [
                "a photo of a normal object",
                "a photo of an anomalous object",
            ]
        return self.encode_text(prompt_templates, normalize=normalize)

    @torch.no_grad()
    def get_image_text_similarity(
        self,
        image_features: torch.Tensor,
        text_features: torch.Tensor,
    ) -> torch.Tensor:
        """计算图像与文本的余弦相似度。

        Args:
            image_features: [B, 512] 归一化图像特征
            text_features: [N, 512] 归一化文本特征

        Returns:
            similarity: [B, N] 相似度矩阵
        """
        similarity = torch.matmul(image_features, text_features.T)
        return similarity

    def get_device(self) -> torch.device:
        return self.device

    def to(self, device):
        self.device = device
        self.model = self.model.to(device)
        return self

    def eval(self):
        self.model.eval()
        return self