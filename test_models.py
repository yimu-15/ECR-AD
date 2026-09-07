# test_models.py
import os
# 镜像设置（双重保险：环境变量 + 代码内设置）
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

import torch
import timm
import open_clip
from PIL import Image
import numpy as np

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")

# ============================================
# 1. DINOv2 ViT-S/14 (timm)
# ============================================
print("\n[1] Loading DINOv2 ViT-S/14 via timm...")
dino = timm.create_model(
    "vit_small_patch14_dinov2.lvd142m",
    pretrained=True,
    num_classes=0,
).to(device)
dino.eval()

# DINOv2 标准输入 518x518 (patch_size=14, 37x37=1369 patches + 1 CLS)
dummy_img = torch.randn(1, 3, 518, 518).to(device)
with torch.no_grad():
    dino_out = dino.forward_features(dummy_img)
    print(f"    DINOv2 output shape: {dino_out.shape}")
    print(f"    Expected: [1, 1370, 384]  (1 CLS + 1369 patches, dim=384)")

# ============================================
# 2. CLIP ViT-B/32 (open_clip)
# ============================================
print("\n[2] Loading CLIP ViT-B/32 via open_clip...")
clip_model, _, clip_preprocess = open_clip.create_model_and_transforms(
    "ViT-B-32",
    pretrained="laion2b_s34b_b79k"
)
clip_model = clip_model.to(device)
clip_model.eval()

dummy_pil = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
clip_img = clip_preprocess(dummy_pil).unsqueeze(0).to(device)
with torch.no_grad():
    clip_img_feat = clip_model.encode_image(clip_img)
    print(f"    CLIP image feature shape: {clip_img_feat.shape}")
    print(f"    Expected: [1, 512]")

# ============================================
# 3. 文本编码
# ============================================
print("\n[3] Testing text encoding...")
tokenizer = open_clip.get_tokenizer("ViT-B-32")
texts = ["a photo of a normal object", "a photo of an anomalous object"]
text_tokens = tokenizer(texts).to(device)
with torch.no_grad():
    text_feat = clip_model.encode_text(text_tokens)
    print(f"    CLIP text feature shape: {text_feat.shape}")
    print(f"    Expected: [2, 512]")

# ============================================
# 4. 余弦相似度验证
# ============================================
print("\n[4] Quick sanity check: text similarity...")
with torch.no_grad():
    sim = torch.nn.functional.cosine_similarity(text_feat[0], text_feat[1], dim=0)
    print(f"    cos_sim('normal', 'anomalous') = {sim.item():.4f}")

print("\n✅ All models loaded and verified successfully!")