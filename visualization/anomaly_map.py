
"""
可视化模块
========
用于生成异常热力图、路由权重可视化、分歧度可视化等。
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from PIL import Image
from typing import Optional, Tuple


def visualize_anomaly_map(
    rgb_image: np.ndarray,
    anomaly_map: np.ndarray,
    save_path: Optional[str] = None,
    cmap: str = "jet",
    alpha: float = 0.5,
    threshold: Optional[float] = None,
) -> plt.Figure:
    """可视化异常热力图。

    Args:
        rgb_image: [H, W, 3] 原始 RGB 图像
        anomaly_map: [H, W] 异常分数图 (值范围 [0, 1])
        save_path: 保存路径 (可选)
        cmap: 颜色映射
        alpha: 热力图透明度
        threshold: 可选的阈值，超过阈值的区域用红色标记

    Returns:
        fig: matplotlib Figure 对象
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # 原始图像
    axes[0].imshow(rgb_image)
    axes[0].set_title("Original Image")
    axes[0].axis('off')

    # 异常热力图
    im = axes[1].imshow(anomaly_map, cmap=cmap)
    axes[1].set_title("Anomaly Map")
    axes[1].axis('off')
    plt.colorbar(im, ax=axes[1], shrink=0.8)

    # 叠加图
    if threshold is not None:
        # 二值化掩码
        binary_mask = (anomaly_map > threshold).astype(np.uint8)
        # 用红色标记异常区域
        overlay = rgb_image.copy().astype(np.float32)
        overlay[binary_mask == 1] = np.array([255, 0, 0])
        axes[2].imshow(overlay.astype(np.uint8))
        axes[2].set_title(f"Anomaly Mask (threshold={threshold})")
    else:
        # 透明度叠加
        overlay = (rgb_image.astype(np.float32) * (1 - alpha) +
                   anomaly_map[:, :, np.newaxis] * alpha * 255)
        axes[2].imshow(np.clip(overlay, 0, 255).astype(np.uint8))
        axes[2].set_title(f"Overlay (alpha={alpha})")
    axes[2].axis('off')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

    return fig


def visualize_routing_weights(
    rgb_image: np.ndarray,
    weight_v: np.ndarray,
    weight_t: np.ndarray,
    anomaly_map: np.ndarray,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """可视化路由权重分配。

    展示模型在不同区域对视觉/语义的依赖程度。

    Args:
        rgb_image: [H, W, 3] 原始 RGB 图像
        weight_v: [H, W] 视觉权重图
        weight_t: [H, W] 语义权重图
        anomaly_map: [H, W] 异常分数图
        save_path: 保存路径

    Returns:
        fig: matplotlib Figure 对象
    """
    fig, axes = plt.subplots(1, 5, figsize=(24, 5))

    # 原始图像
    axes[0].imshow(rgb_image)
    axes[0].set_title("Original Image")
    axes[0].axis('off')

    # 视觉权重
    im_v = axes[1].imshow(weight_v, cmap="Blues")
    axes[1].set_title("Visual Weight")
    axes[1].axis('off')
    plt.colorbar(im_v, ax=axes[1], shrink=0.8)

    # 语义权重
    im_t = axes[2].imshow(weight_t, cmap="Reds")
    axes[2].set_title("Semantic Weight")
    axes[2].axis('off')
    plt.colorbar(im_t, ax=axes[2], shrink=0.8)

    # 异常图
    im_a = axes[3].imshow(anomaly_map, cmap="jet")
    axes[3].set_title("Anomaly Map")
    axes[3].axis('off')
    plt.colorbar(im_a, ax=axes[3], shrink=0.8)

    # 权重差异 (视觉 - 语义)
    diff = weight_v - weight_t
    im_d = axes[4].imshow(diff, cmap="RdBu_r")
    axes[4].set_title("Weight Diff (Vis - Sem)")
    axes[4].axis('off')
    plt.colorbar(im_d, ax=axes[4], shrink=0.8)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

    return fig


def visualize_disagreement(
    rgb_image: np.ndarray,
    disagreement_map: np.ndarray,
    anomaly_map: np.ndarray,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """可视化跨模态分歧度。

    展示模型在哪些区域对视觉和语义的判断存在显著分歧。

    Args:
        rgb_image: [H, W, 3] 原始 RGB 图像
        disagreement_map: [H, W] 分歧度图
        anomaly_map: [H, W] 异常分数图
        save_path: 保存路径

    Returns:
        fig: matplotlib Figure 对象
    """
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))

    axes[0].imshow(rgb_image)
    axes[0].set_title("Original Image")
    axes[0].axis('off')

    im_d = axes[1].imshow(disagreement_map, cmap="hot")
    axes[1].set_title("Cross-Modal Disagreement")
    axes[1].axis('off')
    plt.colorbar(im_d, ax=axes[1], shrink=0.8)

    im_a = axes[2].imshow(anomaly_map, cmap="jet")
    axes[2].set_title("Anomaly Map")
    axes[2].axis('off')
    plt.colorbar(im_a, ax=axes[2], shrink=0.8)

    # 叠加
    overlay = (rgb_image.astype(np.float32) * 0.5 +
               disagreement_map[:, :, np.newaxis] * 0.5 * 255)
    axes[3].imshow(np.clip(overlay, 0, 255).astype(np.uint8))
    axes[3].set_title("Disagreement Overlay")
    axes[3].axis('off')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

    return fig