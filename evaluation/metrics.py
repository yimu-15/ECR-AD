# evaluation/metrics.py
import numpy as np
from sklearn.metrics import roc_auc_score

def compute_metrics(gt_mask, anomaly_map):
    """
    计算图像级和像素级 AUROC
    Args:
        gt_mask: numpy array, 像素级掩码 (H, W), 0/1
        anomaly_map: numpy array, 异常热力图 (H, W)
    Returns:
        dict: 包含 i_auroc 和 p_auroc
    """
    # 1. Image-level AUROC
    # 如果 gt_mask 全为 0，说明是正常样本
    image_label = 1 if np.max(gt_mask) > 0.5 else 0
    
    # 2. Pixel-level AUROC
    # 只对包含异常和正常的混合区域计算（避免全0或全1导致的NaN）
    if np.min(gt_mask) == 0 and np.max(gt_mask) == 1:
        p_auroc = roc_auc_score(gt_mask.flatten(), anomaly_map.flatten())
    else:
        p_auroc = 0.0
        
    return {
        "i_auroc": image_label,
        "p_auroc": p_auroc
    }