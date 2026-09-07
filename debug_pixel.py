# 临时诊断脚本：检查单张图像的像素级方向/数值
import os
import sys
sys.path.insert(0, os.path.abspath('.'))
import torch
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import roc_auc_score
from models.dataset import IndustrialADDataset
from inference.zero_shot import ECRADInferencePipeline

def main():
    pipe = ECRADInferencePipeline(
        config_path='configs/dataset.yaml',
        dataset_name='visa',
        category='candle',
        projection_weights_path='weights/projection_aligned.pth',
        device='cuda'
    )
    ds = IndustrialADDataset('configs/dataset.yaml', 'visa', 'candle', split='test')
    print(f"[debug] dataset size: {len(ds)}")
    labels = ds.labels
    print(f"[debug] #label1(anomaly)={sum(labels)} #label0={len(labels)-sum(labels)}")
    gt_exist = sum(1 for p in ds.gt_paths if p is not None and os.path.exists(p))
    print(f"[debug] gt_paths non-None={sum(1 for p in ds.gt_paths if p is not None)}, exist={gt_exist}")
    # 找一个 label=1 样本检查其 GT 非零像素数
    for j in range(len(ds)):
        if ds.labels[j] == 1:
            s = ds[j]
            print(f"[debug] first anomaly idx={j} path={s['img_path']} gt_sum={int(s['gt'].sum().item())}")
            break

    found = 0
    for i in range(len(ds)):
        sample = ds[i]
        if sample['label'] != 1:
            continue
        img = sample['image'].unsqueeze(0).to(pipe.device)
        img_clip = F.interpolate(img, size=(224, 224), mode='bilinear', align_corners=False)
        img_clip = pipe._renormalize_for_clip(img_clip)

        with torch.no_grad():
            amap, iscore = pipe.compute_anomaly_score(img, img_clip)

        scores = amap[0, 0].cpu().numpy()          # [518, 518]
        gt = sample['gt'][0].numpy() > 0.5          # [518, 518]

        if gt.sum() == 0 or (~gt).sum() == 0:
            continue
        auc = roc_auc_score(gt.ravel(), scores.ravel())
        auc_inv = roc_auc_score(gt.ravel(), -scores.ravel())
        print(f"[img {i}] {os.path.basename(sample['img_path'])} label={sample['label']} "
              f"img_score={iscore.item():.4f} "
              f"pixel_auc={auc:.4f} pixel_auc_inv={auc_inv:.4f} "
              f"mean_inside={scores[gt].mean():.4f} mean_outside={scores[~gt].mean():.4f} "
              f"map_min={scores.min():.4f} map_max={scores.max():.4f} gt_px={gt.sum()}")
        found += 1
        if found >= 8:
            break
    print("[debug] done")

if __name__ == "__main__":
    main()
