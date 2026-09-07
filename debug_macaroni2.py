# 临时诊断脚本：单独排查 macaroni2 图像级 AUC 偏低(0.54)的根因
import os
import sys
sys.path.insert(0, os.path.abspath('.'))
import csv
import torch
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader
from models.dataset import IndustrialADDataset
from scripts.evaluate_all import BatchEvaluator

CFG = 'configs/dataset.yaml'
DSN, CAT = 'visa', 'macaroni2'


def csv_train_normal_count():
    csv_path = os.path.join('datasets', 'visa', 'VisA', 'data', 'VisA_20220922', 'split_csv', '1cls.csv')
    n = 0
    with open(csv_path, 'r') as f:
        for row in csv.DictReader(f):
            if row['object'] == CAT and row['split'] == 'train' and row['label'] == 'normal':
                n += 1
    return n


def main():
    bev = BatchEvaluator(CFG, 'weights/projection_aligned.pth', 'cuda')
    vb, sb = bev.load_memory_banks(DSN, CAT)
    print(f"\n[banks] visual={tuple(vb.features.shape)} semantic={tuple(sb.features.shape)}")
    print(f"[csv] macaroni2 train normal images = {csv_train_normal_count()}")

    tds = IndustrialADDataset(CFG, DSN, CAT, split='test')
    loader = DataLoader(tds, batch_size=8, shuffle=False, num_workers=4, pin_memory=True)
    n_norm = tds.labels.count(0); n_anom = tds.labels.count(1)
    print(f"[data] test total={len(tds)} normal={n_norm} anomaly={n_anom}")

    # 每张图的多种聚合分数
    rec = {'path': [], 'label': [], 'top1_518': [], 'p1_37': [], 'p5_37': [],
           'p10_37': [], 'max_37': [], 'mean_37': [], 'vis_p1': [], 'sem_p1': []}
    per_img_pixel = []  # (name, label, pixel_auc or None, gt_px)

    with torch.no_grad():
        for batch in loader:
            imgs = batch['image'].to(bev.device)
            gts = batch['gt'].to(bev.device)
            labs = batch['label'].tolist()
            paths = batch['img_path']
            B = imgs.shape[0]

            imgs_clip = F.interpolate(imgs, size=(224, 224), mode='bilinear', align_corners=False)
            imgs_clip = (imgs_clip * bev.dino_std + bev.dino_mean - bev.clip_mean) / bev.clip_std

            dino_out = bev.dino(imgs)
            dino_patches = dino_out[:, 1:, :]                     # [B,1369,384]
            clip_out = bev.clip(imgs_clip)
            clip_patches = clip_out[:, 1:, :]                     # [B,49,512]

            Nv = dino_patches.shape[1]; Nt = clip_patches.shape[1]
            zv = bev.proj.forward_visual(dino_patches.reshape(B * Nv, -1)).reshape(B, Nv, -1)
            zt = bev.proj.forward_text(clip_patches.reshape(B * Nt, -1)).reshape(B, Nt, -1)

            sv = vb.compute_anomaly_score(zv, k=5)               # [B,1369]
            st = sb.compute_anomaly_score(zt, k=5)               # [B,49]
            stm = st.reshape(B, 7, 7).unsqueeze(1)
            stm = F.interpolate(stm, size=(37, 37), mode='bilinear', align_corners=False).squeeze(1)
            svm = sv.reshape(B, 37, 37)                          # [B,37,37]
            fused = 0.5 * svm + 0.5 * stm                        # [B,37,37]
            amap = F.interpolate(fused.unsqueeze(1), size=(518, 518), mode='bilinear',
                                 align_corners=False)            # [B,1,518,518]

            for i in range(B):
                lab = int(labs[i])
                rec['path'].append(os.path.basename(paths[i]))
                rec['label'].append(lab)

                fm = fused[i].reshape(-1)                        # 1369
                for key, kp in (('p1_37', 1), ('p5_37', 5), ('p10_37', 10)):
                    kk = max(int(fm.numel() * kp / 100), 1)
                    tv, _ = torch.topk(fm, k=kk)
                    rec[key].append(float(tv.mean()))
                rec['max_37'].append(float(fm.max()))
                rec['mean_37'].append(float(fm.mean()))

                svm_i = svm[i].reshape(-1); stm_i = stm[i].reshape(-1)
                kk = max(int(svm_i.numel() * 0.01), 1)
                tv, _ = torch.topk(svm_i, k=kk); rec['vis_p1'].append(float(tv.mean()))
                tv, _ = torch.topk(stm_i, k=kk); rec['sem_p1'].append(float(tv.mean()))

                a518 = amap[i, 0]
                kk = max(int(a518.numel() * 0.01), 1)
                tv, _ = torch.topk(a518.reshape(-1), k=kk)
                rec['top1_518'].append(float(tv.mean()))

                # per-image pixel AUC
                if lab == 1:
                    gt = gts[i, 0].cpu().numpy() > 0.5
                    gpx = int(gt.sum())
                    if gpx == 0:
                        per_img_pixel.append((os.path.basename(paths[i]), lab, None, 0))
                    else:
                        pauc = roc_auc_score(gt.ravel(), a518.cpu().numpy().ravel())
                        per_img_pixel.append((os.path.basename(paths[i]), lab, float(pauc), gpx))
                else:
                    per_img_pixel.append((os.path.basename(paths[i]), lab, None, -1))

    labs = np.array(rec['label'])
    print("\n===== Image-level AUROC by aggregation strategy =====")
    print(f"{'agg':<12}{'AUROC':>8}")
    for key in ('top1_518', 'p1_37', 'p5_37', 'p10_37', 'max_37', 'mean_37', 'vis_p1', 'sem_p1'):
        s = np.array(rec[key])
        auc = roc_auc_score(labs, s)
        print(f"{key:<12}{auc:>8.4f}")

    print("\n===== top1_518 score distribution =====")
    s = np.array(rec['top1_518'])
    for name, m in (('normal', labs == 0), ('anomaly', labs == 1)):
        ss = s[m]
        print(f"{name:<8} n={len(ss):>3} min={ss.min():.4f} p25={np.percentile(ss,25):.4f} "
              f"med={np.median(ss):.4f} p75={np.percentile(ss,75):.4f} p90={np.percentile(ss,90):.4f} "
              f"max={ss.max():.4f}")

    print("\n===== pixel-level (per anomaly image, top1_518 map vs GT) =====")
    anom = [(nm, pa, g) for nm, _, pa, g in per_img_pixel if pa is not None]
    empty = sum(1 for _, l, pa, g in per_img_pixel if l == 1 and pa is None)
    print(f"anomaly with GT={len(anom)}, anomaly with EMPTY GT mask={empty}")
    if anom:
        aucs = np.array([pa for _, pa, _ in anom])
        rev = int((aucs < 0.5).sum())
        print(f"pixel_auc<0.5 (方向颠倒图) count = {rev} / {len(anom)}")
        order = np.argsort(aucs)
        print("worst-8 (name, gt_px, pixel_auc):")
        for idx in order[:8]:
            nm, pa, g = anom[idx]
            print(f"  {nm:<24} gt_px={g:<7} pixel_auc={pa:.4f}")
    if empty:
        names = [nm for nm, l, pa, g in per_img_pixel if l == 1 and pa is None]
        print("empty-GT anomaly images:", names[:10])

    print("\n[debug] done")


if __name__ == "__main__":
    main()
