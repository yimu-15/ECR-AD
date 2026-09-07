# scripts/analyze_failures.py
"""
失败案例挖掘（Temp 1.3 延伸）：在代表性类别上找出典型误检并可视化。

口径与评估完全一致（A4 无投影 + w_v=0.5 等权）：
  sv/st : DINOv2/CLIP k-NN 异常分数图 [37,37]
  Image score = bilinear->518 -> top-1% mean（= evaluate_all._aggregate_records）
  Pixel AUROC = fused map @518 与评估同款分层子采样（seed=42, 异常<=50k/正常<=10k）

每类别挑三类案例：
  FN      : 异常图图像级分数最低            （漏检：缺陷被判正常）
  FP      : 正常图图像级分数最高            （误报：正常被判缺陷）
  PIXEL   : 异常图 per-image pixel AUROC 最低（定位失败：区域漏/误）

用法:
  python scripts/analyze_failures.py [--dataset mvtec] [--categories screw transistor zipper]
默认挖掘对: mvtec(screw,transistor,zipper) + visa(macaroni2,macaroni1,capsules)。
输出:
  results/failure_cases.csv                        全案例清单（每类型 top n_csv）
  results/figures/failures/{ds}__{cat}/*.png       每 case 原图/GT/融合/DINO/CLIP 热图
"""
import argparse
import csv
import os
import sys
from collections import OrderedDict

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(__file__))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from evaluate_all import BatchEvaluator          # noqa: E402
from models.dataset import IndustrialADDataset   # noqa: E402

CONFIG = 'configs/dataset.yaml'
DEFAULT_DATASETS = {
    'mvtec': ['screw', 'transistor', 'zipper'],
    'visa': ['macaroni2', 'macaroni1', 'capsules'],
}
W_V = 0.5          # A4 等权融合
RES = 518          # 图像级分辨率
OUT_CSV = os.path.join('results', 'failure_cases.csv')
OUT_FIG_DIR = os.path.join('results', 'figures', 'failures')


@torch.no_grad()
def extract_category(evaluator, ds, cat, batch_size=8, k_nn=5):
    """前向整个测试集，返回样本列表：path/label/sv/st/gt(518x518 0/1)。"""
    vb, sb = evaluator.load_memory_banks(ds, cat, use_global=False)
    test_ds = IndustrialADDataset(CONFIG, ds, cat, split='test')
    loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                        num_workers=4, pin_memory=True)
    samples = []
    for batch in tqdm(loader, desc=f'  [{ds}/{cat}]', leave=False):
        images_dino = batch['image'].to(evaluator.device)
        gt = batch['gt'].squeeze(1).cpu().numpy()                # [B,518,518]
        labels = batch['label'].numpy()
        paths = batch['img_path']

        images_clip = F.interpolate(images_dino, size=(224, 224),
                                    mode='bilinear', align_corners=False)
        images_clip = ((images_clip * evaluator.dino_std + evaluator.dino_mean
                        - evaluator.clip_mean) / evaluator.clip_std)

        B = images_dino.shape[0]
        dino_patches = evaluator.dino(images_dino)[:, 1:, :]     # [B,1369,384]
        clip_patches = evaluator.clip(images_clip)[:, 1:, :]     # [B,49,512]
        # A4：无投影原始特征
        sv = vb.compute_anomaly_score(dino_patches, k=k_nn).reshape(B, 37, 37)
        st = sb.compute_anomaly_score(clip_patches, k=k_nn).reshape(B, 7, 7)
        st = F.interpolate(st.unsqueeze(1), size=(37, 37), mode='bilinear',
                           align_corners=False).squeeze(1)
        sv = sv.float().cpu().numpy()
        st = st.float().cpu().numpy()
        for i in range(B):
            samples.append({'path': paths[i], 'label': int(labels[i]),
                            'sv': sv[i], 'st': st[i], 'gt': gt[i]})
    return samples


def img_score_from_map(m):
    """37x37 map -> top1% mean @518（与评估一致）。"""
    x = torch.from_numpy(m.astype(np.float32)).view(1, 1, 37, 37)
    x = F.interpolate(x, size=(RES, RES), mode='bilinear', align_corners=False)
    x = x.reshape(-1)
    k = max(int(x.numel() * 0.01), 1)
    return float(torch.topk(x, k=k).values.mean().item())


def pixel_auc_from_map(m, gt):
    """fused 37x37 -> @518，与评估同款子采样(seed=42, anom<=50k, norm<=10k)的逐图 AUROC。"""
    x = torch.from_numpy(m.astype(np.float32)).view(1, 1, 37, 37)
    x = F.interpolate(x, size=(RES, RES), mode='bilinear', align_corners=False)
    fs = x.reshape(-1).numpy()
    pl = (gt > 0.5).ravel()
    if pl.sum() == 0:
        return 1.0                       # 该图 GT 无异常区域，无从谈定位
    rng = np.random.RandomState(42)
    anom = np.where(pl)[0]
    norm = np.where(~pl)[0]
    if anom.size > 50000:
        anom = rng.choice(anom, 50000, replace=False)
    if norm.size > 10000:
        norm = rng.choice(norm, 10000, replace=False)
    y = np.concatenate([np.ones(len(anom), dtype=int),
                        np.zeros(len(norm), dtype=int)])
    s = np.concatenate([fs[anom], fs[norm]])
    return float(roc_auc_score(y, s))


def analyze_category(evaluator, ds, cat, n_csv=5):
    samples = extract_category(evaluator, ds, cat)
    for s in samples:
        s['s_dino'] = img_score_from_map(s['sv'])
        s['s_clip'] = img_score_from_map(s['st'])
        s['fused'] = W_V * s['sv'] + (1 - W_V) * s['st']
        s['s_fused'] = img_score_from_map(s['fused'])
        s['pixel_auc'] = (pixel_auc_from_map(s['fused'], s['gt'])
                          if s['label'] == 1 else float('nan'))

    def pick(sel, key, k, asc=True):
        return sorted(sel, key=key, reverse=not asc)[:k]

    anom = [s for s in samples if s['label'] == 1]
    norm = [s for s in samples if s['label'] == 0]
    cases = []   # (type, sample)，按 FN/FP/PIXEL 分组
    cases += [('FN', s) for s in pick(anom, lambda s: s['s_fused'], n_csv, asc=True)]
    cases += [('FP', s) for s in pick(norm, lambda s: s['s_fused'], n_csv, asc=False)]
    cases += [('PIXEL', s) for s in pick(anom, lambda s: s['pixel_auc'], n_csv, asc=True)]
    return samples, cases


def save_visual(ds, cat, cases, n_plot=1):
    """每 type 画 n_plot 张 1x5 对比条：原图 / GT / 融合 / DINO / CLIP。"""
    dir_out = os.path.join(OUT_FIG_DIR, f'{ds}__{cat}')
    os.makedirs(dir_out, exist_ok=True)
    grouped = OrderedDict()
    for itype, s in cases:
        grouped.setdefault(itype, []).append(s)
    # 图内文字用英文（避免 CJK 字体缺失），说明在终端打印中给出
    labels = {'FN': 'FN: miss (anomaly scored normal)',
              'FP': 'FP: false alarm (normal scored anomaly)',
              'PIXEL': 'PIXEL: worst localization'}
    saved = []
    for itype, sel in grouped.items():
        for rank, s in enumerate(sel[:n_plot], 1):
            img = plt.imread(s['path'])
            h, w = img.shape[:2]
            fig, axes = plt.subplots(1, 5, figsize=(20, 4.4))
            score_txt = (f'img score fused={s["s_fused"]:.4f} | '
                         f'dino={s["s_dino"]:.4f} | clip={s["s_clip"]:.4f}')
            if s['label'] == 1:
                score_txt += f' | pixel AUROC={s["pixel_auc"]:.4f}'
            fig.suptitle(f'{ds.upper()} / {cat}  [{labels[itype]} #{rank}]\n'
                         f'{os.path.basename(s["path"])}\n{score_txt}',
                         fontsize=11)
            titles = ['image', 'GT', f'fused w={W_V}', 'DINOv2', 'CLIP']
            maps = {'GT': s['gt'], 'fused w=%g' % W_V: s['fused'],
                    'DINOv2': s['sv'], 'CLIP': s['st']}
            for j, ax in enumerate(axes):
                ax.imshow(img)
                ax.set_title(titles[j], fontsize=9)
                if j >= 1:
                    m = maps[titles[j]]
                    if j == 1 and m.max() == 0:
                        ax.set_title('GT  (no defect)', fontsize=9)
                    else:
                        mn, mx = float(m.min()), float(m.max())
                        show = (m - mn) / (mx - mn) if mx > mn else m
                        ax.imshow(show, cmap='jet', alpha=0.55,
                                  interpolation='bilinear')
                ax.set_xticks([])
                ax.set_yticks([])
                ax.set_xlim(0, w)
                ax.set_ylim(h, 0)
            fig.tight_layout()
            p = os.path.join(dir_out, f'{itype}_{rank:02d}.png')
            fig.savefig(p, dpi=150, bbox_inches='tight')
            plt.close(fig)
            saved.append(p)
    return saved


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', default=None)
    ap.add_argument('--categories', nargs='*', default=None)
    ap.add_argument('--n_csv', type=int, default=5)
    ap.add_argument('--n_plot', type=int, default=1)
    ap.add_argument('--batch_size', type=int, default=8)
    ap.add_argument('--k_nn', type=int, default=5)
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--weights', default='weights/projection_aligned.pth')
    args = ap.parse_args()

    pairs = {}
    if args.dataset:
        pairs[args.dataset] = (args.categories
                               or DEFAULT_DATASETS[args.dataset])
    else:
        pairs = dict(DEFAULT_DATASETS)

    evaluator = BatchEvaluator(CONFIG, args.weights, args.device,
                               use_projection=False)
    os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)

    all_rows = []
    with open(OUT_CSV, 'w', newline='') as f:
        w_csv = csv.DictWriter(f, fieldnames=['dataset', 'category', 'type',
                                              'img_path', 'label',
                                              'img_score_fused',
                                              'img_score_dino',
                                              'img_score_clip', 'pixel_auroc'])
        w_csv.writeheader()
        for ds, cats in pairs.items():
            for cat in cats:
                print(f'\n===== {ds} / {cat} =====')
                _, cases = analyze_category(evaluator, ds, cat, args.n_csv)
                saved = save_visual(ds, cat, cases, args.n_plot)
                for itype, s in cases[:args.n_csv * 3]:
                    row = {'dataset': ds, 'category': cat, 'type': itype,
                           'img_path': s['path'], 'label': s['label'],
                           'img_score_fused': f"{s['s_fused']:.4f}",
                           'img_score_dino': f"{s['s_dino']:.4f}",
                           'img_score_clip': f"{s['s_clip']:.4f}",
                           'pixel_auroc': (f"{s['pixel_auc']:.4f}"
                                           if s['label'] == 1 else '')}
                    w_csv.writerow(row)
                    all_rows.append(row)
                for p in saved:
                    print('  saved', p)
    print(f'\nCSV saved: {OUT_CSV}  ({len(all_rows)} case rows)')


if __name__ == '__main__':
    main()
