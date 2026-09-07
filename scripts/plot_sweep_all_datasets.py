# scripts/plot_sweep_all_datasets.py
"""
四数据集 Weight Sweep MEAN 曲线合并图（论文 Figure）。
输入: results/summary_sweep_curves.csv 中 category==MEAN 的行（mvtec/visa/btad/mpdd 每数据集 11 个 w 点）
输出: results/figures/sweep_curves_all_datasets.png
      左面板 Image-level AUROC，右面板 Pixel-level AUROC；
      实线+星标=各数据集峰值，虚线=等权 w=0.5 参考。
"""
import csv
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

WEIGHTS = [round(x, 2) for x in np.arange(0.0, 1.01, 0.1)]
DATASETS = ['mvtec', 'visa', 'btad', 'mpdd']
COLORS = {'mvtec': '#c0392b', 'visa': '#2980b9', 'btad': '#27ae60', 'mpdd': '#8e44ad'}
LABELS = {'mvtec': 'MVTec (15)', 'visa': 'VisA (12)', 'btad': 'BTAD (3)', 'mpdd': 'MPDD (6)'}


def load_mean_curves(path):
    """dataset -> {w: (image_auroc, pixel_auroc)}，仅取跨类别 MEAN 行。"""
    curves = {}
    with open(path, newline='') as f:
        for r in csv.DictReader(f):
            if r['category'] != 'MEAN':
                continue
            curves.setdefault(r['dataset'], {})[float(r['weight_visual'])] = (
                float(r['image_auroc']), float(r['pixel_auroc']))
    return curves


def main():
    curves = load_mean_curves(os.path.join('results', 'summary_sweep_curves.csv'))
    out = os.path.join('results', 'figures')
    os.makedirs(out, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2), sharey=True)
    for (metric, idx, ylab), ax in zip(
            [('image', 0, 'AUROC (image-level)'), ('pixel', 1, 'AUROC (pixel-level)')], axes):
        for ds in DATASETS:
            c = curves[ds]
            vals = [c[w][idx] for w in WEIGHTS]
            best_w = WEIGHTS[int(np.argmax(vals))]
            ax.plot(WEIGHTS, vals, color=COLORS[ds], lw=2.2, marker='o', ms=3.5,
                    label=f"{LABELS[ds]}  (peak w={best_w:.1f})")
            ax.plot(best_w, max(vals), marker='*', color=COLORS[ds], ms=14,
                    zorder=5)  # 峰值星标
        ax.axvline(0.5, color='0.4', ls='--', lw=1.2, label='equal  $w_v$=0.5')
        ax.set_xlabel(r'visual weight  $w_v$  (DINOv2)    [$1-w_v$ = CLIP]')
        ax.set_ylabel(ylab)
        ax.set_title(f'{ylab.replace("AUROC", "")}  mean AUROC vs $w_v$  (no-projection)', fontsize=11)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8.5, loc='lower left')
    fig.suptitle('Weight sweep across 4 datasets (per-category memory bank, no projection)', fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    p = os.path.join(out, 'sweep_curves_all_datasets.png')
    fig.savefig(p, dpi=160)
    plt.close(fig)
    print('saved', p)


if __name__ == '__main__':
    main()
