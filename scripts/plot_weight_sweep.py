# scripts/plot_weight_sweep.py
"""
Weight Sweep 绘图（Temp 1.2 / 论文图）：Image AUROC vs visual weight w_v。
输入: results/{ds}_results_noproj_sweep.csv（每类别 11 个 w 点，w_v=1 即纯 DINOv2）
用法: python -m scripts.plot_weight_sweep            # 默认 mvtec visa
      python -m scripts.plot_weight_sweep btad mpdd   # 任意数据集组合
输出（文件名带数据集后缀，默认 mvtec_visa 不带后缀）:
  results/figures/sweep_img_auroc_curves[_{ds...}].png
  results/figures/sweep_best_w_distribution[_{ds...}].png
"""
import csv
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

WEIGHTS = [round(x, 2) for x in np.arange(0.0, 1.01, 0.1)]


def load(path):
    """category -> {weight_visual: (image_auroc, pixel_auroc)}"""
    curves = {}
    with open(path, newline='') as f:
        for r in csv.DictReader(f):
            curves.setdefault(r['category'], {})[float(r['weight_visual'])] = (
                float(r['image_auroc']), float(r['pixel_auroc']))
    return curves


def main():
    datasets = sys.argv[1:] or ['mvtec', 'visa']
    tag = '' if datasets == ['mvtec', 'visa'] else '_' + '_'.join(datasets)
    out = os.path.join('results', 'figures')
    os.makedirs(out, exist_ok=True)

    # ---- 图 1：Image AUROC vs w_v 曲线 ----
    fig, axes = plt.subplots(1, len(datasets), figsize=(6.5 * len(datasets), 5.2), sharey=True)
    for ax, ds in zip(axes, datasets):
        curves = load(os.path.join('results', f'{ds}_results_noproj_sweep.csv'))
        for c in curves:
            ws = sorted(curves[c])
            img = [curves[c][w][0] for w in ws]
            ax.plot(ws, img, color='0.78', lw=0.9, alpha=0.9)
        # 跨类别均值曲线
        means = [float(np.mean([curves[c][w][0] for c in curves])) for w in WEIGHTS]
        max_w = WEIGHTS[int(np.argmax(means))]
        ax.plot(WEIGHTS, means, color='#c0392b', lw=2.6, marker='o', ms=4,
                label=f'Mean image AUROC (best {max_w:.1f})')
        ax.axvline(0.5, color='#2c3e50', ls='--', lw=1.2, label='equal  w=0.5')
        ax.axvline(max_w, color='#27ae60', ls=':', lw=1.4, label=f'best mean  w={max_w:.1f}')
        ax.set_title(f'{ds.upper()}  ({len(curves)} categories, no-projection)')
        ax.set_xlabel('visual weight  $w_v$  (DINOv2)   [$1-w_v$ = CLIP]')
        ax.grid(alpha=0.3)
        if ds == datasets[0]:
            ax.set_ylabel('Image AUROC')
        ax.legend(fontsize=8, loc='lower left')
    fig.tight_layout()
    p1 = os.path.join(out, f'sweep_img_auroc_curves{tag}.png')
    fig.savefig(p1, dpi=160)
    plt.close(fig)
    print('saved', p1)

    # ---- 图 2：每类别最优 w_v 分布（图像级选优）----
    fig, axes = plt.subplots(1, len(datasets), figsize=(6.5 * len(datasets), 4.6))
    for ax, ds in zip(axes, datasets):
        curves = load(os.path.join('results', f'{ds}_results_noproj_sweep.csv'))
        cats = sorted(curves, key=lambda c: -max(curves[c][w][0] for w in curves[c]))
        best_ws = []
        for c in cats:
            best_ws.append(max(sorted(curves[c]), key=lambda w: curves[c][w][0]))
        ax.bar(range(len(cats)), best_ws, color='#2980b9', alpha=0.85)
        ax.axhline(0.5, color='#e74c3c', ls='--', lw=1.2, label='equal 0.5')
        ax.set_xticks(range(len(cats)))
        ax.set_xticklabels(cats, rotation=60, fontsize=7)
        ax.set_ylim(0, 1.08)
        ax.set_title(f'{ds.upper()}  best $w_v$ per category (by image AUROC)')
        ax.grid(alpha=0.3, axis='y')
        ax.legend(fontsize=8)
    fig.tight_layout()
    p2 = os.path.join(out, f'sweep_best_w_distribution{tag}.png')
    fig.savefig(p2, dpi=160)
    plt.close(fig)
    print('saved', p2)


if __name__ == '__main__':
    main()
