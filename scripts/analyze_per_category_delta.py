# scripts/analyze_per_category_delta.py
"""
逐类别 Δ 分析（Temp 1.3）：DINOv2 vs CLIP vs A4(等权融合) 逐类别量化。
输入: results/summary_sweep_curves.csv（mvtec/visa 每类别 11 个 w 点；w=1.0≈DINOv2, w=0.0≈CLIP, w=0.5≈A4）
对每个类别计算:
  delta_img = img(w=1.0) - img(w=0.0)      >0 DINO 主导 / <0 CLIP wins
  eq_gain   = img(w=0.5) - max(img单模态)   等权融合相对最优单模态的增益（负 = 融合反而受损）
  best_gain = img(best w) - max(img单模态)  理论最优融合增益上限（类别依赖选权的潜力）
输出:
  results/per_category_delta.csv            逐类明细
  results/figures/per_category_delta.png    每数据集 Δ_img 棒棒糖图 + 等权/最优融合增益对比
"""
import csv
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

SRC = os.path.join('results', 'summary_sweep_curves.csv')
OUT_CSV = os.path.join('results', 'per_category_delta.csv')
FIG = os.path.join('results', 'figures', 'per_category_delta.png')

FIELDS = [
    'dataset', 'category',
    'clip_img', 'dino_img', 'eq_img', 'best_img', 'best_w_img',
    'delta_dino_minus_clip_img', 'clip_wins_img',
    'eq_gain_vs_best_img', 'best_gain_vs_best_img',
    'clip_pix', 'dino_pix', 'eq_pix', 'best_pix', 'best_w_pix',
    'delta_pix', 'clip_wins_pix',
    'eq_gain_vs_best_pix', 'best_gain_vs_best_pix',
]


def load(path):
    """(dataset, category) -> {w: (img, pix)}；跳过 summarize 追加的跨类别 MEAN 行"""
    data = {}
    with open(path, newline='') as f:
        for r in csv.DictReader(f):
            if r['category'].strip().upper() == 'MEAN':
                continue
            key = (r['dataset'], r['category'])
            data.setdefault(key, {})[float(r['weight_visual'])] = (
                float(r['image_auroc']), float(r['pixel_auroc']))
    return data


def per_cat(row, curves):
    w2 = curves[row]
    clip_img, clip_pix = w2[0.0]
    dino_img, dino_pix = w2[1.0]
    eq_img, eq_pix = w2[0.5]
    # 图像级选最优 w（与 sweep 报表口径一致）
    best_w_img = max(sorted(w2), key=lambda w: w2[w][0])
    best_img, _ = w2[best_w_img]
    best_w_pix = max(sorted(w2), key=lambda w: w2[w][1])
    _, best_pix = w2[best_w_pix]

    delta_img = dino_img - clip_img
    delta_pix = dino_pix - clip_pix
    best_single_img = max(clip_img, dino_img)
    best_single_pix = max(clip_pix, dino_pix)

    return {
        'dataset': row[0], 'category': row[1],
        'clip_img': clip_img, 'dino_img': dino_img, 'eq_img': eq_img,
        'best_img': best_img, 'best_w_img': best_w_img,
        'delta_dino_minus_clip_img': delta_img,
        'clip_wins_img': int(delta_img < 0),
        'eq_gain_vs_best_img': eq_img - best_single_img,
        'best_gain_vs_best_img': best_img - best_single_img,
        'clip_pix': clip_pix, 'dino_pix': dino_pix, 'eq_pix': eq_pix,
        'best_pix': best_pix, 'best_w_pix': best_w_pix,
        'delta_pix': delta_pix, 'clip_wins_pix': int(delta_pix < 0),
        'eq_gain_vs_best_pix': eq_pix - best_single_pix,
        'best_gain_vs_best_pix': best_pix - best_single_pix,
    }


def write_csv(rows):
    with open(OUT_CSV, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print('saved', OUT_CSV)


def mean_fmt(name, vals, n=4):
    print(f'  {name:<28} mean = {np.mean(vals):.{n}f}')


def main():
    data = load(SRC)
    rows = []
    for key in sorted(data, key=lambda k: (k[0], k[1])):
        rows.append(per_cat(key, data))
    write_csv(rows)

    # ---- 控制台汇总（每数据集）----
    print('\n===== Per-category delta summary =====')
    for ds in ['mvtec', 'visa']:
        sub = [r for r in rows if r['dataset'] == ds]
        d = [r['delta_dino_minus_clip_img'] for r in sub]
        clip_wins = [r['category'] for r in sub if r['clip_wins_img']]
        eq_g = [r['eq_gain_vs_best_img'] for r in sub]
        bg = [r['best_gain_vs_best_img'] for r in sub]
        print(f'\n[{ds.upper()}] {len(sub)} categories')
        mean_fmt('delta_img (DINO-CLIP)', d)
        mean_fmt('eq_gain_img  (0.5 vs best single)', eq_g)
        mean_fmt('best_gain_img (oracle w vs best single)', bg)
        print(f'  CLIP-wins (delta_img<0): {clip_wins if clip_wins else "none"}')
        print(f'  eq fusion 净受损 (eq_gain<0): {[r["category"] for r in sub if r["eq_gain_vs_best_img"] < 0] or "none"}')

        # 跨类别均值：单模态 vs 等权 vs oracle（图像级/像素级）
        for m in ['clip', 'dino', 'eq', 'best']:
            print(f'  mean {m:<5} img={np.mean([r[f"{m}_img"] for r in sub]):.4f}  '
                  f'pix={np.mean([r[f"{m}_pix"] for r in sub]):.4f}')
        eq_gp = [r['eq_gain_vs_best_pix'] for r in sub]
        bgp = [r['best_gain_vs_best_pix'] for r in sub]
        mean_fmt('eq_gain_pix (0.5 vs best single)', eq_gp)
        mean_fmt('best_gain_pix (oracle w)', bgp)
        pix_wins = [r['category'] for r in sub if r['clip_wins_pix']]
        print(f'  CLIP-wins pixel (delta_pix<0): {pix_wins if pix_wins else "none"}')
        print(f'  eq fusion 像素级增益类别 (eq_gain_pix>0): '
              f'{[r["category"] for r in sub if r["eq_gain_vs_best_pix"] > 0] or "none"}')

    # ---- 绘图：棒棒糖 Δ + 增益对比 ----
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.4))
    for ax, ds in zip(axes, ['mvtec', 'visa']):
        sub = sorted([r for r in rows if r['dataset'] == ds],
                     key=lambda r: r['delta_dino_minus_clip_img'])
        cats = [r['category'] for r in sub]
        deltas = [r['delta_dino_minus_clip_img'] for r in sub]
        wins = [r['clip_wins_img'] for r in sub]
        y = np.arange(len(cats))
        colors = ['#e74c3c' if w else '#2980b9' for w in wins]
        # 棒棒糖（stem）
        ax.hlines(y, 0, deltas, color=[c for c in colors], lw=1.6)
        ax.scatter(deltas, y, color=colors, s=42, zorder=3)
        # 红色向左 = CLIP wins；零线
        ax.axvline(0, color='k', lw=0.9)
        ax.set_yticks(y)
        ax.set_yticklabels(cats, fontsize=8)
        ax.set_xlabel(r'$\Delta$ image AUROC  =  DINOv2(w=1) $-$ CLIP(w=0)')
        ax.set_title(f'{ds.upper()}  single-modality gap per category '
                     f'({"red = CLIP wins" if any(wins) else "DINO wins everywhere"})')
        ax.grid(alpha=0.3, axis='x')
        # 标注 clip_wins 计数
        n_wins = sum(wins)
        ax.text(0.02, 0.98, f'CLIP wins: {n_wins}/{len(cats)}',
                transform=ax.transAxes, ha='left', va='top', fontsize=9,
                bbox=dict(fc='white', ec='#bdc3c7', alpha=0.9))
    fig.tight_layout()
    os.makedirs(os.path.dirname(FIG), exist_ok=True)
    fig.savefig(FIG, dpi=160)
    plt.close(fig)
    print('saved', FIG)


if __name__ == '__main__':
    main()
