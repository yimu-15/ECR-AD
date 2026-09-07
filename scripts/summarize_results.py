# scripts/summarize_results.py
"""
整理 results/ 下所有已完成实验的 AUROC 数据，输出结构化汇总表：
  1) results/summary_long_all.csv      : 每 (dataset, setting, category) 一行（逐类明细）
  2) results/summary_means.csv         : 每 (dataset, setting) 的类别均值 + 类别数
  3) results/summary_sweep_curves.csv  : 合并所有 *_noproj_sweep.csv 曲线（含跨类别 MEAN 行）

文件名→设置 映射与 scripts/evaluate_all.py 的 csv_tag 规则一致。
"""
import csv
import os

# (文件名后缀, 设置名)；顺序即汇总表显示顺序
SETTINGS = [
    ('', 'Full'),
    ('_global', 'A3-global'),
    ('_visual', 'A1-DINOv2'),
    ('_text', 'A2-CLIP'),
    ('_noproj', 'A4-noproj'),
]
DATASETS = ['mvtec', 'visa', 'btad', 'mpdd']


def parse_plain(path):
    """解析非 sweep 的每类别结果 CSV。"""
    rows = []
    with open(path, newline='') as f:
        for r in csv.DictReader(f):
            rows.append({'category': r['category'],
                         'image_auroc': float(r['image_auroc']),
                         'pixel_auroc': float(r['pixel_auroc'])})
    return rows


def main():
    long_rows, mean_rows, sweep_rows = [], [], []
    for ds in DATASETS:
        for suffix, setting in SETTINGS:
            path = os.path.join('results', f'{ds}_results{suffix}.csv')
            if not os.path.exists(path):
                continue
            per = parse_plain(path)
            for p in per:
                long_rows.append({'dataset': ds, 'setting': setting, **p})
            img = [p['image_auroc'] for p in per]
            pix = [p['pixel_auroc'] for p in per]
            mean_rows.append({'dataset': ds, 'setting': setting,
                              'mean_image_auroc': sum(img) / len(img),
                              'mean_pixel_auroc': sum(pix) / len(pix),
                              'n_categories': len(per)})

        # 无投影 weight sweep 曲线（A4 配置，每类别 11 个 w 点）
        spath = os.path.join('results', f'{ds}_results_noproj_sweep.csv')
        if not os.path.exists(spath):
            continue
        acc_w = {}
        with open(spath, newline='') as f:
            for r in csv.DictReader(f):
                row = {'dataset': ds, 'category': r['category'],
                       'weight_visual': float(r['weight_visual']),
                       'image_auroc': float(r['image_auroc']),
                       'pixel_auroc': float(r['pixel_auroc'])}
                sweep_rows.append(row)
                acc_w.setdefault(row['weight_visual'], []).append(row)
        # 每个 w 点上的跨类别 MEAN 行（用于画均值曲线）
        for w, rows_w in sorted(acc_w.items()):
            sweep_rows.append({'dataset': ds, 'category': 'MEAN',
                               'weight_visual': w,
                               'image_auroc': sum(r['image_auroc'] for r in rows_w) / len(rows_w),
                               'pixel_auroc': sum(r['pixel_auroc'] for r in rows_w) / len(rows_w)})

    os.makedirs('results', exist_ok=True)
    for fname, fields, rows in [
        ('summary_long_all.csv',
         ['dataset', 'setting', 'category', 'image_auroc', 'pixel_auroc'], long_rows),
        ('summary_means.csv',
         ['dataset', 'setting', 'mean_image_auroc', 'mean_pixel_auroc', 'n_categories'], mean_rows),
        ('summary_sweep_curves.csv',
         ['dataset', 'category', 'weight_visual', 'image_auroc', 'pixel_auroc'], sweep_rows),
    ]:
        path = os.path.join('results', fname)
        with open(path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
        print(f'saved {path} ({len(rows)} rows)')


if __name__ == '__main__':
    main()
