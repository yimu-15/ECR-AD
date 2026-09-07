#!/usr/bin/env python3
"""
P0-3b: 导师 §11 四列表 — compactness-CAME 生死分支判定
==========================================================
零新跑实验：纯 CSV 聚合（image-level），判定"相关性低 ≠ 端到端必输"。

列含义:
  fixed05 / fixed08 : 固定融合权重 w=0.5 / 0.8 的 image AUROC (对 sweep 曲线插值)
  oracle            : 每类 sweep 曲线峰值 (oracle w 的 AUROC)
  proposed          : w=w_predicted (compactness 预测) 处插值 AUROC
  plateau_hit       : proposed >= oracle - TAU (0.001) → 落在平台区
  prop-08           : proposed - fixed08
  plateau%          : 每数据集 plateau_hit 占比

分支判据:
  prop-08 >= -0.002 且 plateau% >= 60 → compactness-CAME 保留为 safe adaptive baseline
  prop-08 <  -0.005 → compactness 版作废, 进入预测器 v2/v3
"""
import numpy as np
import pandas as pd

SWEEP = "results/summary_sweep_curves.csv"   # evaluate_all --weight_sweep 真实产物
PRED = "p0_results/p0_3_oracle_vs_predicted.csv"
TAU = 0.001
OUT = "p0_results/p0_3b_sec11_table.csv"

sweep = pd.read_csv(SWEEP)
pred = pd.read_csv(PRED)

# 列名修正: 原始 CSV 列为 weight_visual; 剔除跨类汇总 MEAN 行
sweep = sweep.rename(columns={"weight_visual": "w_dino"})
sweep = sweep[sweep["category"] != "MEAN"].reset_index(drop=True)

recs = []
for _, p in pred.iterrows():
    g = sweep[(sweep.dataset == p.dataset) & (sweep.category == p.category)].sort_values("w_dino")
    if g.empty:
        continue
    w, a = g["w_dino"].to_numpy(), g["image_auroc"].to_numpy()
    oracle = a.max()
    recs.append(dict(
        dataset=p.dataset, category=p.category,
        fixed05=float(np.interp(0.5, w, a)),
        fixed08=float(np.interp(0.8, w, a)),
        oracle=float(oracle),
        proposed=float(np.interp(p.w_predicted, w, a)),
        plateau_hit=int(np.interp(p.w_predicted, w, a) >= oracle - TAU),
    ))

t = pd.DataFrame(recs)
s = t.groupby("dataset")[["fixed05", "fixed08", "oracle", "proposed"]].mean()
s.loc["GLOBAL"] = t[["fixed05", "fixed08", "oracle", "proposed"]].mean()
s["prop-08"] = s.proposed - s.fixed08
s["plateau%"] = (t.groupby("dataset").plateau_hit.mean() * 100).round(1)
s.loc["GLOBAL", "plateau%"] = (100.0 * t.plateau_hit.mean())

print("=" * 78)
print("P0-3b 导师 §11 四列表 (image-level, 零新跑实验, 纯 CSV 聚合)")
print("=" * 78)
print(s.round(4).to_string())

print("\n--- 平台命中率按类别明细 (plateau_hit=1 表示 proposed 落在 oracle-TAU 内) ---")
miss = t[~t.plateau_hit.astype(bool)]
print(f"总计 {len(t)} 类, plateau 命中 {int(t.plateau_hit.sum())} 类 "
      f"({100.0 * t.plateau_hit.mean():.1f}%), 未命中 {len(miss)} 类:")
if len(miss):
    print(miss[["dataset", "category", "fixed05", "fixed08", "oracle", "proposed"]].round(4).to_string(index=False))

# 分支判据 (用 GLOBAL 行)
prop_08_global = float(s.loc["GLOBAL", "prop-08"])
plateau_global = float(s.loc["GLOBAL", "plateau%"])
print("\n--- 分支判定 (GLOBAL) ---")
print(f"prop-08 = {prop_08_global:+.5f}   plateau% = {plateau_global:.1f}")
if prop_08_global >= -0.002 and plateau_global >= 60:
    print("✅ 保留 compactness-CAME 为 safe adaptive baseline "
          "('无标签、无训练、不掉点' + plateau-hit 叙事)")
elif prop_08_global < -0.005:
    print("❌ compactness 版作废 (prop-08 < -0.005) → 进入预测器 v2/v3")
else:
    print("⚠️ 灰色地带: prop-08 与 plateau% 未同时达标, 需人工权衡")

t.to_csv(OUT, index=False)
print(f"\n已保存: {OUT}")
