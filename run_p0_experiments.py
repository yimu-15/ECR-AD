#!/usr/bin/env python3
"""
P0 实验执行脚本 — CAME-AD 论文核心实验
=============================================
P0-1: Pixel-level Full Weight Sweep (w_v ∈ {0.0, 0.1, ..., 1.0})
P0-2: Per-category Optimal Weight Analysis
P0-3: Oracle vs Predicted Weight (Compactness-based)

放置位置: 项目根目录 (与 memory_banks/, datasets/ 同级)
运行方式: python run_p0_experiments.py
"""

import os
import sys
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm

# ============================================================
#  全局配置 — 根据你的实际路径修改这里
# ============================================================
CONFIG = {
    # 数据集名称 (小写, 用于 memory_banks 子目录名)
    "datasets": ["mvtec", "visa", "btad", "mpdd"],

    # 数据集原始路径 (用于加载测试图像和 GT mask; 与 configs/dataset.yaml 一致)
    "dataset_roots": {
        "mvtec": "./datasets/MVTec AD",
        "visa":  "./datasets/visa/VisA/data/VisA_20220922",
        "btad":  "./datasets/BTAD",
        "mpdd":  "./datasets/MPDD",
    },

    # Memory Bank 特征路径 (A4 无投影库)
    "memory_bank_dir": "./weights/memory_banks_noproj",

    # Image-level sweep CSV (P0-2 需要; summarize_results 汇总产物)
    "image_sweep_csv": "./results/summary_sweep_curves.csv",

    # 输出目录
    "output_dir": "./p0_results",

    # 权重扫描范围
    "weights": [round(w, 1) for w in np.arange(0.0, 1.05, 0.1)],

    # 设备
    "device": "cuda" if torch.cuda.is_available() else "cpu",

    # 数值稳定性
    "eps": 1e-8,

    # Pixel-level 评估: 特征图 resize 到的目标尺寸
    # (anomaly map 会被 resize 到 GT mask 的尺寸再计算 AUROC)
    "pixel_eval_resize": None,  # None = 使用 GT mask 原始尺寸

    # Memory Bank 文件名 (A4 无投影库: 原始 DINOv2/CLIP patch 特征, [M, C] tensor)
    "dino_mb_filename": "visual.pth",
    "clip_mb_filename": "semantic.pth",

    # 测试集特征缓存目录 (避免重复提取; P0-1 暂未启用)
    "test_feat_cache_dir": "./p0_results/test_cache",
}

os.makedirs(CONFIG["output_dir"], exist_ok=True)
os.makedirs(CONFIG["test_feat_cache_dir"], exist_ok=True)


# ============================================================
#  工具函数
# ============================================================

def check_paths():
    """启动前检查所有关键路径是否存在"""
    errors = []
    mb_dir = Path(CONFIG["memory_bank_dir"])
    if not mb_dir.exists():
        errors.append(f"❌ Memory Bank 目录不存在: {mb_dir}")

    for ds in CONFIG["datasets"]:
        ds_path = Path(CONFIG["dataset_roots"][ds])
        if not ds_path.exists():
            errors.append(f"❌ 数据集目录不存在: {ds_path} ({ds})")

        mb_ds_path = mb_dir / ds
        if not mb_ds_path.exists():
            errors.append(f"❌ Memory Bank 子目录不存在: {mb_ds_path}")

    csv_path = Path(CONFIG["image_sweep_csv"])
    if not csv_path.exists():
        errors.append(f"⚠️  Image-level sweep CSV 不存在: {csv_path} (P0-2 将跳过)")

    if errors:
        print("\n".join(errors))
        print("\n请修改 CONFIG 中的路径后重新运行。")
        sys.exit(1)

    print("✅ 所有路径检查通过。")


def load_memory_bank(dataset, category):
    """
    加载指定类别的 DINO 和 CLIP Memory Bank 特征。
    返回: dict with keys 'dino' and 'clip', 每个是 [M, C] 的 tensor
    """
    mb_path = Path(CONFIG["memory_bank_dir"]) / dataset / category
    dino_file = mb_path / CONFIG["dino_mb_filename"]
    clip_file = mb_path / CONFIG["clip_mb_filename"]

    if not dino_file.exists():
        raise FileNotFoundError(f"DINO MB 不存在: {dino_file}")
    if not clip_file.exists():
        raise FileNotFoundError(f"CLIP MB 不存在: {clip_file}")

    dino_feat = torch.load(dino_file, map_location=CONFIG["device"])
    clip_feat = torch.load(clip_file, map_location=CONFIG["device"])

    # 确保是 2D [M, C] (如果是 3D [M, H*W, C] 则需要 reshape)
    if dino_feat.dim() == 3:
        # Memory Bank 应该是 coreset 后的 patch 集合 [M, C]
        # 如果是 [M, H*W, C]，取 mean pooling
        dino_feat = dino_feat.mean(dim=1)
    if clip_feat.dim() == 3:
        clip_feat = clip_feat.mean(dim=1)

    return {"dino": dino_feat.float(), "clip": clip_feat.float()}


# ============================================================
#  P0-1: Pixel-level Full Weight Sweep (聚合真实测量 + 双面板)
# ============================================================

def run_p0_1_pixel_sweep():
    """
    P0-1: Pixel-level Full Weight Sweep。

    数据来源: results/summary_sweep_curves.csv —— 由 evaluate_all.py --weight_sweep
    在 A4(无投影) 口径下对 4 数据集 36 类 × 11 个 w 已完成真实推理, 每行包含:
        image_auroc = 图像级 Top1%@518 AUROC (逐类)
        pixel_auroc = seed42 固定分层抽样(异常<=50k / 正常<=10k)的全局 Pixel AUROC (逐类)

    本函数不重复推理, 而是聚合这些真实测量结果:
        1) 输出 p0_1_pixel_sweep.csv: 逐数据集 × w 的类别均值曲线 (image + pixel 双侧)
        2) 绘制 Image-level vs Pixel-level 双面板曲线图
        3) 打印假说量化小结 (图像级峰值 w*≈? / 像素级峰值区间?)
    """
    print("=" * 70)
    print("[P0-1] Pixel-level Full Weight Sweep (聚合真实 pixel 测量)")
    print("=" * 70)

    csv_path = Path(CONFIG["image_sweep_csv"])
    if not csv_path.exists():
        print(f"[ERR] sweep CSV 不存在: {csv_path}")
        print("请先运行: python -m scripts.evaluate_all --dataset <ds> --no_projection --weight_sweep")
        return None

    df = pd.read_csv(csv_path)
    if "w_dino" not in df.columns and "weight_visual" in df.columns:
        df = df.rename(columns={"weight_visual": "w_dino"})

    # 逐类别真实行 (剔除 CSV 内的 MEAN 汇总行), 仅保留 4 个目标数据集
    cat_df = df[df["category"] != "MEAN"].copy()
    cat_df = cat_df[cat_df["dataset"].isin(CONFIG["datasets"])].copy()

    # 自查类别均值 vs CSV MEAN 行 → 聚合口径一致性校验
    mean_rows = df[df["category"] == "MEAN"]
    if len(mean_rows) > 0:
        agg_self = cat_df.groupby(["dataset", "w_dino"], as_index=False)[
            ["image_auroc", "pixel_auroc"]].mean()
        chk = agg_self.merge(
            mean_rows[["dataset", "w_dino", "image_auroc", "pixel_auroc"]],
            on=["dataset", "w_dino"], suffixes=("_self", "_csvmean"))
        d_img = (chk["image_auroc_self"] - chk["image_auroc_csvmean"]).abs().max()
        d_pix = (chk["pixel_auroc_self"] - chk["pixel_auroc_csvmean"]).abs().max()
        print(f"[校验] 自查类别均值 vs CSV MEAN 行 -> img Δmax={d_img:.2e}, pix Δmax={d_pix:.2e}")

    # 逐数据集 × w: 类别级 image/pixel AUROC 均值曲线
    curve = cat_df.groupby(["dataset", "w_dino"], as_index=False).agg(
        n_categories=("category", "count"),
        mean_image_auroc=("image_auroc", "mean"),
        mean_pixel_auroc=("pixel_auroc", "mean"),
    )
    curve = curve.sort_values(["dataset", "w_dino"]).reset_index(drop=True)

    out_path = Path(CONFIG["output_dir"]) / "p0_1_pixel_sweep.csv"
    curve.to_csv(out_path, index=False)
    print(f"[P0-1] 数据表保存至: {out_path}")

    # ---- 逐数据集曲线 + 峰值汇总 ----
    print("\n" + "-" * 96)
    print("逐数据集曲线汇总 (w=0.0 纯CLIP / w=0.5 等权 / w=1.0 纯DINO / peak)")
    print("-" * 96)
    print(f"{'Dataset':<8}{'Level':<8}{'#cat':<6}{'w=0.0':<9}{'w=0.5':<9}{'w=1.0':<9}{'peak w*':<10}{'peak':<10}")
    img_peaks, pix_peaks = {}, {}
    for ds in CONFIG["datasets"]:
        sub = curve[curve["dataset"] == ds].sort_values("w_dino")
        if len(sub) == 0:
            continue
        ncat = int(sub["n_categories"].iloc[0])
        for level, col in [("Image", "mean_image_auroc"), ("Pixel", "mean_pixel_auroc")]:
            v0 = sub.iloc[0][col]
            v05row = sub[sub["w_dino"] == 0.5]
            v05 = v05row[col].iloc[0] if len(v05row) else float("nan")
            v1 = sub.iloc[-1][col]
            imax = sub[col].idxmax()
            prow = sub.loc[imax]
            print(f"{ds.upper():<8}{level:<8}{ncat:<6}{v0:<9.4f}{v05:<9.4f}{v1:<9.4f}"
                  f"{prow['w_dino']:<10.1f}{prow[col]:<10.4f}")
            if level == "Image":
                img_peaks[ds] = (prow["w_dino"], prow[col])
            else:
                pix_peaks[ds] = (prow["w_dino"], prow[col])
    print("-" * 96)

    if img_peaks and pix_peaks:
        mean_img_w = float(np.mean([v[0] for v in img_peaks.values()]))
        mean_pix_w = float(np.mean([v[0] for v in pix_peaks.values()]))
        print(f"图像级峰值 w* 跨数据集均值 = {mean_img_w:.2f}   (假说: ≈0.8 视觉主导)")
        print(f"像素级峰值 w* 跨数据集均值 = {mean_pix_w:.2f}   (假说: ≈0.5-0.8 多模态互补)")

    # ---- GLOBAL (4 数据集等权平均) 曲线 ----
    global_curve = curve.groupby("w_dino", as_index=False)[
        ["mean_image_auroc", "mean_pixel_auroc"]].mean().sort_values("w_dino").reset_index(drop=True)
    print("\nGLOBAL (4 数据集等权平均) 曲线:")
    print(f"{'w':<6}{'Image':<10}{'Pixel':<10}")
    for _, r in global_curve.iterrows():
        print(f"{r['w_dino']:<6.1f}{r['mean_image_auroc']:<10.4f}{r['mean_pixel_auroc']:<10.4f}")

    def _peak_row(cc, col):
        return cc.loc[cc[col].idxmax()]

    g_img = _peak_row(global_curve, "mean_image_auroc")
    g_pix = _peak_row(global_curve, "mean_pixel_auroc")
    print(f"\nGLOBAL 图像级峰值 @ w*={g_img['w_dino']:.1f}: {g_img['mean_image_auroc']:.4f}")
    print(f"GLOBAL 像素级峰值 @ w*={g_pix['w_dino']:.1f}: {g_pix['mean_pixel_auroc']:.4f}")

    # ---- 假说量化: 融合增益 (像素级融合互补性检查) ----
    print("\n假说量化 (像素级多模态互补: 融合区间是否 ≥ 两单模态端点):")
    for ds in CONFIG["datasets"]:
        sub = curve[curve["dataset"] == ds].sort_values("w_dino")
        if len(sub) == 0:
            continue
        p0 = sub.iloc[0]["mean_pixel_auroc"]
        p1 = sub.iloc[-1]["mean_pixel_auroc"]
        pbest = sub["mean_pixel_auroc"].max()
        bw = sub.loc[sub["mean_pixel_auroc"].idxmax(), "w_dino"]
        gain_best = pbest - max(p0, p1)
        eq05row = sub[sub["w_dino"] == 0.5]
        eq05 = eq05row["mean_pixel_auroc"].iloc[0] if len(eq05row) else float("nan")
        note = "OK(融合≥双端点)" if pbest >= max(p0, p1) - 1e-12 else "FALL"
        print(f"  {ds.upper():<8} pix: 端点({p0:.4f},{p1:.4f}) 等权@0.5={eq05:.4f} "
              f"峰值{pbest:.4f}@w={bw:.1f} 融合增益vs最好单模态={gain_best:+.4f} [{note}]")

    # 绘制双面板图
    plot_dual_panel_curves(curve, global_curve)
    return curve


def plot_dual_panel_curves(curve_df, global_df=None):
    """
    绘制双面板 Figure:
        左: Image-level AUROC vs w_v
        右: Pixel-level AUROC vs w_v
    每条曲线 = 数据集内类别均值的均值; 数字标注 = 各数据集峰值位置;
    灰色虚线 = 等权 w=0.5。
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 6.5))
    colors = {"mvtec": "#1f77b4", "visa": "#ff7f0e", "btad": "#2ca02c", "mpdd": "#d62728"}
    markers = ["o", "s", "^", "D"]
    panels = [("Image-level AUROC", "mean_image_auroc", "Image-level"),
              ("Pixel-level AUROC", "mean_pixel_auroc", "Pixel-level")]

    for (ylabel, col, title) in panels:
        ax = axes[0 if col.startswith("mean_image") else 1]
        for i, ds in enumerate(CONFIG["datasets"]):
            sub = curve_df[curve_df["dataset"] == ds].sort_values("w_dino")
            if len(sub) == 0:
                continue
            c = colors.get(ds, "gray")
            ax.plot(sub["w_dino"], sub[col], markers[i % len(markers)] + "-",
                    color=c, label=ds.upper(), linewidth=2, markersize=5)
            imax = sub[col].idxmax()
            p = sub.loc[imax]
            ax.annotate(f"{p['w_dino']:.1f}", xy=(p["w_dino"], p[col]),
                        xytext=(0, 8), textcoords="offset points",
                        ha="center", color=c, fontsize=9, fontweight="bold")
        if global_df is not None and len(global_df) > 0:
            gg = global_df.sort_values("w_dino")
            ax.plot(gg["w_dino"], gg[col], "-", color="#444444",
                    linewidth=1.8, alpha=0.6, label="GLOBAL (4-ds avg)")
        ax.axvline(0.5, color="gray", linestyle="--", alpha=0.5, linewidth=1)
        ax.text(0.5, 0.98, "w=0.5", transform=ax.transAxes,
                ha="center", va="top", color="gray", fontsize=9)
        ax.set_xlabel("$w_v$ (DINO weight)", fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.set_title(title, fontsize=14)
        ax.legend(fontsize=10, loc="lower right")
        ax.grid(True, alpha=0.3)
        ax.set_xlim(-0.05, 1.05)

    fig.suptitle("P0-1: Fusion Weight Sweep on 4 Datasets (A4 no-projection, real inference)",
                 fontsize=13, y=1.02)
    plt.tight_layout()
    out_path = Path(CONFIG["output_dir"]) / "p0_1_dual_panel_curves.png"
    plt.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"[P0-1] 双面板曲线图保存至: {out_path}")


# ============================================================
#  P0-2: Per-category Optimal Weight Analysis
# ============================================================

def run_p0_2_per_category_analysis():
    """
    P0-2: 从已有的 Image-level sweep CSV 中提取每个类别的最优权重。
    """
    print("\n" + "=" * 70)
    print("🚀 [P0-2] Per-category Optimal Weight Analysis")
    print("=" * 70)

    csv_path = Path(CONFIG["image_sweep_csv"])
    if not csv_path.exists():
        print(f"⚠️  Image-level sweep CSV 不存在: {csv_path}")
        print("   请先运行 Image-level sweep 或将 CSV 放到正确路径。")
        return None

    df = pd.read_csv(csv_path)

    # 兼容字段名: 本项目 sweep CSV 用 weight_visual 表示 DINO(visual) 权重
    if "w_dino" not in df.columns and "weight_visual" in df.columns:
        df = df.rename(columns={"weight_visual": "w_dino"})

    # 剔除汇总用的 MEAN 行, 并只保留当前配置中的数据集
    df = df[df["category"] != "MEAN"].copy()
    df = df[df["dataset"].isin(CONFIG["datasets"])].copy()

    # 检查必要列
    required_cols = ["dataset", "category", "w_dino", "image_auroc"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        print(f"❌ CSV 缺少必要列: {missing}")
        print(f"   现有列: {list(df.columns)}")
        return None

    best_records = []
    for (ds, cat), group in df.groupby(["dataset", "category"]):
        best_idx = group["image_auroc"].idxmax()
        best_row = group.loc[best_idx]

        # 获取端点值
        pure_clip = group[group["w_dino"] == 0.0]["image_auroc"].values
        pure_dino = group[group["w_dino"] == 1.0]["image_auroc"].values

        best_records.append({
            "dataset": ds,
            "category": cat,
            "best_w_dino": best_row["w_dino"],
            "best_image_auroc": round(best_row["image_auroc"], 4),
            "pure_clip_auroc": round(pure_clip[0], 4) if len(pure_clip) > 0 else None,
            "pure_dino_auroc": round(pure_dino[0], 4) if len(pure_dino) > 0 else None,
            "equal_weight_auroc": round(
                group[group["w_dino"] == 0.5]["image_auroc"].values[0], 4
            ) if len(group[group["w_dino"] == 0.5]) > 0 else None,
        })

    result_df = pd.DataFrame(best_records)
    out_path = Path(CONFIG["output_dir"]) / "p0_2_per_category_best_w.csv"
    result_df.to_csv(out_path, index=False)

    # 打印统计摘要
    print(f"\n{'Dataset':<10} {'Categories':<12} {'w* Range':<15} {'Mean w*':<10} {'Std w*':<10}")
    print("-" * 57)
    for ds in CONFIG["datasets"]:
        sub = result_df[result_df["dataset"] == ds]
        if len(sub) > 0:
            w_range = f"[{sub['best_w_dino'].min():.1f}, {sub['best_w_dino'].max():.1f}]"
            print(f"{ds.upper():<10} {len(sub):<12} {w_range:<15} "
                  f"{sub['best_w_dino'].mean():<10.2f} {sub['best_w_dino'].std():<10.2f}")

    print(f"\n✅ P0-2 完成。结果保存至: {out_path}")
    return result_df


# ============================================================
#  P0-3: Oracle vs Predicted Weight
# ============================================================

def compute_compactness_corrected(memory_bank):
    """
    计算 Normal Reference 的紧凑度 ρ (修正版)。

    导师公式: ρ_c = E_{x ∈ M_c}[d(f(x), M_c \ {x})]
    这里 d = 样本 x 到 Memory Bank 中除自身外最近邻的 1-NN 距离。

    实际 Memory Bank 是 patch 级特征, M 可达 10 万 (如 MVTec bottle DINO 100k),
    直接两两计算 [M, M] 不可行 → 固定种子随机子采样 S 行,
    再分块计算"排除自身"的 1-NN 平均距离。
    dino / clip 使用同一组种子样本索引, 保证 ρ 可比。

    返回: rho_dino, rho_clip (float)
    """
    SAMPLE_N = 20000   # 每模态子采样规模 (超过则随机抽)
    CHUNK = 1024       # 分块行数, 控制峰值显存

    def _mean_self_nn_dist(feat):
        M = feat.shape[0]
        if M <= SAMPLE_N:
            sub = feat
        else:
            rng = np.random.RandomState(0)
            idx = rng.choice(M, SAMPLE_N, replace=False)
            sub = feat[idx]
        n = sub.shape[0]
        sub = sub.to(CONFIG["device"]).float()

        with torch.no_grad():
            total = 0.0
            for r0 in range(0, n, CHUNK):
                r1 = min(r0 + CHUNK, n)
                q = sub[r0:r1]                       # [b, C]
                d = torch.cdist(q, sub, p=2)         # [b, n]
                # 排除自身: 全局行号 r0+i 的对角元置 inf
                rows = torch.arange(r1 - r0, device=d.device)
                d[rows, r0 + rows] = float('inf')
                total += d.min(dim=-1).values.sum().item()
        return total / n

    rho_dino = _mean_self_nn_dist(memory_bank["dino"])
    rho_clip = _mean_self_nn_dist(memory_bank["clip"])
    return rho_dino, rho_clip


def compute_predicted_weight(rho_dino, rho_clip):
    """
    根据导师公式计算预测权重:
        r_v = 1 / (ρ_v + ε)
        r_t = 1 / (ρ_t + ε)
        w_v = r_v / (r_v + r_t)
    """
    eps = CONFIG["eps"]
    r_dino = 1.0 / (rho_dino + eps)
    r_clip = 1.0 / (rho_clip + eps)
    w_pred = r_dino / (r_dino + r_clip)
    return w_pred, r_dino, r_clip


def run_p0_3_oracle_vs_predicted(per_cat_df):
    """
    P0-3: Oracle vs Predicted Weight 相关性分析。
    """
    print("\n" + "=" * 70)
    print("🚀 [P0-3] Oracle vs Predicted Weight Analysis")
    print("=" * 70)

    if per_cat_df is None:
        print("❌ P0-2 结果为空，无法运行 P0-3。")
        return None

    records = []
    for _, row in tqdm(per_cat_df.iterrows(), total=len(per_cat_df),
                        desc="Computing compactness"):
        try:
            mb = load_memory_bank(row["dataset"], row["category"])
        except FileNotFoundError as e:
            print(f"   ⚠️ 跳过 {row['dataset']}/{row['category']}: {e}")
            continue

        rho_dino, rho_clip = compute_compactness_corrected(mb)
        w_pred, r_dino, r_clip = compute_predicted_weight(rho_dino, rho_clip)

        records.append({
            "dataset": row["dataset"],
            "category": row["category"],
            "w_oracle": row["best_w_dino"],
            "w_predicted": round(w_pred, 4),
            "rho_dino": round(rho_dino, 4),
            "rho_clip": round(rho_clip, 4),
            "r_dino": round(r_dino, 4),
            "r_clip": round(r_clip, 4),
            "best_image_auroc": row["best_image_auroc"],
        })

    result_df = pd.DataFrame(records)
    out_path = Path(CONFIG["output_dir"]) / "p0_3_oracle_vs_predicted.csv"
    result_df.to_csv(out_path, index=False)

    # --- 统计分析 ---
    print(f"\n{'Dataset':<10} {'N':<5} {'Pearson r':<12} {'Spearman r':<12} {'MAE':<10}")
    print("-" * 49)

    from scipy.stats import spearmanr

    global_corr = result_df["w_oracle"].corr(result_df["w_predicted"])
    global_spearman, _ = spearmanr(result_df["w_oracle"], result_df["w_predicted"])
    global_mae = np.abs(result_df["w_oracle"] - result_df["w_predicted"]).mean()

    for ds in CONFIG["datasets"]:
        sub = result_df[result_df["dataset"] == ds]
        if len(sub) >= 2:
            pearson_r = sub["w_oracle"].corr(sub["w_predicted"])
            spearman_r, _ = spearmanr(sub["w_oracle"], sub["w_predicted"])
            mae = np.abs(sub["w_oracle"] - sub["w_predicted"]).mean()
            print(f"{ds.upper():<10} {len(sub):<5} {pearson_r:<12.4f} {spearman_r:<12.4f} {mae:<10.4f}")

    print(f"{'GLOBAL':<10} {len(result_df):<5} {global_corr:<12.4f} {global_spearman:<12.4f} {global_mae:<10.4f}")

    # --- 绘制散点图 ---
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    colors_map = {"mvtec": "#1f77b4", "visa": "#ff7f0e", "btad": "#2ca02c", "mpdd": "#d62728"}

    for ax, ds in zip(axes.flatten(), CONFIG["datasets"]):
        sub = result_df[result_df["dataset"] == ds]
        if len(sub) == 0:
            ax.set_visible(False)
            continue

        color = colors_map.get(ds, 'gray')
        ax.scatter(sub["w_predicted"], sub["w_oracle"],
                   c=color, alpha=0.8, edgecolors='k', linewidth=0.5, s=60)

        # 完美预测线
        ax.plot([0, 1], [0, 1], 'r--', linewidth=1.5, label='Perfect (y=x)', alpha=0.7)

        # 固定 w=0.8 参考线
        ax.axhline(y=0.8, color='gray', linestyle=':', alpha=0.5, label='Fixed w=0.8')

        ax.set_xlim(-0.05, 1.05)
        ax.set_ylim(-0.05, 1.05)
        ax.set_xlabel("$w^{pred}$ (Compactness-based)", fontsize=11)
        ax.set_ylabel("$w^{oracle}$ (Sweep-based)", fontsize=11)

        ds_corr = sub["w_oracle"].corr(sub["w_predicted"])
        ds_mae = np.abs(sub["w_oracle"] - sub["w_predicted"]).mean()
        ax.set_title(f"{ds.upper()}  (r={ds_corr:.3f}, MAE={ds_mae:.3f})", fontsize=12)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal')

    plt.suptitle("Oracle vs Predicted Fusion Weight (CAME-AD)", fontsize=14, y=1.02)
    plt.tight_layout()
    scatter_path = Path(CONFIG["output_dir"]) / "p0_3_oracle_vs_predicted_scatter.png"
    plt.savefig(scatter_path, dpi=200, bbox_inches='tight')
    plt.close()

    # --- 额外: 绘制 ρ_dino vs ρ_clip 的分布图 ---
    fig2, axes2 = plt.subplots(2, 2, figsize=(14, 12))
    for ax, ds in zip(axes2.flatten(), CONFIG["datasets"]):
        sub = result_df[result_df["dataset"] == ds]
        if len(sub) == 0:
            ax.set_visible(False)
            continue

        color = colors_map.get(ds, 'gray')
        ax.scatter(sub["rho_dino"], sub["rho_clip"],
                   c=color, alpha=0.8, edgecolors='k', linewidth=0.5, s=60)
        ax.plot([0, max(sub["rho_dino"].max(), sub["rho_clip"].max()) * 1.1],
                [0, max(sub["rho_dino"].max(), sub["rho_clip"].max()) * 1.1],
                'r--', alpha=0.5, label='ρ_dino = ρ_clip')
        ax.set_xlabel("ρ_dino (DINO compactness)", fontsize=11)
        ax.set_ylabel("ρ_clip (CLIP compactness)", fontsize=11)
        ax.set_title(f"{ds.upper()}", fontsize=12)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    plt.suptitle("DINO vs CLIP Compactness per Category", fontsize=14, y=1.02)
    plt.tight_layout()
    compact_path = Path(CONFIG["output_dir"]) / "p0_3_compactness_distribution.png"
    plt.savefig(compact_path, dpi=200, bbox_inches='tight')
    plt.close()

    print(f"\n✅ P0-3 完成。")
    print(f"   📄 数据表: {out_path}")
    print(f"   📊 散点图: {scatter_path}")
    print(f"   📊 紧凑度分布: {compact_path}")

    return result_df


# ============================================================
#  主入口
# ============================================================

def main():
    print("╔══════════════════════════════════════════════════════════╗")
    print("║         CAME-AD P0 Experiments Runner                   ║")
    print("║  Category-Adaptive Multimodal Evidence for IAD          ║")
    print("╚══════════════════════════════════════════════════════════╝")
    print(f"\nDevice: {CONFIG['device']}")
    print(f"Output: {CONFIG['output_dir']}")

    # Step 0: 路径检查
    check_paths()

    # Step 1: P0-1 (聚合 evaluate_all --weight_sweep 的真实逐类测量, 读 CSV, 秒级)
    pixel_curve_df = run_p0_1_pixel_sweep()

    # Step 2: P0-2 (逐类别最优权重分析, 同样读 sweep CSV)
    per_cat_df = run_p0_2_per_category_analysis()

    # Step 3: P0-3 (依赖 P0-2，需要加载 MB 特征 + 分块计算紧凑度, 最慢)
    if per_cat_df is not None:
        oracle_pred_df = run_p0_3_oracle_vs_predicted(per_cat_df)
    else:
        print("\n⚠️  P0-3 跳过 (P0-2 无结果)")

    print("\n" + "=" * 70)
    print("🎉 P0 实验执行完毕!")
    print(f" 所有结果保存在: {os.path.abspath(CONFIG['output_dir'])}")
    print("=" * 70)
    print("""
📋 下一步行动:
  1. 检查 p0_1_dual_panel_curves.png
     → 图像级峰值是否 ≈0.8 (视觉主导) / 像素级峰值是否落在 0.5-0.8 (多模态互补)
  2. 检查 p0_2_per_category_best_w.csv
     → 确认各类别最优 w 的跨度是否足够大 (non-universal)
  3. 检查 p0_3_oracle_vs_predicted_scatter.png
     → 如果 Pearson r > 0.5，说明 compactness-based 预测有效
  4. 将结果发给导师确认
""")


if __name__ == "__main__":
    main()