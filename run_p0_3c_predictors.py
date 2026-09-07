#!/usr/bin/env python3
"""
P0-3c: 预测器 v2 (z-compactness) + v3 (pseudo-anomaly probing)
============================================================
严格 normal-only: 只用 target-normal Memory Bank, 零异常标签, 零训练。
统一评估协议 (与 p0_3b §11 表一致, image-level):
  plateau% : proposed 落在 [oracle - TAU, oracle] 平台区的类别占比 (TAU=0.001)
  prop-08  : mean(proposed AUROC) - mean(fixed w=0.8 AUROC)

预测器家族:
  oracle_upperbound  : w = w_oracle (P0-2 sweep argmax)   → 校验管线 (plateau%=100, prop-08≈+0.003)
  v0_raw_compactness : w = r_d/(r_d+r_c), r=1/(rho+eps)    → 复现 p0_3 (prop-08≈-0.0047, plateau≈33%)
  v2_z_compactness   : 模态内跨类别 z(rho) 后 logistic     → 治死因1 (尺度偏置)
  v3_pseudo_probe    : 特征空间合成伪异常, AUC 分离度 → z → logistic (治死因2, 主推)
                       gamma ∈ {0.5, 1.0, 2.0} 扫描, 默认 1.0

产物:
  p0_results/p0_3c_predictor_leaderboard.csv   ← (predictor, scope) × 指标
  p0_results/p0_3c_predictor_scatter.png
  p0_results/p0_3c_detail_{predictor}.csv      ← 逐类明细 (v3 默认档 = v3_pseudo_probe)

运行: python run_p0_3c_predictors.py
"""
import os
import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

# ==================== 配置 (与 run_p0_experiments.py / p0_3b 对齐) ====================
CFG = dict(
    datasets=["mvtec", "visa", "btad", "mpdd"],
    memory_bank_dir="./weights/memory_banks_noproj",   # A4 无投影库
    dino_mb_filename="visual.pth",                     # DINO patch 特征 [M,384]
    clip_mb_filename="semantic.pth",                   # CLIP patch 特征 [M,512]
    sweep_csv="./results/summary_sweep_curves.csv",    # dataset,category,weight_visual,image_auroc,pixel_auroc
    pred_csv="./p0_results/p0_3_oracle_vs_predicted.csv",  # 含 w_oracle / w_predicted / rho_*
    out_dir="./p0_results",
    device="cuda" if torch.cuda.is_available() else "cpu",
    eps=1e-8,
    seed=42,
    # --- compactness 复现 p0_3 口径 ---
    sample_n=20000,   # 超过则 RandomState(0) 固定种子子采样 (与 p0_3 逐位同源)
    chunk=1024,       # LOO 1-NN 分块行数, 控显存
    # --- v3 伪异常合成参数 ---
    n_mixup_pairs=2000,
    n_noise_samples=2000,
    mixup_lam_range=(0.3, 0.7),
    noise_gammas=[0.5, 1.0, 2.0, 4.0, 8.0],   # 可扫描; 噪声外推幅度 = gamma * median(normal LOO)
    default_gamma=1.0,
    # --- plateau 判据 (与 p0_3b 一致) ---
    tau_plateau=0.001,
)
torch.manual_seed(CFG["seed"]); np.random.seed(CFG["seed"])
os.makedirs(CFG["out_dir"], exist_ok=True)


# ==================== 数据加载 (复刻 p0_3 compute_compactness_corrected 的采样) ====================
def load_mb(dataset, category):
    """加载 DINO/CLIP MB 并做与 p0_3 完全一致的子采样 (M>20000 → RandomState(0) 抽 20000)."""
    p = Path(CFG["memory_bank_dir"]) / dataset / category
    dino = torch.load(p / CFG["dino_mb_filename"], map_location="cpu").float()
    clip = torch.load(p / CFG["clip_mb_filename"], map_location="cpu").float()
    if dino.dim() == 3: dino = dino.mean(dim=1)
    if clip.dim() == 3: clip = clip.mean(dim=1)

    def _subsample(feat):
        if feat.shape[0] > CFG["sample_n"]:
            rng = np.random.RandomState(0)
            idx = rng.choice(feat.shape[0], CFG["sample_n"], replace=False)
            feat = feat[idx]
        return feat

    dino = _subsample(dino).to(CFG["device"])
    clip = _subsample(clip).to(CFG["device"])
    return dino, clip


def get_categories():
    """三方交集: sweep CSV + pred CSV + MB 目录. 返回 [(dataset, category), ...]"""
    sweep = pd.read_csv(CFG["sweep_csv"])
    pred = pd.read_csv(CFG["pred_csv"])
    keys_sweep = set(zip(sweep.dataset[sweep.category != "MEAN"], sweep.category[sweep.category != "MEAN"]))
    keys_pred = set(zip(pred.dataset, pred.category))
    out = []
    for ds in CFG["datasets"]:
        mb_root = Path(CFG["memory_bank_dir"]) / ds
        if not mb_root.exists(): continue
        for cat in sorted(d.name for d in mb_root.iterdir() if d.is_dir()):
            if (ds, cat) in keys_pred and (ds, cat) in keys_sweep:
                out.append((ds, cat))
    return out


# ==================== 基础几何量 (分块 LOO 1-NN, 排除自身; 与 p0_3 同构) ====================
@torch.no_grad()
def loo_self_distances(feat):
    """每行到库内最近邻(排除自身)的距离 [n]. 分块 cdist, 与 p0_3 chunk 逻辑一致."""
    n = feat.shape[0]
    out = torch.empty(n, device=feat.device, dtype=torch.float32)
    for r0 in range(0, n, CFG["chunk"]):
        r1 = min(r0 + CFG["chunk"], n)
        d = torch.cdist(feat[r0:r1], feat, p=2)          # [b, n]
        rows = torch.arange(r1 - r0, device=d.device)
        d[rows, r0 + rows] = float("inf")                # 排除自身 (全局行号)
        out[r0:r1] = d.min(dim=-1).values
    return out


@torch.no_grad()
def nn_dist_to(q, mb):
    """查询集 q [P,C] 到 mb [M,C] 的最近邻距离 [P]."""
    return torch.cdist(q, mb, p=2).min(dim=-1).values


# ==================== v3: pseudo-anomaly probing ====================
def pseudo_probe_errs(mb, gammas, s_normal=None):
    """
    合成伪异常 (mixup 流形内 + 噪声流形外), 测该模态把伪异常推出 normal 流形的能力。
    返回 {gamma: err=1-AUC}, err 越小越可靠 (AUC: pseudo=1 vs normal=0, score=NN 距离)。
    s_normal 可预传 (loo_self_distances 结果), 避免与 ρ 计算重复做 LOO。
    """
    M = mb.shape[0]
    dev = mb.device
    if s_normal is None:
        s_normal = loo_self_distances(mb)
    med = float(s_normal.median().item())

    # --- mixup 伪异常 (与 gamma 无关, 算一次) ---
    P = CFG["n_mixup_pairs"]
    i = torch.randint(0, M, (P,))
    j = torch.randint(0, M, (P,))
    lam = torch.empty(P).uniform_(*CFG["mixup_lam_range"])
    pm = (lam[:, None].to(dev) * mb[i.to(dev)] + (1 - lam[:, None]).to(dev) * mb[j.to(dev)])
    s_mix = nn_dist_to(pm, mb)

    # --- 噪声伪异常 (沿单位随机方向外推 g*med; med=正常1-NN距离中位数) ---
    # 注意: n0 必须 L2 归一化, 否则位移范数被 √d 放大 ~20 倍, 噪声永远完美可分,
    #       导致 gamma 档恒等 (sanity 已证实 g=0.5/1/2 的 err 完全相同)。
    S = min(CFG["n_noise_samples"], M)
    n0 = torch.randn(S, mb.shape[1], device=dev)
    n0 = n0 / n0.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    base = mb[:S]
    s_normal_np = s_normal.cpu().numpy()
    errs = {}
    for g in gammas:
        pn = base + g * med * n0
        s_noise = nn_dist_to(pn, mb).cpu().numpy()
        labels = np.concatenate([np.ones(P + S), np.zeros(M)])
        scores = np.concatenate([s_mix.cpu().numpy(), s_noise, s_normal_np])
        try:
            auc = roc_auc_score(labels, scores)
        except ValueError:
            auc = 0.5
        errs[g] = 1.0 - auc
    return errs


# ==================== 信号收集 ====================
def collect_signals(keys):
    """返回 dict: {rho_dino, rho_clip, err_dino{g}, err_clip{g}} 均为 {(ds,cat): val}"""
    sig = {"rho_dino": {}, "rho_clip": {}, "err_dino": {}, "err_clip": {}}
    for g in CFG["noise_gammas"]:
        sig["err_dino"][g] = {}
        sig["err_clip"][g] = {}
    for ds, cat in tqdm(keys, desc="Collecting signals (LOO + pseudo-probe)"):
        dino, clip = load_mb(ds, cat)
        # 每模态只算一次 LOO: ρ=mean, s_normal 复用于伪异常探针
        s_d = loo_self_distances(dino)
        s_c = loo_self_distances(clip)
        sig["rho_dino"][(ds, cat)] = float(s_d.mean().item())
        sig["rho_clip"][(ds, cat)] = float(s_c.mean().item())
        e_d = pseudo_probe_errs(dino, CFG["noise_gammas"], s_normal=s_d)
        e_c = pseudo_probe_errs(clip, CFG["noise_gammas"], s_normal=s_c)
        for g in CFG["noise_gammas"]:
            sig["err_dino"][g][(ds, cat)] = e_d[g]
            sig["err_clip"][g][(ds, cat)] = e_c[g]
        del dino, clip, s_d, s_c
    return sig


# ==================== 预测器 (严格 normal-only, 无训练) ====================
def sigmoid(x): return 1.0 / (1.0 + np.exp(-np.clip(x, -50, 50)))

def zscore(vals_by_key, keys):
    a = np.array([vals_by_key[k] for k in keys])
    mu, sd = a.mean(), a.std() + CFG["eps"]
    return {k: (vals_by_key[k] - mu) / sd for k in keys}


def build_predictions(keys, sig, oracle_w):
    """返回 {name: {(ds,cat): w_pred}}"""
    z_rd = zscore(sig["rho_dino"], keys)
    z_rc = zscore(sig["rho_clip"], keys)
    out = {"oracle_upperbound": oracle_w, "v0_raw_compactness": {}}
    for k in keys:
        rd = 1.0 / (sig["rho_dino"][k] + CFG["eps"])
        rc = 1.0 / (sig["rho_clip"][k] + CFG["eps"])
        out["v0_raw_compactness"][k] = rd / (rd + rc)
    out["v2_z_compactness"] = {k: sigmoid(z_rc[k] - z_rd[k]) for k in keys}
    for g in CFG["noise_gammas"]:
        ze_d = zscore(sig["err_dino"][g], keys)
        ze_c = zscore(sig["err_clip"][g], keys)
        name = "v3_pseudo_probe" if g == CFG["default_gamma"] else f"v3_pseudo_probe_gamma{g}"
        out[name] = {k: sigmoid(ze_c[k] - ze_d[k]) for k in keys}
    return out


# ==================== 统一评估 (image-level, p0_3b §11 协议) ====================
def build_detail_frames(keys, predictions, oracle_w, sweep_df):
    """逐类别明细表 {name: DataFrame}"""
    frames = {}
    for name, w_map in predictions.items():
        rows = []
        for ds, cat in keys:
            g = sweep_df[(sweep_df.dataset == ds) & (sweep_df.category == cat)].sort_values("w")
            if g.empty: continue
            w, a = g["w"].to_numpy(), g["image_auroc"].to_numpy()
            w_o = oracle_w[(ds, cat)]
            a_o = float(a.max())                       # oracle AUROC = 曲线峰值 (p0_3b 口径)
            w_p = float(w_map[(ds, cat)])
            a_p = float(np.interp(w_p, w, a))
            rows.append(dict(
                dataset=ds, category=cat,
                w_pred=round(w_p, 4), w_oracle=float(w_o),
                fixed05=round(float(np.interp(0.5, w, a)), 6),
                fixed08=round(float(np.interp(0.8, w, a)), 6),
                oracle_auroc=round(a_o, 6),
                proposed=round(a_p, 6),
                plateau_hit=int(a_p >= a_o - CFG["tau_plateau"]),
            ))
        frames[name] = pd.DataFrame(rows)
    return frames


def _safe_corr(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    if len(x) < 2 or x.std() == 0 or y.std() == 0: return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def aggregate(detail, scope_keys):
    """按 scope 聚合成一行指标 (scope 用 detail 子集的均值/占比)."""
    t = detail
    if scope_keys is not None:
        keep = set(scope_keys)
        t = t[[(r.dataset, r.category) in keep for _, r in t.iterrows()]].copy()
    n = len(t)
    if n == 0: return None
    pearson = _safe_corr(t.w_pred, t.w_oracle)
    try:
        sp, _ = spearmanr(t.w_pred, t.w_oracle)
        sp = float(sp) if sp == sp else float("nan")
    except Exception:
        sp = float("nan")
    return dict(
        N=n,
        pearson=pearson, spearman=sp,
        mae=float(np.abs(t.w_pred - t.w_oracle).mean()),
        plateau_pct=float(100.0 * t.plateau_hit.mean()),
        fixed05=float(t.fixed05.mean()), fixed08=float(t.fixed08.mean()),
        oracle=float(t.oracle_auroc.mean()), proposed=float(t.proposed.mean()),
        prop_minus_08=float(t.proposed.mean() - t.fixed08.mean()),
    )


def format_cell(v, fmt=".4f"):
    return "nan" if (v is None or (isinstance(v, float) and v != v)) else f"{v:{fmt}}"


# ==================== 绘图 ====================
def plot_scatter(frames, keys, best_names):
    colors = {"mvtec": "#1f77b4", "visa": "#ff7f0e", "btad": "#2ca02c", "mpdd": "#d62728"}
    row_titles = ["GLOBAL (all)", "MVTec", "VisA", "BTAD", "MPDD"]
    fig, axes = plt.subplots(5, len(best_names), figsize=(6.5 * len(best_names), 15), squeeze=False)
    for c, name in enumerate(best_names):
        det = frames[name]
        # GLOBAL
        ax = axes[0][c]
        for ds in CFG["datasets"]:
            sub = det[det.dataset == ds]
            if len(sub):
                ax.scatter(sub.w_pred, sub.w_oracle, c=colors[ds], s=42,
                           edgecolors="k", linewidth=0.4, alpha=0.85, label=ds.upper())
        ax.plot([0, 1], [0, 1], "r--", alpha=0.5, linewidth=1)
        r = _safe_corr(det.w_pred, det.w_oracle)
        ax.set_title(f"{name}\nGLOBAL r={format_cell(r, '.3f')}", fontsize=10)
        ax.set_xlim(-0.05, 1.05); ax.set_ylim(-0.05, 1.05)
        ax.set_xlabel("w_pred", fontsize=9); ax.set_ylabel("w_oracle", fontsize=9)
        ax.legend(fontsize=7); ax.grid(alpha=0.3)
        # per dataset
        for r_i, ds in enumerate(CFG["datasets"]):
            ax = axes[r_i + 1][c]
            sub = det[det.dataset == ds]
            if len(sub):
                ax.scatter(sub.w_pred, sub.w_oracle, c=colors[ds], s=48,
                           edgecolors="k", linewidth=0.4, alpha=0.9)
                r = _safe_corr(sub.w_pred, sub.w_oracle)
                ax.set_title(f"{ds.upper()} r={format_cell(r, '.3f')}", fontsize=9)
            ax.plot([0, 1], [0, 1], "r--", alpha=0.5, linewidth=1)
            ax.set_xlim(-0.05, 1.05); ax.set_ylim(-0.05, 1.05)
            ax.set_xlabel("w_pred", fontsize=9); ax.set_ylabel("w_oracle", fontsize=9)
            ax.grid(alpha=0.3)
    fig.suptitle("P0-3c: Oracle vs Predicted Fusion Weight (predictors v0 / v2 / v3)",
                 fontsize=14, y=1.0)
    plt.tight_layout()
    out = Path(CFG["out_dir"]) / "p0_3c_predictor_scatter.png"
    plt.savefig(out, dpi=150, bbox_inches="tight"); plt.close()
    print(f"[图] 散点图已保存: {out}")
    return out


# ==================== 主流程 ====================
def main():
    print("=" * 74)
    print("P0-3c: Predictor v2 (z-compactness) + v3 (pseudo-anomaly probing)")
    print(f"Device: {CFG['device']}")
    print("=" * 74)

    keys = get_categories()
    print(f"有效类别数 (三方交集): {len(keys)}")
    if len(keys) == 0:
        raise SystemExit("未找到三方交集类别, 检查路径/列名")

    # oracle 权重映射 (P0-2 口径)
    pred_df = pd.read_csv(CFG["pred_csv"])
    ocol = "w_oracle" if "w_oracle" in pred_df.columns else "best_w_dino"
    oracle_w = {(r.dataset, r.category): float(r[ocol]) for _, r in pred_df.iterrows()}

    # sweep 曲线 (剔除 MEAN 汇总行, 列重命名 weight_visual -> w)
    sweep_df = pd.read_csv(CFG["sweep_csv"])
    sweep_df = sweep_df[sweep_df.category != "MEAN"].rename(columns={"weight_visual": "w"}).copy()

    # 1) 收集信号 (ρ + v3 伪异常分离度)
    sig = collect_signals(keys)

    # 2) 预测器
    predictions = build_predictions(keys, sig, oracle_w)

    # 3) 逐类明细
    frames = build_detail_frames(keys, predictions, oracle_w, sweep_df)

    # 4) 聚合 (GLOBAL + 各数据集) → leaderboard
    scopes = {"GLOBAL": None}
    for ds in CFG["datasets"]:
        scopes[ds.upper()] = [(a, b) for (a, b) in keys if a == ds]

    lb_rows = []
    for name in frames:
        det = frames[name]
        for scope, scope_keys in scopes.items():
            m = aggregate(det, scope_keys)
            if m is None: continue
            lb_rows.append(dict(predictor=name, scope=scope, **m))
    lb = pd.DataFrame(lb_rows)
    lb_path = Path(CFG["out_dir"]) / "p0_3c_predictor_leaderboard.csv"
    lb.to_csv(lb_path, index=False)
    print(f"[CSV] Leaderboard: {lb_path}")

    # 5) 打印 GLOBAL 对照表
    glob = lb[lb.scope == "GLOBAL"].set_index("predictor")
    print("\n" + "-" * 74)
    print("GLOBAL (36 类) 对照表:   [PASS 判据: plateau%>=60 且 prop-08>=-0.002]")
    print("-" * 74)
    print(f"{'predictor':<28}{'N':<4}{'Pearson':<9}{'MAE':<8}{'plateau%':<9}"
          f"{'fixed08':<9}{'proposed':<9}{'oracle':<9}{'prop-08':<9}")
    for name in frames:
        if name not in glob.index: continue
        r = glob.loc[name]
        print(f"{name:<28}{int(r.N):<4}{format_cell(r.pearson):<9}{format_cell(r.mae):<8}"
              f"{format_cell(r.plateau_pct, '.1f'):<9}{format_cell(r.fixed08):<9}"
              f"{format_cell(r.proposed):<9}{format_cell(r.oracle):<9}{format_cell(r.prop_minus_08, '+.4f'):<9}")

    # 6) 达标判定 (决策树)
    print("\n" + "=" * 74)
    print("决策树判定 (GLOBAL)")
    print("=" * 74)

    def check(name, want_plateau=60):
        r = glob.loc[name]
        p_ok = float(r.plateau_pct) >= want_plateau
        m_ok = float(r.prop_minus_08) >= -0.002
        return p_ok, m_ok, r

    # oracle 管线校验
    r = glob.loc["oracle_upperbound"]
    bug = float(r.plateau_pct) < 99.9 or abs(float(r.prop_minus_08) - 0.003) > 0.001
    print(f"[oracle_upperbound] plateau%={r.plateau_pct:.1f} prop-08={r.prop_minus_08:+.4f} "
          f"({'❌ 评估管线/列名有 bug, 先修管线!' if bug else '✅ 校验通过'})")

    # v0 同源校验
    r = glob.loc["v0_raw_compactness"]
    src_ok = abs(float(r.prop_minus_08) - (-0.0047)) < 0.0015 and abs(float(r.plateau_pct) - 33.3) < 5
    # ρ 与 p0_3 CSV 逐类核对
    recomputed = {}
    for k in keys:
        rd = 1.0 / (sig["rho_dino"][k] + CFG["eps"])
        rc = 1.0 / (sig["rho_clip"][k] + CFG["eps"])
        recomputed[k] = rd / (rd + rc)
    csv_w = {(r.dataset, r.category): float(r.w_predicted) for _, r in pred_df.iterrows()}
    dmax = max(abs(recomputed[k] - csv_w[k]) for k in keys)
    print(f"[v0_raw_compactness] plateau%={r.plateau_pct:.1f} prop-08={r.prop_minus_08:+.4f} "
          f"|w_recompute-w_p0_3csv|max={dmax:.2e} "
          f"({'❌ 信号收集与 p0_3 不同源 → 检查 load_mb 采样种子' if (not src_ok or dmax > 1e-3) else '✅ 同源'})")

    for name in ["v2_z_compactness"] + [n for n in frames if n.startswith("v3_pseudo_probe")]:
        r = glob.loc[name]
        p_ok, m_ok, _ = check(name)
        print(f"[{name}] plateau%={r.plateau_pct:.1f} prop-08={r.prop_minus_08:+.4f} "
              f"(Pearson={r.pearson:+.3f}) → {'✅ PASS → CAME 采用该预测器' if (p_ok and m_ok) else '❌ FAIL'}")

    # 7) 逐数据集 prop-08 / plateau% (p0_3b 口径, 查 visa 等弱点)
    print("\n逐数据集 prop-08 / plateau%:")
    print(f"{'predictor':<28}{'':<4}" + "".join(f"{ds.upper():>14}" for ds in CFG["datasets"]))
    for name in frames:
        line = f"{name:<32}"
        for ds in CFG["datasets"]:
            sub = lb[(lb.predictor == name) & (lb.scope == ds.upper())]
            if len(sub):
                line += f"{sub.iloc[0].prop_minus_08:+.3f}/{sub.iloc[0].plateau_pct:.0f}%".rjust(14)
        print(line)

    # 8) 逐类明细 CSV
    for name, det in frames.items():
        det.to_csv(Path(CFG["out_dir"]) / f"p0_3c_detail_{name}.csv", index=False)
    print(f"\n[CSV] 逐类明细已保存: p0_results/p0_3c_detail_*.csv "
          f"(含 v3 默认档 p0_3c_detail_v3_pseudo_probe.csv)")

    # 9) 散点图
    best_names = ["v0_raw_compactness", "v2_z_compactness", "v3_pseudo_probe"]
    plot_scatter(frames, keys, best_names)

    print("\n" + "=" * 74)
    print("解读指南")
    print("=" * 74)
    print("""
  • oracle plateau%=100 & prop-08=+0.003 → 管线/列名 OK
  • v0 复现 p0_3 (prop-08≈-0.0047, plateau≈33%) → 同源 OK
  • v2 达标(plateau>=60 & prop-08>=-0.002) → z-score 够用, CAME 用 v2
  • v2 败 / v3 达标            → compactness 信息不足, probing 有效, CAME 用 v3
  • v2、v3 均败                → normal-only 信号无法预测 oracle, 走导师 §12 研究路线
      (fixed w≈0.8 捕获 oracle 大部分增益 + empirical/mechanistic 为主贡献)
""")


if __name__ == "__main__":
    main()
