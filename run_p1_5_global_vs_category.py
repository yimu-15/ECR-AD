#!/usr/bin/env python3
"""
P1-5: Global vs Category Memory Bank — mechanism analysis (导师 §25)
====================================================================
回答: 为什么 category-specific MB 比 global MB 高 (VisA 曾达 ~45pp)?
机制假说: global MB 中其他类别的 normal 特征会"吸收"anomaly 的 1-NN,
          压低 anomaly 分数 → normal/anomaly 分布重叠 → 可分性坍塌;
          而 normal 测试样本总能找到本类邻居, 分数几乎不动 (不对称位移)。

三组设置 (两个对照):
  1. category        : 本类 MB (主方法)
  2. global_full     : 数据集内全部类别 MB 直接拼接 (无尺寸控制)
  3. global_matched  : 每类等量下采样 (seed=42), 总规模 = 单类 MB 中位数
                       → 尺寸对照: 分离"类别特化增益"与"MB 规模效应"

模态 (默认 dino; 开关 clip / fused):
  dino : 只用 visual.pth (DINOv2 384D 无投影)
  clip : 只用 semantic.pth (CLIP 512D 无投影)
  fused: 双模态分数层融合, 权重 CFG["fused_w_dino"] (默认 0.8 DINO / 0.2 CLIP)

机制指标:
  image_auroc       : 逐图 Top-1% mean (与 evaluate_all 图像级口径完全一致)
  d_prime           : (μ_anom - μ_norm) / sqrt((σ²_anom + σ²_norm)/2)
  xcat_nn_ratio     : global MB 下, 测试 patch 的 1-NN 落在其他类别的比例
                      (normal 图 / anomaly 图分开统计)
  不对称位移         : (mean_norm/mean_anom) category vs global 的逐类差

口径一致性保障 (与 evaluate_all 同源):
  - 测试特征缓存 test_features_cache/{ds}/{cat}/(dino|clip)_test.pth
    由 evaluate_all 的 BatchEvaluator 编码器 + 相同 transform 现推生成。
    缓存缺失 → FileNotFoundError 拒绝运行 (严禁随机占位)。
  - 打分: L2 归一化查询 × (L2 归一化 bank) → mm → 1-sim → top-k(5) 均值
    = MemoryBank.compute_anomaly_score 逐位同构。
  - 图像级聚合 = bilinear 上采样到 518x518 → top 1% → mean (CFG["agg"]="topk_mean").
  - 运行前可用 --verify 对 mvtec/bottle 做单类校准: category-DINO image AUROC
    必须与 results/summary_sweep_curves.csv 中 w=1.0 行 |Δ|<0.002 才放行全量。

产物 (out_dir = p1_5_results/):
  p1_5_global_vs_category.csv     逐类 × 逐设置
  p1_5_dataset_summary.csv        数据集级汇总 (AUROC / d' / xcat / 位移)
  p1_5_dist_{dataset}.png         §25 分布图 (3 面板: category | matched | full)
  p1_5_xcat_{dataset}.png         跨类 NN 干扰条形图 (matched vs full)

用法:
  python run_p1_5_global_vs_category.py --datasets mvtec --verify        # 单类校准
  python run_p1_5_global_vs_category.py --datasets mvtec visa            # 全量 (默认 dino)
  python run_p1_5_global_vs_category.py --modality fused --datasets mvtec visa
"""

import os
import sys
import csv
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

from sklearn.metrics import roc_auc_score

# ==================== 配置 ====================
CFG = dict(
    datasets=["mvtec", "visa"],
    modality="dino",               # "dino" | "clip" | "fused"
    fused_w_dino=0.8,              # modality="fused" 时的 DINO 权重
    use_projection=False,          # False: memory_banks_noproj (A4 最优配置, 默认)

    # ---- 打分 / 聚合口径 (必须与 scripts/evaluate_all.py 一致) ----
    k_nn=5,                        # MemoryBank.compute_anomaly_score 的 top-k
    agg="topk_mean",               # 图像级聚合: evaluate_all = top 1% mean (不是 max)
    topk_ratio=0.01,

    bank_root_proj="weights/memory_banks",        # 投影 256D 库
    bank_root_noproj="weights/memory_banks_noproj",  # 无投影 384/512D 库
    test_feat_cache_dir=os.path.join(PROJECT_ROOT, "test_features_cache"),
    out_dir=os.path.join(PROJECT_ROOT, "p1_5_results"),
    config_path=os.path.join(PROJECT_ROOT, "configs/dataset.yaml"),
    projection_weights=os.path.join(PROJECT_ROOT, "weights/projection_aligned.pth"),

    device="cuda" if torch.cuda.is_available() else "cpu",
    seed=42,
    eps=1e-8,
    batch_size=8,
    mm_bytes_budget=0.9e9,         # GPU 上单块 mm 输出上限 (防 OOM, 8GB 卡)
)

os.makedirs(CFG["out_dir"], exist_ok=True)
os.makedirs(CFG["test_feat_cache_dir"], exist_ok=True)

_EVAL = None  # 惰性单例 BatchEvaluator (与 evaluate_all 完全同源)


def get_evaluator():
    """惰性构建 evaluate_all 的 BatchEvaluator (use_projection 跟随 CFG)。"""
    global _EVAL
    if _EVAL is None:
        from scripts.evaluate_all import BatchEvaluator
        _EVAL = BatchEvaluator(CFG["config_path"], CFG["projection_weights"],
                               device=CFG["device"],
                               use_projection=CFG["use_projection"])
    return _EVAL


def bank_root():
    return CFG["bank_root_proj"] if CFG["use_projection"] else CFG["bank_root_noproj"]


def bank_fname(modality):
    """每个模态对应的 bank 文件名 (与 build_memory_bank_per_class.py 一致)。"""
    return "visual.pth" if modality in ("dino", "fused") else "semantic.pth"


# ==================== 类别列表 ====================
def get_categories(dataset):
    import yaml
    with open(CFG["config_path"], "r", encoding="utf-8") as f:
        cats = yaml.safe_load(f)[dataset]["categories"]
    return cats


# ==================== 测试特征缓存 (必须与 evaluate_all 同源) ====================
def _load_cache_blob(dataset, category, feat_key):
    """feat_key: 'dino' | 'clip' → {ds}/{cat}/{feat_key}_test.pth"""
    p = os.path.join(CFG["test_feat_cache_dir"], dataset, category, f"{feat_key}_test.pth")
    if not os.path.exists(p):
        raise FileNotFoundError(
            f"[P1-5] 测试特征缓存缺失: {p}\n"
            f"  先用 --build_cache 生成 (或跑完整 run 时会自动生成)。\n"
            f"  缓存必须由 evaluate_all 同源编码器产生 — 严禁随机特征占位!")
    blob = torch.load(p, map_location="cpu")
    feats = blob["feats"].float()          # [N, HW, D]
    labels = blob["labels"].long()         # [N]
    assert feats.dim() == 3 and labels.numel() == feats.shape[0]
    return feats, labels


def load_test_features(dataset, category):
    """加载测试特征与标签。
    dino  : feats [N, 1369, 384], labels [N]
    clip  : feats [N, 49, 512],   labels [N]
    fused : 两者都要 (返回 (dino, clip, labels))
    """
    m = CFG["modality"]
    if m == "fused":
        fd, labels = _load_cache_blob(dataset, category, "dino")
        fc, _ = _load_cache_blob(dataset, category, "clip")
        return fd, fc, labels
    if m == "dino":
        f, labels = _load_cache_blob(dataset, category, "dino")
        return f, None, labels
    f, labels = _load_cache_blob(dataset, category, "clip")
    return f, None, labels


def _load_feats(dataset, category, key):
    """按指定 key 显式加载缓存特征 (与全局 CFG['modality'] 解耦, 供 fused 使用)。"""
    feats, labels = _load_cache_blob(dataset, category, key)
    return feats, labels


def build_feature_cache(dataset, categories, need_clip):
    """用 BatchEvaluator (与 evaluate_all 同一编码器/transform) 现推 test 特征并存盘。
    dino 特征: images 518 -> [N, 1369, 384]   (含 evaluate_all 完全一致的前处理)
    clip 特征: 224 bilinear + CLIP 归一化 -> [N, 49, 512]
    """
    ev = get_evaluator()
    from models.dataset import IndustrialADDataset
    from torch.utils.data import DataLoader
    from tqdm import tqdm

    device = torch.device(CFG["device"])

    for cat in categories:
        cache_dir = os.path.join(CFG["test_feat_cache_dir"], dataset, cat)
        os.makedirs(cache_dir, exist_ok=True)
        want = ["dino"] + (["clip"] if need_clip else [])
        todo = [k for k in want
                if not os.path.exists(os.path.join(cache_dir, f"{k}_test.pth"))]
        if not todo:
            continue
        print(f"[cache] {dataset}/{cat}: 提取测试特征 {todo} ...")

        test_ds = IndustrialADDataset(CFG["config_path"], dataset, cat, split="test")
        loader = DataLoader(test_ds, batch_size=CFG["batch_size"], shuffle=False,
                            num_workers=4, pin_memory=True)

        feats_cache = {k: [] for k in todo}
        labels_all = []
        for batch in tqdm(loader, desc=f"  {cat}", leave=False):
            images_dino = batch["image"].to(device)     # [B,3,518,518]
            labels_all.extend(int(x) for x in batch["label"].numpy())

            if "dino" in todo:
                dino_out = ev.dino(images_dino)
                feats_cache["dino"].append(dino_out[:, 1:, :].float().cpu())   # [B,1369,384]

            if "clip" in todo:
                images_clip = F.interpolate(images_dino, size=(224, 224),
                                            mode="bilinear", align_corners=False)
                images_clip = (images_clip * ev.dino_std + ev.dino_mean
                               - ev.clip_mean) / ev.clip_std
                clip_out = ev.clip(images_clip)
                feats_cache["clip"].append(clip_out[:, 1:, :].float().cpu())   # [B,49,512]

        labels_t = torch.tensor(labels_all, dtype=torch.long)   # [N]
        for k in todo:
            Fk = torch.cat(feats_cache[k], dim=0).float()        # [N, P, D]
            out_p = os.path.join(cache_dir, f"{k}_test.pth")
            torch.save({"feats": Fk, "labels": labels_t}, out_p)
            print(f"  [cache] saved {os.path.relpath(out_p)} {tuple(Fk.shape)}")
    print(f"[cache] {dataset}: test 特征缓存就绪 ({len(categories)} 类, clip={need_clip})")


def ensure_test_features(dataset, categories, modality):
    """本次运行所需缓存缺失时，用 evaluate_all 同源编码器现推并存盘。

    与 evaluate_all.evaluate_category 逐字段同源：
      dino: images 518 -> dino_out[:, 1:, :]        (137x... 实为 [N,1369,384])
      clip: 224 bilinear + dino->clip 反归一化 -> clip_out[:, 1:, :] ([N,49,512])
    缓存文件为 {"feats": [N,P,D], "labels": [N]} —— 原始特征（未归一化），
    打分时再归一化（与 MemoryBank.compute_anomaly_score 一致）。
    """
    need_clip = modality in ("clip", "fused")
    missing = 0
    for cat in categories:
        keys = ["dino"] + (["clip"] if need_clip else [])
        for k in keys:
            p = os.path.join(CFG["test_feat_cache_dir"], dataset, cat, f"{k}_test.pth")
            if not os.path.exists(p):
                missing += 1
    if missing:
        print(f"[cache] {missing} 个缓存缺失 -> 调用 evaluate_all 同源提取器现推...")
        build_feature_cache(dataset, categories, need_clip=need_clip)


# ==================== 记忆库加载 / Global Bank 构建 ====================
def load_category_bank(dataset, category, key):
    """加载类别特化 MB（与 evaluate_all.load_memory_banks 同目录/同格式）。
    key: 'dino' -> visual.pth (384D, 已 L2 归一化)
         'clip' -> semantic.pth (512D, 已 L2 归一化)
    返回 [M, D] (CPU)。
    """
    fname = "visual.pth" if key == "dino" else "semantic.pth"
    p = os.path.join(bank_root(), dataset, category, fname)
    if not os.path.exists(p):
        raise FileNotFoundError(f"[P1-5] 类别 MB 缺失: {p}")
    bank = torch.load(p, map_location="cpu").float()
    assert bank.dim() == 2, f"unexpected bank shape {tuple(bank.shape)}"
    return bank


def build_global_bank(dataset, categories, key, matched):
    """数据集内全部类别 MB 拼接 → global bank。

    - global_full: 直接 cat (无尺寸控制)。MVTec 15 类视觉特征 ~1.48M 行。
    - global_matched: 每类用 seed42 的 randperm 等量下采样,
        总规模 ≈ 单类 MB 规模的中位数 → 尺寸对照 (分离"类别特化增益"
        与"MB 规模效应")。

    返回 (bank [M',D] 已归一化, cat_ids [M'] 每行所属类别下标)。
    """
    banks, sizes = [], []
    for c in categories:
        b = load_category_bank(dataset, c, key)
        banks.append(b)
        sizes.append(b.shape[0])
    if matched:
        target = int(np.median(sizes))          # 对照目标: 单类 MB 中位规模
        budget = max(target // len(categories), 1)
        keep_plan = []
        for i, b in enumerate(banks):
            n = b.shape[0]
            keep_plan.append(min(budget, n))
        print(f"[global] matched: median={target}, per-class budget={budget}"
              f", 实际保留={keep_plan}")
        sel, sel_ids = [], []
        g = torch.Generator().manual_seed(CFG["seed"])   # 确定性下采样
        for i, b in enumerate(banks):
            keep = keep_plan[i]
            if keep < b.shape[0]:
                idx = torch.randperm(b.shape[0], generator=g)[:keep]
                sel.append(b[idx])
            else:
                sel.append(b)
            sel_ids.append(torch.full((sel[-1].shape[0],), i, dtype=torch.long))
        bank = torch.cat(sel, dim=0)
        cat_ids = torch.cat(sel_ids, dim=0)
    else:
        bank = torch.cat(banks, dim=0)
        cat_ids = torch.cat([torch.full((b.shape[0],), i, dtype=torch.long)
                             for i, b in enumerate(banks)], dim=0)
    return bank, cat_ids


# ==================== kNN 打分 (与 MemoryBank 逐位同构 + 返回 NN 类别) ====================
@torch.no_grad()
def _topk_distances(query, bank, k=5, return_ids=False):
    """query [Q, D] (原始特征, 未归一化), bank [M, D] (已归一化)。
    与 MemoryBank.compute_anomaly_score 一致:
      query 再 F.normalize → mm → distances = 1 - sim → top-k 最小距离。
    返回 (dists [Q,k], ids [Q,k] 或 None); ids = bank 行号 (升序最近在前)。
    """
    query = F.normalize(query, dim=-1)
    Q, M = query.shape[0], bank.shape[0]
    kk = min(k, M)
    rmax = max(1, int(CFG["mm_bytes_budget"] / (M * 4)))   # 单块 mm 输出 [rmax, M]
    dists_all, ids_all = [], []
    for i in range(0, Q, rmax):
        q = query[i:i + rmax]
        sim = torch.mm(q, bank.T)                       # [rmax, M]
        dist = 1.0 - sim
        vals, idx = torch.topk(dist, k=kk, dim=1, largest=False)
        dists_all.append(vals)
        ids_all.append(idx)
    dists = torch.cat(dists_all, dim=0)
    ids = torch.cat(ids_all, dim=0) if return_ids else None
    return dists, ids


@torch.no_grad()
def patch_score_maps(feat3, bank, grid_src, grid_dst, k=5, return_ids=False):
    """feat3 [N, P, D] 原始测试特征 -> 每 patch 异常分数 [N, grid_dst, grid_dst]。

    - dino: grid_src=37, grid_dst=37 (直接 reshape)
    - clip: grid_src=7  → 按 evaluate_all 上采样到 37 再进统一流水线
    bank [M, D]。返回 (maps [N,g,g], ids [N,P,k] 或 None)。
    """
    N, P, D = feat3.shape
    dists, ids = _topk_distances(feat3.reshape(N * P, D).to(CFG["device"]),
                                 bank.to(CFG["device"]), k=k, return_ids=return_ids)
    scores = dists.mean(dim=1).reshape(N, grid_src, grid_src)   # 越大越异常
    if grid_src != grid_dst:
        scores = F.interpolate(scores.unsqueeze(1), size=(grid_dst, grid_dst),
                               mode="bilinear", align_corners=False).squeeze(1)
    if ids is not None:
        ids = ids.reshape(N, P, ids.shape[1])
    return scores.cpu(), ids


def _image_scores_from_maps(maps):
    """maps [N, 37, 37] (cpu) -> 每图 Top-1% mean 图像分数 [N]。
    与 evaluate_all 图像级口径一致: 37x37 bilinear -> 518x518 -> top 1% mean。
    """
    amap = F.interpolate(maps.unsqueeze(1), size=(518, 518), mode="bilinear",
                         align_corners=False).squeeze(1).flatten(1)   # [N, 518*518]
    k = max(int(amap.shape[1] * CFG["topk_ratio"]), 1)
    topk = torch.topk(amap, k=k, dim=1).values.mean(dim=1)
    return topk.numpy()


def _grid_from_P(P):
    g = int(round(P ** 0.5))
    assert g * g == P, f"P={P} 不是平方数 -> 无法 reshape 成正方形 patch grid"
    return g


# ==================== 指标 (AUROC / d' / xcat / 位移) ====================
def _stats_from_scores(image_scores, labels):
    """image_scores [N], labels [N] (0=normal, 1=anomaly) -> dict。
    d' = (μ_anom - μ_norm) / sqrt((σ²_anom + σ²_norm)/2), 基于逐图 Top-1% mean 分数。
    """
    s = np.asarray(image_scores, dtype=np.float64)
    y = np.asarray(labels)
    idx_n, idx_a = y == 0, y == 1
    mu_n, sd_n = (float(s[idx_n].mean()), float(s[idx_n].std(ddof=0))) if idx_n.any() else (float("nan"), float("nan"))
    mu_a, sd_a = (float(s[idx_a].mean()), float(s[idx_a].std(ddof=0))) if idx_a.any() else (float("nan"), float("nan"))
    if idx_n.any() and idx_a.any():
        try:
            auroc = float(roc_auc_score(y, s))
        except ValueError:
            auroc = float("nan")
        dprime = (mu_a - mu_n) / np.sqrt((sd_a ** 2 + sd_n ** 2) / 2 + CFG["eps"])
    else:
        auroc = float("nan")
        dprime = float("nan")
    return dict(image_auroc=auroc, d_prime=float(dprime),
                mean_norm=mu_n, mean_anom=mu_a, std_norm=sd_n, std_anom=sd_a)


def _xcat_from_nn_ids(nn_ids, cat_ids, cat_idx, labels):
    """跨类 1-NN 比例。

    nn_ids [N,P]: 每个 test patch 的 1-NN 在 global bank 中的行号
                  (已由上游从 top-k 索引取第一列);
    cat_ids [M]:  global bank 每行所属的类别下标 (build_global_bank 返回,
                  多类拼接后行号 ≠ 类别下标, 必须经此映射);
    cat_idx:      当前测试类别在 dataset 类别列表中的下标。
    返回 (ratio_norm, ratio_anom): normal/anomaly patch 的 1-NN 落在
    其他类别(非 cat_idx)的比例。
    """
    y = labels.numpy()
    nn = nn_ids.numpy()                                    # [N, P]
    cc = (cat_ids.detach().cpu().numpy() if torch.is_tensor(cat_ids)
          else np.asarray(cat_ids))                        # [M]
    out = {}
    for grp, m in (("norm", y == 0), ("anom", y == 1)):
        if int(m.sum()) == 0:
            out[grp] = float("nan")
            continue
        own = cc[nn[m]] == cat_idx   # 1-NN 落在本类 bank → 不算跨类
        out[grp] = float(1.0 - own.mean()) if own.size else float("nan")
    return out["norm"], out["anom"]


# ==================== 单类别 × 单设置 打分 ====================
_BANK_CACHE = {}


def _get_bank(dataset, categories, key, setting, category):
    """返回 (bank, cat_ids)。bank 缓存到 GPU (device), cat_ids 保留 CPU。
    每个 (dataset,key,setting) 只构建一次; global 构建时也顺带校验单类库齐全。"""
    ck = (dataset, key, setting, category if setting == "category" else "")
    if ck not in _BANK_CACHE:
        if setting == "category":
            b = load_category_bank(dataset, category, key)
            _BANK_CACHE[ck] = (b.to(CFG["device"]), None)
        else:
            bank, ids = build_global_bank(dataset, categories, key,
                                          matched=(setting == "global_matched"))
            _BANK_CACHE[ck] = (bank.to(CFG["device"]), ids)
    return _BANK_CACHE[ck]


@torch.no_grad()
def score_setting(dataset, categories, cat_idx, category, setting, modality):
    """一个 (类别, 设置) 的完整打分与机制指标。

    模态:
      dino : dino 特征 vs visual bank (37 grid)
      clip : clip 特征 vs semantic bank (7 -> 37 grid)
      fused: 双模态分数层融合 0.8*dino + 0.2*clip (37 grid, 与 evaluate_all 同)
    xcat 仅在 global 设置 & 单一模态时定义 (融合分数无单一 bank 行归属)。

    返回 dict: modality/setting/bank_size/n_norm/n_anom/
               image_auroc/d_prime/mean_norm/mean_anom/std_norm/std_anom/
               xcat_norm/xcat_anom
    """
    need_xcat = (setting != "category") and (modality != "fused")

    # 显式按 key 加载 (解耦 CFG['modality'])；labels 两模态同源, 只取一次
    dino_feat = clip_feat = labels = None
    if modality in ("dino", "fused"):
        dino_feat, labels = _load_feats(dataset, category, "dino")
        dino_feat = dino_feat.to(CFG["device"])
    if modality in ("clip", "fused"):
        clip_feat, labels_c = _load_feats(dataset, category, "clip")
        clip_feat = clip_feat.to(CFG["device"])
        labels = labels_c if labels is None else labels

    maps = ids_all = cat_ids = None
    bank_M = None
    if modality in ("dino", "fused"):
        bank, cat_ids = _get_bank(dataset, categories, "dino", setting, category)
        bank_M = int(bank.shape[0])
        g_src = _grid_from_P(dino_feat.shape[1])
        maps, ids_all = patch_score_maps(dino_feat, bank, g_src, 37,
                                         k=CFG["k_nn"], return_ids=need_xcat)
    if modality in ("clip", "fused"):
        bank, cids = _get_bank(dataset, categories, "clip", setting, category)
        if cat_ids is None:
            cat_ids = cids
        if bank_M is None:
            bank_M = int(bank.shape[0])
        g_src = _grid_from_P(clip_feat.shape[1])
        mc, idsc = patch_score_maps(clip_feat, bank, g_src, 37,
                                    k=CFG["k_nn"], return_ids=need_xcat)
        if maps is None:
            maps, ids_all = mc, idsc
        else:
            # 融合在 37x37 分数层完成 (与 evaluate_all._aggregate_records 同构),
            # 此后无单一 bank 行归属 → xcat 不定义
            maps = CFG["fused_w_dino"] * maps + (1.0 - CFG["fused_w_dino"]) * mc
            ids_all = None

    imgs = _image_scores_from_maps(maps)
    y = labels.numpy()
    st = _stats_from_scores(imgs, y)
    if need_xcat and ids_all is not None:
        xc_n, xc_a = _xcat_from_nn_ids(ids_all[:, :, 0].cpu(), cat_ids,
                                       cat_idx, labels.cpu())
    else:
        xc_n = xc_a = float("nan")

    return dict(modality=modality, setting=setting, bank_size=bank_M,
                n_norm=int((y == 0).sum()), n_anom=int((y == 1).sum()),
                xcat_norm=xc_n, xcat_anom=xc_a, **st)


# ==================== 汇总: run_dataset + 不对称位移 ====================
def _compute_displacement(rows):
    """为 global 设置行补 disp_norm/disp_anom (相对同类的 category 行):
    disp = mean(global) - mean(category) → normal≈0 而 anomaly 显著为负
    (被其他类 normal 吸收) 即为不对称位移假说的证据。"""
    cat_row = next(r for r in rows if r["setting"] == "category")
    for r in rows:
        if r["setting"] == "category":
            r["disp_norm"] = r["disp_anom"] = float("nan")
        else:
            r["disp_norm"] = r["mean_norm"] - cat_row["mean_norm"]
            r["disp_anom"] = r["mean_anom"] - cat_row["mean_anom"]
    return rows


SETTINGS = ["category", "global_matched", "global_full"]


def run_dataset(dataset, modality):
    """一个数据集 × 一种模态的完整三臂跑分 (category / global_matched / global_full)。

    返回 (rows, summary): 逐类逐设置明细行 + 数据集级汇总行。
    明细行列见 score_setting; summary 为各设置均值 (AUROC/d'/xcat/位移)。
    """
    categories = get_categories(dataset)
    print(f"\n===== P1-5 [{dataset}] modality={modality} 类别数={len(categories)} =====")

    # 预热 global bank 进 _BANK_CACHE (每 (dataset,key,setting) 只构建一次),
    # 顺带校验所有单类库齐全
    for key in (["dino"] if modality == "dino" else
                ["clip"] if modality == "clip" else ["dino", "clip"]):
        for s in ("global_matched", "global_full"):
            _get_bank(dataset, categories, key, s, None)

    rows = []
    for ci, cat in enumerate(categories):
        for setting in SETTINGS:
            r = score_setting(dataset, categories, ci, cat, setting, modality)
            r.update(dataset=dataset, category=cat)
            rows.append(r)
        _compute_displacement([r for r in rows if r["category"] == cat])
        cr = next(r for r in rows if r["category"] == cat and r["setting"] == "category")
        gf = next(r for r in rows if r["category"] == cat and r["setting"] == "global_full")
        print(f"  [{cat:<14}] cat={cr['image_auroc']:.4f} | full={gf['image_auroc']:.4f}"
              f" | d_anom(full-cat)={gf['disp_anom']:+.4f} d_norm={gf['disp_norm']:+.4f}"
              f" | xcat_a={gf['xcat_anom']:.3f}")

    summary = _summarize(rows, dataset, modality)
    return rows, summary


def _summarize(rows, dataset, modality):
    """数据集级汇总: 每设置一个均值行 (AUROC/d' 取有效均值; xcat/disp 仅 global)。"""
    out = []
    for setting in SETTINGS:
        rs = [r for r in rows if r["setting"] == setting]
        row = dict(dataset=dataset, modality=modality, setting=setting,
                   n_categories=len(rs), bank_size=int(np.mean([r["bank_size"] for r in rs])))
        for field in ("image_auroc", "d_prime", "xcat_norm", "xcat_anom",
                      "disp_norm", "disp_anom"):
            vals = [r[field] for r in rs if r[field] == r[field]]
            row[field] = float(np.mean(vals)) if vals else float("nan")
        out.append(row)
    m = {r["setting"]: r for r in out}
    print(f"  >> AUROC mean: category={m['category']['image_auroc']:.4f}"
          f" | matched={m['global_matched']['image_auroc']:.4f}"
          f" | full={m['global_full']['image_auroc']:.4f}")
    if m["global_full"]["disp_anom"] == m["global_full"]["disp_anom"]:
        print(f"  >> 不对称位移 (full-cat): mean d_anom={m['global_full']['disp_anom']:+.4f}"
              f" d_norm={m['global_full']['disp_norm']:+.4f}"
              f" | xcat_anom={m['global_full']['xcat_anom']:.3f}"
              f" xcat_norm={m['global_full']['xcat_norm']:.3f}")
    return out


# ==================== 图 (p1_5_dist_* / p1_5_xcat_*) ====================
def plot_distributions(rows, dataset):
    """§25 不对称位移分布图: 每类 mean 分数 (norm/anom) 沿 3 设置的变化。
    预期: normal 三条线几乎重合; anomaly 线在 global_matched/full 显著下移。"""
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.2), sharey=True)
    cat_order = sorted({r["category"] for r in rows})
    x = np.arange(len(cat_order))
    styles = {"category": ("#2c7fb8", "-"), "global_matched": ("#f0a500", "--"),
              "global_full": ("#d62728", ":")}
    for ax, setting in zip(axes, SETTINGS):
        norm = [next(r for r in rows if r["category"] == c and r["setting"] == setting)["mean_norm"]
                for c in cat_order]
        anom = [next(r for r in rows if r["category"] == c and r["setting"] == setting)["mean_anom"]
                for c in cat_order]
        col, ls = styles[setting]
        ax.plot(x, norm, color=col, ls=ls, marker="o", ms=4, label="normal mean")
        ax.plot(x, anom, color=col, ls=ls, marker="s", ms=4, label="anomaly mean")
        ax.set_xticks(x)
        ax.set_xticklabels(cat_order, rotation=90, fontsize=7)
        ax.set_title(setting)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    axes[0].set_ylabel("per-image Top-1% mean score")
    fig.suptitle(f"P1-5 asymmetric shift — {dataset} (dotted=anomaly 被吸收?)", fontsize=12)
    fig.tight_layout()
    p = os.path.join(CFG["out_dir"], f"p1_5_dist_{dataset}.png")
    fig.savefig(p, dpi=150)
    plt.close(fig)
    print(f"[fig] {p}")


def plot_xcat(rows, dataset):
    """跨类 1-NN 干扰: global_matched vs global_full 的 xcat_norm / xcat_anom 条形图。"""
    fig, axes = plt.subplots(1, 2, figsize=(17, 4.6), sharey=True)
    cat_order = sorted({r["category"] for r in rows})
    x = np.arange(len(cat_order))
    w = 0.38
    for ax, metric, title in ((axes[0], "xcat_norm", "normal patches"),
                              (axes[1], "xcat_anom", "anomaly patches")):
        for j, s in enumerate(("global_matched", "global_full")):
            vals = [next(r for r in rows if r["category"] == c and r["setting"] == s)[metric]
                    for c in cat_order]
            ax.bar(x + (j - 0.5) * w, vals, w, label=s, color=("#f0a500", "#d62728")[j])
        ax.set_xticks(x)
        ax.set_xticklabels(cat_order, rotation=90, fontsize=7)
        ax.set_title(f"{title} — 1-NN 落在其他类别比例")
        ax.grid(alpha=0.3, axis="y")
        ax.legend(fontsize=8)
    fig.suptitle(f"P1-5 cross-category NN interference — {dataset}", fontsize=12)
    fig.tight_layout()
    p = os.path.join(CFG["out_dir"], f"p1_5_xcat_{dataset}.png")
    fig.savefig(p, dpi=150)
    plt.close(fig)
    print(f"[fig] {p}")


# ==================== 保存 CSV ====================
def _save_rows(rows, path):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"[csv] {path} ({len(rows)} 行)")


# ==================== 单类校准 (--verify) ====================
def verify_bottle(modality):
    """mvtec/bottle × category setting: DINO image AUROC 必须与
    results/summary_sweep_curves.csv 中 (mvtec,bottle,w=1.0) |Δ|<0.002 才放行。"""
    dataset, cat, target_w = "mvtec", "bottle", 1.0
    if modality == "clip":
        target_w = 0.0
    if modality == "fused":
        raise SystemExit("--verify 仅对单模态 (dino/clip) 有意义 (fused 需双权重列)")

    categories = get_categories(dataset)
    print(f"[verify] {dataset}/{cat}  modality={modality}  (w={target_w} 纯DINO侧参照)")
    row = score_setting(dataset, categories, categories.index(cat), cat,
                        "category", modality)

    csv_path = os.path.join(PROJECT_ROOT, "results/summary_sweep_curves.csv")
    with open(csv_path, newline="") as f:
        ref = next((r for r in csv.DictReader(f)
                    if r["dataset"] == dataset and r["category"] == cat
                    and abs(float(r["weight_visual"]) - target_w) < 1e-9), None)
    if ref is None:
        raise SystemExit(f"[verify] {csv_path} 中找不到 (mvtec,bottle,w={target_w}) 行")
    ref_auc = float(ref["image_auroc"])
    got = row["image_auroc"]
    diff = abs(got - ref_auc)
    print(f"[verify] computed image_auroc={got:.6f} | sweep w={target_w}="
          f"{ref_auc:.6f} | Δ={diff:.6f}")
    if diff < 0.002:
        print("[verify] PASS (Δ<0.002) → 口径与 evaluate_all 一致, 可放行全量")
        return True
    print(f"[verify] FAIL (Δ={diff:.4f} >= 0.002) → 请检查聚合/特征口径后再全量")
    return False


# ==================== main ====================
def main():
    ap = argparse.ArgumentParser(
        description="P1-5 Global vs Category Memory Bank 机制分析")
    ap.add_argument("--datasets", nargs="+", default=CFG["datasets"],
                    choices=["mvtec", "visa", "btad", "mpdd"])
    ap.add_argument("--modality", default=CFG["modality"],
                    choices=["dino", "clip", "fused"])
    ap.add_argument("--fused_w_dino", type=float, default=CFG["fused_w_dino"])
    ap.add_argument("--agg", default=CFG["agg"],
                    help="图像级聚合; P1-5 与 evaluate_all 对齐固定为 topk_mean(top 1% mean),"
                         "传 max 会直接报错保护口径")
    ap.add_argument("--topk_ratio", type=float, default=CFG["topk_ratio"])
    ap.add_argument("--k_nn", type=int, default=CFG["k_nn"])
    ap.add_argument("--use_projection", action="store_true",
                    help="投影模式 (默认 False = A4 无投影 memory_banks_noproj)")
    ap.add_argument("--build_cache", action="store_true",
                    help="只构建本次所需 test 特征缓存后退出")
    ap.add_argument("--verify", action="store_true",
                    help="先跑 mvtec/bottle 单类 category 校准 (Δ<0.002 才继续)")
    ap.add_argument("--verify_only", action="store_true",
                    help="校准通过后即退出, 不跑全量 (用于快速过闸门)")
    ap.add_argument("--no_plots", action="store_true", help="跳过绘图")
    args = ap.parse_args()

    CFG["modality"] = args.modality
    CFG["fused_w_dino"] = args.fused_w_dino
    CFG["use_projection"] = args.use_projection
    CFG["k_nn"] = args.k_nn
    CFG["topk_ratio"] = args.topk_ratio
    if args.agg != "topk_mean":
        raise SystemExit(
            f"[P1-5] agg 必须为 topk_mean (evaluate_all 图像级口径 = top 1% mean),"
            f" 收到 '{args.agg}' → 拒绝运行, 防 AUROC 与 summary_sweep_curves 对不上")

    print(f"P1-5  Global vs Category Memory Bank | modality={args.modality}"
          f" | fused_w_dino={CFG['fused_w_dino']} | datasets={args.datasets}")

    for ds in args.datasets:
        cats = get_categories(ds)
        need_clip = args.modality in ("clip", "fused")
        ensure_test_features(ds, cats, args.modality)
        if args.build_cache:
            continue

    if args.build_cache:
        print("[build_cache] 完成.")
        return

    if args.verify:
        ok = verify_bottle("dino" if args.modality == "fused" else args.modality)
        if not ok:
            raise SystemExit(1)
        print("[verify] 通过 → 继续全量跑\n")
        if args.verify_only:
            print("[verify_only] 校准通过, 退出 (未跑全量)")
            return

    all_rows, all_sum = [], []
    for ds in args.datasets:
        rows, summary = run_dataset(ds, args.modality)
        all_rows.extend(rows)
        all_sum.extend(summary)
        if not args.no_plots:
            plot_distributions([r for r in rows if r["dataset"] == ds], ds)
            plot_xcat([r for r in rows if r["dataset"] == ds], ds)

    rows_csv = os.path.join(CFG["out_dir"], "p1_5_global_vs_category.csv")
    sum_csv = os.path.join(CFG["out_dir"], "p1_5_dataset_summary.csv")
    _save_rows(all_rows, rows_csv)
    _save_rows(all_sum, sum_csv)

    print("\n===== P1-5 数据集级汇总 (image AUROC) =====")
    hdr = "{:<7}{:<15}{:>8}{:>9}{:>8}{:>8}{:>8}{:>8}".format(
        "dataset", "setting", "AUROC", "d_prime", "xcat_n", "xcat_a", "disp_n", "disp_a")
    print(hdr)
    print("-" * len(hdr))
    for r in all_sum:
        if r["modality"] == args.modality:
            print(f"{r['dataset']:<7}{r['setting']:<15}"
                  f"{r['image_auroc']:>8.4f}{r['d_prime']:>9.2f}"
                  f"{r['xcat_norm']:>8.3f}{r['xcat_anom']:>8.3f}"
                  f"{r['disp_norm']:>+8.3f}{r['disp_anom']:>+8.3f}")


if __name__ == "__main__":
    main()