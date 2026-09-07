# models/memory_bank.py
"""
特征记忆库（Memory Bank）
用于异常检测：存储正常样本的投影特征，推理时通过 k-NN 距离度量异常程度。
核心思想来自 PatchCore (Roth et al., 2022)。

- build: 收集正常 patch 特征 -> L2 归一化 -> Coreset 随机下采样（内存/计算友好，
  避免在全数据集上直接跑大规模 MiniBatchKMeans 造成 OOM）
- compute_anomaly_score: 查询 patch 与库中最近 k 个特征的平均余弦距离作为异常分数
"""
import torch
import torch.nn.functional as F
import numpy as np


class MemoryBank:
    """
    正常特征记忆库
    存储正常样本的 patch-level 投影特征，支持 k-NN 距离计算和 Coreset 下采样。
    """

    def __init__(self, device='cuda'):
        self.device = device
        self.features = None      # [N, D] 存储的特征（L2 归一化）
        self.is_built = False

    def build(self, feature_list, coreset_ratio=0.1, random_seed=42,
              max_bank_size=100000):
        """
        从正常样本特征列表构建记忆库
        Args:
            feature_list: list of [N_i, D] tensors，每个元素是一个 batch 的 patch 特征
            coreset_ratio: Coreset 保留比例（0.1 = 保留 10%）
            random_seed: 随机种子（固定种子保证可复现）
            max_bank_size: 记忆库规模上限（防止跨数据集通用库过大导致 kNN 太慢 / 显存不足）
        """
        # 分段均匀随机抽样（确定性 seed），避免全量 torch.cat 导致 OOM：
        # 先算总量 -> 抽取全局下标 -> 按区间从各 chunk 收集对应行
        n_total = sum(t.shape[0] for t in feature_list)
        print(f"[MemoryBank] Total patches collected: {n_total}, dim={feature_list[0].shape[1]}")

        n_keep = int(n_total * coreset_ratio)
        n_keep = min(max(n_keep, 1000), max_bank_size)   # 至少 1000，至多 max_bank_size
        n_keep = min(n_keep, n_total)

        if n_total > n_keep:
            print(f"[MemoryBank] Coreset random selection: {n_total} -> {n_keep}")
            rng = np.random.default_rng(random_seed)
            global_idx = np.sort(rng.choice(n_total, size=n_keep, replace=False))

            selected = []
            start = 0
            for t in feature_list:
                n = t.shape[0]
                local = global_idx[(global_idx >= start) & (global_idx < start + n)] - start
                if local.size > 0:
                    selected.append(t[local])
                start += n
            all_features = torch.cat(selected, dim=0) if selected else torch.empty(0, feature_list[0].shape[1])
        else:
            all_features = torch.cat(feature_list, dim=0)

        # L2 归一化（使距离计算等价于余弦距离）
        all_features = F.normalize(all_features, dim=1)

        self.features = all_features.to(self.device)
        self.is_built = True
        print(f"[MemoryBank] Memory bank built: {self.features.shape}")

    @torch.no_grad()
    def compute_anomaly_score(self, query_features, k=5):
        """
        计算查询特征到记忆库的 k-NN 距离作为异常分数
        Args:
            query_features: [B, N, D] 查询的 patch 特征（已 L2 归一化）
            k: 最近邻数量
        Returns:
            scores: [B, N] 每个 patch 的异常分数（k-NN 平均距离，越大越异常）
        """
        if not self.is_built:
            raise RuntimeError("Memory bank has not been built yet!")

        B, N, D = query_features.shape

        # 归一化查询特征
        query_features = F.normalize(query_features, dim=-1)

        # 展平: [B*N, D]
        query_flat = query_features.reshape(B * N, D)

        # 分块计算距离（避免 OOM）
        chunk_size = 1024
        all_scores = []

        bank = self.features  # [M, D]

        for i in range(0, query_flat.shape[0], chunk_size):
            chunk = query_flat[i:i+chunk_size]  # [C, D]

            # 计算余弦距离: 1 - cosine_similarity
            # 由于特征已归一化，点积 = 余弦相似度
            similarities = torch.mm(chunk, bank.T)  # [C, M]
            distances = 1.0 - similarities  # [C, M] 余弦距离，越大越异常

            # 取 top-k 最小距离（最近的 k 个邻居）
            topk_distances, _ = torch.topk(distances, k=k, dim=1, largest=False)

            # 平均距离作为异常分数
            scores = topk_distances.mean(dim=1)  # [C]
            all_scores.append(scores)

        all_scores = torch.cat(all_scores, dim=0)  # [B*N]
        all_scores = all_scores.reshape(B, N)  # [B, N]

        return all_scores
