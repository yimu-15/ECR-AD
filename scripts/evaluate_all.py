# scripts/evaluate_all.py
"""
ECR-AD 批量评估脚本（Visual Dominance 重构后）
一键跑完指定数据集的所有类别，汇总 Image-level 和 Pixel-level AUROC。

支持消融开关：
  --global_bank    A3：强制使用全局记忆库（非类别特化）
  --modality       A1/A2：visual=仅 DINOv2，text=仅 CLIP，both=双模态融合
  --no_projection  A4：跳过投影层，使用原始 DINOv2/CLIP 特征做 k-NN
  --weight_sweep   1.2 实验：融合权重扫描 w_v∈[0,1]（特征只提一次，复用算 AUROC）
"""
import os
import sys
import argparse
import torch
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader
from torchvision import transforms as T
from tqdm import tqdm
import yaml
import csv
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models.dataset import IndustrialADDataset
from models.dino_encoder import DINOv2Encoder
from models.clip_encoder import CLIPEncoder
from models.projection import ProjectionHead
from models.memory_bank import MemoryBank


class BatchEvaluator:
    """批量评估器：加载一次模型，遍历所有类别"""

    def __init__(self, config_path, projection_weights, device='cuda', use_projection=True):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.use_projection = use_projection

        # 加载模型（只加载一次）
        print("[INFO] Loading models...")
        self.dino = DINOv2Encoder().to(self.device).eval()
        self.clip = CLIPEncoder().to(self.device).eval()

        self.proj = None
        if use_projection:
            self.proj = ProjectionHead().to(self.device).eval()
            self.proj.load_state_dict(torch.load(projection_weights, map_location=self.device))
        self.global_bank_dir = os.path.dirname(projection_weights)

        # 数据变换
        self.transform_dino = T.Compose([
            T.Resize((518, 518)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        # CLIP 归一化参数
        self.dino_mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(self.device)
        self.dino_std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(self.device)
        self.clip_mean = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(1, 3, 1, 1).to(self.device)
        self.clip_std = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(1, 3, 1, 1).to(self.device)

        print("[INFO] Models loaded.")

    def load_memory_banks(self, dataset_name, category, use_global=False):
        """加载记忆库：
        - 默认：类别特化库（投影→memory_banks / 无投影 A4→memory_banks_noproj）
        - use_global=True（A3）：强制加载全局库 weights/memory_bank_*.pth
        """
        if use_global:
            if not self.use_projection:
                raise RuntimeError("--global_bank 与 --no_projection 不兼容：全局库是投影特征(256D)，"
                                   "无投影原始特征(384/512D)维度不匹配")
            visual_path = os.path.join(self.global_bank_dir, 'memory_bank_visual.pth')
            semantic_path = os.path.join(self.global_bank_dir, 'memory_bank_semantic.pth')
        else:
            bank_root = os.path.join('weights',
                                     'memory_banks' if self.use_projection else 'memory_banks_noproj')
            bank_dir = os.path.join(bank_root, dataset_name, category)
            visual_path = os.path.join(bank_dir, 'visual.pth')
            semantic_path = os.path.join(bank_dir, 'semantic.pth')

            # 类别特化库缺失时回退到全局库（仅投影模式维度兼容）
            if not os.path.exists(visual_path) and self.use_projection:
                print(f"  [WARN] Class-specific memory bank not found at {bank_dir}")
                print(f"  [WARN] Falling back to global memory bank")
                visual_path = os.path.join(self.global_bank_dir, 'memory_bank_visual.pth')
                semantic_path = os.path.join(self.global_bank_dir, 'memory_bank_semantic.pth')

        if not os.path.exists(visual_path) or not os.path.exists(semantic_path):
            raise FileNotFoundError(
                f"Memory bank not found (projection={self.use_projection}, use_global={use_global}): "
                f"{visual_path} / {semantic_path}")

        visual_bank = MemoryBank(device=self.device)
        visual_bank.features = torch.load(visual_path, map_location=self.device)
        visual_bank.is_built = True

        semantic_bank = MemoryBank(device=self.device)
        semantic_bank.features = torch.load(semantic_path, map_location=self.device)
        semantic_bank.is_built = True

        return visual_bank, semantic_bank

    @torch.no_grad()
    def evaluate_category(self, config_path, dataset_name, category, batch_size=8, k_nn=5,
                          modality='both', use_global=False, weight_sweep=False):
        """评估单个类别。
        modality: both / visual(A1) / text(A2)
        use_global: True 时强制使用全局记忆库（A3）
        weight_sweep: True 时返回该类别全部测试图像的 (37x37) 双模态分数记录，
                     由 _run_weight_sweep 在多种融合权重下复用计算 AUROC（特征只提一次）。
        """
        # 加载记忆库
        visual_bank, semantic_bank = self.load_memory_banks(dataset_name, category, use_global=use_global)

        # 加载测试集
        test_dataset = IndustrialADDataset(config_path, dataset_name, category, split='test')
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

        records = []
        for batch in tqdm(test_loader, desc=f"  {category}", leave=False):
            images_dino = batch['image'].to(self.device)
            gt_masks = batch['gt'].to(self.device)      # [B, 1, 518, 518]
            labels = batch['label'].numpy()              # (B,)

            # CLIP 输入
            images_clip = F.interpolate(images_dino, size=(224, 224), mode='bilinear', align_corners=False)
            images_clip = (images_clip * self.dino_std + self.dino_mean - self.clip_mean) / self.clip_std

            B = images_dino.shape[0]

            # 特征提取
            dino_out = self.dino(images_dino)
            dino_patches = dino_out[:, 1:, :]            # [B, 1369, 384]

            clip_out = self.clip(images_clip)
            clip_patches = clip_out[:, 1:, :]            # [B, 49, 512]

            if self.use_projection:
                # 投影（显式分支 + L2 归一化，与记忆库构建 forward_visual / forward_text 保持一致）
                N_v = dino_patches.shape[1]
                z_v = self.proj.forward_visual(dino_patches.reshape(B * N_v, -1)).reshape(B, N_v, -1)   # [B, 1369, 256]

                N_t = clip_patches.shape[1]
                z_t = self.proj.forward_text(clip_patches.reshape(B * N_t, -1)).reshape(B, N_t, -1)     # [B, 49, 256]
            else:
                # A4：无投影，直接用原始 DINOv2/CLIP patch 特征（384 / 512 维）
                z_v = dino_patches
                z_t = clip_patches

            # k-NN 异常分数（两模态都计算，融合在分数层完成）
            score_v = visual_bank.compute_anomaly_score(z_v, k=k_nn).reshape(B, 37, 37)    # [B, 37, 37]
            score_t = semantic_bank.compute_anomaly_score(z_t, k=k_nn).reshape(B, 7, 7)     # [B, 7, 7]
            score_t = F.interpolate(score_t.unsqueeze(1), size=(37, 37), mode='bilinear',
                                    align_corners=False).squeeze(1)                          # [B, 37, 37]

            sv = score_v.float().cpu()                 # [B, 37, 37]
            st = score_t.float().cpu()                 # [B, 37, 37]
            gt = gt_masks.squeeze(1).cpu().numpy()     # [B, 518, 518]

            # 像素级子采样索引（与旧版逐 batch 逻辑完全一致：每 batch seed=42，异常<=50k / 正常<=10k）。
            # 索引只依赖 GT 与 rng 调用序列、与融合权重无关 → 在这里一次性固化，
            # weight_sweep 的所有 w 复用同一组采样，逐点可比且与旧评估口径一致。
            rng = np.random.RandomState(42)
            for i in range(B):
                pl = gt[i].flatten()
                anom_idx = np.where(pl > 0.5)[0]
                norm_idx = np.where(pl <= 0.5)[0]
                if anom_idx.size > 50000:
                    anom_idx = rng.choice(anom_idx, 50000, replace=False)
                if norm_idx.size > 10000:
                    norm_idx = rng.choice(norm_idx, 10000, replace=False)
                records.append({
                    'label': int(labels[i]),
                    'sv': sv[i],
                    'st': st[i],
                    'anom_idx': anom_idx.astype(np.int64),
                    'norm_idx': norm_idx.astype(np.int64),
                })

        if weight_sweep:
            return records

        # 非 sweep 模式：模态 -> 融合权重，等价于旧版
        #   both=0.5/0.5, visual=纯 DINO, text=纯 CLIP
        w_v = 1.0 if modality == 'visual' else (0.0 if modality == 'text' else 0.5)
        return _aggregate_records(records, w_v=w_v)


def _aggregate_records(records, w_v=0.5, chunk_size=64):
    """从缓存的双模态 37x37 分数图按权重 w_v 线性融合，重算 Image / Pixel AUROC。

    - Image : fused -> bilinear 上采样 518x518 -> Top-1% -> mean（与旧版一致）
    - Pixel : 使用 evaluate_category 已固化的 GT 子采样索引（逐 batch seed=42，
              异常<=50k / 正常<=10k），保证扫描所有 w 都基于同一组采样像素。
    """
    img_labels, img_scores = [], []
    pix_labels, pix_scores = [], []
    use_cuda = torch.cuda.is_available()

    for start in range(0, len(records), chunk_size):
        recs = records[start:start + chunk_size]
        sv = torch.stack([r['sv'] for r in recs])                 # [C,37,37]
        st = torch.stack([r['st'] for r in recs])                 # [C,37,37]
        fused = (w_v * sv + (1.0 - w_v) * st).unsqueeze(1)        # [C,1,37,37]
        if use_cuda:
            fused = fused.cuda()
        amap = F.interpolate(fused, size=(518, 518), mode='bilinear',
                             align_corners=False).squeeze(1)      # [C,518,518]
        amap_flat = amap.reshape(amap.shape[0], -1)
        k = max(int(amap_flat.shape[1] * 0.01), 1)
        topk_vals, _ = torch.topk(amap_flat, k=k, dim=1)
        img_scores.extend(topk_vals.mean(dim=1).cpu().numpy().tolist())
        img_labels.extend(r['label'] for r in recs)

        amap_np = amap.cpu().numpy()
        for j, r in enumerate(recs):
            fs = amap_np[j].flatten()
            anom = fs[r['anom_idx']]
            norm = fs[r['norm_idx']]
            pix_scores.extend(np.concatenate([anom, norm]).tolist())
            pix_labels.extend([1] * anom.size + [0] * norm.size)

    try:
        image_auc = roc_auc_score(img_labels, img_scores)
    except ValueError:
        image_auc = float('nan')
    try:
        pixel_auc = roc_auc_score(pix_labels, pix_scores)
    except ValueError:
        pixel_auc = float('nan')
    return image_auc, pixel_auc


def _run_weight_sweep(records, chunk_size=64):
    """1.2 融合权重扫描：对同一批缓存的双模态 37x37 分数图做 w_v 扫描。

    w_v = DINOv2 权重，CLIP 权重 = 1 - w_v；所有点复用同一组
    GT 子采样像素索引与同一份分数图（特征只提取一次），因此逐点可比。
    返回 [(w_v, image_auc, pixel_auc), ...]，w_v = 0.0, 0.1, ..., 1.0。
    """
    weights = [round(float(x), 2) for x in np.arange(0.0, 1.01, 0.1)]
    out = []
    for w_v in weights:
        img_auc, pix_auc = _aggregate_records(records, w_v=w_v, chunk_size=chunk_size)
        out.append((w_v, img_auc, pix_auc))
    return out


def _run_standard(evaluator, args, categories):
    """非 sweep：逐类别评估（等价旧版 A3 / A1 / A2 / A4 路径）。"""
    results = []
    for cat in categories:
        print(f'\n[{cat}] Evaluating...')
        try:
            img_auc, pix_auc = evaluator.evaluate_category(
                args.config, args.dataset, cat,
                batch_size=args.batch_size, k_nn=args.k_nn,
                modality=args.modality, use_global=args.global_bank)
        except Exception as e:
            print(f'  ERROR: {e}')
            img_auc, pix_auc = -1.0, -1.0
        results.append({'category': cat, 'image_auroc': img_auc, 'pixel_auroc': pix_auc})
        if img_auc >= 0:
            print(f'  Image AUROC: {img_auc:.4f} | Pixel AUROC: {pix_auc:.4f}')

    valid = [r for r in results if r['image_auroc'] >= 0]
    mean_img = float(np.mean([r['image_auroc'] for r in valid])) if valid else 0.0
    mean_pix = float(np.mean([r['pixel_auroc'] for r in valid])) if valid else 0.0

    print('\n' + '=' * 60)
    print(f"{'Category':<20} {'Image AUROC':>12} {'Pixel AUROC':>12}")
    print('-' * 60)
    for r in results:
        if r['image_auroc'] >= 0:
            print(f"{r['category']:<20} {r['image_auroc']:>12.4f} {r['pixel_auroc']:>12.4f}")
        else:
            print(f"{r['category']:<20} {'ERROR':>12} {'ERROR':>12}")
    print('-' * 60)
    print(f"{'Mean':<20} {mean_img:>12.4f} {mean_pix:>12.4f}")
    print('=' * 60)
    return results


def _run_sweep(evaluator, args, categories):
    """weight_sweep：每类别只前向一次，缓存双模态分数图后在 w_v∈[0,1] 上复用算 AUROC。"""
    rows = []          # CSV 行：(category, w_v, image_auroc, pixel_auroc)
    per_cat_best = []  # 汇总用：(category, best_w, best_img, best_pix)

    for cat in categories:
        print(f'\n[{cat}] Extracting features once, sweeping w_v...')
        try:
            records = evaluator.evaluate_category(
                args.config, args.dataset, cat,
                batch_size=args.batch_size, k_nn=args.k_nn,
                modality='both', use_global=args.global_bank,
                weight_sweep=True)
            sweep_rows = _run_weight_sweep(records)   # [(w_v, img_auc, pix_auc), ...]

            valid = [t for t in sweep_rows if t[1] == t[1] and t[2] == t[2]]
            best = max(valid, key=lambda t: t[1]) if valid else sweep_rows[0]
            by_w = {w_v: (img_auc, pix_auc) for w_v, img_auc, pix_auc in sweep_rows}
            img00, pix00 = by_w.get(0.0, (float('nan'), float('nan')))
            img50, pix50 = by_w.get(0.5, (float('nan'), float('nan')))
            img10, pix10 = by_w.get(1.0, (float('nan'), float('nan')))

            print(f'  w=0.0(CLIP):  img {img00:.4f} pix {pix00:.4f}'
                  f' | w=0.5(等权): img {img50:.4f} pix {pix50:.4f}'
                  f' | w=1.0(DINO): img {img10:.4f} pix {pix10:.4f}')
            print(f'  >> Best w_v={best[0]:.1f}: img {best[1]:.4f} pix {best[2]:.4f}')

            rows.extend({'category': cat, 'weight_visual': w_v,
                         'image_auroc': img_auc, 'pixel_auroc': pix_auc}
                        for w_v, img_auc, pix_auc in sweep_rows)
            per_cat_best.append({'category': cat, 'weight_visual': best[0],
                                 'image_auroc': best[1], 'pixel_auroc': best[2]})
        except Exception as e:
            print(f'  ERROR: {e}')
            rows.append({'category': cat, 'weight_visual': float('nan'),
                         'image_auroc': -1.0, 'pixel_auroc': -1.0})

    # ---- 汇总表：单类别最优 w_v + w=0.5 复现 A4 ----
    means = {}
    for key in ['img_best', 'pix_best', 'img_05', 'pix_05', 'img_00', 'pix_00', 'img_10', 'pix_10']:
        means[key] = float('nan')
    by_cat = {}
    for r in rows:
        by_cat.setdefault(r['category'], {})[r['weight_visual']] = (r['image_auroc'], r['pixel_auroc'])
    agg_img, agg_pix = [], []
    agg_best_img, agg_best_pix = [], []
    agg_00, agg_10 = [], []
    for r in per_cat_best:
        wmap = by_cat[r['category']]
        agg_best_img.append(r['image_auroc'])
        agg_best_pix.append(r['pixel_auroc'])
        if 0.0 in wmap and wmap[0.0][0] == wmap[0.0][0]:
            agg_00.append(wmap[0.0][0])
        if 1.0 in wmap and wmap[1.0][0] == wmap[1.0][0]:
            agg_10.append(wmap[1.0][0])
        if 0.5 in wmap and wmap[0.5][0] == wmap[0.5][0]:
            agg_img.append(wmap[0.5][0])
            agg_pix.append(wmap[0.5][1])

    means['img_05'] = float(np.mean(agg_img)) if agg_img else float('nan')
    means['pix_05'] = float(np.mean(agg_pix)) if agg_pix else float('nan')
    means['img_best'] = float(np.mean(agg_best_img)) if agg_best_img else float('nan')
    means['pix_best'] = float(np.mean(agg_best_pix)) if agg_best_pix else float('nan')
    means['img_00'] = float(np.mean(agg_00)) if agg_00 else float('nan')
    means['img_10'] = float(np.mean(agg_10)) if agg_10 else float('nan')

    print('\n' + '=' * 60)
    print('Weight Sweep Summary  [category 级最优 w_v]')
    print('-' * 60)
    print(f"{'Category':<16} {'best w_v':>8} {'Img(best)':>10} {'Pix(best)':>10}"
          f" {'Img@0.5':>10} {'Pix@0.5':>10}")
    for r in per_cat_best:
        wmap = by_cat[r['category']]
        img50, pix50 = wmap.get(0.5, (float('nan'), float('nan')))
        print(f"{r['category']:<16} {r['weight_visual']:>8.1f} {r['image_auroc']:>10.4f}"
              f" {r['pixel_auroc']:>10.4f} {img50:>10.4f} {pix50:>10.4f}")
    print('-' * 60)
    fmt = lambda v: f'{v:.4f}' if v == v else '  n/a '
    print(f"{'Mean':<16} {'':>8} {fmt(means['img_best']):>10} {fmt(means['pix_best']):>10}"
          f" {fmt(means['img_05']):>10} {fmt(means['pix_05']):>10}")
    print('=' * 60)
    print(f"[Means] w=0.0(纯CLIP) img={fmt(means['img_00'])} | w=1.0(纯DINO) img={fmt(means['img_10'])}")
    return rows


def main():
    parser = argparse.ArgumentParser(description='ECR-AD 批量评估 / 融合权重扫描')
    parser.add_argument('--config', type=str, default='configs/dataset.yaml')
    parser.add_argument('--dataset', type=str, default='visa', choices=['mvtec', 'visa', 'btad', 'mpdd'])
    parser.add_argument('--weights', type=str, default='weights/projection_aligned.pth')
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--k_nn', type=int, default=5)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--global_bank', action='store_true',
                        help='A3 消融：强制使用全局记忆库（非类别特化）')
    parser.add_argument('--modality', type=str, default='both', choices=['both', 'visual', 'text'],
                        help='A1/A2 消融：visual=仅 DINOv2，text=仅 CLIP，both=双模态等权融合')
    parser.add_argument('--no_projection', action='store_true',
                        help='A4 消融：跳过投影层，使用原始 DINOv2/CLIP 特征做 k-NN')
    parser.add_argument('--weight_sweep', action='store_true',
                        help='1.2 实验：融合权重扫描 w_v∈[0,1]（特征只提一次，复用算 AUROC）')
    args = parser.parse_args()

    # 实验命名（按顺序覆盖，与旧版打印口径一致）
    exp_name = 'baseline'
    if args.global_bank:
        exp_name = 'A3-global_bank'
    if args.modality == 'visual':
        exp_name = 'A1-visual'
    if args.modality == 'text':
        exp_name = 'A2-text'
    if args.no_projection:
        exp_name = 'A4-no_projection'
    if args.weight_sweep:
        exp_name += ' / weight_sweep'

    with open(args.config, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)[args.dataset]
    categories = cfg['categories']

    print('=' * 60)
    print(f'ECR-AD Batch Evaluation  [{exp_name}]')
    print(f'Dataset: {args.dataset} | Categories: {len(categories)}')
    print('=' * 60 + '\n')

    evaluator = BatchEvaluator(args.config, args.weights, args.device,
                               use_projection=not args.no_projection)

    if args.weight_sweep:
        results = _run_sweep(evaluator, args, categories)
    else:
        results = _run_standard(evaluator, args, categories)

    # ---- 保存 CSV（sweep 模式含 weight_visual 列）----
    csv_tag = ''
    if args.global_bank:
        csv_tag += '_global'
    if args.modality != 'both':
        csv_tag += '_' + args.modality
    if args.no_projection:
        csv_tag += '_noproj'
    if args.weight_sweep:
        csv_tag += '_sweep'
    csv_path = f'results/{args.dataset}_results{csv_tag}.csv'

    os.makedirs('results', exist_ok=True)
    with open(csv_path, 'w', newline='') as f:
        if args.weight_sweep:
            fieldnames = ['category', 'weight_visual', 'image_auroc', 'pixel_auroc']
        else:
            fieldnames = ['category', 'image_auroc', 'pixel_auroc']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)
    print(f'\nResults saved to {csv_path}')


if __name__ == '__main__':
    main()