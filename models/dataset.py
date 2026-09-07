# models/dataset.py
import os
import glob
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms as T
import yaml

class IndustrialADDataset(Dataset):
    """
    统一工业异常检测数据集加载器
    支持: MVTec AD, VisA, BTAD, MPDD
    """
    def __init__(self, config_path, dataset_name, category, split='test', transform=None):
        """
        Args:
            config_path: yaml配置文件路径
            dataset_name: 'mvtec', 'visa', 'btad', 'mpdd'
            category: 类别名，如 'bottle', 'candle'
            split: 'train' (仅正常) 或 'test' (正常+异常)
            transform: 图像变换
        """
        with open(config_path, 'r', encoding='utf-8') as f:
            self.cfg = yaml.safe_load(f)[dataset_name]
        
        self.dataset_name = dataset_name
        self.category = category
        self.split = split
        self.transform = transform or T.Compose([
            T.Resize((518, 518)),  # DINOv2 标准尺寸
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        self.image_paths = []
        self.gt_paths = []
        self.labels = []  # 0: normal, 1: anomaly
        
        self._load_paths()

    @staticmethod
    def _list_images(directory):
        """列出目录下所有图片（避免 Windows 大小写不敏感导致重复匹配）。"""
        images = []
        for ext in ('*.jpg', '*.jpeg', '*.png', '*.bmp'):
            # 小写扩展名模式在 Windows 上大小写不敏感，可匹配 .JPG/.JPEG/.PNG
            images.extend(sorted(glob.glob(os.path.join(directory, ext))))
        return images

    def _load_paths(self):
        root = self.cfg['root']
        cat_path = os.path.join(root, self.category)
        
        if self.dataset_name == 'mvtec':
            if self.split == 'train':
                # 仅加载 train/good
                pattern = os.path.join(cat_path, self.cfg['train_dir'], '*.png')
                self.image_paths = sorted(glob.glob(pattern))
                self.labels = [0] * len(self.image_paths)
                self.gt_paths = [None] * len(self.image_paths)
            else:
                # 加载 test 下所有子文件夹
                test_root = os.path.join(cat_path, self.cfg['test_dir'])
                for defect_type in sorted(os.listdir(test_root)):
                    defect_path = os.path.join(test_root, defect_type)
                    if not os.path.isdir(defect_path): continue
                    imgs = sorted(glob.glob(os.path.join(defect_path, '*.png')))
                    self.image_paths.extend(imgs)
                    is_anomaly = 0 if defect_type == self.cfg['normal_label'] else 1
                    self.labels.extend([is_anomaly] * len(imgs))
                    
                    # 处理 GT
                    gt_root = os.path.join(cat_path, self.cfg['gt_dir'])
                    for img in imgs:
                        if is_anomaly == 0:
                            self.gt_paths.append(None)
                        else:
                            # MVTec GT 命名: 000.png -> 000_mask.png 或同名
                            gt_name = os.path.basename(img).replace('.png', '_mask.png')
                            gt_path = os.path.join(gt_root, defect_type, gt_name)
                            if not os.path.exists(gt_path):
                                gt_path = os.path.join(gt_root, defect_type, os.path.basename(img))
                            self.gt_paths.append(gt_path)

        elif self.dataset_name == 'visa':
            # VisA 结构特殊，正常和异常在不同文件夹，且官方划分在 split_csv/1cls.csv 中
            root = self.cfg['root']
            cat_path = os.path.join(root, self.category)
            csv_path = os.path.join(root, 'split_csv', '1cls.csv')
            normal_dir = os.path.join(cat_path, self.cfg['normal_dir'])
            anomaly_dir = os.path.join(cat_path, self.cfg['anomaly_dir'])

            if os.path.exists(csv_path):
                import csv as _csv
                n_imgs, a_imgs, a_masks = [], [], []
                with open(csv_path, 'r') as f:
                    reader = _csv.DictReader(f)
                    for row in reader:
                        if row['object'] != self.category or row['split'] != self.split:
                            continue
                        img = os.path.join(root, row['image'].replace('/', os.sep))
                        if row['label'] == 'normal':
                            n_imgs.append(img)
                        else:
                            a_imgs.append(img)
                            a_masks.append(os.path.join(root, row['mask'].replace('/', os.sep)))
                # 官方 csv 中 normal 列在 test 下包含正常样本；异常按 mask 列对应
                self.image_paths = n_imgs + a_imgs
                self.labels = [0] * len(n_imgs) + [1] * len(a_imgs)
                self.gt_paths = [None] * len(n_imgs) + a_masks
            else:
                # 回退: 无官方 csv 时读取整个 Normal/Anomaly 目录
                if self.split == 'train':
                    imgs = self._list_images(normal_dir)
                    self.image_paths = imgs
                    self.labels = [0] * len(imgs)
                    self.gt_paths = [None] * len(imgs)
                else:
                    # 注意: Windows 下 glob 大小写不敏感，只需统一小写扩展名模式，
                    # 若再补大写模式会重复匹配同一文件导致样本翻倍
                    n_imgs = self._list_images(normal_dir)
                    a_imgs = self._list_images(anomaly_dir)

                    self.image_paths = n_imgs + a_imgs
                    self.labels = [0] * len(n_imgs) + [1] * len(a_imgs)

                    # GT 只有异常有
                    mask_dir = os.path.join(cat_path, self.cfg['mask_dir'])
                    self.gt_paths = [None] * len(n_imgs)
                    for img in a_imgs:
                        # VisA mask 命名通常是 .png
                        mask_name = os.path.basename(img).rsplit('.', 1)[0] + '.png'
                        self.gt_paths.append(os.path.join(mask_dir, mask_name))

        elif self.dataset_name in ['btad', 'mpdd']:
            # BTAD 和 MPDD 结构与 MVTec 类似（BTAD 部分类别图片为 .bmp）
            if self.split == 'train':
                imgs = self._list_images(os.path.join(cat_path, self.cfg['train_dir']))
                self.image_paths = sorted(imgs)
                self.labels = [0] * len(self.image_paths)
                self.gt_paths = [None] * len(self.image_paths)
            else:
                test_root = os.path.join(cat_path, self.cfg['test_dir'])
                gt_root = os.path.join(cat_path, self.cfg['gt_dir'])
                for defect_type in sorted(os.listdir(test_root)):
                    defect_path = os.path.join(test_root, defect_type)
                    if not os.path.isdir(defect_path): continue
                    imgs = sorted(self._list_images(defect_path))
                    self.image_paths.extend(imgs)
                    is_anomaly = 0 if defect_type == self.cfg['normal_label'] else 1
                    self.labels.extend([is_anomaly] * len(imgs))

                    for img in imgs:
                        if is_anomaly == 0:
                            self.gt_paths.append(None)
                        else:
                            # GT 命名约定不一，按优先级回退：
                            #   1) 与图像同名同扩展名（BTAD 02/03 等）
                            #   2) <stem>_mask.png（MPDD）
                            #   3) <stem>.png（BTAD 01：图像 .bmp、GT .png）
                            stem = os.path.splitext(os.path.basename(img))[0]
                            gt_dir = os.path.join(gt_root, defect_type)
                            candidates = [
                                os.path.join(gt_dir, os.path.basename(img)),
                                os.path.join(gt_dir, stem + '_mask.png'),
                                os.path.join(gt_dir, stem + '.png'),
                            ]
                            gt_path = next((c for c in candidates if os.path.exists(c)), candidates[-1])
                            self.gt_paths.append(gt_path)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        label = self.labels[idx]
        
        # 图像变换
        image = self.transform(image)
        
        # GT 处理
        gt = torch.zeros([1, image.shape[1], image.shape[2]])  # 默认全0
        gt_path = self.gt_paths[idx]
        if gt_path is not None and os.path.exists(gt_path):
            mask = Image.open(gt_path).convert('L')
            # 注意：GT 需要和图像做同样的 Resize，但用 NEAREST 保持 0/1
            mask = T.functional.resize(mask, [image.shape[1], image.shape[2]], interpolation=T.InterpolationMode.NEAREST)
            mask = T.functional.to_tensor(mask)
            # VisA 的 GT 是小整数标注图 (0,1,2,...,255 均代表缺陷区域而非 255 才为异常)，
            # 因此用 >0 而非 >0.5 判定；MVTec/BTAD/MPDD 为 0/255 二值图，>0 同样成立
            gt = (mask > 0).float()
        
        return {
            'image': image,
            'gt': gt,
            'label': label,
            'img_path': img_path
        }