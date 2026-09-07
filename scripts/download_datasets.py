#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
scripts/download_datasets.py

为 ECR-AD 一键下载并整理四个工业异常检测数据集：MVTec AD / VisA / BTAD / MPDD。

背景：
    datasets/ 体积过大（MVTec AD ~4.9GB、VisA ~2GB、BTAD ~1GB、MPDD ~1.7GB），
    无法直接提交进 GitHub 仓库，因此本仓库只收录“可复现下载脚本”。
    本脚本负责：下载官方压缩包（断点续传 + 自动重试）→ 解压 → 自动整理成
    configs/dataset.yaml 中 root 所期望的目录结构。

用法（在项目根目录 d:/ECR-AD 下执行）：
    python -m scripts.download_datasets --dataset mvtec       # 下载并整理单个数据集
    python -m scripts.download_datasets --dataset visa
    python -m scripts.download_datasets --dataset btad
    python -m scripts.download_datasets --dataset mpdd        # 无公开匿名直链，仅打印引导
    python -m scripts.download_datasets --all                 # 依次处理全部数据集
    python -m scripts.download_datasets --url <镜像URL> --dataset mvtec   # 指定镜像链接
    python -m scripts.download_datasets --check               # 仅校验本地目录结构（不联网）

说明：
    * 下载的压缩包缓存在 datasets/_archives/ 下（已被 .gitignore 排除）；
      若下载中断可重跑同一命令继续（断点续传）。
    * 默认整理成功后保留压缩包（便于日后重装）；可用 --no-keep-archive 删除。
    * MVTec 整包链接取自 mvtec.com 官网下载页，若链接失效可用 --url 指定镜像。
    * MPDD 官方需通过网页填写信息获取，无匿名直链，脚本只做目录校验与引导。

许可：
    MVTec AD、VisA：CC BY-NC-SA 4.0（非商用）
    BTAD、MPDD：官方声明仅供研究使用
"""

import argparse
import os
import shutil
import socket
import sys
import tarfile
import time
import urllib.error
import urllib.request
import zipfile

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASETS_ROOT = os.path.join(PROJECT_ROOT, 'datasets')
ARCHIVES_DIR = os.path.join(DATASETS_ROOT, '_archives')

USER_AGENT = ('Mozilla/5.0 (Windows NT 10.0; Win64; x64) '
              'AppleWebKit/537.36 (KHTML, like Gecko) '
              'Chrome/120.0 Safari/537.36')

_IMAGE_EXTS = {'.jpg', '.jpeg', '.png', '.bmp'}


def _rel(*parts):
    """返回相对于项目 datasets 根目录的绝对路径。"""
    return os.path.join(DATASETS_ROOT, *parts)


# 每个数据集: 官方链接 / 压缩包名 / 目标目录(相对 datasets/) / 结构探针 / 类别列表
# 其中 train_dir / test_dir / gt_dir 与 configs/dataset.yaml 的语义保持一致。
DATASETS = {
    'visa': dict(
        name='VisA',
        url='https://amazon-visual-anomaly.s3.us-west-2.amazonaws.com/VisA_20220922.tar',
        archive='VisA_20220922.tar',
        kind='tar',
        target=_rel('visa', 'VisA', 'data', 'VisA_20220922'),
        hint='split_csv',
        categories=['candle', 'capsules', 'cashew', 'chewinggum', 'fryum',
                    'macaroni1', 'macaroni2', 'pcb1', 'pcb2', 'pcb3', 'pcb4',
                    'pipe_fryum'],
        train_dir='Data/Images/Normal',
        test_dir='Data/Images/Anomaly',
        gt_dir='Data/Masks/Anomaly',
        license_note='CC BY-NC-SA 4.0（官方 tar 内含 LICENSE-DATASET）',
        note='保留官方原始目录树（对象/Data/Images|Masks + split_csv/1cls.csv），'
             'models/dataset.py 直接消费 split_csv 划分。',
    ),
    'mvtec': dict(
        name='MVTec AD',
        url='https://www.mydrive.ch/shares/150996/b52ecdcbf521176e9db9c731f2304b27/'
            'download/420938113-1629960298/mvtec_anomaly_detection.tar.xz',
        archive='mvtec_anomaly_detection.tar.xz',
        kind='tar',
        target=_rel('MVTec AD'),
        hint='bottle',
        categories=['bottle', 'cable', 'capsule', 'hazelnut', 'metal_nut',
                    'pill', 'screw', 'toothbrush', 'transistor', 'zipper',
                    'carpet', 'grid', 'leather', 'tile', 'wood'],
        train_dir='train/good',
        test_dir='test',
        gt_dir='ground_truth',
        license_note='CC BY-NC-SA 4.0',
        note='整包来自 mvtec.com 下载页（约 4.9GB）；单个类别包可自行从官网分类下载。',
    ),
    'btad': dict(
        name='BTAD',
        url='https://avires.dimi.uniud.it/papers/btad/btad.zip',
        archive='btad.zip',
        kind='zip',
        target=_rel('BTAD'),
        hint='01',
        categories=['01', '02', '03'],
        train_dir='train/ok',
        test_dir='test',
        gt_dir='ground_truth',
        license_note='仅供研究使用（VT-ADL 官方）',
        note='每类产品目录含 train/test 与类别内部的 ground_truth/ko 掩码。',
    ),
    'mpdd': dict(
        name='MPDD',
        url=None,  # 无公开匿名直链，需官网/镜像手动获取
        archive=None,
        kind='manual',
        target=_rel('MPDD'),
        hint=None,
        categories=['bracket_black', 'bracket_brown', 'bracket_white',
                    'connector', 'metal_plate', 'tubes'],
        train_dir='train/good',
        test_dir='test',
        gt_dir='ground_truth',
        license_note='仅供研究使用（VUT Brno 官方）',
        note='官方通过网页分发（需填写申请），本脚本仅提供引导与目录自检。',
    ),
}


# --------------------------------------------------------------------------- #
# 基础工具
# --------------------------------------------------------------------------- #
def _info(msg):
    print(msg)


def _warn(msg):
    print('[警告] ' + msg)


def _ok(msg):
    print('[OK] ' + msg)


def _fail(msg):
    print('[失败] ' + msg)


def _count_images(folder):
    """统计目录下图片数量（含 jpg/jpeg/png/bmp，大小写不敏感）。"""
    if not os.path.isdir(folder):
        return 0
    n = 0
    for root, _dirs, files in os.walk(folder):
        for f in files:
            if os.path.splitext(f)[1].lower() in _IMAGE_EXTS:
                n += 1
    return n


def _locate_content_root(start, hint):
    """
    在解压目录中定位“内容根目录”：
    不断沿“唯一子目录”下行，直到直接包含 hint 探针（如 split_csv / bottle / 01），
    返回该目录；找不到 hint 时返回尽头目录，由后续校验报错兜底。
    """
    node = start
    while True:
        try:
            entries = os.listdir(node)
        except OSError:
            break
        entries = [e for e in entries if e != '__MACOSX']
        if hint and hint in entries:
            return node
        dirs = [e for e in entries if os.path.isdir(os.path.join(node, e))]
        if len(dirs) == 1:
            node = os.path.join(node, dirs[0])
            continue
        return node


def _merge_into(dst_root, src_root):
    """
    把 src_root 下的全部条目并入 dst_root（逐条目移动，同名目录递归覆盖合并）。
    目标目录尚不存在时相当于整体移动，避免多余拷贝。
    """
    os.makedirs(dst_root, exist_ok=True)
    for name in sorted(os.listdir(src_root)):
        src = os.path.join(src_root, name)
        dst = os.path.join(dst_root, name)
        if name == '__MACOSX':
            shutil.rmtree(src, ignore_errors=True)
            continue
        if os.path.isdir(src):
            if os.path.isdir(dst) and not os.listdir(dst):
                os.rmdir(dst)  # 空目录先移除，便于整体 rename
            if os.path.exists(dst):
                _copytree_overlay(src, dst)
                shutil.rmtree(src, ignore_errors=True)
            else:
                shutil.move(src, dst)
        else:
            if os.path.exists(dst):
                os.remove(dst)  # 同名文件直接覆盖
            shutil.move(src, dst)


def _copytree_overlay(src, dst):
    """递归把 src 覆盖拷贝进 dst（目录合并、文件覆盖）。"""
    os.makedirs(dst, exist_ok=True)
    for name in os.listdir(src):
        s = os.path.join(src, name)
        d = os.path.join(dst, name)
        if os.path.isdir(s):
            _copytree_overlay(s, d)
        else:
            shutil.copy2(s, d)


def _rm(path):
    shutil.rmtree(path, ignore_errors=True)


# --------------------------------------------------------------------------- #
# 目录自检（与 configs/dataset.yaml + models/dataset.py 的读取口径一致）
# --------------------------------------------------------------------------- #
def check_dataset(ds_key):
    ds = DATASETS[ds_key]
    target = ds['target']
    problems = []
    notes = []
    n_img = 0

    if not os.path.isdir(target):
        problems.append('目标目录不存在: %s' % target)
        return False, problems, notes, 0

    if ds_key == 'visa':
        csv = os.path.join(target, 'split_csv', '1cls.csv')
        if not os.path.exists(csv):
            problems.append('缺少划分文件 split_csv/1cls.csv')
        else:
            notes.append('找到划分文件 split_csv/1cls.csv')
        n_img = _count_images(target)
        notes.append('图片总数(全目录): %d' % n_img)
        return (not problems), problems, notes, n_img

    # mvtec / btad / mpdd: 逐类别校验 train 与 test
    for cat in ds['categories']:
        cat_dir = os.path.join(target, cat)
        train_dir = os.path.join(cat_dir, ds['train_dir'])
        test_dir = os.path.join(cat_dir, ds['test_dir'])
        gt_dir = os.path.join(cat_dir, ds['gt_dir'])
        if not os.path.isdir(train_dir):
            problems.append('%s: 缺少训练目录 %s' % (cat, ds['train_dir']))
            continue
        if not os.path.isdir(test_dir):
            problems.append('%s: 缺少测试目录 %s' % (cat, ds['test_dir']))
            continue
        nt = _count_images(train_dir)
        ne = _count_images(test_dir)
        n_img += nt + ne
        notes.append('%s: train=%d 张 / test=%d 张' % (cat, nt, ne))
        if not os.path.isdir(gt_dir):
            notes.append('%s: 未发现 ground_truth 目录（像素级评估将退化为全 0）' % cat)

    return (not problems), problems, notes, n_img


def report_check(ds_key):
    ok, problems, notes, n_img = check_dataset(ds_key)
    ds = DATASETS[ds_key]
    _info('---- %s -> %s ----' % (ds['name'], ds['target']))
    if ok:
        _ok('目录结构就绪')
    else:
        _warn('目录结构不完整')
    for n in notes:
        _info('      ' + n)
    for p in problems:
        _info('      - ' + p)
    return ok


# --------------------------------------------------------------------------- #
# 下载（断点续传 / 重试）
# --------------------------------------------------------------------------- #
def download_file(url, dest, timeout=60, retries=5):
    tmp = dest + '.part'
    attempt = 0
    while True:
        attempt += 1
        try:
            existing = os.path.getsize(tmp) if os.path.exists(tmp) else 0
            req = urllib.request.Request(url, headers={'User-Agent': USER_AGENT})
            if existing:
                req.add_header('Range', 'bytes=%d-' % existing)
            resp = urllib.request.urlopen(req, timeout=timeout)
            code = resp.getcode() or 200

            if existing and code == 200:
                # 服务器不支持断点续传 → 从头重下
                _warn('服务器不支持 Range，重新下载 %s' % os.path.basename(dest))
                os.remove(tmp)
                existing = 0
                req = urllib.request.Request(url, headers={'User-Agent': USER_AGENT})
                resp = urllib.request.urlopen(req, timeout=timeout)
                code = resp.getcode() or 200
            if code == 416:
                # 本地 .part 已等于完整大小
                os.replace(tmp, dest)
                return dest

            total = None
            length = resp.headers.get('Content-Length')
            if length and code == 206:
                total = int(length) + existing
            elif length and code == 200:
                total = int(length)

            mode = 'ab' if existing else 'wb'
            got = existing
            last_t = time.time()
            with open(tmp, mode) as f:
                while True:
                    chunk = resp.read(1 << 20)
                    if not chunk:
                        break
                    f.write(chunk)
                    got += len(chunk)
                    now = time.time()
                    if now - last_t >= 1.0:
                        last_t = now
                        if total:
                            pct = got * 100.0 / total
                            sys.stdout.write('\r    已下载 %.1f / %.1f MB (%.1f%%)   '
                                             % (got / 1e6, total / 1e6, pct))
                        else:
                            sys.stdout.write('\r    已下载 %.1f MB   ' % (got / 1e6))
                        sys.stdout.flush()
            sys.stdout.write('\r%s 完成，共 %.1f MB\n'
                             % (os.path.basename(dest), got / 1e6))
            resp.close()
            os.replace(tmp, dest)
            return dest
        except (urllib.error.URLError, urllib.error.HTTPError,
                socket.timeout, ConnectionError, OSError) as exc:
            if attempt > retries:
                _fail('下载失败（已重试 %d 次）: %s' % (retries, exc))
                return None
            _warn('下载出错 %s（第 %d/%d 次尝试），5s 后重试...'
                  % (exc, attempt, retries))
            time.sleep(5)


# --------------------------------------------------------------------------- #
# 解压并整理
# --------------------------------------------------------------------------- #
def extract_and_arrange(ds_key, archive_path, url):
    ds = DATASETS[ds_key]
    target = ds['target']
    tmp = os.path.join(ARCHIVES_DIR, 'tmp_' + ds_key)
    _rm(tmp)
    os.makedirs(tmp, exist_ok=True)

    _info('解压 %s ...' % os.path.basename(archive_path))
    try:
        if ds['kind'] == 'tar':
            with tarfile.open(archive_path, 'r:*') as tf:
                tf.extractall(tmp)
        else:
            with zipfile.ZipFile(archive_path) as zf:
                zf.extractall(tmp)
    except (tarfile.TarError, zipfile.BadZipFile, OSError) as exc:
        _fail('解压失败 %s（压缩包可能损坏，请删除后重新下载）' % exc)
        _rm(tmp)
        return False

    # 定位内容根目录并整体并入目标目录
    root = _locate_content_root(tmp, ds['hint'])
    entries = [e for e in os.listdir(root) if e != '__MACOSX']
    _info('定位内容根目录: %s（条目: %s）'
          % (os.path.relpath(root, ARCHIVES_DIR), ', '.join(sorted(entries)[:8]) + ' ...'))
    _info('整理到目标目录: %s' % target)
    _merge_into(target, root)
    _rm(tmp)

    # 整理后自检
    _info('整理完成，执行目录自检：')
    report_check(ds_key)
    return True


# --------------------------------------------------------------------------- #
# 主流程
# --------------------------------------------------------------------------- #
def process_dataset(ds_key, url_override=None, keep_archive=True):
    ds = DATASETS[ds_key]
    _info('')
    _info('=' * 70)
    _info('处理数据集: %s' % ds['name'])
    _info('目标目录  : %s' % ds['target'])
    _info('许可      : %s' % ds['license_note'])
    _info('说明      : %s' % ds['note'])
    _info('=' * 70)

    # 1) 已安装则跳过
    ok, _problems, _notes, n_img = check_dataset(ds_key)
    if ok:
        _ok('%s 已就绪（共 %d 张图片），跳过下载。如需重下请先删除目标目录。'
            % (ds['name'], n_img))
        return 0

    # 2) 手动数据集（MPDD）：仅打印引导
    if ds['kind'] == 'manual':
        _warn('%s 没有公开匿名直链，无法自动下载。' % ds['name'])
        _info('请通过以下渠道手动获取后按上述“目标目录”结构放置：')
        _info('  1) 官方分发页: https://www.fit.vut.cz/research/publication/12183/ '
              '(填写申请后获得下载链接)')
        _info('  2) 若已知压缩包链接，可用: python -m scripts.download_datasets '
              '--dataset mpdd --url <下载链接>')
        _info('  3) 放置要求（与 configs/dataset.yaml 的 mpdd.root 对齐）：')
        _info('     datasets/MPDD/<类别>/train/good/         正常训练图')
        _info('     datasets/MPDD/<类别>/test/<缺陷类型>/    测试图（含 good 与缺陷）')
        _info('     datasets/MPDD/<类别>/ground_truth/<缺陷类型>/*_mask.png  像素掩码')
        _info('其中 <类别> ∈ {%s}' % ', '.join(ds['categories']))
        return 1

    # 3) 下载
    os.makedirs(ARCHIVES_DIR, exist_ok=True)
    archive_path = os.path.join(ARCHIVES_DIR, ds['archive'])
    url = url_override or ds['url']
    if not url:
        _warn('未配置下载链接（可用 --url 指定）。')
        return 1

    if os.path.exists(archive_path) and os.path.getsize(archive_path) > 0:
        _info('复用已有压缩包 %s（约 %.1f MB），如需强制重下请删除该文件。'
              % (archive_path, os.path.getsize(archive_path) / 1e6))
    else:
        _info('开始下载: %s' % url)
        _info('保存到   : %s' % archive_path)
        if download_file(url, archive_path) is None:
            return 1

    # 4) 解压整理
    if not extract_and_arrange(ds_key, archive_path, url):
        return 1

    # 5) 收尾
    if not keep_archive:
        _info('删除压缩包（--no-keep-archive）...')
        os.remove(archive_path)
    return 0


def main():
    parser = argparse.ArgumentParser(
        description='下载并整理 ECR-AD 的四个工业异常检测数据集',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__)
    parser.add_argument('-d', '--dataset', nargs='*', choices=list(DATASETS),
                        help='数据集子集: visa/mvtec/btad/mpdd（默认全部）')
    parser.add_argument('--all', action='store_true',
                        help='处理全部数据集（等价 -d visa mvtec btad mpdd）')
    parser.add_argument('--check', action='store_true',
                        help='仅自检本地目录结构，不联网不下载')
    parser.add_argument('--url', default=None,
                        help='覆盖下载链接（用于镜像/失效链接）')
    parser.add_argument('--no-keep-archive', action='store_true',
                        help='整理成功后删除压缩包（默认保留以便断点续传与重装）')
    args = parser.parse_args()

    if args.all or not args.dataset:
        keys = list(DATASETS)
    else:
        keys = args.dataset

    if args.check:
        _info('仅自检模式：校验以下数据集的本地目录结构')
        for k in keys:
            report_check(k)
        return

    if args.url:
        if len(keys) != 1:
            parser.error('--url 仅支持同时指定一个数据集（-d）')
        DATASETS[keys[0]]['url'] = args.url

    socket.setdefaulttimeout(60)
    ret = 0
    for k in keys:
        ret = max(ret, process_dataset(k, url_override=args.url,
                                       keep_archive=not args.no_keep_archive))
    _info('')
    _info('全部处理完成。验证加载：python -c "from models.dataset import '
          'IndustrialADDataset as D; import yaml; '
          'c=yaml.safe_load(open(\'configs/dataset.yaml\',encoding=\'utf-8\')); '
          'print(len(D(\'configs/dataset.yaml\',\'mvtec\',c[\'mvtec\'][\'categories\'][0],split=\'train\')))"')
    sys.exit(ret)


if __name__ == '__main__':
    main()
