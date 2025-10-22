#!/usr/bin/env python3
"""Utility to train SegFormer-B0 on the Echodynamic dataset."""
from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

CONFIG_TEMPLATE = """# Auto-generated configuration file.
# This file was created by train_echodynamic_segformer.py to fine-tune SegFormer-B0
# on the Echodynamic cardiac segmentation dataset.

norm_cfg = dict(type='BN', requires_grad=True)
model = dict(
    type='EncoderDecoder',
    pretrained=None,
    backbone=dict(type='mit_b0', style='pytorch'),
    decode_head=dict(
        type='SegFormerHead',
        in_channels=[32, 64, 160, 256],
        in_index=[0, 1, 2, 3],
        feature_strides=[4, 8, 16, 32],
        channels=128,
        dropout_ratio=0.1,
        num_classes=2,
        norm_cfg=norm_cfg,
        align_corners=False,
        decoder_params=dict(embed_dim=256),
        loss_decode=dict(type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)
    ),
    train_cfg=dict(),
    test_cfg=dict(mode='whole')
)

dataset_type = 'CustomDataset'
classes = {classes}
palette = {palette}
data_root = r'{data_root}'

img_norm_cfg = dict(mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
crop_size = ({crop_size}, {crop_size})

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(type='Resize', img_scale=crop_size, keep_ratio=False),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PhotoMetricDistortion'),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_semantic_seg'])
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=crop_size,
        flip=False,
        transforms=[
            dict(type='Resize', img_scale=crop_size, keep_ratio=False),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img'])
        ]
    )
]

data = dict(
    samples_per_gpu={batch_size},
    workers_per_gpu={workers},
    train=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='images/train',
        ann_dir='annotations/train',
        pipeline=train_pipeline,
        classes=classes,
        palette=palette
    ),
    val=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='images/val',
        ann_dir='annotations/val',
        pipeline=test_pipeline,
        classes=classes,
        palette=palette
    ),
    test=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='images/val',
        ann_dir='annotations/val',
        pipeline=test_pipeline,
        classes=classes,
        palette=palette
    )
)

evaluation = dict(interval=1, metric=['mIoU', 'mDice'])
checkpoint_config = dict(interval=1)
log_config = dict(interval=10, hooks=[dict(type='TextLoggerHook')])
optimizer = dict(
    type='AdamW',
    lr={learning_rate},
    betas=(0.9, 0.999),
    weight_decay=0.01,
    paramwise_cfg=dict(
        custom_keys={{
            'pos_block': dict(decay_mult=0.0),
            'norm': dict(decay_mult=0.0),
            'head': dict(lr_mult=10.0)
        }}
    )
)
optimizer_config = dict()
lr_config = dict(policy='poly', power=1.0, min_lr=0.0, by_epoch=True)
runner = dict(type='EpochBasedRunner', max_epochs={epochs})
workflow = [('train', 1)]

cudnn_benchmark = False
work_dir = r'{work_dir}'
"""

DATASET_PATTERN = re.compile(
    r"(?P<stem>.+?)_(?P<phase>ED|ES)_(?P<split>train|val|test)(?P<mask>_mask)?\\.png",
    re.IGNORECASE,
)

CLASSES = ('background', 'left_ventricle')
PALETTE = ([0, 0, 0], [255, 255, 255])


@dataclass
class Args:
    repo_url: str
    repo_dir: Path
    work_dir: Path
    epochs: int
    batch_size: int
    workers: int
    crop_size: int
    learning_rate: float
    limit_per_split: Optional[int]
    force_download: bool
    force_reprepare: bool
    skip_training: bool
    no_validate: bool
    python: str
    branch: Optional[str]


class CommandError(RuntimeError):
    """Raised when a subprocess exits with a non-zero code."""

    def __init__(self, command: Iterable[str], exit_code: int) -> None:
        super().__init__(f"Command {' '.join(map(str, command))} exited with status {exit_code}")
        self.command = list(command)
        self.exit_code = exit_code


def parse_args() -> Args:
    project_root = Path(__file__).resolve().parent
    default_repo_dir = project_root / 'external' / 'SegFormer'
    default_work_dir = project_root / 'work_dirs' / 'segformer_b0_echodynamic'

    parser = argparse.ArgumentParser(
        description='Download the Echodynamic dataset and fine-tune SegFormer-B0 using the official training code.'
    )
    parser.add_argument('--repo-url', default='https://github.com/NVlabs/SegFormer.git', help='SegFormer repository to clone.')
    parser.add_argument('--repo-dir', type=Path, default=default_repo_dir,
                        help='Location where the SegFormer repository will be stored.')
    parser.add_argument('--branch', default=None, help='Optional branch or tag to checkout in the SegFormer repository.')
    parser.add_argument('--work-dir', type=Path, default=default_work_dir,
                        help='Directory where training logs and checkpoints will be written.')
    parser.add_argument('--epochs', type=int, default=2, help='Number of training epochs to run.')
    parser.add_argument('--batch-size', type=int, default=8, help='Mini-batch size per GPU.')
    parser.add_argument('--workers', type=int, default=2, help='Number of data loading workers per GPU.')
    parser.add_argument('--crop-size', type=int, default=112, help='Spatial resolution used for training and evaluation.')
    parser.add_argument('--learning-rate', type=float, default=6e-5, help='Learning rate for AdamW.')
    parser.add_argument('--limit-per-split', type=int, default=None,
                        help='Optionally limit the number of samples used from each split for quick experiments.')
    parser.add_argument('--force-download', action='store_true',
                        help='Force re-downloading the Kaggle dataset even if it exists in the cache.')
    parser.add_argument('--force-reprepare', action='store_true',
                        help='Rebuild the local dataset directory even if it already exists.')
    parser.add_argument('--skip-training', action='store_true', help='Only prepare data and configs without running training.')
    parser.add_argument('--no-validate', action='store_true', help='Disable validation during training when running tools/train.py.')
    parser.add_argument('--python', default=sys.executable, help='Python interpreter used to invoke training.')

    raw_args = parser.parse_args()
    return Args(
        repo_url=raw_args.repo_url,
        repo_dir=raw_args.repo_dir,
        work_dir=raw_args.work_dir,
        epochs=raw_args.epochs,
        batch_size=raw_args.batch_size,
        workers=raw_args.workers,
        crop_size=raw_args.crop_size,
        learning_rate=raw_args.learning_rate,
        limit_per_split=raw_args.limit_per_split,
        force_download=raw_args.force_download,
        force_reprepare=raw_args.force_reprepare,
        skip_training=raw_args.skip_training,
        no_validate=raw_args.no_validate,
        python=raw_args.python,
        branch=raw_args.branch,
    )


def run_command(command: Iterable[str], cwd: Optional[Path] = None) -> None:
    completed = subprocess.run(command, cwd=str(cwd) if cwd else None)
    if completed.returncode != 0:
        raise CommandError(command, completed.returncode)


def ensure_dependency(module: str, install_hint: str) -> None:
    try:
        __import__(module)
    except ImportError as exc:  # pragma: no cover - import guard
        raise RuntimeError(f"Missing required dependency '{module}'. {install_hint}") from exc


def ensure_segformer_repo(args: Args) -> Path:
    repo_dir = args.repo_dir
    if repo_dir.exists():
        print(f"[segformer] Reusing existing repository at {repo_dir}")
    else:
        print(f"[segformer] Cloning {args.repo_url} into {repo_dir}")
        repo_dir.parent.mkdir(parents=True, exist_ok=True)
        run_command(['git', 'clone', args.repo_url, str(repo_dir)])
        if args.branch:
            run_command(['git', 'checkout', args.branch], cwd=repo_dir)
    return repo_dir


def download_dataset(force: bool) -> Path:
    ensure_dependency('kagglehub', "Install it with 'pip install kagglehub'.")
    import kagglehub  # type: ignore

    if force:
        print('[dataset] Forcing fresh download of the Echodynamic dataset.')
        path = kagglehub.dataset_download('caramellattedecaf/echodynamic', force=True)
    else:
        path = kagglehub.dataset_download('caramellattedecaf/echodynamic')
    dataset_root = Path(path) / 'content' / 'echonet_frames'
    if not dataset_root.exists():
        raise FileNotFoundError(f'Expected dataset directory {dataset_root} to exist after download.')
    print(f"[dataset] Dataset downloaded to {dataset_root}")
    return dataset_root


def prepare_dataset(raw_dataset: Path, output_root: Path, limit_per_split: Optional[int], force: bool) -> Path:
    ensure_dependency('numpy', "Install it with 'pip install numpy'.")
    ensure_dependency('PIL', "Install it with 'pip install pillow'.")
    import numpy as np  # type: ignore
    from PIL import Image  # type: ignore

    prepared_root = output_root
    image_root = prepared_root / 'images'
    mask_root = prepared_root / 'annotations'

    if force and prepared_root.exists():
        print(f"[dataset] Removing existing prepared dataset at {prepared_root}")
        shutil.rmtree(prepared_root)

    if prepared_root.exists():
        print(f"[dataset] Using existing prepared dataset at {prepared_root}")
        return prepared_root

    print(f"[dataset] Preparing dataset in {prepared_root}")
    for split in ('train', 'val', 'test'):
        (image_root / split).mkdir(parents=True, exist_ok=True)
        (mask_root / split).mkdir(parents=True, exist_ok=True)

    counters = {'train': 0, 'val': 0, 'test': 0}
    for file in sorted(raw_dataset.glob('*.png')):
        match = DATASET_PATTERN.fullmatch(file.name)
        if not match:
            continue
        split = match.group('split').lower()
        if limit_per_split is not None and counters[split] >= limit_per_split:
            continue

        stem = match.group('stem')
        phase = match.group('phase').lower()
        sample_stem = f"{stem}_{phase}_{split}"
        if match.group('mask'):
            destination = mask_root / split / f'{sample_stem}.png'
            if destination.exists():
                continue
            array = np.array(Image.open(file))
            binary = (array > 0).astype('uint8')
            Image.fromarray(binary).save(destination)
        else:
            destination = image_root / split / f'{sample_stem}.png'
            if destination.exists():
                continue
            shutil.copy2(file, destination)
            counters[split] += 1

    if counters['train'] == 0 or counters['val'] == 0:
        raise RuntimeError('Dataset preparation produced empty train/val splits. Check the source files.')

    print('[dataset] Summary: ' + ', '.join(f"{split}={count}" for split, count in counters.items()))
    return prepared_root


def write_config(segformer_repo: Path, prepared_dataset: Path, args: Args) -> Path:
    config_dir = segformer_repo / 'local_configs' / 'segformer' / 'echodynamic'
    config_dir.mkdir(parents=True, exist_ok=True)
    config_path = config_dir / 'segformer.b0.{}x{}.echodynamic.py'.format(args.crop_size, args.crop_size)

    config_content = CONFIG_TEMPLATE.format(
        classes=CLASSES,
        palette=PALETTE,
        data_root=prepared_dataset.as_posix(),
        crop_size=args.crop_size,
        batch_size=args.batch_size,
        workers=args.workers,
        learning_rate=args.learning_rate,
        epochs=args.epochs,
        work_dir=args.work_dir.as_posix(),
    )
    config_path.write_text(config_content)
    print(f"[config] Wrote training config to {config_path}")
    return config_path


def run_training(segformer_repo: Path, config_path: Path, args: Args) -> None:
    ensure_dependency('torch',
                      "Install PyTorch first, for example 'pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121'.")
    ensure_dependency('mmcv',
                      "Install mmcv-full that matches your PyTorch/CUDA setup, e.g. 'pip install -U openmim && mim install mmcv-full==1.3.0'.")

    args.work_dir.mkdir(parents=True, exist_ok=True)

    command = [
        args.python,
        'tools/train.py',
        str(config_path),
        '--work-dir',
        str(args.work_dir),
    ]
    if args.no_validate:
        command.append('--no-validate')
    print('[train] Launching training:\n  ' + ' '.join(command))
    run_command(command, cwd=segformer_repo)


def main() -> None:
    args = parse_args()
    segformer_repo = ensure_segformer_repo(args)

    raw_dataset = download_dataset(force=args.force_download)
    script_root = Path(__file__).resolve().parent
    prepared_dataset = prepare_dataset(
        raw_dataset=raw_dataset,
        output_root=script_root / 'data' / 'echodynamic',
        limit_per_split=args.limit_per_split,
        force=args.force_reprepare,
    )
    config_path = write_config(segformer_repo, prepared_dataset, args)

    if args.skip_training:
        print('[train] Skipping training as requested. Config and data are ready.')
        return

    run_training(segformer_repo, config_path, args)


if __name__ == '__main__':
    try:
        main()
    except CommandError as exc:
        print(f'Error: {exc}', file=sys.stderr)
        sys.exit(exc.exit_code)
    except Exception as exc:  # pragma: no cover - top-level guard for clarity
        print(f'Error: {exc}', file=sys.stderr)
        sys.exit(1)
