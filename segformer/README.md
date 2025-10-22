# SegFormer-B0 Training on the Echodynamic Dataset

This folder contains utilities that automate preparing the
[Echodynamic](https://www.kaggle.com/datasets/caramellattedecaf/echodynamic)
cardiac ultrasound dataset and fine-tuning the official
[NVlabs/SegFormer](https://github.com/NVlabs/SegFormer) implementation of
SegFormer-B0 for binary left ventricle segmentation.

## Prerequisites

1. **Python environment** – create and activate a virtual environment that has
   access to your GPU drivers if you plan to train with CUDA.
2. **PyTorch** – install a CUDA-enabled build that matches your GPU. Example for
   CUDA 12.1:

   ```bash
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
   ```

3. **MMCV** – SegFormer relies on `mmcv-full`. Install the wheel that matches
   your PyTorch/CUDA combination (the `openmim` helper handles the selection):

   ```bash
   pip install -U openmim
   mim install mmcv-full==1.3.0
   ```

4. **Supporting utilities** – install the lightweight Python packages used by
   the helper script:

   ```bash
   pip install kagglehub numpy pillow
   ```

   The Kaggle dataset is downloaded with the public `kagglehub` client so no
   Kaggle API credentials are required.

## Quick start

Run the helper script to download the dataset, clone the official SegFormer
repository, generate a config, and launch two epochs of training:

```bash
python segformer/train_echodynamic_segformer.py
```

The first run performs the following steps:

1. Downloads and caches the Kaggle dataset (~360 MB).
2. Converts it into the directory layout expected by MMSegmentation under
   `segformer/data/echodynamic/` with binary masks (0 background, 1 ventricle).
3. Clones `https://github.com/NVlabs/SegFormer` into
   `segformer/external/SegFormer/`.
4. Writes an auto-generated config that fine-tunes SegFormer-B0 for two epochs
   at 112×112 resolution and stores it in the cloned repository.
5. Invokes `tools/train.py` from the official codebase so the training workflow
   remains unmodified.

All outputs (logs, checkpoints, configs) are saved into
`segformer/work_dirs/segformer_b0_echodynamic/` by default.

## Useful command-line options

The script exposes several optional flags:

- `--limit-per-split N` – use only the first `N` samples from each split for a
  quick smoke test.
- `--force-download` – re-download the dataset even if it is already cached.
- `--force-reprepare` – rebuild the prepared dataset from scratch.
- `--skip-training` – stop after preparing data and configs.
- `--repo-dir` / `--work-dir` – customise where the SegFormer repo is cloned and
  where outputs are stored.

Run `python segformer/train_echodynamic_segformer.py --help` for the full list
of options.

## Notes

- The auto-generated config disables SyncBN to remain compatible with a single
  GPU setup.
- If you want to resume or extend training you can edit the generated config in
  the cloned repository directly or rerun the helper script with different
  arguments (for example `--epochs 50`).
