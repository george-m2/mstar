# Adversarial SAR ATR Across a Domain Gap

Scaling our MSTAR proof-of-concept up to [ATRNet-STAR][atrnet] on an HPC
cluster of L40 nodes. Three backbones (ResNet-50, EfficientNet-B3,
ViT-B/16) are trained across five seeds each, then evaluated against
FGSM, PGD, and CW adversarial attacks.

[atrnet]: https://github.com/waterdisappear/ATRNet-STAR

## Repo Layout

```
mstar/
├── train.py                # CLI: train one (dataset, model, seed)
├── attack.py               # CLI: evaluate one (model, seed, attack, eps)
├── src/sar_atr/            # shared library (datasets, models, attacks, engine)
├── slurm/
│   ├── env_setup.sh        # sourced by every SLURM job
│   ├── slurm_train.sh      # array job: 3 models x 5 seeds = 15 tasks
│   ├── slurm_attack.sh     # array job: 3 x 5 x 9 (attack, eps) = 135 tasks
│   ├── submit.sh           # chains train -> attack with --dependency=afterok
│   └── aggregate_results.sh# merge per-task attack CSVs into one file
├── train.ipynb / eval.ipynb# original MSTAR proof-of-concept notebooks
├── pyproject.toml, uv.lock # reproducible dependency pins (managed by uv)
└── requirements.txt        # pip fallback
```

Outputs land in two top-level folders, both configurable via env var:

- `checkpoints/{dataset}/{model}/seed_{seed}/` -- best/final weights,
  resume state, history + metrics JSON, training log.
- `results/` -- `train_summary.csv`, per-task attack CSVs under
  `results/attacks/...`, aggregated `attack_results.csv`.

## 1. One-time Cluster Setup

On a login node with internet access (GPU not required):

```bash
module load python/3.12           # or your site's equivalent
cd /path/to/mstar
uv sync                           # creates .venv/ from pyproject.toml + uv.lock
```

If your site prefers `pip`:

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip install --index-url https://download.pytorch.org/whl/cu124 \
    torch torchvision  # replace cu124 with your site's CUDA
```

### Stage the dataset onto parallel storage

ATRNet-STAR ships as 7z archives on Hugging Face / SciDB. Stage the
**SOC-40** experimental configuration onto your site's parallel
filesystem so the dataloader doesn't saturate shared NFS:

```bash
# Example using huggingface_hub; substitute your preferred download flow.
huggingface-cli download waterdisappear/ATRNet-STAR \
    --repo-type dataset \
    --include "Ground_Range/amplitude_uint8/SOC-40*" \
    --local-dir "$SAR_ATR_DATA_ROOT/atrnet_star"

# Expected layout after extraction:
#   $SAR_ATR_DATA_ROOT/atrnet_star/SOC-40/train/<class>/*.tif
#   $SAR_ATR_DATA_ROOT/atrnet_star/SOC-40/test/<class>/*.tif
```

The loader accepts either `$SAR_ATR_DATA_ROOT/atrnet_star` (archive
root) or the `SOC-40/` directory directly; override the config with
`--atrnet_config` if you stage a different split (e.g. `EOC-Azimuth`).

## 2. Submit the HPC Pipeline

Once `.venv/` exists and data is staged, run from the project root:

```bash
./slurm/submit.sh                # full pipeline: train array -> attack array
./slurm/submit.sh --train-only   # just training
./slurm/submit.sh --attacks-only # evaluate existing checkpoints
```

`submit.sh` returns the `JOBID` of each submission so you can
`squeue --me` / `scontrol show job` / `scancel` them explicitly.

### Defaults (override by editing `slurm/slurm_*.sh` or exporting env vars)

| Variable | Default | Purpose |
| --- | --- | --- |
| `SAR_ATR_PROJECT_DIR` | `$PWD` | Repo root on shared FS. |
| `SAR_ATR_DATA_ROOT` | `/scratch/$USER/datasets` | Parent of `atrnet_star/`, `mstar/`. |
| `SAR_ATR_DATA_DIR` | `$SAR_ATR_DATA_ROOT/atrnet_star` | Direct override for `--data_dir`. |
| `SAR_ATR_CHECKPOINT_DIR` | `$SAR_ATR_PROJECT_DIR/checkpoints` | Where `train.py` writes. |
| `SAR_ATR_RESULTS_DIR` | `$SAR_ATR_PROJECT_DIR/results` | Where CSVs / JSON land. |
| `SAR_ATR_DATASET` | `atrnet_star` | Switch to `mstar` for a smoke test. |
| `SAR_ATR_EPOCHS` | `50` | Training epochs per array task. |

### SLURM directives worth re-checking at your site

- `#SBATCH --gres=gpu:l40:1` -- BIH / many sites use this syntax. If
  your scheduler expects `--gres=gpu:nvidia_l40s:1` or the newer
  `--gpus=l40:1` style, edit both `slurm_train.sh` and `slurm_attack.sh`.
- `#SBATCH --partition=gpu` -- replace with your L40 partition name.
- `#SBATCH --time=` -- wall-time budgets are generous defaults; tune
  downwards once you have timing data.
- `#SBATCH --cpus-per-task=8` / `--mem=64G` -- sized for L40 nodes with
  8-worker dataloaders and full batches in VRAM.

## 3. Standalone Invocation (e.g. during debugging)

`train.py` and `attack.py` are plain CLIs; you can run them without
SLURM on any GPU host:

```bash
python train.py \
    --dataset atrnet_star --model resnet50 --seed 0 \
    --epochs 50 --batch_size 64 \
    --data_dir /path/to/atrnet_star

python attack.py \
    --dataset atrnet_star --model resnet50 --seed 0 \
    --attack_type pgd --epsilon 0.02 \
    --checkpoint_path checkpoints/atrnet_star/resnet50/seed_0/best_model.pth \
    --data_dir /path/to/atrnet_star \
    --results_csv results/attack_results.csv
```

For a pure-MSTAR smoke test (the original proof-of-concept) swap
`--dataset atrnet_star` for `--dataset mstar` and point `--data_dir` at
the flat `Padded_imgs/` folder from the Kaggle upload.

## 4. Aggregating Attack Results

Each SLURM array task writes its own CSV to avoid concurrent-write
corruption. After `slurm_attack.sh` finishes:

```bash
./slurm/aggregate_results.sh
# -> results/attack_results.csv
```

Schema (one row per (dataset, model, seed, attack, epsilon)):

```
dataset, model, seed, attack_type, epsilon,
pgd_steps, pgd_alpha, cw_c, cw_steps,
clean_acc, adv_acc, attack_success_rate, attack_sec, checkpoint
```

## 5. Memory / OOM Fallbacks

- ViT-B/16 uses `--grad_accum_steps 2` and `batch_size=32` by default
  in `slurm_train.sh`, preserving an effective batch of 64 while fitting
  comfortably in 48 GB of L40 VRAM.
- If CW on ViT-B still OOMs during attack generation, drop
  `slurm_attack.sh`'s ViT-B branch further to `BATCH_SIZE=8`.
- Set `SAR_ATR_DETERMINISTIC=1` for bit-exact reproducibility at the
  cost of ~2x slower training (cuDNN non-deterministic kernels off).

## 6. Preserving the MSTAR Proof-of-Concept

The original `train.ipynb` / `eval.ipynb` notebooks are untouched --
they remain the reference implementation for MSTAR results cited in
the proposal (99.93% clean accuracy, 0.28% under PGD eps=0.02). The
new modular pipeline reproduces those numbers via
`--dataset mstar --model resnet50 --seed 42`.
