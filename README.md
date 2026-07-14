# NAPTIME

**Neural Astrophysical Photometric Transient Identification and Modeling Engine**

NAPTIME is a neural-process framework for photometric transient classification
under sparse and partial observational context. It targets the alert-stream
regime anticipated for the Vera C. Rubin Observatory Legacy Survey of Space and
Time (LSST), where light curves are irregularly sampled, spectroscopic
confirmation is limited, and classifications must be useful before the source
evolution is fully observed.

The model uses a ConvGNP-style (Convolutional Gaussian Neural Process)
architecture that jointly reconstructs light curves and produces a class
posterior. Context photometry is projected onto a regular temporal grid via
Gaussian set convolution, processed by a convolutional backbone, and decoded into
a predictive flux distribution. A global latent path captures object-level
variability. An optional metadata branch fuses host-galaxy context and
photometric redshift near the classifier head.

The primary motivation is recovering tidal disruption events (TDEs) within a
broader alert stream. TDEs are rare, often nuclear, and overlap photometrically
with AGN, supernovae, and other nuclear variability. NAPTIME treats
broad-population alert classification and TDE retrieval as coupled tasks. The
same multiclass posterior that defines the family-level taxonomy also provides a
TDE ranking score.

> [!NOTE]  
> The paper describing this work is currently under review. A preprint link will
> be added here when available.

## Results

The values below come from fixed 85/15 train-validation splits with seed 42.
ELAsTiCC2 is the primary benchmark used for model selection and comparison.
MALLORN is smaller and is reported as a secondary photometry-only check.

**ELAsTiCC2** (15-family Rubin-like benchmark, primary):

| Model | Macro F1 | Macro AUROC | TDE Avg Precision |
|---|---|---|---|
| With metadata | 0.903 | 0.991 | 0.985 |
| Photometry only | 0.874 | 0.986 | 0.979 |

At the earliest 10% of detected observations, macro F1 is approximately 0.42
with metadata and 0.34 with photometry only. The metadata gain is largest when
the light curve is still short.

**MALLORN** (photometry-only TDE-focused benchmark, secondary):

| Macro F1 | Macro AUROC |
|---|---|
| 0.693 | 0.958 |

The MALLORN validation split has few TDE examples, so its class-specific metrics
have higher sampling variance than the ELAsTiCC2 results.

## Requirements

- Python 3.13
- `uv`
- PyTorch-compatible CPU or GPU environment

GPU use is optional. Pass `--device cuda` for CUDA, `--device mps` for Apple
Silicon, or `--device cpu` for CPU.

## Installation

Clone the repository and install the locked environment:

```bash
git clone https://github.com/<owner>/naptime.git
cd naptime
uv sync
```

Run the test suite:

```bash
uv sync --group test
uv run pytest
```

Check the command-line interface:

```bash
naptime --help
```

If you are working directly from a source checkout without activating the
environment, prefix commands with `uv run`.

## Inference Service

NAPTIME includes a stateless FastAPI service for direct model inference. It
accepts photometric detections and optional host metadata, then returns class
probabilities and a TDE ranking score. This is being built toward broker 
integration.

```bash
uv sync --extra serve
NAPTIME_CHECKPOINT=/path/to/checkpoint.pt \
  uvicorn naptime.serve:app --host 127.0.0.1 --port 8000
```

Example request:

```json
POST /classify
{
  "object_id": "obj-001",
  "detections": [
    {"mjd": 60000.0, "flux": 120.3, "flux_err": 4.1, "band": "r"},
    {"mjd": 60003.5, "flux": 185.7, "flux_err": 5.8, "band": "g"},
    {"mjd": 60007.1, "flux": 210.2, "flux_err": 6.0, "band": "r"},
    {"mjd": 60010.0, "flux": 198.4, "flux_err": 5.5, "band": "i"}
  ],
  "redshift": 0.12,
  "metadata": {
    "mwebv": 0.03,
    "host_snsep": 0.4,
    "host_logmass": 10.2
  }
}
```

Environment variables:

| Variable | Default | Description |
|---|---|---|
| `NAPTIME_CHECKPOINT` | `/models/latest.pt` | Path to checkpoint file |
| `NAPTIME_DEVICE` | `cpu` | `cpu`, `cuda`, or `mps` |

## Checkpoints

Training commands write checkpoints under the selected output directory. The
primary checkpoint is:

```text
best_primary_checkpoint.pt
```

Checkpoints are not stored in git. For reproducible releases, attach them as
GitHub Release assets or store them in another artifact repository.

## ELAsTiCC2 Training

Use lazy loading for full ELAsTiCC2 runs to avoid loading all photometry into
memory at once.

```bash
naptime train-elasticc-focus-baseline \
  --data-dir /path/to/ELASTICC2_TRAIN_02 \
  --out-dir output/elasticc2_families_full_photoz \
  --device cuda \
  --num-workers 0 \
  --epochs 40 \
  --patience 10 \
  --batch-size 16 \
  --lr 3e-4 \
  --weight-decay 1e-5 \
  --lambda-recon 1.0 \
  --lambda-cls 1.0 \
  --beta-kl 5e-4 \
  --kl-warmup-epochs 20 \
  --use-latent \
  --latent-dim 32 \
  --latent-hidden-dim 128 \
  --grid-feat-dim 256 \
  --point-feat-dim 128 \
  --conv-layers 8 \
  --conv-dropout 0.1 \
  --use-metadata \
  --use-redshift \
  --redshift-source photoz \
  --elasticc-taxonomy families \
  --num-classes 15 \
  --lazy-elasticc \
  --max-cached-shards 1 \
  --checkpoint-metric macro_f1
```

For the photometry-only comparison model, replace `--use-metadata` with
`--no-use-metadata` and write to a separate output directory.

## ELAsTiCC2 Evaluation

Evaluate a trained checkpoint:

```bash
naptime evaluate-elasticc-focus-baseline \
  --data-dir /path/to/ELASTICC2_TRAIN_02 \
  --checkpoint output/elasticc2_families_full_photoz/best_primary_checkpoint.pt \
  --out-dir output/elasticc2_families_full_photoz/eval \
  --device cuda \
  --batch-size 32 \
  --num-workers 0 \
  --seed 42 \
  --val-frac 0.15 \
  --elasticc-taxonomy families \
  --lazy-elasticc \
  --max-cached-shards 1 \
  --full-context-eval
```

Run a prefix-context sweep (mimics progressively accumulating alert history):

```bash
naptime evaluate-elasticc-focus-prefix-sweep \
  --data-dir /path/to/ELASTICC2_TRAIN_02 \
  --checkpoint output/elasticc2_families_full_photoz/best_primary_checkpoint.pt \
  --out-dir output/elasticc2_families_full_photoz/prefix_sweep \
  --device cuda \
  --batch-size 32 \
  --num-workers 0 \
  --seed 42 \
  --val-frac 0.15 \
  --elasticc-taxonomy families \
  --context-fractions 0.1 0.2 0.4 0.6 0.8 0.9 0.95 1.0
```

Run a fixed-context sweep:

```bash
naptime evaluate-elasticc-focus-context-sweep \
  --data-dir /path/to/ELASTICC2_TRAIN_02 \
  --checkpoint output/elasticc2_families_full_photoz/best_primary_checkpoint.pt \
  --out-dir output/elasticc2_families_full_photoz/context_sweep \
  --device cuda \
  --batch-size 32 \
  --num-workers 0 \
  --seed 42 \
  --val-frac 0.15 \
  --elasticc-taxonomy families
```

## MALLORN Training

```bash
naptime train-mallorn-baseline \
  --data-dir /path/to/mallorn \
  --out-dir output/mallorn_multiclass \
  --num-classes 6 \
  --epochs 120 \
  --patience 25 \
  --batch-size 32 \
  --device cuda
```

## MALLORN Evaluation

```bash
naptime evaluate-mallorn-baseline \
  --data-dir /path/to/mallorn \
  --checkpoint output/mallorn_multiclass/best_primary_checkpoint.pt \
  --out-dir output/mallorn_multiclass/eval \
  --num-classes 6 \
  --batch-size 32 \
  --full-context-eval \
  --device cuda
```

Run cross-validation:

```bash
naptime crossval-mallorn-baseline \
  --data-dir /path/to/mallorn \
  --out-dir output/mallorn_crossval \
  --num-classes 6 \
  --folds 5 \
  --epochs 120 \
  --batch-size 32 \
  --device cuda
```

## Lazy ELAsTiCC2 Loading

Use `--lazy-elasticc --max-cached-shards 1` for memory-constrained training or
evaluation. The lazy path stores only an object index and loads photometry shards
on demand. The eager path is faster when enough memory is available.

## Command Reference

```bash
naptime --help
naptime train-elasticc-focus-baseline --help
naptime evaluate-elasticc-focus-baseline --help
naptime train-mallorn-baseline --help
naptime evaluate-mallorn-baseline --help
```
