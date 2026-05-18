# Hybrid Three-Branch Deepfake Detector

This repository contains the in-progress implementation of a deepfake face detection project built around a three-branch discriminator. The current codebase includes the full Week 1 baseline and the Week 2 Dev 1 Branch B / Phase 2 stack: a working Branch A spatial model, a working Branch B spatiotemporal branch, a frozen-Branch-A Phase 2 discriminator, the CelebA pair dataset pipeline, an optical-flow precompute utility, and tests for the current model and training paths.

The long-term design is documented in [docs/master-plan.md](docs/master-plan.md), but the repository is not at full three-branch parity yet. The `README` below describes what is implemented now.

## Current Status

- Implemented: Branch A baseline training on paired CelebA images
- Implemented: Branch B spatiotemporal features and `DiscriminatorPhase2`
- Implemented: Phase 2 A+B trainer CLI and checkpoint/report writing
- Implemented: CelebA dataloader with real/fake pair construction
- Implemented: offline Farneback optical-flow precompute utility
- Implemented: metric computation, checkpointing, run summaries, and optional TensorBoard logging
- Verified: Branch B regression tests, Branch A freeze tests, a 1-epoch Phase 2 dry-run, and a full gate-clearing Phase 2 training run
- Not implemented yet: Branch C, Phase 3+, and the final fused three-branch discriminator
- Completed: `checkpoints/phase2_a_b.pt` with Phase 2 best validation metrics of balanced accuracy `1.0000` and F1 `1.0000`

## Repository Layout

```text
deepfake_detector/
├── config/
│   └── config.yaml
├── data/
│   ├── augmentations.py
│   ├── celeba_loader.py
│   └── precompute_flow.py
├── docs/
│   ├── build-plan.md
│   └── master-plan.md
├── evaluation/
│   └── metrics.py
├── models/
│   ├── branch_a.py
│   ├── branch_b.py
│   └── discriminator.py
├── scripts/
│   └── download_celeba.sh
├── tests/
│   ├── test_bootstrap_and_imports.py
│   ├── test_branch_a_baseline.py
│   ├── test_data_pipeline.py
│   └── test_model.py
├── training/
│   ├── branch_a_trainer.py
│   ├── phase2_trainer.py
│   ├── tracker.py
│   ├── train_branch_a.py
│   └── phase2_train.py
├── pyrightconfig.json
├── requirements.txt
└── README.md
```

## Implemented Models

The current codebase includes two model stages.

Branch A in [models/branch_a.py](models/branch_a.py):

- Input: two `64 x 64` RGB face frames
- Encoder: five convolution blocks with spectral normalization and LeakyReLU
- Classifier: concatenated twin-frame features passed through an MLP
- Output: one real/fake logit for the frame pair

Branch B and Phase 2 in [models/branch_b.py](models/branch_b.py) and [models/discriminator.py](models/discriminator.py):

- `EmbedCNN`: tied lightweight 4-block CNN per frame, projected to a 64-D embedding
- `BranchB_Spatiotemporal`: computes committed temporal proxies from `(frame_a, frame_b)`
- Output layout: 8-D summary `[velocity(mean,std,max), curvature(mean,std,max), acceleration(mean,max)]`
- `DiscriminatorPhase2`: frozen Branch A encoder on `frame_a` only, concatenated with Branch B's 8-D output, then fused through a `2056 -> 512 -> 128 -> 1` head

The Phase 2 load path reuses only `encoder.*` weights from `checkpoints/phase1_branch_a_best.pt`; the Week 1 classifier head is discarded.

## Dataset Pipeline

The dataset code in [data/celeba_loader.py](data/celeba_loader.py) builds pair-labeled samples from CelebA.

Real pairs:

- If `identity_CelebA.txt` is present, the loader pairs images from the same identity.
- If the identity file is missing, it falls back to adjacent-image pairing.

Fake pairs:

- The current baseline does not use GAN-generated or diffusion-generated fakes.
- Instead, it duplicates one image and injects Gaussian noise into the second frame.

Transforms from [data/augmentations.py](data/augmentations.py):

- Resize to `64 x 64`
- Random horizontal flip during training
- Color jitter during training
- Normalize tensors to `[-1, 1]`

This means current metrics are only useful as a smoke-tested baseline and are not representative of real deepfake performance.

## Configuration

The default config lives at [config/config.yaml](config/config.yaml).

Key defaults:

- Dataset size target: `202,599` images
- Native resolution target: `178 x 218`
- Batch size: `64`
- Epochs: `100`
- Learning rate: `2e-4`
- Scheduler: `CosineAnnealingLR`
- Checkpoint metric: `balanced_accuracy`
- Default checkpoint name: `phase1_branch_a_best.pt`

The file now also contains a dedicated `phase2:` block:

- Epochs: `20`
- Learning rate: `2e-4`
- Scheduler: `CosineAnnealingLR`
- Pretrained Branch A checkpoint: `phase1_branch_a_best.pt`
- Default Phase 2 checkpoint name: `phase2_a_b.pt`
- Targets: balanced accuracy `>= 0.88`, F1 `>= 0.88`

By default, outputs are written to:

- `checkpoints/phase1_branch_a_best.pt`
- `runs/<run-name>/benchmark_summary.json`
- `runs/<run-name>/benchmark_summary.md`
- `runs/<run-name>/metrics_history.json`

## Setup

Create an environment and install dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

Dependencies from [requirements.txt](requirements.txt):

- `torch`
- `torchvision`
- `opencv-python`
- `numpy`
- `scikit-learn`
- `Pillow`
- `tqdm`
- `tensorboard`
- `pyyaml`

## Download CelebA

Use the bootstrap script:

```bash
bash scripts/download_celeba.sh
```

Prerequisites:

- Kaggle CLI installed
- Kaggle credentials at `~/.kaggle/kaggle.json`

The tests include explicit failure checks for missing Kaggle CLI and missing credentials.

## Train The Branch A Baseline

Run the baseline trainer:

```bash
python3 -m training.train_branch_a --config config/config.yaml --run-name branch_a_baseline
```

Useful flags:

- `--train-limit`: cap train samples for smoke runs
- `--val-limit`: cap validation samples for smoke runs
- `--epochs-override`: override configured epochs
- `--device cpu|cuda|mps`: force a device
- `--tracker-backend tensorboard`: emit TensorBoard logs under `runs/<run-name>/tensorboard`

Example short smoke run:

```bash
python3 -m training.train_branch_a \
  --config config/config.yaml \
  --run-name smoke \
  --train-limit 512 \
  --val-limit 128 \
  --epochs-override 1 \
  --device cpu
```

## Train Phase 2 A+B

Run the Phase 2 trainer:

```bash
python3 -m training.phase2_train --config config/config.yaml --run-name phase2_a_b
```

Useful flags:

- `--train-limit`: cap train samples for smoke runs
- `--val-limit`: cap validation samples for smoke runs
- `--epochs-override`: override configured epochs
- `--max-batches`: cap batches per split for very short dry-runs
- `--checkpoint-name-override`: avoid overwriting the final checkpoint during smoke runs
- `--device cpu|cuda|mps`: force a device

Example smoke run:

```bash
python3 -m training.phase2_train \
  --config config/config.yaml \
  --run-name phase2_a_b_smoke \
  --train-limit 256 \
  --val-limit 64 \
  --epochs-override 1 \
  --max-batches 2 \
  --checkpoint-name-override phase2_a_b_smoke.pt \
  --device cpu
```

## Precompute Optical Flow

The flow utility in [data/precompute_flow.py](data/precompute_flow.py) currently supports Farneback flow only.

```bash
python3 -m data.precompute_flow \
  --img-dir data/celeba/img_align_celeba \
  --out-dir data/flow_cache \
  --method farneback \
  --image-size 64
```

This produces one `*_flow.pt` tensor per image. The current Branch A baseline does not consume these tensors yet; they are groundwork for later branches.

## Evaluation And Targets

The implemented evaluation in [evaluation/metrics.py](evaluation/metrics.py) reports:

- Balanced accuracy
- F1 score
- Loss

The Week 1 trainer uses baseline targets:

- Balanced accuracy: `>= 0.77`
- F1: `>= 0.70`

The Phase 2 trainer uses:

- Balanced accuracy: `>= 0.88`
- F1: `>= 0.88`

These are still in-domain noise-duplicate gates, not realistic deepfake benchmarks.

## Tests

Run the test suite with:

```bash
python -m unittest discover -s tests
```

Coverage currently includes:

- Branch A forward-pass shape checks
- Branch B output shape and golden numerical regression
- Phase 2 discriminator output shape
- Branch A frozen-after-step verification for Phase 2
- Phase 1 checkpoint load/remap verification for Phase 2
- Metric computation sanity checks
- Scheduler configuration checks
- End-to-end training smoke test with checkpoint and report generation
- Dataset shape, label balance, normalization, and pairing behavior
- Optical-flow precompute smoke test
- Import/bootstrap checks
- TensorBoard tracker smoke test when TensorBoard is installed

## Limitations

- Branch C and later ensemble phases are not implemented yet.
- Current fake samples are Gaussian-noise duplicates, not actual deepfakes.
- Out-of-domain evaluation is not implemented.
- If `identity_CelebA.txt` is missing, real pairs fall back to adjacent-image pairing.
- `phase2_a_b.pt` is an in-domain noise-duplicate checkpoint, not a realistic deepfake benchmark.
- The checked-in `.venv` may be stale; in this workspace `pytest` was not available in the active interpreter.

## Roadmap

The planned next steps are:

1. Implement Branch C physics-based features from flow and photometrics.
2. Add Phase 3+ fusion training for the multi-branch classifier.
3. Replace synthetic noise-duplicate negatives with stronger fake-generation sources.
4. Add out-of-domain evaluation.
5. Repair or recreate the local Python environment if exact `pytest` workflow parity is required.
