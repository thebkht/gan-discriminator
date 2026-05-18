# Hybrid Three-Branch Deepfake Detector

This repository contains the early implementation of a deepfake face detection project built around a planned three-branch discriminator. The current codebase implements the Week 1 baseline only: a working Branch A spatial model, a CelebA pair dataset pipeline, an optical-flow precompute utility, and tests for the baseline training path.

The long-term design is documented in [docs/master-plan.md](docs/master-plan.md), but the repository is not at full three-branch parity yet. The `README` below describes what is implemented now.

## Current Status

- Implemented: Branch A baseline training on paired CelebA images
- Implemented: CelebA dataloader with real/fake pair construction
- Implemented: offline Farneback optical-flow precompute utility
- Implemented: metric computation, checkpointing, run summaries, and optional TensorBoard logging
- Not implemented yet: Branch B, Branch C, and the final fused three-branch discriminator

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
│   └── branch_a.py
├── scripts/
│   └── download_celeba.sh
├── tests/
│   ├── test_bootstrap_and_imports.py
│   ├── test_branch_a_baseline.py
│   └── test_data_pipeline.py
├── training/
│   ├── branch_a_trainer.py
│   ├── tracker.py
│   └── train_branch_a.py
├── pyrightconfig.json
├── requirements.txt
└── README.md
```

## Implemented Baseline

The current model in [models/branch_a.py](models/branch_a.py) is a Branch A baseline:

- Input: two `64 x 64` RGB face frames
- Encoder: five convolution blocks with spectral normalization and LeakyReLU
- Classifier: concatenated twin-frame features passed through an MLP
- Output: one real/fake logit for the frame pair

This is a pair classifier, not a full GAN discriminator and not the final three-branch model described in the planning docs.

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
python -m training.train_branch_a --config config/config.yaml --run-name branch_a_baseline
```

Useful flags:

- `--train-limit`: cap train samples for smoke runs
- `--val-limit`: cap validation samples for smoke runs
- `--epochs-override`: override configured epochs
- `--device cpu|cuda|mps`: force a device
- `--tracker-backend tensorboard`: emit TensorBoard logs under `runs/<run-name>/tensorboard`

Example short smoke run:

```bash
python -m training.train_branch_a \
  --config config/config.yaml \
  --run-name smoke \
  --train-limit 512 \
  --val-limit 128 \
  --epochs-override 1 \
  --device cpu
```

## Precompute Optical Flow

The flow utility in [data/precompute_flow.py](data/precompute_flow.py) currently supports Farneback flow only.

```bash
python -m data.precompute_flow \
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

The trainer currently uses internal baseline targets:

- Balanced accuracy: `>= 0.77`
- F1: `>= 0.70`

Those targets are for the Week 1 noise-duplicate baseline only. They are not equivalent to a realistic deepfake benchmark.

## Tests

Run the test suite with:

```bash
python -m unittest discover -s tests
```

Coverage currently includes:

- Branch A forward-pass shape checks
- Metric computation sanity checks
- Scheduler configuration checks
- End-to-end training smoke test with checkpoint and report generation
- Dataset shape, label balance, normalization, and pairing behavior
- Optical-flow precompute smoke test
- Import/bootstrap checks
- TensorBoard tracker smoke test when TensorBoard is installed

## Limitations

- Only Branch A is implemented.
- Current fake samples are Gaussian-noise duplicates, not actual deepfakes.
- Out-of-domain evaluation is not implemented.
- If `identity_CelebA.txt` is missing, real pairs fall back to adjacent-image pairing.
- The planning docs describe a broader system than the code currently provides.

## Roadmap

The planned next steps are:

1. Implement Branch B temporal features.
2. Implement Branch C physics-based features from flow and photometrics.
3. Add fusion training for the multi-branch classifier.
4. Replace synthetic noise-duplicate negatives with stronger fake-generation sources.
5. Add out-of-domain evaluation.
