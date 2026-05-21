# Hybrid Three-Branch Deepfake Detector

This repository tracks an in-progress deepfake face detection project based on the project proposal: a hybrid discriminator that combines spatial CNN features, spatiotemporal embedding derivatives, and physics-based dynamics.

The codebase is not at full proposal parity yet. Today it includes the completed Branch A baseline, the current Branch B / Phase 2 stack, the CelebA pair dataset pipeline, the optical-flow precompute utility, and tests around the implemented training path. The docs below distinguish between the proposal target and the code that exists now.

## Current Status

- Implemented: Branch A spatial encoder and paired-frame baseline classifier
- Implemented: Branch B spatiotemporal summary branch and `DiscriminatorPhase2`
- Implemented: Phase 2 A+B trainer CLI and checkpoint/report writing
- Implemented: CelebA dataloader with real/fake pair construction
- Implemented: validation-loss overfitting stop logic for Branch A and Phase 2
- Implemented: offline Farneback optical-flow precompute utility
- Implemented: standalone Branch A test-split confusion-matrix evaluation
- Implemented: metric computation, checkpointing, run summaries, and optional TensorBoard logging
- Verified: Branch B regression tests, Branch A freeze tests, data pipeline tests, overfit-stop unit tests, and Branch A evaluation smoke tests
- Not implemented yet: Branch C, Phase 3+, Hinge-loss fine-tuning, and the final fused three-branch discriminator from the proposal
- Historical checkpoint: `checkpoints/phase2_a_b.pt` was trained on the legacy pre-Run 3 Branch B architecture and should not be treated as the current baseline

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
│   ├── branch_a_eval.py
│   └── eval.py
├── models/
│   ├── branch_a.py
│   ├── branch_b.py
│   └── discriminator.py
├── scripts/
│   └── download_celeba.sh
├── tests/
│   ├── test_bootstrap_and_imports.py
│   ├── test_branch_a_baseline.py
│   ├── test_data.py
│   └── test_model.py
├── training/
│   ├── eval_branch_a.py
│   ├── overfit_stop.py
│   ├── phase1_train.py
│   ├── phase2_train.py
│   ├── phase2_trainer.py
│   ├── tracker.py
│   └── trainer.py
├── pyrightconfig.json
├── requirements.txt
└── README.md
```

## Architecture

### Proposal target

The proposal defines a three-branch discriminator over consecutive `64 x 64` RGB face frames:

- Branch A: spatial CNN encoder, `2048-D`
- Branch B: spatiotemporal derivative summary, `8-D`
- Branch C: optical-flow + photometric dynamics, `28-D`
- Final fusion head: concatenated `2084-D -> 512 -> 128 -> 1`

The intended training order is:

1. Train Branch A end-to-end.
2. Add Branch B with Branch A partially frozen, then finetune the shared encoder tail.
3. Add Branch C with earlier branches frozen.
4. Fine-tune the fused model and evaluate branch ensembles, with B+C treated as the strongest OOD-oriented configuration in the proposal.

### Current implementation

The repository currently implements Branch A and the A+B Phase 2 stack.

Branch A in [models/branch_a.py](/Users/thebkht/Sejong Uni/Pattern Regocnition/deepfake_detector/models/branch_a.py):

- `BranchAEncoder`: five convolution blocks with spectral normalization and LeakyReLU, outputting a `2048-D` feature from one frame
- `BranchABaseline`: applies the encoder to both frames and classifies the concatenated `4096-D` pair representation

Branch B and Phase 2 in [models/branch_b.py](/Users/thebkht/Sejong Uni/Pattern Regocnition/deepfake_detector/models/branch_b.py) and [models/discriminator.py](/Users/thebkht/Sejong Uni/Pattern Regocnition/deepfake_detector/models/discriminator.py):

- `BranchB_Spatiotemporal` now reuses the pretrained `BranchAEncoder` for both frames instead of a separate `EmbedCNN`
- Committed temporal summary: `8-D` `[vel_mean, vel_std, vel_max, vel_min, cos_sim, l2_dist, sign_consistency, abs_vel_mean]`
- The `8-D` summary is normalized with `LayerNorm(8)` and expanded to a learned `32-D` feature before fusion
- `DiscriminatorPhase2` keeps `feat_a` on a `no_grad()` path, finetunes only the last two Branch A blocks through Branch B, and fuses `2048 + 32 = 2080` features through a stronger dropout head

This means the current code is still not the proposal's final `2048 + 8 + 28 = 2084` fused discriminator, but Run 3 now aligns Branch B to the same pretrained feature space as Branch A.

## Dataset Pipeline

The dataset code in [data/celeba_loader.py](data/celeba_loader.py) builds pair-labeled samples from CelebA.

Real pairs:

- If `identity_CelebA.txt` is present, the loader pairs images from the same identity.
- If the identity file is missing, it falls back to adjacent-image pairing.

Fake pairs:

- The current baseline does not use GAN-generated or diffusion-generated fakes.
- With `identity_CelebA.txt` present, it pairs the anchor with a frame from a different identity.
- If the identity file is missing, it falls back to a deterministic distant-index pair.

Transforms from [data/augmentations.py](data/augmentations.py):

- Resize to `64 x 64`
- Random horizontal flip during training
- Color jitter during training
- Normalize tensors to `[-1, 1]`

This means current metrics are still only useful as a proxy-task baseline and are not representative of real deepfake performance. The task is still "same identity vs. different identity", not detection over real generative manipulations or OOD deepfakes.

## Configuration

The default config lives at [config/config.yaml](config/config.yaml).

Key defaults:

- Dataset size target: `202,599` images
- Native resolution target: `178 x 218`
- Batch size: `64`
- Epochs: `100`
- Learning rate: `1.5e-4`
- Shared-backbone finetuning: last `2` Branch A blocks trainable at `0.1x` LR
- Scheduler: `CosineAnnealingLR`
- Checkpoint metric: `balanced_accuracy`
- Default checkpoint name: `phase1_branch_a_best.pt`
- Early-stop defaults: warmup `3`, overfit patience `5`, val-loss ceiling patience `3`, Branch A ceiling `0.35`

The file now also contains a dedicated `phase2:` block:

- Epochs: `20`
- Learning rate: `2e-4`
- Scheduler: `CosineAnnealingLR`
- Pretrained Branch A checkpoint: `phase1_branch_a_best.pt`
- Default Phase 2 checkpoint name: `phase2_a_b.pt`
- Recommended Run 3 checkpoint name: `phase2_a_b_run3.pt`
- Targets: balanced accuracy `>= 0.88`, F1 `>= 0.88`
- Early-stop defaults: warmup `3`, overfit patience `5`, val-loss ceiling patience `3`, validation balanced-accuracy patience `4`, Phase 2 ceiling `0.40`

By default, outputs are written to:

- `checkpoints/phase1_branch_a_best.pt`
- `runs/<run-name>/benchmark_summary.json`
- `runs/<run-name>/benchmark_summary.md`
- `runs/<run-name>/metrics_history.json`
- `runs/<run-name>/train_batch0.jpg` through `train_batch2.jpg`
- `runs/<run-name>/val_batch0_labels.jpg` through `val_batch2_labels.jpg`
- `runs/<run-name>/val_batch0_pred.jpg` through `val_batch2_pred.jpg`
- `runs/<run-name>/confusion_matrix.png`
- `runs/<run-name>/confusion_matrix_normalized.png`
- `runs/<run-name>/results.png`

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
- `matplotlib`

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
python3 -m training.phase1_train --config config/config.yaml --run-name branch_a_baseline
```

Useful flags:

- `--train-limit`: cap train samples for smoke runs
- `--val-limit`: cap validation samples for smoke runs
- `--epochs-override`: override configured epochs
- `--device cpu|cuda|mps`: force a device
- `--tracker-backend tensorboard`: emit TensorBoard logs under `runs/<run-name>/tensorboard`

Example short smoke run:

```bash
python3 -m training.phase1_train \
  --config config/config.yaml \
  --run-name smoke \
  --train-limit 512 \
  --val-limit 128 \
  --epochs-override 1 \
  --device cpu
```

The Branch A trainer now stops early if either of these sustained patterns appears:

- validation loss worsens while train loss improves for `5` consecutive epochs
- after a `3`-epoch warmup, validation loss stays above `0.35` for `3` consecutive epochs while `train_loss < val_loss`

Each Branch A run also saves:

- training preview grids for the first three train batches
- validation label/prediction preview grids for the first three validation batches
- `confusion_matrix.png` and `confusion_matrix_normalized.png` from the best validation epoch
- `results.png` with train loss, validation loss, validation balanced accuracy, and validation F1 curves

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

The Phase 2 trainer uses the same overfit trend rule and a Phase 2 loss ceiling of `0.40`.

Each Phase 2 run writes preview JPGs during training and also attempts the same plotting artifacts as Branch A:

- `train_batch*.jpg`
- `val_batch*_labels.jpg`
- `val_batch*_pred.jpg`
- `confusion_matrix.png`
- `confusion_matrix_normalized.png`
- `results.png`

Notes:
- Preview JPGs are saved from the first few train and validation batches during the run.
- Plot PNGs require `matplotlib`. If it is not installed in the active Python environment, training still completes but the plotting files are skipped.

## Precompute Optical Flow

The flow utility in [data/precompute_flow.py](data/precompute_flow.py) currently supports Farneback flow only.

```bash
python3 -m data.precompute_flow \
  --img-dir data/celeba/img_align_celeba \
  --out-dir data/flow_cache \
  --method farneback \
  --image-size 64
```

This produces one `*_flow.pt` tensor per image. These tensors are the intended input substrate for the proposal's Branch C, which is not implemented yet.

## Evaluation And Targets

The implemented evaluation in [evaluation/eval.py](evaluation/eval.py) reports:

- Balanced accuracy
- F1 score
- Loss

The repository also includes a standalone Branch A test evaluator in [training/eval_branch_a.py](training/eval_branch_a.py). It loads a saved Branch A checkpoint, runs the `test` split, and writes:

- `runs/<run-name>/confusion_matrix.json`
- `runs/<run-name>/confusion_matrix.png`
- `runs/<run-name>/eval_report.md`

Run it with:

```bash
python3 -m training.eval_branch_a --config config/config.yaml --run-name branch_a_test_eval
```

Optional checkpoint override:

```bash
python3 -m training.eval_branch_a \
  --config config/config.yaml \
  --checkpoint checkpoints/phase1_branch_a_best.pt \
  --run-name branch_a_test_eval
```

The Week 1 trainer uses baseline targets:

- Balanced accuracy: `>= 0.77`
- F1: `>= 0.70`

The Phase 2 trainer uses:

- Balanced accuracy: `>= 0.88`
- F1: `>= 0.88`

These are still in-domain proxy-task gates, not realistic deepfake benchmarks. The proposal's headline `94.4%` balanced-accuracy result refers to the recommended B+C ensemble on difficult OOD content, which this repository does not implement yet.

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
- End-to-end training smoke test with checkpoint, preview-image, and plot generation
- Dataset shape, label balance, normalization, and pairing behavior
- Optical-flow precompute smoke test
- Import/bootstrap checks
- TensorBoard tracker smoke test when TensorBoard is installed

## Limitations

- Branch C and later ensemble phases are not implemented yet.
- Current fake samples are cross-identity proxy negatives, not actual deepfakes.
- Out-of-domain evaluation is not implemented.
- If `identity_CelebA.txt` is missing, real pairs fall back to adjacent-image pairing.
- If `identity_CelebA.txt` is missing, fake pairs fall back to deterministic distant-index pairing.
- `phase2_a_b.pt` was trained on the earlier trivial proxy task and is not a realistic deepfake benchmark.
- The checked-in pseudo-identity file may be attribute-derived rather than true CelebA identity labels, depending on local workspace state.
- The checked-in `.venv` may be stale; in this workspace `pytest` was not available in the active interpreter.

## Roadmap

The planned next steps are:

1. Implement Branch C physics-based features from flow and photometrics.
2. Add Phase 3 and Phase 4 training to reach the proposal's three-branch fused model.
3. Decide whether Branch B should stay at the current learned `32-D` expansion or be reduced back to the proposal's direct `8-D` fusion contract.
4. Replace cross-identity proxy negatives with stronger fake-generation sources.
5. Add out-of-domain evaluation and branch-combination experiments, especially the proposal's recommended B+C ensemble.
