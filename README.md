# Hybrid Three-Branch Deepfake Detector

This repository tracks an in-progress deepfake face detection project based on the project proposal: a hybrid discriminator that combines spatial CNN features, spatiotemporal embedding derivatives, and physics-based dynamics.

The codebase is not at full proposal parity yet. Today it includes the completed Branch A baseline, the current Branch B / Phase 2 stack, the trained Branch C / Phase 3 path, the CelebA pair dataset pipeline, the optical-flow precompute utility, and tests around the implemented training path. The docs below distinguish between the proposal target, the code that exists now, and the later-phase work that is still open.

## Current Status

- Implemented: Branch A spatial encoder and paired-frame baseline classifier
- Implemented: Branch B spatiotemporal summary branch and `DiscriminatorPhase2`
- Implemented: Branch C physics branch and `DiscriminatorPhase3`
- Implemented: Phase 2 A+B trainer CLI and checkpoint/report writing
- Implemented: Phase 3 A+B+C trainer CLI, resume wiring, and flow-cache preflight checks
- Implemented: CelebA dataloader with real/fake pair construction
- Implemented: validation-loss overfitting stop logic for Branch A and Phase 2
- Implemented: generic checkpoint save/load helpers and standalone hinge-loss module
- Implemented: offline Farneback optical-flow precompute utility
- Implemented: standalone Branch A test-split confusion-matrix evaluation
- Implemented: metric computation, checkpointing, confusion-matrix plotting, run summaries, and optional TensorBoard logging
- Verified: Branch B regression tests, Branch C golden-feature tests, Branch A/B freeze tests, data pipeline tests, overfit-stop unit tests, and Branch A evaluation smoke tests
- Verified: `checkpoints/phase3_a_b_c.pt` and `runs/phase3_a_b_c_w2/` clear the configured Phase 3 gate at epoch `8` with balanced accuracy `0.8741`, F1 `0.9067`, AUC-ROC `0.9484`, and validation loss `0.2726`
- Implemented: Phase 4 fine-tuning path, staged-unfreeze `DiscriminatorPhase4`, fake-positive asymmetric BCE+hinge loss, and `inference_contract.json` handoff artifact generation
- Verified: the final Phase 4 run improved balanced accuracy only marginally (`0.8790 -> 0.8850`) and AUC-ROC (`0.9480 -> 0.9499`), but reduced F1 (`0.9072 -> 0.8955`) and real-class TNR (`0.78 -> 0.72`); Phase 3 is the current best deployment candidate under the balanced objective
- Not implemented yet: RF branch-combination experiments and OOD evaluation
- Active runtime contract: `2048 + 32 + 28 = 2108`; the proposal-parity `2048 + 8 + 28 = 2084` fusion contract is not the current load-compatible path
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
│   ├── branch_c.py
│   └── discriminator.py
├── scripts/
│   ├── download_celeba.sh
│   └── eval_pred_all_branches.py
├── tests/
│   ├── test_bootstrap_and_imports.py
│   ├── test_branch_a_baseline.py
│   ├── test_data.py
│   ├── test_model.py
│   └── test_overfit_stop.py
├── training/
│   ├── batch_preview.py
│   ├── checkpointing.py
│   ├── eval_branch_a.py
│   ├── losses.py
│   ├── overfit_stop.py
│   ├── phase1_train.py
│   ├── phase2_train.py
│   ├── phase2_trainer.py
│   ├── phase3_train.py
│   ├── phase3_trainer.py
│   ├── phase4_finetune.py
│   ├── phase4_trainer.py
│   ├── run_artifacts.py
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
- Proposal fusion head: concatenated `2084-D -> 512 -> 128 -> 1`
- Active runtime fusion head: concatenated `2108-D -> 512 -> 128 -> 1`

The intended training order is:

1. Train Branch A end-to-end.
2. Add Branch B with Branch A partially frozen, then finetune the shared encoder tail.
3. Add Branch C with earlier branches frozen.
4. Fine-tune the fused model and evaluate branch ensembles, with B+C treated as the strongest OOD-oriented configuration in the proposal.

### Current implementation

The repository currently implements Branch A, the A+B Phase 2 stack, and the A+B+C Phase 3 code path.

Branch A in [models/branch_a.py](models/branch_a.py):

- `BranchAEncoder`: five convolution blocks with spectral normalization and LeakyReLU, outputting a `2048-D` feature from one frame
- `BranchABaseline`: applies the encoder to both frames and classifies the concatenated `4096-D` pair representation

Branch B and Phase 2 in [models/branch_b.py](models/branch_b.py) and [models/discriminator.py](models/discriminator.py):

- `BranchB_Spatiotemporal` now reuses the pretrained `BranchAEncoder` for both frames instead of a separate `EmbedCNN`
- Committed temporal summary: `8-D` `[vel_mean, vel_std, vel_max, vel_min, cos_sim, l2_dist, sign_consistency, abs_vel_mean]`
- The `8-D` summary is normalized with `LayerNorm(8)` and expanded to a learned `32-D` feature before fusion
- `DiscriminatorPhase2` keeps `feat_a` on a `no_grad()` path, finetunes only the last two Branch A blocks through Branch B, and fuses `2048 + 32 = 2080` features through a stronger dropout head

Branch C and Phase 3 in [models/branch_c.py](models/branch_c.py), [models/discriminator.py](models/discriminator.py), and [training/phase3_trainer.py](training/phase3_trainer.py):

- `BranchC_Physics` is a deterministic `28-D` feature extractor: `20-D` flow summaries plus `8-D` HSV photometric summaries
- `DiscriminatorPhase3` loads frozen Phase 2 Branch A+B weights, concatenates `2048 + 32 + 28 = 2108` features, and trains only Branch C plus a fresh fusion head
- Phase 3 enforces the current cache contract by requiring `include_flow=True` and `pairing_mode="adjacent_cache"` dataloaders, then verifying the cached flow directory before training starts
- The repository also ships `training/checkpointing.py` and `training/losses.py`, including resume support and a standalone `HingeLoss` module for later phases

This means the current code now reaches the proposal's three-branch structure and includes the Phase 4 fine-tuning path, but not the full evaluation parity. The active Phase 3 and Phase 4 path uses the current `32-D` Branch B expansion, so the runtime contract remains `2048 + 32 + 28 = 2108` and the later ensemble workflow is still pending.

Phase 4 in [models/discriminator.py](models/discriminator.py), [training/phase4_trainer.py](training/phase4_trainer.py), and [training/losses.py](training/losses.py):

- `DiscriminatorPhase4` loads the Phase 3 `2108-D` contract but starts in a fusion-only trainable state
- Training uses staged unfreezing: fusion head first, Branch B expander + Branch C next, then the last two Branch A blocks at a lower LR
- The Phase 4 loss is `AsymmetricCombinedLoss`, which keeps the repository's fake-positive logit convention and upweights real-class mistakes with `real_weight`
- The old standalone `CombinedBCEHingeLoss` remains available for tests and comparison, but it is not the active Phase 4 trainer loss
- Final Phase 4 evaluation did not become the deployment path: it increased fake recall (`TPR 0.94 -> 0.97`) while lowering real specificity (`TNR 0.78 -> 0.72`) and F1, consistent with overfitting the proxy training distribution rather than improving generalization

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

The file now also contains dedicated `phase2:` and `phase3:` blocks.

Phase 2 defaults:

- Epochs: `20`
- Learning rate: `1.5e-4`
- Scheduler: `CosineAnnealingLR`
- Pretrained Branch A checkpoint: `phase1_branch_a_best.pt`
- Default Phase 2 checkpoint name: `phase2_a_b.pt`
- Targets: balanced accuracy `>= 0.88`, F1 `>= 0.88`
- Early-stop defaults: warmup `3`, overfit patience `5`, val-loss ceiling patience `3`, validation balanced-accuracy patience `4`, Phase 2 ceiling `0.40`

Phase 3 defaults:

- Epochs: `20`
- Learning rate: `2e-4`
- Scheduler: `CosineAnnealingLR`
- Pretrained Phase 2 checkpoint: `phase2_a_b.pt`
- Default Phase 3 checkpoint name: `phase3_a_b_c.pt`
- Pairing mode: `adjacent_cache`
- Targets: balanced accuracy `>= 0.83`, F1 `>= 0.80`
- Early-stop defaults currently include overfit patience `5` and validation-loss ceiling `0.45`

Phase 4 defaults:

- Epochs: `30`
- Base learning rate: `5e-5`
- Scheduler: `CosineAnnealingLR`
- Pretrained Phase 3 checkpoint: `phase3_a_b_c.pt`
- Default Phase 4 checkpoint name: `phase4_ensemble.pt`
- Pairing mode: `adjacent_cache`
- Loss: `AsymmetricCombinedLoss` with `bce_weight=0.7`, `hinge_weight=0.3`, `real_weight=1.5`, `fake_weight=1.0`, and margin `0.8`
- Staged unfreezing: 10 epochs fusion-only at `5e-5`, 10 epochs Branch B expander + Branch C at `2e-5`, then 10 epochs with the last two Branch A blocks at `5e-6`
- Early stopping is stage-aware: a plateau in Stage 1 or Stage 2 advances to the next stage instead of ending the full Phase 4 run

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
- `--device cpu|cuda|mps`: force a device; default is `mps`, then `cuda`, then `cpu`
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
- `--device cpu|cuda|mps`: force a device; default is `mps`, then `cuda`, then `cpu`

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

This produces one `*_flow.pt` tensor per image. These tensors are the required input substrate for the implemented Phase 3 Branch C path.

## Train Phase 3 A+B+C

Run the Phase 3 trainer:

```bash
python3 -m training.phase3_train --config config/config.yaml --run-name phase3_a_b_c
```

Useful flags:

- `--train-limit`: cap train samples for smoke runs
- `--val-limit`: cap validation samples for smoke runs
- `--epochs-override`: override configured epochs
- `--max-batches`: cap batches per split for very short dry-runs
- `--num-workers`: override the Phase 3 dataloader worker count
- `--checkpoint-name-override`: avoid overwriting the final checkpoint during smoke runs
- `--resume`: resume from a saved Phase 3 checkpoint
- `--device cpu|cuda|mps`: force a device; default comes from `phase3.device`, then falls back through the usual resolver

Example smoke run:

```bash
python3 -m training.phase3_train \
  --config config/config.yaml \
  --run-name phase3_a_b_c_smoke \
  --train-limit 256 \
  --val-limit 64 \
  --epochs-override 1 \
  --max-batches 2 \
  --checkpoint-name-override phase3_a_b_c_smoke.pt \
  --num-workers 0 \
  --device cpu
```

Phase 3 currently trains with `BCEWithLogitsLoss`, not the proposal's later fine-tuning loss mix. Before the first epoch it also:

- verifies the flow cache stem set against the image tree
- requires cached flow tensors to be available for `adjacent_cache` pairing
- loads Branch A+B weights from the configured Phase 2 checkpoint and freezes those branches

## Evaluation And Targets

The implemented evaluation in [evaluation/eval.py](evaluation/eval.py) reports:

- Balanced accuracy
- F1 score
- AUC-ROC
- Loss
- Confusion-matrix plots

The repository also includes a standalone Branch A test evaluator in [training/eval_branch_a.py](training/eval_branch_a.py). It loads a saved Branch A checkpoint, runs the `test` split, and writes:

- `runs/<run-name>/confusion_matrix.json`
- `runs/<run-name>/confusion_matrix.png`
- `runs/<run-name>/eval_report.md`

Run the Branch A evaluator with:

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

The branch comparison script in [scripts/eval_pred_all_branches.py](scripts/eval_pred_all_branches.py) exports prediction CSVs and confusion matrices for available Branch A, Phase 2, Phase 3, and Phase 4 checkpoints. For Phase 3 and Phase 4 it keeps `pairing_mode="adjacent_cache"` so cached flow tensors still match their adjacent partners, then balances the exported evaluation rows by class. The summary records both the balanced eval count and the original source class counts.

Run the branch comparison evaluator with:

```bash
python3 scripts/eval_pred_all_branches.py --config config/config.yaml --run-dir runs/eval_pred_all_branches --device cpu
```

The Week 1 trainer uses baseline targets:

- Balanced accuracy: `>= 0.77`
- F1: `>= 0.70`

The Phase 2 trainer uses:

- Balanced accuracy: `>= 0.88`
- F1: `>= 0.88`

These are still in-domain proxy-task gates, not realistic deepfake benchmarks. The proposal's headline `94.4%` balanced-accuracy result refers to the recommended B+C ensemble on difficult OOD content, which this repository has not reproduced yet.

## Tests

Run the test suite with:

```bash
python -m unittest discover -s tests
```

Coverage currently includes:

- Branch A forward-pass shape checks
- Branch B output shape and golden numerical regression
- Branch C output shape and golden feature regression
- Phase 2 discriminator output shape
- Phase 3 discriminator output shape
- Branch A frozen-after-step verification for Phase 2
- Branch A + Branch B frozen-after-step verification for Phase 3
- Phase 1 checkpoint load/remap verification for Phase 2
- Phase 2-to-Phase 3 load verification and Phase 3 resume helper coverage
- Metric computation sanity checks
- Hinge-loss label-convention test
- Scheduler configuration checks
- End-to-end training smoke test with checkpoint, preview-image, and plot generation
- Dataset shape, label balance, normalization, and pairing behavior
- Optical-flow precompute smoke test
- Import/bootstrap checks
- TensorBoard tracker smoke test when TensorBoard is installed

## Limitations

- Branch C and Phase 3 are implemented and trained, but the current result is still an in-domain proxy task rather than a real deepfake benchmark.
- Phase 4 fine-tuning is implemented and has been run, but its asymmetric loss worsened the TNR/TPR balance and should not replace the Phase 3 checkpoint for deployment-style evaluation.
- Current fake samples are cross-identity proxy negatives, not actual deepfakes.
- Out-of-domain evaluation is not implemented.
- Branch-combination ensemble experiments are not implemented in this branch.
- Phase 3/4 flow-aware evaluation must keep `adjacent_cache`; switching those phases to default pairing would attach cached flow tensors to the wrong frame pair unless the cache is regenerated.
- The proposal's direct `2048 + 8 + 28 = 2084` fusion contract is still not the active runtime contract; the current Phase 3 and Phase 4 stack uses `2048 + 32 + 28 = 2108`.
- If `identity_CelebA.txt` is missing, real pairs fall back to adjacent-image pairing.
- If `identity_CelebA.txt` is missing, fake pairs fall back to deterministic distant-index pairing.
- `phase2_a_b.pt` may be a legacy pre-Run 3 checkpoint; verify the checkpoint provenance before treating it as a Phase 3 base.
- The checked-in pseudo-identity file may be attribute-derived rather than true CelebA identity labels, depending on local workspace state.
- The checked-in `.venv` may be stale; in this workspace `pytest` was not available in the active interpreter.

## Roadmap

The planned next steps are:

1. Run the RF branch-combination ensemble experiments, with B+C as the priority configuration.
2. Sweep decision thresholds on the Phase 3 checkpoint to recover the best TNR/TPR operating point.
3. Add out-of-domain evaluation on real deepfake data; this is the only meaningful test of the proposal's `94.4%` claim.
4. Replace cross-identity proxy negatives with stronger fake-generation sources.
