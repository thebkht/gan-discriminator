# Hybrid Three-Branch GAN Discriminator for Deepfake Face Detection

This repository contains the project plan and dataset assets for a deepfake face detection system based on a hybrid three-branch discriminator. The design follows the master plan in [docs/master-plan.md](/docs/master-plan.md) and targets improved out-of-domain robustness compared with single-branch CNN discriminators.

## Overview

The proposed detector combines three complementary signals:

- Branch A: CNN spatial features from a single face frame
- Branch B: Spatiotemporal embedding derivatives from consecutive frame pairs
- Branch C: Physics-based dynamics using optical flow and HSV photometrics

These branch outputs are fused by a fully connected head to classify samples as real or fake. The recommended deployment configuration from the plan is the `B + C` ensemble, which is expected to provide the strongest out-of-domain performance.

## Project Goal

Standard single-branch discriminators can perform well on in-distribution samples but degrade heavily on unseen generation methods. This project aims to recover robust out-of-domain detection performance, with a target of:

- Balanced accuracy: `>= 94%`
- F1 score: `>= 0.93`

## Repository Status

The current repository contains:

- `docs/master-plan.md`: full architecture, build plan, metrics, and dependency specification
- `data/celeba/`: CelebA dataset assets used for training and evaluation

The implementation scaffold described in the plan has not been added yet.

## Planned Project Structure

```text
deepfake_detector/
├── config/
│   └── config.yaml
├── data/
│   ├── celeba_loader.py
│   ├── precompute_flow.py
│   └── augmentations.py
├── models/
│   ├── discriminator.py
│   ├── branch_a.py
│   ├── branch_b.py
│   └── branch_c.py
├── training/
│   ├── trainer.py
│   ├── losses.py
│   ├── phase1_train.py
│   ├── phase2_train.py
│   ├── phase3_train.py
│   └── phase4_finetune.py
├── evaluation/
│   ├── eval.py
│   ├── ensemble.py
│   └── ood_eval.py
├── checkpoints/
├── runs/
├── scripts/
│   └── download_celeba.sh
├── tests/
│   ├── test_model.py
│   └── test_data.py
├── requirements.txt
└── README.md
```

## Architecture Summary

### Branch A: CNN Spatial

- Input: single `64 x 64 x 3` frame
- Output: `2048-D` feature vector
- Backbone: 5 convolution blocks with spectral normalization and LeakyReLU

### Branch B: Spatiotemporal

- Input: consecutive frame pair
- Output: `8-D` temporal descriptor
- Signal: embedding velocity, curvature, and related temporal statistics

### Branch C: Physics-Based Dynamics

- Input: raw or precomputed frame-pair dynamics
- Output: `28-D` descriptor
- Signal: optical flow statistics and HSV photometric consistency

### Fusion Head

- Concatenated input: `2084-D`
- MLP: `2084 -> 512 -> 128 -> 1`
- Output: real/fake logit

## Dataset

The project is based on CelebA.

| Property          | Value                        |
| ----------------- | ---------------------------- |
| Total images      | 202,599                      |
| Identities        | 10,177                       |
| Native resolution | 178 x 218                    |
| Target resolution | 64 x 64                      |
| Attributes        | 40 binary labels per image   |
| License           | Non-commercial research only |

Download command:

```bash
./scripts/download_celeba.sh
```

Requirements for the download script:

- Kaggle CLI installed
- Kaggle API credentials available at `~/.kaggle/kaggle.json`

Planned split:

| Split |   Count |
| ----- | ------: |
| Train | 162,770 |
| Val   |  19,867 |
| Test  |  19,962 |

## Data Pipeline

Planned real/fake sampling:

- Real pair: two images from the same identity
- Fake pair: one image plus a perturbed or synthetic counterpart

Train-time augmentation:

- Random horizontal flip
- Color jitter
- Normalize to `[-1, 1]`

Optical flow is intended to be precomputed offline and cached as `.pt` tensors for faster training.

Example command:

```bash
python data/precompute_flow.py \
  --img-dir /data/celeba/img_align_celeba \
  --out-dir /data/celeba/flow_cache \
  --method farneback
```

## Training Plan

The training strategy is split into four phases to reduce unstable co-adaptation between branches:

1. Train Branch A plus the fusion head
2. Freeze Branch A and train Branch B plus the fusion head
3. Freeze Branches A and B and train Branch C plus the fusion head
4. Unfreeze all branches and fine-tune the full system with a lower learning rate

Key hyperparameters from the plan:

| Parameter        | Value                          |
| ---------------- | ------------------------------ |
| Image size       | `64 x 64`                      |
| Batch size       | `64`                           |
| Optimizer        | `Adam(beta1=0.5, beta2=0.999)` |
| LR (phases 1-3)  | `2e-4`                         |
| LR (phase 4)     | `5e-5`                         |
| Epochs per phase | `20`                           |
| Scheduler        | `CosineAnnealingLR`            |
| Dropout          | `0.3`                          |

## Evaluation

Primary metrics:

- Balanced accuracy
- F1 score
- AUC-ROC
- Confusion matrix

Planned out-of-domain evaluation includes:

- Style-transferred faces
- Face reenactment outputs
- Diffusion-based face synthesis

## Expected Performance

| Configuration      | Balanced Accuracy |   F1 |
| ------------------ | ----------------: | ---: |
| Branch A only      |             77.8% | 0.70 |
| Branch B only      |    88.9% to 94.4% | 0.91 |
| Branch C only      |             83.3% | 0.80 |
| A + B ensemble     |             89.5% | 0.88 |
| A + C ensemble     |             88.9% | 0.85 |
| B + C ensemble     |             94.4% | 0.93 |
| A + B + C ensemble |             89.5% | 0.86 |

## Planned Dependencies

```text
torch>=2.1.0
torchvision>=0.16.0
opencv-python>=4.8.0
numpy>=1.24.0
scikit-learn>=1.3.0
Pillow>=10.0.0
tqdm>=4.66.0
tensorboard>=2.14.0
pyyaml>=6.0
```

Install command once `requirements.txt` exists:

```bash
pip install -r requirements.txt
```

## Getting Started

At the current stage, the repository is documentation-first. Recommended next steps are:

1. Create the planned source tree from the master plan
2. Add `requirements.txt` and `config/config.yaml`
3. Implement the CelebA loader and flow precomputation pipeline
4. Build and test Phase 1 training for Branch A

## Reference Document

For the full technical specification, milestones, risks, and citations, see [docs/master-plan.md](/docs/master-plan.md).
