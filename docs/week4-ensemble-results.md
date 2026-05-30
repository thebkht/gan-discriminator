# Week 4 Ensemble Results

## Status

Week 4 forensics OOD evaluation is complete on all four local forensics datasets under `data/forensics/`.

Primary handoff artifact for Dev 2:

- `runs/forensics_eval/summary.json`
- `runs/forensics_eval/summary.md`
- `runs/forensics_eval/per_dataset/*/ensemble_per_image_scores.csv`
- `runs/forensics_eval/per_dataset/*/*_confusion_matrix.png`
- `runs/forensics_eval/pooled/*_confusion_matrix.png`

The baseline run used the full CelebA train transfer cache at `runs/celeba_features/phase3_train_adjacent_cache.npz`, the Phase 3 checkpoint at `checkpoints/phase3_a_b_c.pt`, split `test`, and the `adjacent_same_class` forensics pairing contract.

The inference-only recovery harness now defaults new forensics evaluations to degenerate pairing, supports MTCNN-aligned caches from `scripts/preprocess_forensics_faces.py`, uses forensics validation threshold artifacts from `evaluation.forensics_threshold_sweep`, and exposes `--branch-b-invert-logits` plus horizontal-flip `--tta`. These are calibration and inference changes only; they do not change the negative-transfer conclusion below unless a new locked final run is generated and reported.

## In-Domain Proxy

Source: `runs/ensemble_ablation/summary.md`, balanced CelebA proxy test subset, `13,074` examples. The RF probe trained on an in-memory 80/20 split of test-split branch features, so it is an in-domain proxy ablation rather than the forensics transfer protocol.

| # | Config | Classifier | Balanced accuracy | F1 | AUC-ROC |
| -: | ------ | ---------- | ----------------: | --: | ------: |
| 1 | A only | Logistic | 0.6529 | 0.6224 | 0.6860 |
| 2 | B only | Logistic | 0.8930 | 0.8897 | 0.9471 |
| 3 | C only | Logistic | 0.5530 | 0.5516 | 0.5717 |
| 4 | A+B | RF | 0.8939 | 0.8913 | 0.9463 |
| 5 | A+C | RF | 0.6636 | 0.6338 | 0.6966 |
| 6 | B+C | RF | 0.8869 | 0.8837 | 0.9440 |
| 7 | A+B+C | RF | 0.8992 | 0.8962 | 0.9471 |

The B+C proposal gate was not cleared on the proxy table: balanced accuracy was `0.8869` versus the `0.944` target, and F1 was `0.8837` versus the `0.93` target.

## Forensics Neural Baseline

Source: `runs/forensics_eval/summary.md`. Threshold `0.61` is the best proxy-task threshold from `runs/ensemble_ablation/threshold_sweep.json`.

| Dataset | N | Bal Acc @0.5 | Bal Acc @0.61 | F1 | AUC-ROC |
| ------- | -: | -----------: | -------------: | --: | ------: |
| Data Set 1 | 5227 | 0.5177 | 0.5153 | 0.3724 | 0.5255 |
| Data Set 2 | 5226 | 0.4422 | 0.4444 | 0.3316 | 0.4189 |
| Data Set 3 | 5226 | 0.4557 | 0.4369 | 0.3203 | 0.4317 |
| Data Set 4 | 5226 | 0.5013 | 0.4935 | 0.3248 | 0.4989 |
| Pooled | 20905 | 0.4792 | 0.4725 | 0.3370 | 0.4675 |

The neural Phase 3 checkpoint does not transfer to the forensics distribution. Pooled balanced accuracy is below random at both thresholds.

## Forensics Transfer Ensemble

Source: `runs/forensics_eval/summary.md`, balanced accuracy.

| Dataset | A | B | C | A+B | A+C | B+C | A+B+C |
| ------- | --: | --: | --: | --: | --: | --: | ----: |
| Data Set 1 | 0.5040 | 0.5021 | 0.4996 | 0.5180 | 0.4991 | 0.4978 | 0.5135 |
| Data Set 2 | 0.5012 | 0.4319 | 0.5000 | 0.4536 | 0.5037 | 0.4380 | 0.4551 |
| Data Set 3 | 0.5012 | 0.4374 | 0.4996 | 0.4642 | 0.5029 | 0.4516 | 0.4646 |
| Data Set 4 | 0.4991 | 0.5018 | 0.4992 | 0.4921 | 0.5028 | 0.4991 | 0.5041 |
| Pooled | 0.5014 | 0.4683 | 0.4996 | 0.4820 | 0.5021 | 0.4716 | 0.4843 |

No branch combination clears the proposal OOD gate. The recommended B+C architecture from the proposal reaches only `0.4716` pooled balanced accuracy and `0.4981` F1 on the full transfer protocol.

## Pooled Confusion Matrix Counts

Counts use the repository label convention: real=`0`, fake=`1`; `TN` means real predicted real, `FP` means real predicted fake, `FN` means fake predicted real, and `TP` means fake predicted fake.

| Config | TN | FP | FN | TP | Pattern |
| ------ | --: | --: | --: | --: | ------- |
| A only | 59 | 10354 | 31 | 10461 | Predicts almost everything fake |
| B only | 4735 | 5678 | 5436 | 5056 | Opposite-bias / near class-prior flip |
| C only | 10 | 10403 | 18 | 10474 | Predicts almost everything fake |
| A+B | 3125 | 7288 | 3527 | 6965 | Strong fake bias |
| A+C | 203 | 10210 | 160 | 10332 | Near-total fake prediction |
| B+C | 4384 | 6029 | 5013 | 5479 | Below-random transfer |
| A+B+C | 3195 | 7218 | 3548 | 6944 | Strong fake bias |

## Interpretation

The full OOD result contradicts the proposal deployment assumption. Every transfer ensemble trained on CelebA branch features is ineffective on the four forensics datasets. Branch A and Branch C collapse almost completely into the fake class. A+B, A+C, and A+B+C inherit the same real-class failure mode. Branch B is the most diagnostic failure: its velocity features do not preserve polarity on the forensics distribution, so adding C in B+C does not recover the signal.

The likely failure mode is that the transfer RFs learned CelebA-specific decision boundaries, dominated by identity-pair statistics and local proxy artifacts, rather than GAN artifact detectors. These boundaries do not transfer to the forensics images.

## Deployment Recommendation

Do not ship B+C, A+B+C, or the Phase 3 neural checkpoint as a forensics detector from this training run.

The defensible conclusion is negative: the current CelebA-trained feature stack is useful as an architecture/proxy-task experiment, but not as a deployed OOD deepfake detector. A production candidate would need training data with real generative manipulations or explicit domain adaptation before the Week 4 OOD gate can be reopened.

## Architecture Checklist

- Automated architecture contracts live in `tests/test_architecture_contracts.py`.
- Phase 3 and Phase 4 forward paths return `a`, `b`, `c`, and `logit` with dimensions `2048`, `32`, `28`, and `(B,)`.
- The inference contract at `runs/phase4_ensemble/inference_contract.json` sums to `2108`.
- Branch B and Branch C normalization guards are covered by finite-output tests.
- Phase freeze behavior remains covered by the existing `tests/test_model.py` Phase 2/3/4 tests.
- No training imports are required inside `models/`.
- Forensics evaluation intentionally uses on-the-fly Farneback flow because the forensics folders do not provide a CelebA-style flow cache.
