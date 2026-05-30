# Final Report: Forensics OOD Evaluation

## Executive Summary

The full forensics OOD evaluation is complete across all four local datasets, totaling `20,905` images. The result is a clear negative finding: the CelebA-trained neural model and all seven transfer ensembles fail to generalize to the forensics distribution.

**The proposal target of 94.4% balanced accuracy and F1 ≥ 0.93 on forensics OOD content is not achievable with inference-only fixes (MTCNN alignment, threshold calibration, degenerate pairing, TTA) applied to the CelebA-trained checkpoint. Closing the gap to 94.4% requires training or adaptation on forensics or equivalent manipulated-face data — which is explicitly out of scope for this recovery pass.**

The primary Dev 2 handoff artifact is `runs/forensics_eval/summary.json`. Per-image scores and confusion-matrix images are under `runs/forensics_eval/per_dataset/`, with pooled confusion matrices under `runs/forensics_eval/pooled/`.

## Protocol

| Item | Value |
| ---- | ----- |
| Evaluation root | `data/forensics` |
| Split | `test` |
| Pairing | `adjacent_same_class` |
| Neural checkpoint | `checkpoints/phase3_a_b_c.pt` |
| Phase 4 comparison checkpoint | `checkpoints/phase4_ensemble.pt` |
| CelebA transfer cache | `runs/celeba_features/phase3_train_adjacent_cache.npz` |
| Output directory | `runs/forensics_eval` |

## Inference-Only Recovery Harness

This repository now includes the reduced-scope recovery path without forensics training:

| Fix | Artifact / option |
| --- | --- |
| MTCNN-aligned cache | `scripts/preprocess_forensics_faces.py --output-root data/forensics_aligned` |
| Degenerate pairing | default `--pairing degenerate` in `scripts/run_forensics_eval.py` |
| Forensics val thresholds | `python -m evaluation.forensics_threshold_sweep --split validation` |
| Per-dataset thresholds | `runs/forensics_threshold/per_dataset_thresholds.json` with `--threshold-mode per_dataset` |
| Branch B polarity flag | `--branch-b-invert-logits` |
| Horizontal flip TTA | `--tta` |

Final claims must be gated on the measured MTCNN detection rate in `data/forensics_aligned/alignment_report.json`. If any dataset is below 95% detection, report measured lift only and do not extrapolate the optimistic MTCNN trajectory.

## Neural Result

| Dataset | N | Bal Acc @0.5 | Bal Acc @0.61 | F1 | AUC-ROC |
| ------- | -: | -----------: | -------------: | --: | ------: |
| Data Set 1 | 5227 | 0.5177 | 0.5153 | 0.3724 | 0.5255 |
| Data Set 2 | 5226 | 0.4422 | 0.4444 | 0.3316 | 0.4189 |
| Data Set 3 | 5226 | 0.4557 | 0.4369 | 0.3203 | 0.4317 |
| Data Set 4 | 5226 | 0.5013 | 0.4935 | 0.3248 | 0.4989 |
| Pooled | 20905 | 0.4792 | 0.4725 | 0.3370 | 0.4675 |

The neural Phase 3 checkpoint is below random on the pooled forensics set.

## Transfer Ensemble Result

Balanced accuracy:

| Dataset | A | B | C | A+B | A+C | B+C | A+B+C |
| ------- | --: | --: | --: | --: | --: | --: | ----: |
| Data Set 1 | 0.5040 | 0.5021 | 0.4996 | 0.5180 | 0.4991 | 0.4978 | 0.5135 |
| Data Set 2 | 0.5012 | 0.4319 | 0.5000 | 0.4536 | 0.5037 | 0.4380 | 0.4551 |
| Data Set 3 | 0.5012 | 0.4374 | 0.4996 | 0.4642 | 0.5029 | 0.4516 | 0.4646 |
| Data Set 4 | 0.4991 | 0.5018 | 0.4992 | 0.4921 | 0.5028 | 0.4991 | 0.5041 |
| Pooled | 0.5014 | 0.4683 | 0.4996 | 0.4820 | 0.5021 | 0.4716 | 0.4843 |

B+C, the proposal-recommended configuration, reaches only `0.4716` pooled balanced accuracy and `0.4981` F1. It does not clear the `0.944` balanced-accuracy or `0.93` F1 target.

## Confusion-Matrix Story

| Config | TN | FP | FN | TP | Failure mode |
| ------ | --: | --: | --: | --: | ------------ |
| A only | 59 | 10354 | 31 | 10461 | Real-class collapse; predicts almost everything fake |
| B only | 4735 | 5678 | 5436 | 5056 | Opposite-bias / polarity mismatch on forensics |
| C only | 10 | 10403 | 18 | 10474 | Real-class collapse; predicts almost everything fake |
| A+B | 3125 | 7288 | 3527 | 6965 | Strong fake bias |
| A+C | 203 | 10210 | 160 | 10332 | Near-total fake prediction |
| B+C | 4384 | 6029 | 5013 | 5479 | Below-random transfer |
| A+B+C | 3195 | 7218 | 3548 | 6944 | Strong fake bias |

The story is consistent across the pooled matrices: CelebA-trained transfer ensembles are not detecting general GAN artifacts. They learned proxy boundaries tied to CelebA pair statistics. On forensics, those boundaries either collapse into fake predictions or, for Branch B, lose polarity and produce the opposite bias.

## Conclusion

OOD evaluation is complete and the gate fails. The current CelebA-trained feature stack should not be presented as a deployable forensics detector.

The next defensible path is to train or adapt on true manipulated-face data, then rerun the same OOD protocol. Until then, the project result should be reported as a negative transfer finding rather than as a successful reproduction of the proposal's B+C OOD claim.
