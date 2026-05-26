# Hybrid Three-Branch GAN Discriminator — Build Plan

> Last updated: 2026-05-26
> Status: **Week 1 and Week 2 are gate-cleared.** The repository now includes a trained `phase3_a_b_c.pt` checkpoint plus matching run artifacts. **Week 3 remains split**: the Phase 4 fine-tuning path is implemented on the active `2108-D` contract, while branch-combination ensemble experiments and OOD evaluation are still open.

> **2 Engineers · 4 Weeks · OOD Robustness Target: 94.4% balanced accuracy**

---

## Overview

| Phase | Week | Focus                       | Gate                                                            |
| ----- | ---- | --------------------------- | --------------------------------------------------------------- |
| 1     | 1    | Setup + Branch A            | Branch A val acc ≥ 77%, F1 ≥ 0.70; flow cache complete         |
| 2     | 2    | Branches B & C (parallel)   | `phase2_a_b.pt` and `phase3_a_b_c.pt` both saved               |
| 3     | 3    | Phase 4 fine-tune + separate ensemble follow-up | B+C ensemble ≥ 94.4% balanced acc, F1 ≥ 0.93                   |
| 4     | 4    | Eval & hardening            | OOD eval complete; final report written                         |

---

## Progress snapshot (living)

| Phase | Where the repo is now | Next gate |
| ----- | --------------------- | --------- |
| 1 | CelebA at `data/celeba/img_align_celeba` (202,599 images); Branch A encoder and baseline classifier implemented; flow cache complete at `data/flow_cache` (202,599 `*_flow.pt` files, ~7.0 GB, shape `(2, 64, 64)` float32); `test_flow_precompute_smoke` passing; loader supports same-identity real pairs, cross-identity proxy fakes, singleton-adjacent fallback, and attribute-derived pseudo-identities when true identity labels are unavailable | Keep the saved Branch A baseline and reports aligned with the stronger proxy-task configuration |
| 2 | Branch B (`models/branch_b.py`) and the full Phase 2 A+B stack are implemented; Run 3 now shares Branch A's encoder, uses the committed 8-D summary `[vel_mean, vel_std, vel_max, vel_min, cos_sim, l2_dist, sign_consistency, abs_vel_mean]`, expands it to 32-D before fusion, and partially unfreezes the shared encoder tail. Branch C (`models/branch_c.py`), `DiscriminatorPhase3`, hinge loss, flow-aware `adjacent_cache` loading, checkpoint resume helpers, and Phase 3 CLI/trainer wiring are implemented and trained. `checkpoints/phase3_a_b_c.pt` matches `runs/phase3_a_b_c_w2/benchmark_summary.json`, with the best validation result at epoch `8`: balanced accuracy `0.8741`, F1 `0.9067`, AUC-ROC `0.9484`, loss `0.2726`. | Start Week 3 ensemble work from the trained Phase 3 baseline |

Update this table when a gate flips so the plan stays honest for the next work session.

---

## Delivery process

**How this plan is used**

- **Checklists** track scope; **gates** (Overview table) decide when to move to the next week. Do not start Week 3 ensemble work until both `phase2_a_b.pt` and `phase3_a_b_c.pt` exist and clear their gates.
- **Checklist truthfulness:** when a commit materially completes a checklist item, update that checkbox in the same commit. Do not leave completed work unchecked because the broader week gate is still open.
- **Dependency direction:** `data/` → `models/` → `training/` → `evaluation/`. Do not import training scripts from model modules.
- **Vertical slices:** prefer a thin working slice (one branch implemented, wired into a training script, producing a checkpoint) over "all model code first, training later."
- **Freeze discipline:** Branch A conv weights must be verifiably frozen before Phase 2 training begins. Add a unit test that confirms no weight change after an optimizer step. Same rule applies to Branch A + B before Phase 3.
- **Proposal parity discipline:** when docs say "proposal", keep the exact tensor contracts from the project proposal. When code deviates temporarily, call it out explicitly instead of silently rewriting the proposal in documentation.
- **Cache contract:** cached flow filenames are `{frame_a_stem}_flow.pt`, computed against the adjacent-index partner rule used by `data/precompute_flow.py`. The loader's real/fake pair sampling can now diverge from that rule. Before Branch C training, either keep Branch C explicitly on adjacent-index pairing or regenerate the cache for the chosen pairing strategy. Do not silently mix cached flow built for one pairing rule with frame pairs sampled by another.
- **Commit discipline:** land implementation in small commits by subsystem. Default split: shared data contract change, then model implementation, then training script, then tests. Each commit should answer: what boundary advanced, what verification was run.
- **Docs move with the stage:** if a stage changes a runtime contract (e.g. `__getitem__` return signature), update this file's Progress snapshot and any affected docstrings in the same commit.

**Local verification (before calling a task done)**

- Model forward-pass tests: `python -m pytest tests/test_model.py -v`
- Data loader tests: `python -m unittest tests.test_data -v` — includes `test_flow_precompute_smoke`
- Overfit-stop tests: `python -m unittest tests.test_overfit_stop -v`
- Branch A eval tests: `python -m unittest tests.test_branch_a_baseline -v`
- Training dry-run (2 batches): add `--max-batches 2` flag or equivalent guard before a full run
- Checkpoint integrity: load the saved `.pt` and confirm the metric in `benchmark_summary.json` matches the training log
- Early-stop integrity: confirm any truncated run reports `stopped_early=true` and includes a `stop_reason`
- Confusion-matrix integrity: run `python -m training.eval_branch_a --config config/config.yaml --run-name branch_a_test_eval` and confirm both JSON and Markdown reports are written under `runs/`

**Definition of done (default)**

- Unit tests pass for all affected modules.
- Output shape assertion exists for every new branch and committed interface: Branch B summary → `(B, 8)`, current Branch B module output → `(B, 32)`, Branch C → `(B, 28)`, Phase 3 logits → `(B,)`.
- No frozen branch weights change after an optimizer step — verified by test.
- Checkpoint saved with epoch, `model_state_dict`, `optimizer_state_dict`, and best metric.
- `runs/` log updated with the training run for the phase.
- Progress snapshot updated when a gate flips.

**Plan hygiene**

- When a **gate** is met, update the **Progress snapshot** and the phase **Done when** if reality diverged from the original wording.
- **End of each week:** note what shipped and what slipped — one short list in the progress snapshot is enough.

---

## Week 1 — Setup + Branch A `[COMPLETE]`

### Dev 1 — Model & Training

**Goal:** Branch A trains end-to-end. Checkpoint saved and gate-cleared.

- [x] Project scaffold, `config.yaml`, `requirements.txt`
- [x] Experiment tracking setup (TensorBoard or W&B)
- [x] Implement `BranchA_CNN` (`models/branch_a.py`) — 5 conv blocks, SpectralNorm + BN, LeakyReLU(0.2) throughout, 2048-D flatten output
- [x] Implement `DiscriminatorPhase1` (`models/discriminator.py`) — Branch A + fusion FC head (2048 → 512 → 128 → 1)
- [x] Core training loop (`training/trainer.py`), BCE loss
- [x] Unit tests: forward-pass output shape `(B, 1)`, no NaN activations
- [x] Train Branch A — **val balanced acc 1.0000, F1 1.0000 @ epoch 34**
- [x] Save `checkpoints/phase1_branch_a_best.pt` + `runs/branch_a_baseline/benchmark_summary.json`

### Dev 2 — Data & Eval

**Goal:** Dataset validated, flow cache complete, eval module interface defined.

- [x] Validate CelebA: 202,599 images, 178×218 native resolution
- [x] Implement `CelebAFramePairDataset` with adjacent-index fallback pairing (`data/celeba_loader.py`)
- [x] Augmentation pipeline: random horizontal flip, ColorJitter (brightness ±0.1, contrast ±0.1, saturation ±0.05), normalize to [-1, 1]
- [x] Data loader unit tests: shape checks, label balance, no NaN
- [x] Launch Farnebäck flow pre-computation (`data/precompute_flow.py`)
- [x] Flow cache verified: 202,599 files, 0 missing, 0 extra, shape `(2, 64, 64)` float32, ~7.0 GB
- [x] `tests.test_data.DataPipelineTestCase.test_flow_precompute_smoke` passing
- [x] Eval module skeleton (`evaluation/eval.py`) — `compute_balanced_accuracy`, `compute_f1`, `compute_auc_roc` stubs defined

**Done when:** `phase1_branch_a_best.pt` reports val balanced acc ≥ 77% and F1 ≥ 0.70 in `benchmark_summary.json`; flow cache contains exactly 202,599 files and smoke test passes.

**Actual result:** acc 1.0000, F1 1.0000 — gate cleared.

---

## Week 2 — Branches B & C `[COMPLETE]`

Dev 1 owns Branch B. Dev 2 owns Branch C. Both run in parallel, but the remaining architectural question is whether Branch B's implemented 32-D learned expansion is a temporary Phase 2 convenience or the intended long-term contract.

### Dev 1 — Branch B (Spatiotemporal)

**Goal:** Branch B implemented and trained with Branch A frozen. `phase2_a_b.pt` saved.

- [x] Implement `BranchB_Spatiotemporal` (`models/branch_b.py`)
  - Shared embed CNN: `frame_t`, `frame_t1` → 64-D each (tied weights, independent forward passes); LeakyReLU(0.2)
  - `velocity = e_t1 − e_t` (64-D)
  - `curvature = velocity / ‖velocity‖` (64-D, L2-normalized)
  - `acceleration` ≈ second-order approximation (64-D)
  - Aggregate `(mean, std, max)` over each of the three quantities → proposal-level **8-D summary**
  - Current implementation expands that summary through a small learned head to a **32-D output**
- [x] Implement `DiscriminatorPhase2` (`models/discriminator.py`)
  - Load `phase1_branch_a_best.pt`; set Branch A conv `requires_grad = False`
  - Current implementation concat `[branch_a_2048, branch_b_32]` → 2080-D into fusion head (2080 → 512 → 128 → 1)
  - Proposal target for the eventual three-branch model remains `[branch_a_2048, branch_b_8, branch_c_28]` → 2084-D
- [x] Write Phase 2 training script (`training/phase2_train.py`)
  - Optimizer: Adam (β₁=0.5, β₂=0.999), LR = 2e-4; only Branch B + fusion head params
  - Scheduler: CosineAnnealingLR; 20 epochs, batch size 64; loss: BCE
- [x] Unit tests (`tests/test_model.py`)
  - Branch B output shape `(B, 8)` ✓
  - Full Phase 2 forward pass output `(B, 1)` ✓
  - Branch A weights unchanged after optimizer step ✓
- [x] Train Branch B; save `checkpoints/phase2_a_b.pt`

**Gate:** val balanced acc ≥ 88%, F1 ≥ 0.88; Branch A freeze verified by test.

---

### Dev 2 — Branch C (Physics Dynamics)

**Goal:** Branch C implemented and trained with Branch A + B frozen. `phase3_a_b_c.pt` saved.

> **Cache contract:** do not rename or regenerate `*_flow.pt` files during this week unless `identity_CelebA.txt` is explicitly introduced and cache regeneration is intentional. Keep Branch C on adjacent-index pairing to match the existing cache.

- [x] Update `CelebAFramePairDataset.__getitem__` to return `(frame_t, frame_t1, flow_tensor, label)`
  - Load `{frame_a_stem}_flow.pt` from `data/flow_cache/`
  - Unit test: returned flow tensor shape `(2, 64, 64)` ✓, no NaN ✓
- [x] Implement `BranchC_Physics` (`models/branch_c.py`)
  - **Optical flow features (20-D):** load cached dx/dy tensor; compute divergence, curl, gradient magnitude per pixel; aggregate `(mean, std, max, min, range)` over each → 15-D; global stats (mean magnitude, max magnitude, dominant direction histogram bins) → 5-D
  - **HSV photometrics (8-D):** convert `frame_t` and `frame_t1` from [-1,1] to [0,1] → RGB → HSV; per frame: `(mean_H, std_H, mean_S, mean_V)` → 4-D × 2 frames = 8-D
  - **Total: 28-D output**
- [x] Implement `DiscriminatorPhase3` (`models/discriminator.py`)
  - Load `phase2_a_b.pt`; freeze Branch A + Branch B (`requires_grad = False`)
  - If Branch B is reduced back to proposal form: concat `[branch_a_2048, branch_b_8, branch_c_28]` → 2084-D
  - If Branch B keeps the current learned expansion: concat `[branch_a_2048, branch_b_32, branch_c_28]` → 2108-D
  - Pick one contract explicitly before implementation; do not leave Phase 3 ambiguous
- [x] Implement Hinge loss (`training/losses.py`)
  - `L_hinge = E[max(0, 1 − D(x))] + E[max(0, 1 + D(G(z)))]`
- [x] Write Phase 3 training script (`training/phase3_train.py`)
  - Optimizer: Adam (β₁=0.5, β₂=0.999), LR = 2e-4; only Branch C + fusion head params
  - Scheduler: CosineAnnealingLR; 20 epochs, batch size 64; loss: BCE
- [x] Implement checkpoint save/resume (`training/checkpointing.py`, `training/phase3_trainer.py`)
  - Save: epoch, `model_state_dict`, `optimizer_state_dict`, best metric
  - Resume: `--resume checkpoints/<path>.pt`
- [x] Finalize eval module (`evaluation/eval.py`) — replace stubs with real implementations; add `plot_confusion_matrix(y_true, y_pred, save_path)`
- [x] Unit tests (`tests/test_model.py`, `tests/test_data.py`)
  - Branch C output shape `(B, 28)` ✓
  - Full Phase 3 forward pass output `(B, 1)` ✓
  - Branch A + B weights unchanged after Phase 3 optimizer step ✓
- [x] Train Branch C; save `checkpoints/phase3_a_b_c.pt` and matching run reports

**Actual result:** best val balanced acc `0.8741`, F1 `0.9067`, AUC-ROC `0.9484`, loss `0.2726` @ epoch `8` in `runs/phase3_a_b_c_w2/`.

**Gate:** val balanced acc ≥ 83%, F1 ≥ 0.80; Branch A + B freeze verified by test; flow cache still contains exactly 202,599 files after the run.

---

### Week 2 Critical Sync Point

> **End of Week 2** — cleared. `phase2_a_b.pt` and `phase3_a_b_c.pt` both exist, so Week 3 work can proceed from the trained Phase 3 baseline.

---

## Week 3 — Full Ensemble Fine-tune

### Dev 1 — End-to-End Fine-tune

**Goal:** All branches unfrozen and fine-tuned together on the active `2108-D` fusion contract. `phase4_ensemble.pt` saved.

- [x] Implement `DiscriminatorPhase4` (`models/discriminator.py`)
  - Load `phase3_a_b_c.pt`; unfreeze all branches
  - Final fusion FC head stays on the active `2108 → 512 → 128 → 1` contract (`2048 + 32 + 28`)
- [x] Implement combined loss (`training/losses.py`): `L_total = 0.7 × L_BCE + 0.3 × L_hinge`
- [x] Write Phase 4 fine-tune script (`training/phase4_finetune.py`)
  - All parameters trainable; Optimizer: Adam (β₁=0.5, β₂=0.999), LR = **5e-5**
  - Scheduler: CosineAnnealingLR; 20 epochs, batch size 64; combined loss
- [ ] Run Phase 4 training and save `checkpoints/phase4_ensemble.pt`
- [ ] Run all 7 ensemble combination experiments in a separate evaluation branch/scope (see table below)
- [x] Prepare inference handoff artifact for Week 4 eval — `runs/<run>/inference_contract.json`

**Ensemble experiment matrix:**

| # | Branches | Classifier |
| - | -------- | ---------- |
| 1 | A only | Logistic on logit |
| 2 | B only | Logistic on logit |
| 3 | C only | Logistic on logit |
| 4 | A + B | Random Forest |
| 5 | A + C | Random Forest |
| **6** | **B + C** | **Random Forest** ⭐ |
| 7 | A + B + C | Random Forest |

**Gate:** B+C ensemble val balanced acc ≥ 94.4%, F1 ≥ 0.93.

---

### Dev 2 — RF Ensemble + Ablation

**Goal:** RF classifiers trained for all 7 configs. Per-branch ablation and confusion matrix output complete.

- [ ] Implement `evaluation/ensemble.py` in the ensemble-eval branch/scope
  - `extract_branch_outputs(model, dataloader, branch) -> np.ndarray` — shape depends on chosen contract
  - `train_rf_ensemble(features, labels) -> RandomForestClassifier` — `n_estimators=100, random_state=42`
  - `evaluate_ensemble(clf, features, labels) -> dict` — balanced acc, F1, AUC-ROC
- [ ] Run RF ensemble for all 7 branch combinations on held-out test split
- [ ] Per-branch ablation: forward each branch independently, zero others, compute balanced acc / F1 / AUC-ROC
- [ ] Save confusion matrices for all 7 configs to `runs/ensemble_ablation/`

**Done when:** All 7 experiment results are logged; B+C RF ensemble clears the gate; ablation table is written to `runs/ensemble_ablation/`.

---

## Week 4 — Eval & Hardening

### Dev 1 — Architecture Review + Experiment Support

- [ ] Finalize all 7 ensemble results table; confirm B+C is the deployment-recommended config
- [ ] Architecture review: no orphaned branches, no unbounded tensor ops, no missing gradient guards
- [ ] Support OOD eval — load `phase4_ensemble.pt` via inference script, accept image dir, output per-image scores

### Dev 2 — OOD Eval, Profiling, Report

**OOD evaluation (`evaluation/ood_eval.py`):**

- [ ] Assemble OOD test sets: style-transferred faces, face reenactment outputs (e.g. First Order Motion Model), diffusion-based face synthesis (e.g. Stable Diffusion inpainting)
- [ ] Implement `evaluate_ood(model, ood_dataloader, config) -> dict` — balanced acc, F1, AUC-ROC, confusion matrix per OOD category
- [ ] Run B+C ensemble on all OOD categories
- [ ] Run Branch A baseline on same OOD sets for comparison

**Inference profiling:**

- [ ] Profile forward pass latency per branch (CPU + GPU): Branch A ms/image, Branch B (embed × 2 + delta) ms/image, Branch C (flow load + feature extraction) ms/image, full ensemble ms/image
- [ ] If flow pre-compute is a bottleneck: parallelize with `multiprocessing.Pool`; evaluate CUDA Farnebäck if GPU is available

**Final eval report:**

- [ ] Consolidated results table — all 7 ensemble configs × in-domain + all OOD categories
- [ ] Per-branch ablation table
- [ ] Confusion matrices (in-domain + per OOD category)
- [ ] Inference time profile
- [ ] Deployment recommendation: **B+C ensemble** confirmed as production config with written rationale

**Done when:** OOD eval complete for all three OOD categories; final report written and committed to `docs/`; inference profile logged to `runs/`.

---

## Architecture Reference

This reference table describes the proposal target. The current repository only implements Branch A and the A+B Phase 2 subset, where Branch B is expanded to `32-D` before fusion.

### Branch Dimensions

| Branch | Dim | Signal |
| ------ | --- | ------ |
| A — CNN Spatial | 2048-D | Static texture & structure (5 conv blocks, SpectralNorm + BN) |
| B — Spatiotemporal | 8-D | Shared-encoder temporal stats over embed delta: velocity summary + cosine/L2/sign consistency |
| C — Physics Dynamics | 28-D | Optical flow div/curl/grad (20-D) + HSV photometrics (8-D) |
| **Concatenated (active runtime)** | **2108-D** | Fusion FC input |

### Hyperparameters

| Parameter | Phases 1–3 | Phase 4 |
| --------- | ---------- | ------- |
| Optimizer | Adam (β₁=0.5, β₂=0.999) | same |
| Learning rate | 2e-4 | **5e-5** |
| Batch size | 64 | 64 |
| Epochs | 20 | 20 |
| Scheduler | CosineAnnealingLR | same |
| Loss | BCE | 0.7 × BCE + 0.3 × Hinge |
| Dropout (fusion) | 0.3 | 0.3 |

---

## Checkpoint Registry

| File | Phase | Contents | Status |
| ---- | ----- | -------- | ------ |
| `phase1_branch_a_best.pt` | 1 | Branch A conv + FC | Gate cleared: acc ≥ 77%, F1 ≥ 0.70 |
| `phase2_a_b.pt` | 2 | Current Phase 2 baseline in this workspace; verify provenance before reuse because older legacy Phase 2 runs also exist locally | Gate cleared in prior run: acc ≥ 88%, F1 ≥ 0.88 |
| `phase3_a_b_c.pt` | 3 | A + B (frozen) + Branch C + FC | Gate cleared at epoch 8: balanced acc `0.8741`, F1 `0.9067`, AUC-ROC `0.9484`, loss `0.2726` |
| `phase4_ensemble.pt` | 4 | All branches unfrozen, fine-tuned | Not created yet; target B+C ≥ 94.4%, F1 ≥ 0.93 |

---

## Expected Final Results

| Configuration | Auth % | Synth % | F1 | Notes |
| ------------- | ------ | ------- | -- | ----- |
| Branch A only | 77.8% | 77.8% | 0.70 | Phase 1 gate |
| Branch B only | 88.9% | 94.4% | 0.91 | |
| Branch C only | 83.3% | 83.3% | 0.80 | |
| A + B | 89.5% | 89.5% | 0.88 | |
| A + C | 88.9% | 88.9% | 0.85 | |
| **B + C** | **94.4%** | **94.4%** | **0.93** | ⭐ Deploy target |
| A + B + C | 89.5% | 89.5% | 0.86 | Branch A dilutes OOD robustness |

> Branch A introduces in-distribution bias that degrades OOD performance when added to the B+C ensemble. B+C is the deployment-recommended configuration.

---

## Risk Register

| Risk | Likelihood | Impact | Mitigation |
| ---- | ---------- | ------ | ---------- |
| Branch A dominates gradients in Phase 4 | High | High | Phased freeze ensures independent feature learning; gradient scaling if Phase 4 still shows Branch A dominance |
| `identity_CelebA.txt` introduced mid-cache | Medium | High | Keep Branch C on adjacent-index pairing OR regenerate cache before Phase 3 training; never silently mix pairing strategies |
| Shared Branch B/Branch A encoder tail overfits or drifts BN stats during Phase 2 | Medium | Medium | Freeze blocks 0-2, keep blocks 3-4 in train mode only, and enforce partial-freeze behavior in unit tests |
| Flow cache corrupted or stems mismatched | Low | High | `test_flow_precompute_smoke` must pass before Phase 3 train; verify file count after every run |
| OOD test sets unavailable in Week 4 | Medium | Medium | Source and stage OOD data during Week 3 in parallel with ensemble training |
| Overfitting to CelebA in Phase 4 | Medium | High | OOD eval mandatory before Phase 4 sign-off; do not close Week 4 without OOD numbers |

---

## Standing Rules

- **Freeze before you train.** No phase training begins without a passing unit test confirming prior branch weights are frozen.
- **Cache contract is inviolable.** `{frame_a_stem}_flow.pt`, shape `(2, 64, 64)`, adjacent-index partner rule. Any deviation is an explicit decision requiring cache regeneration.
- **B+C is the deployment config.** Do not optimize A+B+C metrics at the cost of B+C robustness.
- **OOD eval is not optional.** Week 4 is not done until OOD numbers exist for all three OOD categories.
- **Checkpoints are the handoff artifact.** Each week ends with a saved checkpoint. If the gate is not cleared, the checkpoint is still saved and the miss is noted in the progress snapshot.
- **One training script per phase.** Do not fold phases into one script; each script is its own audit trail.

---

## Weekly rhythm

| Day | Activity |
| --- | -------- |
| **Monday** | Review prior week gate; pick concrete tasks for the week; confirm prior checkpoint loads cleanly before writing new code |
| **Wednesday** | Mid-week check — if a branch is not converging, decide to adjust LR or descope; do not let one failing branch block the other |
| **Thursday** | Integrate risky pieces (freeze tests, data loader contract changes) so Friday is not the first time they run together |
| **Friday** | Full training dry-run or checkpoint validation on physical hardware; update **Progress snapshot**; note what shipped / slipped |
