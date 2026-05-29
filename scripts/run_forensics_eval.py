"""Run Week 4 forensics OOD evaluation.

Smoke:
  python scripts/run_forensics_eval.py \
    --config config/config.yaml \
    --device cpu \
    --dataset "Data Set 1" \
    --limit 128 \
    --max-batches 4

Full:
  python scripts/run_forensics_eval.py \
    --config config/config.yaml \
    --checkpoint checkpoints/phase3_a_b_c.pt \
    --phase4-checkpoint checkpoints/phase4_ensemble.pt \
    --forensics-root data/forensics \
    --split test \
    --device mps
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evaluation.ood_eval import main


if __name__ == "__main__":
    main(sys.argv[1:])
