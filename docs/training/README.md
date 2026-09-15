# Training

This section documents how the DepthWizard depth-estimation backbone was fine-tuned on the GAMUS/DFC2019 overhead dataset, kept as genuine artifacts from the actual training runs.

## Methodology & plans

- [GAMUS training plan](../methodology/GAMUS_TRAINING_PLAN.md) — dataset strategy, loss design, and hyperparameters.
- [RTX 4060 execution plan](RTX4060_EXECUTION_PLAN.md) — GPU/memory planning for the fine-tuning runs.

## Reports

- [Overnight training report](reports/OVERNIGHT_TRAINING_REPORT.md) — 100% GAMUS run (12 epochs, 3.6 h).
- [Precision fine-tuning report](reports/PRECISION_TRAINING_REPORT.md) — Stage 2 precision fine-tune (3 epochs).

## Raw logs (unmodified)

- [training_full_overnight.log](logs/training_full_overnight.log) — raw execution log of the overnight run.
- [training_precision.log](logs/training_precision.log) — raw execution log of the precision run.

## Archives (preserved as-is)

- [run1/](archives/run1/) — first full training run checkpoints/logs.
- [failed-precision-run/](archives/failed-precision-run/) — an earlier precision run that produced a non-converging checkpoint; kept to preserve an honest record of the training history.

## Reproducing

Run the training loop from the repository root:

```bash
python backend/training/train_gamus_full.py
```

Validation and benchmark scripts live in [`backend/training/`](../../backend/training/).