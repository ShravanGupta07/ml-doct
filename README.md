# ML Doctor (`mldoct`) 🩺

ML Doctor is a developer tool for finding common PyTorch training-code problems without executing the inspected source. Phase 1 focuses on a stable analysis core and CI-friendly outputs while preserving the existing `check`, `diagnose`, `watch`, and tuner workflows.

## Install
```bash
pip install mldoct
# For the PyTorch auto-tuner:
pip install "mldoct[torch]"
```

## Static check
```bash
mldoctor check train.py
mldoctor check train.py --format json --output report.json
mldoctor check train.py --format sarif --output report.sarif
mldoctor check train.py --fix
mldoctor check train.py --fix --dry-run
```
`check` never imports or executes the target Python file.

## Rules
- MD001 missing `optimizer.zero_grad()` when `backward()` exists.
- MD002 `torch.no_grad()` without a visible `model.eval()`.
- MD003 possible double-softmax with `CrossEntropyLoss`.
- MD004 raw loss-tensor accumulation.
- MD006 scheduler step ordering.
- MD007 missing gradient clipping advisory.
- MD008 DataLoader `num_workers=0` performance advisory.

Findings include severity, confidence, source location, evidence, and whether an automatic fix is available.

## Configuration
Put this in `pyproject.toml` or `.mldoctor.toml`:
```toml
[tool.mldoct]
ignore = ["MD007"]
fail_on = "ERROR"
confidence = 0.70
```

## Diagnose logs
```bash
mldoctor diagnose metrics.csv
mldoctor diagnose ./logs --export-report
```
Required columns are `train_loss`, `val_loss`, and `grad_norm`.

## Watchdog
```bash
mldoctor watch train.py --epochs 20
```
It runs the training script as a child process and watches its output for NaN/Inf, large losses, validation plateaus, and available NVIDIA VRAM trends.

## Auto-tuner
```python
from mldoct.tuner import tune_batch_size
optimal = tune_batch_size(model, dataset_sample_shape=(3, 224, 224))
```
Install `mldoct[torch]` for this feature.

## Important limitation
ML Doctor is heuristic/static analysis software. A clean report is not proof that a model is correct, and a warning is not proof that a training run will fail. Review findings against the actual model, data, and objective.
