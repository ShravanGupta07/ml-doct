# Changelog

## 0.3.0 — Phase 1
- Added structured diagnostics with stable rule IDs.
- Added JSON and SARIF output for CI.
- Added configuration via `[tool.mldoct]`.
- Made auto-fixing syntax-checked, atomic, and backup-preserving.
- Fixed diagnosis feature extraction to match the training model's ratio-based `t_loss_drop`.
- Added deterministic heuristic fallback when the embedded model cannot load.
- Kept `check`, `diagnose`, `watch`, and tuner APIs compatible.
- Removed `torch` from the base install; tuner users can install `mldoct[torch]`.
