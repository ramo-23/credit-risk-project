# Model Performance Report

## Summary
- Training rows: 1834515
- Validation rows: 426149

## Subsampling and Tuning Rationale
- Subsampling: stratified subsample used to reduce compute burden during hyperparameter search.
- Fraction: 15% of training set used (configurable).
- Tuning: limited grid on `C` (0.1, 1.0, 10.0), L2 penalty, class_weight="balanced", 3-fold CV.
- Reason: dataset is large; full CV across millions of rows is computationally prohibitive. Subsampling preserves target rate and yields stable hyperparameter estimates while saving compute.

## Tuning Results
- Best C (from subsample): 10.0

## Validation Metrics
### Baseline
- AUC: 0.9298825185264974
- KS: 0.7271533702518134
- Precision (0.5): 0.03754041391919233
- Recall (0.5): 0.9491842284160436
- Brier score: 0.24936300758368501

### Tuned
- AUC: 0.9298812685463429
- KS: 0.7271343346433282
- Precision (0.5): 0.03754041391919233
- Recall (0.5): 0.9491842284160436
- Brier score: 0.24936822995621669

## Model Selection Decision
- Tuned model AUC improvement is marginal (delta=-0.0000). Prefer baseline for stability and interpretability.

## Reproducibility
- Random seed used: 42
- Subsample fraction: 0.15
- Parameter grid: C in [0.1, 1.0, 10.0]
