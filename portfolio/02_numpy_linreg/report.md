# Week 2 — NumPy linear-models mini-library

## Comparison with scikit-learn

| Model | ||β_ours − β_sk||₂ | |b_ours − b_sk| |
|---|---|---|
| OLS | 0.000e+00 | 0.000e+00 |
| Ridge | 7.891e-16 | 8.327e-17 |
| Lasso | 1.183e-09 | 2.545e-11 |

All three should be ≤ 10⁻³ on this synthetic problem.

## SGD training loss

first-epoch MSE: 15.1170
last-epoch MSE:  1.4419

## Support recovery (lasso, α = 0.1)

true support: [np.int64(0), np.int64(1), np.int64(2), np.int64(3), np.int64(4)]
recovered:    [np.int64(0), np.int64(1), np.int64(2), np.int64(3), np.int64(4)]